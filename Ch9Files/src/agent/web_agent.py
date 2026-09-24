"""
web_agent.py — Orchestrator for the Supertron multi-vendor web search agent.

Architecture (two-node StateGraph with Send fan-out):

  1. vendor_lookup  [N parallel instances, one per WEBSITES entry]
       Dispatched via LangGraph's Send API from dispatch_to_vendors.
       Each instance runs the selected sub-agent implementation — either
       Nova Act (nova_vendor_agent.py) or Claude ReAct (claude_vendor_agent.py) —
       scoped to one vendor URL. The operator.add reducer on
       OrchestratorState.vendor_results accumulates all per-vendor outputs.

  2. synthesize
       Runs after all vendor_lookup instances finish. Calls Claude to merge all
       per-vendor JSON results into a single consolidated parts array and markdown
       comparison table.

Sub-agent selection is controlled by the sub_agent field in OrchestratorState
("nova-act" or "claude"). Pass it via run_agent(sub_agent=...) or as the
--sub-agent flag in run_agent.py.

The orchestrator graph uses AgentCoreMemorySaver for checkpoint resilience.

Entry points:
  run_agent()  — Synchronous wrapper. Used by run_agent.py for local CLI runs.
  invoke()     — @app.entrypoint registered with BedrockAgentCoreApp. The
                 deployment artifact: AgentCore Runtime calls this via
                 POST /invocations on the container.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import uuid
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from langgraph.checkpoint.memory import MemorySaver
from langgraph_checkpoint_aws import AgentCoreMemorySaver

from agent.claude_vendor_agent import run_vendor_agent as run_claude_vendor_agent
from agent.config import EXAMPLE_MISSIONS, MODEL_ID, REGION, WEBSITES
from agent.state import OrchestratorState, VendorLookupInput
from agent.tools.code_interpreter import new_session_id, stop_session
from agent.nova_vendor_agent import run_vendor_agent as run_nova_act_vendor_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_synthesis_llm():
    """Lazily build (and cache) the model client used by the synthesize node."""
    return init_chat_model(
        MODEL_ID,
        model_provider="bedrock_converse",
        region_name=REGION,
        temperature=0,
        max_tokens=4096,  # Higher than sub-agents: synthesize merges N vendor outputs.
    )


# ── Orchestrator nodes ────────────────────────────────────────────────────────


def dispatch_to_vendors(state: OrchestratorState) -> list[Send]:
    """
    Routing function: fan out to one vendor_lookup node per website.

    Returns a list of Send objects — one per entry in WEBSITES. LangGraph
    runs all vendor_lookup instances in parallel, collecting their outputs
    via the operator.add reducer on OrchestratorState.vendor_results.

    The ci_session_id (a single shared Code Interpreter session for Claude
    runs) is created in _run_async and passed through state, so that the
    orchestrator can stop the session in a finally block after the graph
    completes.
    """
    sub_agent = state["sub_agent"]
    ci_session_id = state["ci_session_id"]
    if sub_agent == "claude":
        logger.info(
            "Dispatching %d Claude vendor sub-agents (CI session: %s).",
            len(WEBSITES),
            ci_session_id,
        )
    else:
        logger.info("Dispatching %d Nova Act vendor sub-agents.", len(WEBSITES))
    return [
        Send(
            "vendor_lookup",
            VendorLookupInput(
                vendor_url=url,
                mission=state["mission"],
                sub_agent=sub_agent,
                ci_session_id=ci_session_id,
            ),
        )
        for url in WEBSITES
    ]


async def vendor_lookup(state: VendorLookupInput) -> dict:
    """
    Run a single-vendor sub-agent and accumulate its result.

    Routes to the Nova Act or Claude implementation based on state["sub_agent"].
    Called once per vendor URL, all instances running in parallel.
    Returns {"vendor_results": [one_result]} — the operator.add reducer
    on OrchestratorState.vendor_results merges all of these lists.
    """
    if state["sub_agent"] == "claude":
        result = await run_claude_vendor_agent(
            vendor_url=state["vendor_url"],
            mission=state["mission"],
            ci_session_id=state["ci_session_id"],
        )
    else:
        result = await run_nova_act_vendor_agent(
            vendor_url=state["vendor_url"],
            mission=state["mission"],
        )
    return {"vendor_results": [result]}


async def synthesize(state: OrchestratorState) -> dict:
    """
    Consolidate all per-vendor results into a single comparison table.

    Runs after all vendor_lookup instances complete. Calls the model with the
    full set of vendor sub-agent outputs and asks it to:
    - Extract and merge all JSON parts arrays.
    - Return a consolidated JSON array and a markdown table.
    """
    model = _get_synthesis_llm()

    vendor_summaries = "\n\n---\n\n".join(
        f"Vendor URL: {r.get('vendor_url', 'unknown')}\n"
        + (r.get("raw") or f"Error: {r.get('error', 'no output')}")
        for r in state["vendor_results"]
    )

    prompt = (
        f"You are consolidating memory component search results from "
        f"{len(state['vendor_results'])} vendor sub-agents.\n\n"
        f"## Original Mission\n{state['mission']}\n\n"
        f"Each sub-agent output below contains either a JSON parts array, a "
        f"no-match status, or an error.\n\n"
        f"{vendor_summaries}\n\n"
        "Extract all JSON parts arrays from the sub-agent outputs above. "
        "Merge them into a single deduplicated list.\n\n"
        "Return the following, in this order:\n\n"
        "1. Restate the original mission in one sentence so the report is "
        "self-contained.\n\n"
        "2. A markdown table summarising all matching parts found:\n"
        "   | Vendor | Part Number | Speed | Package | Detail Page | Datasheet |\n"
        "   |--------|-------------|-------|---------|-------------|----------|\n\n"
        "3. A brief assessment of how well the results match the original mission "
        "requirements — note any gaps (e.g. missing speed grades, vendors that "
        "returned errors, parts that only partially match).\n\n"
        "Do not include a JSON array in your response — the structured data is "
        "written to a separate file.\n\n"
        "If no matching parts were found across any vendor, restate the mission, "
        "list which vendors were searched, and explain why no matches were returned."
    )

    response = await model.ainvoke([HumanMessage(content=prompt)])
    logger.info("Synthesis complete.")
    return {"final_result": response.content}


# ── Graph construction ────────────────────────────────────────────────────────


def build_orchestrator():
    """
    Build and compile the orchestrator StateGraph.

    Topology:
      START
        ↓  [dispatch_to_vendors routing function — Send fan-out]
      vendor_lookup × N  [parallel, one per WEBSITES entry]
        ↓  [all complete → operator.add reducer accumulates results]
      synthesize
        ↓
      END

    The AgentCoreMemorySaver checkpoints the orchestrator state after each
    completed node. If the process crashes mid-run, restart with the same
    thread_id to resume from the last checkpoint. (thread_id and sub_agent
    flow through the runtime config and OrchestratorState respectively, so
    this builder takes no per-run arguments — one compiled graph serves
    every invocation.)

    When AGENTCORE_MEMORY_ID is unset we fall back to in-process MemorySaver,
    which does NOT survive process exit. A deployed runtime should always
    have AGENTCORE_MEMORY_ID injected as an environment variable; if it's
    missing on a deployed run, log a warning so the silent loss of resume
    capability is visible.
    """
    agentcore_memory_id = os.environ.get("AGENTCORE_MEMORY_ID")
    if agentcore_memory_id:
        checkpointer = AgentCoreMemorySaver(agentcore_memory_id, region_name=REGION)
    else:
        # Best-effort detection of "running on the deployed runtime" so we can
        # warn loudly when checkpointing has silently degraded. AWS_EXECUTION_ENV
        # is the convention for Lambda; AgentCore Runtime does not document a
        # dedicated env var, but the CONTAINER_CREDENTIALS_FULL_URI / *_RELATIVE_URI
        # variables are present whenever the container picks up its task role
        # via the ECS metadata endpoint, which is the case here. Either signal
        # is enough to surface the warning; absence just means we stay quiet
        # in normal local development.
        on_deployed_runtime = (
            os.environ.get("AWS_EXECUTION_ENV")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        )
        if on_deployed_runtime:
            logger.warning(
                "AGENTCORE_MEMORY_ID is not set on a deployed runtime. "
                "Falling back to in-process MemorySaver — checkpoints will "
                "NOT survive container restart."
            )
        checkpointer = MemorySaver()

    builder = StateGraph(OrchestratorState)
    builder.add_node("vendor_lookup", vendor_lookup)
    builder.add_node("synthesize", synthesize)

    builder.add_conditional_edges(START, dispatch_to_vendors, ["vendor_lookup"])
    builder.add_edge("vendor_lookup", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)


# ── Run agent ─────────────────────────────────────────────────────────────────


def run_agent(
    mission: str,
    thread_id: str | None = None,
    actor_id: str = "default",
    sub_agent: str = "nova-act",
) -> dict[str, Any]:
    """
    Run the multi-vendor web search orchestrator end-to-end.

    Synchronous wrapper around the async orchestrator. Used by run_agent.py
    for local CLI runs and by the BedrockAgentCoreApp entrypoint below.

    Args:
        mission:    The component specification lookup task.
        thread_id:  LangGraph thread ID for checkpointing. Auto-generated if None.
        actor_id:   AgentCore Memory actor identity. Stable across runs for a
                    given user/tenant; defaults to "default" when unauthenticated.
        sub_agent:  Sub-agent implementation to use: "nova-act" (default) or "claude".

    Returns:
        Dict with 'final_result', 'vendor_results', and 'thread_id'.
    """
    return asyncio.run(_run_async(mission, thread_id, actor_id, sub_agent))


async def _run_async(
    mission: str,
    thread_id: str | None,
    actor_id: str = "default",
    sub_agent: str = "nova-act",
) -> dict[str, Any]:
    thread_id = thread_id or str(uuid.uuid4())
    logger.info("Starting orchestrator. Sub-agent: %s. Thread: %s", sub_agent, thread_id)

    # Provision a single Code Interpreter session up front for Claude runs
    # so all parallel vendor sub-agents share state. Stopping it in the
    # finally block releases the underlying microVM as soon as the run ends
    # rather than waiting on the 15-minute idle timeout. The boto3 call is
    # synchronous so we run it on a worker thread to avoid blocking the loop
    # while the AgentCore service spins up the sandbox.
    ci_session_id = (
        await asyncio.to_thread(new_session_id) if sub_agent == "claude" else ""
    )

    try:
        graph = build_orchestrator()

        config = {"configurable": {"thread_id": thread_id, "actor_id": actor_id}}

        initial_state: OrchestratorState = {
            "mission": mission,
            "sub_agent": sub_agent,
            "ci_session_id": ci_session_id,
            "vendor_results": [],
            "final_result": None,
        }

        final_state = await graph.ainvoke(initial_state, config=config)

        logger.info(
            "Orchestrator complete. Vendors searched: %d. Thread: %s",
            len(final_state.get("vendor_results", [])),
            thread_id,
        )
        return {
            "final_result": final_state.get("final_result"),
            "vendor_results": final_state.get("vendor_results", []),
            "thread_id": thread_id,
        }
    finally:
        if ci_session_id:
            await asyncio.to_thread(stop_session, ci_session_id)


# ── AgentCore Runtime entry point ────────────────────────────────────────────
#
# AgentCore Runtime is an HTTP service, not Lambda. The deployed container
# serves POST /invocations and GET /ping on port 8080. The BedrockAgentCoreApp
# class wires both endpoints automatically; @app.entrypoint registers the
# function the runtime calls when an /invocations request arrives. Streaming
# responses are returned as SSE events on the same /invocations route. The SDK
# also registers a /ws WebSocket route, but it only serves traffic when an
# @app.websocket handler is registered — for our entrypoint-only agent it is
# present but unused.
#
# The runtime injects the session ID as the X-Amzn-Bedrock-AgentCore-Runtime-
# Session-Id HTTP header. The SDK exposes it as context.session_id; the second
# parameter MUST be named `context` — the SDK matches on parameter name (see
# BedrockAgentCoreApp._takes_context) and silently passes only the payload if
# you call it anything else. actor_id is NOT injected by the runtime; we treat
# it as application-level metadata supplied via the payload.

app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload: dict, context: Any) -> dict:
    """
    AgentCore Runtime invocation handler.

    Defined async so we can await _run_async directly. Wrapping it with
    asyncio.run inside a sync handler would crash if the Starlette app moves
    sync entrypoints onto the main event loop in a future SDK release.

    Supported payload fields:
        mission:    Component specification lookup task. Defaults to EXAMPLE_MISSIONS[0].
        thread_id:  LangGraph thread ID for resumable runs. Defaults to the
                    runtime session ID, so a caller that reuses runtimeSessionId
                    automatically resumes from the last checkpoint.
        actor_id:   Application-level identifier for AgentCore Memory namespacing.
                    Defaults to "default".
        sub_agent:  "nova-act" (default) or "claude".
    """
    mission = payload.get("mission", EXAMPLE_MISSIONS[0])
    thread_id = payload.get("thread_id") or getattr(context, "session_id", None)
    actor_id = payload.get("actor_id", "default")
    sub_agent = payload.get("sub_agent", "nova-act")

    logger.info(
        "Runtime invocation. session_id=%s thread_id=%s sub_agent=%s",
        getattr(context, "session_id", None), thread_id, sub_agent,
    )

    return await _run_async(
        mission=mission,
        thread_id=thread_id,
        actor_id=actor_id,
        sub_agent=sub_agent,
    )


if __name__ == "__main__":
    # Starts the HTTP server on :8080 (host 0.0.0.0 inside Docker, 127.0.0.1
    # locally). Use this to test the runtime contract end-to-end without
    # deploying:  curl -X POST http://localhost:8080/invocations -d '{...}'
    app.run()

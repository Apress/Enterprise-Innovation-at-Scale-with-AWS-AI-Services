"""
code_interpreter.py — AgentCore Code Interpreter wrapper.

The Code Interpreter provides a managed, sandboxed Python execution environment
with a standard data science stack (pandas, numpy, matplotlib, json, re).

Key properties:
- Session persistence: pass the same `sessionId` across multiple `executeCode`
  calls and variables, imports, and DataFrames remain in memory.
- File I/O: `writeFiles` saves up to 100 MB inline; larger output goes to S3.
- Six operations exposed by the service: executeCode, executeCommand, readFiles,
  writeFiles, listFiles, removeFiles. This wrapper uses executeCode (always),
  plus writeFiles / readFiles for staging large payloads — see _write_file and
  _read_file below.
- Always check `stderr` and `exitCode` in the response — a non-zero exit code
  means the code failed even if `stdout` has content.

boto3 client name: 'bedrock-agentcore'
Code Interpreter identifier: 'aws.codeinterpreter.v1'
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import re
from typing import Any

import boto3
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from agent.config import MODEL_ID, REGION

CODE_INTERPRETER_ID = "aws.codeinterpreter.v1"

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_codegen_llm():
    """Lazily build (and cache) the LLM used to generate processing scripts.

    Hoisted out of the per-call path so N parallel vendor sub-agents share one
    client instead of constructing one per tool invocation.
    """
    return init_chat_model(
        MODEL_ID,
        model_provider="bedrock_converse",
        region_name=REGION,
        temperature=0,
        max_tokens=2048,
    )


# ── Low-level Code Interpreter helpers ───────────────────────────────────────


@functools.lru_cache(maxsize=4)
def _get_client(region: str = REGION) -> Any:
    """Return a boto3 client for the AgentCore Code Interpreter service.

    Cached per region so that N parallel vendor sub-agents share one client
    instead of building a fresh one on every executeCode/writeFiles call.
    """
    return boto3.client("bedrock-agentcore", region_name=region)


def _execute_code(
    code: str,
    session_id: str,
    region: str = REGION,
) -> dict:
    """
    Execute Python code in the Code Interpreter session and return the result.

    Args:
        code: Python source code to execute.
        session_id: Persistent session ID — reuse across calls to retain state.
        region: AWS region.

    Returns:
        Dict with keys: stdout, stderr, exitCode, executionTime (ms).
        These values are extracted from the structuredContent field of the
        streaming event stream returned by invoke_code_interpreter.
    """
    client = _get_client(region)

    response = client.invoke_code_interpreter(
        codeInterpreterIdentifier=CODE_INTERPRETER_ID,
        sessionId=session_id,
        name="executeCode",
        arguments={"code": code, "language": "python"},
    )

    # The response is a streaming event stream. Each `result` chunk has two
    # parallel views of the same execution:
    #   - `content`: a list of typed text blocks ([{"type":"text","text":...}])
    #     suitable for direct printing to a console.
    #   - `structuredContent`: a dict with stdout/stderr/exitCode/executionTime
    #     fields suitable for programmatic consumption.
    # We use the structured form because we need exitCode and stderr separately
    # to decide whether the script succeeded.
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    exit_code = 0
    execution_time = 0

    for event in response.get("stream", []):
        if "result" in event:
            content = event["result"].get("structuredContent", {})
            if content.get("stdout"):
                stdout_parts.append(content["stdout"])
            if content.get("stderr"):
                stderr_parts.append(content["stderr"])
            if "exitCode" in content:
                exit_code = content["exitCode"]
            if "executionTime" in content:
                execution_time = content["executionTime"]

    result = {
        "stdout": "".join(stdout_parts),
        "stderr": "".join(stderr_parts),
        "exitCode": exit_code,
        "executionTime": execution_time,
    }

    if result["exitCode"] != 0:
        logger.warning(
            "Code Interpreter exit code %d. stderr: %s",
            result["exitCode"],
            result["stderr"][:500],
        )
    return result


def _write_file(
    session_id: str,
    filename: str,
    content_bytes: bytes,
    region: str = REGION,
) -> str:
    """
    Write a binary file into the Code Interpreter session filesystem.

    Use this to stage large payloads (e.g. a multi-page catalog dump) for
    processing instead of embedding them in the generated code string. The
    returned path can then be read from inside executeCode with open().

    All filesystem operations are dispatched through invoke_code_interpreter
    with `name="writeFiles"` (or `"readFiles"`); there is no top-level
    `client.write_files` on the boto3 client.

    Returns the path within the session where the file was written.
    """
    client = _get_client(region)
    client.invoke_code_interpreter(
        codeInterpreterIdentifier=CODE_INTERPRETER_ID,
        sessionId=session_id,
        name="writeFiles",
        arguments={
            "content": [
                {
                    "path": filename,
                    "blob": base64.standard_b64encode(content_bytes).decode("utf-8"),
                }
            ]
        },
    )
    return f"/tmp/{filename}"


def _read_file(
    session_id: str,
    filename: str,
    region: str = REGION,
) -> bytes:
    """Read a file from the Code Interpreter session filesystem.

    Dispatched through invoke_code_interpreter with `name="readFiles"`.
    """
    client = _get_client(region)
    response = client.invoke_code_interpreter(
        codeInterpreterIdentifier=CODE_INTERPRETER_ID,
        sessionId=session_id,
        name="readFiles",
        arguments={"paths": [filename]},
    )
    # Walk the streaming event stream and pull the first file blob out.
    for event in response.get("stream", []):
        if "result" in event:
            content = event["result"].get("structuredContent", {})
            files = content.get("content") or content.get("files") or []
            if files:
                return base64.standard_b64decode(files[0]["blob"])
    raise FileNotFoundError(
        f"Code Interpreter readFiles returned no content for {filename!r}"
    )


# ── LangGraph tool ───────────────────────────────────────────────────────────


def make_code_interpreter_tool(session_id: str):
    """
    Build a `process_with_code_interpreter(data, task)` tool bound to a session.

    The session_id is closed over at construction time rather than being passed
    as a tool argument. Threading the session ID through the model — first
    embedding it in the human message, then trusting the model to copy it
    correctly into every tool call — is fragile (the model can drop it,
    paraphrase it, or invent a new one). Binding it here removes that whole
    failure mode, and shortens the tool signature the model has to reason
    about from three arguments to two.

    Args:
        session_id: Code Interpreter session ID. The same ID is reused for
            every invocation of the returned tool, so imports and intermediate
            DataFrames accumulate within one orchestrator run.

    Returns:
        A LangChain @tool — `process_with_code_interpreter(data, task)`.
    """

    @tool
    async def process_with_code_interpreter(data: str, task: str) -> str:
        """
        Generate and execute Python code to process web-extracted data.

        Calls the model to write a Python script that performs the given task
        on the provided data, then runs that script in the AgentCore Code
        Interpreter sandbox. Returns the structured result.

        Output contract: the generated code must print a JSON result to stdout.
        The tool returns that JSON (parts array or Error object), or an error
        dict if execution fails.

        Args:
            data: Raw text or JSON extracted by a browser tool.
            task: Natural-language description of what to do with the data
                (e.g. "Extract all DDR4 part numbers, speed grades, and
                datasheet URLs into a JSON array").

        Returns:
            JSON string with keys: result (the stdout text, which is the JSON
            the generated script printed) and exitCode. On execution failure:
            JSON with keys 'error' and 'exitCode'.
        """
        llm = _get_codegen_llm()
        DATA_LIMIT = 4000
        if len(data) > DATA_LIMIT:
            logger.info(
                "Data truncated from %d to %d chars for code-gen prompt.",
                len(data), DATA_LIMIT,
            )
        code_prompt = (
            f"Write a Python script that performs the following task on the provided data.\n\n"
            f"Task: {task}\n\n"
            f"Data (may be raw text or JSON):\n```\n{data[:DATA_LIMIT]}\n```\n\n"
            "Requirements:\n"
            "- Import only stdlib modules (json, re, etc.).\n"
            "- Print the final result as a single JSON object or array to stdout using print(json.dumps(...)).\n"
            "- If no matching data is found, print: print(json.dumps({'Error': 'No matching parts found'}))\n"
            "- Output only the Python code — no markdown fences, no explanation."
        )
        response = await llm.ainvoke(code_prompt)
        code = response.content.strip()

        # Strip markdown code fences if the model included them despite being asked not to.
        # re.search (not re.sub with ^) finds the fence anywhere in the string, so this
        # works even when the model prefixes the code block with an explanation sentence.
        fence_match = re.search(r"```(?:\w+)?\n(.*?)```", code, re.DOTALL)
        if fence_match:
            code = fence_match.group(1)
        code = code.strip()

        # boto3 has no native async API, so wrap the blocking call in
        # asyncio.to_thread to keep the event loop free while N parallel
        # vendor sub-agents execute code concurrently.
        logger.info(
            "Executing generated code in Code Interpreter (session %s).", session_id
        )
        exec_result = await asyncio.to_thread(_execute_code, code, session_id)

        if exec_result["exitCode"] != 0:
            return json.dumps(
                {"error": exec_result["stderr"], "exitCode": exec_result["exitCode"]}
            )

        return json.dumps(
            {
                "result": exec_result["stdout"].strip(),
                "exitCode": exec_result["exitCode"],
            }
        )

    return process_with_code_interpreter


# ── Utility ───────────────────────────────────────────────────────────────────


def new_session_id() -> str:
    """Start a new AgentCore Code Interpreter session and return its ID.

    The session ID is assigned by the AgentCore service, not generated locally.
    Callers must pass this ID to every subsequent InvokeCodeInterpreter call to
    retain state (variables, imports, DataFrames) across tool invocations.

    Caller is responsible for stopping the session via stop_session() when done.
    Sessions also auto-terminate after sessionTimeoutSeconds (15 min by default).
    """
    client = _get_client()
    response = client.start_code_interpreter_session(
        codeInterpreterIdentifier=CODE_INTERPRETER_ID,
        name="supertron-code-interpreter",
        sessionTimeoutSeconds=900,  # 15 min; max is 28800 (8 hr)
    )
    return response["sessionId"]


def stop_session(session_id: str, region: str = REGION) -> None:
    """Stop a Code Interpreter session to release the underlying microVM.

    Idle sessions are billed until the 15-minute timeout, so explicitly stopping
    a session as soon as the orchestrator finishes can save compute cost on
    long-running deployments.
    """
    if not session_id:
        return
    client = _get_client(region)
    try:
        client.stop_code_interpreter_session(
            codeInterpreterIdentifier=CODE_INTERPRETER_ID,
            sessionId=session_id,
        )
        logger.info("Stopped Code Interpreter session %s.", session_id)
    except Exception as exc:
        # Don't let cleanup failures mask the original orchestrator result.
        logger.warning("Failed to stop CI session %s: %s", session_id, exc)

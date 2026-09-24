"""
state.py — TypedDicts for the Supertron multi-agent memory search.

There are three state types:

  OrchestratorState — flows through the top-level orchestrator graph.
  VendorLookupInput — the sub-state each vendor_lookup node receives via Send.
  VendorResult      — the output each vendor sub-agent returns.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict


class VendorResult(TypedDict, total=False):
    """
    Output from a single vendor sub-agent invocation.

    On success: vendor_url and raw are set (raw is a JSON string).
    On failure: vendor_url and error are set.
    """

    vendor_url: str
    raw: str    # Sub-agent's final message: JSON array or no-match/error status
    error: str  # Set only if the sub-agent raised an unhandled exception


class VendorLookupInput(TypedDict):
    """
    Sub-state passed to each vendor_lookup node via LangGraph's Send API.

    Contains only the fields each vendor_lookup invocation needs. sub_agent
    selects which implementation to use ("nova-act" or "claude").
    ci_session_id is the shared Code Interpreter session for Claude runs;
    it is an empty string for Nova Act runs.
    """

    vendor_url: str
    mission: str
    sub_agent: str
    ci_session_id: str


class OrchestratorState(TypedDict):
    """
    State flowing through the orchestrator graph.

    vendor_results uses operator.add as its reducer so that parallel
    vendor_lookup nodes can each append their result independently —
    LangGraph merges the lists automatically when all parallel branches
    complete before synthesize runs.

    ci_session_id is set once at orchestrator entry (in _run_async) so
    that all parallel vendor sub-agents in a Claude run share one Code
    Interpreter session. The orchestrator stops the session in a finally
    block after the graph completes.
    """

    mission: str
    sub_agent: str
    ci_session_id: str

    # Each vendor_lookup node returns {"vendor_results": [one_result]}.
    # operator.add concatenates all those single-element lists into one.
    vendor_results: Annotated[list[VendorResult], operator.add]

    final_result: Optional[str]

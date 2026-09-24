"""
nova_vendor_agent.py — Single-vendor Nova Act sub-agent.

run_vendor_agent() provisions an AgentCore Browser session, connects Nova Act
to it via CDP, and uses natural-language act() calls to navigate and extract
structured parts data from the vendor's product catalog.

Nova Act manages the full agent loop internally — no LangGraph ReAct graph is
needed at the sub-agent level. Structured output is returned via a Pydantic
schema passed to nova.act(), giving the orchestrator clean JSON without any
post-processing step.
"""

from __future__ import annotations

import json
import logging

from nova_act.asyncio import NovaAct
from nova_act.types.act_errors import ActAgentFailed, ActError, ActExceededMaxStepsError
from nova_act.types.workflow import Workflow
from pydantic import BaseModel

from agent.config import NOVA_ACT_API_KEY, NOVA_ACT_MODEL, NOVA_ACT_PAGE_TIMEOUT, REGION
from agent.tools.browser import browser_cdp_session

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────


def _navigation_prompt(mission: str) -> str:
    return (
        "Navigate to the memory products section of this website and find a parts catalog or "
        "product listing that allows us to browse relevant parts. "
        "The goal is to find the section of the site that would list memory products matching "
        "the following specification:\n\n"
        f"{mission}\n\n"
        "Do not use the site-wide search box — navigate directly to the relevant product "
        "category or parts catalog page. If this site does not sell those parts, stop here."
    )


def _extraction_prompt(mission: str) -> str:
    return (
        "Use the catalog and filters to find all memory parts that match the following "
        f"specification:\n\n"
        f"{mission}\n\n"
        "For each matching part collect: vendor name, part number, speed grade, package type, "
        "datasheet URL, and the HTML product detail page URL (detail_url). "
        "If no dedicated detail page exists, use the catalog page URL for detail_url.\n\n"
        "Important guidelines:\n"
        "- Do not apply attribute filters (density, speed, package) unless you are certain "
        "they include all parts that could match. Show all results first and page through them.\n"
        "- Check for pagination controls (numbered pages, 'Next') and click through every "
        "page — collect all results before finishing.\n"
        "- Do not navigate to PDF or binary file URLs. Record the PDF link directly as "
        "datasheet_url without navigating to it.\n"
        "If no parts match the specification, set status to 'no_matching_parts' and explain "
        "why in the reason field. "
    )


# ── Output schema ─────────────────────────────────────────────────────────────


class MemoryPart(BaseModel):
    vendor: str
    part_number: str
    speed: str
    package: str
    datasheet_url: str = ""
    detail_url: str = ""


class PartsList(BaseModel):
    parts: list[MemoryPart] = []
    status: str = "found"   # "found" | "no_matching_parts" | "error"
    reason: str = ""        # populated when status != "found"


# ── Agent entry point ─────────────────────────────────────────────────────────


_MAX_ATTEMPTS = 2


async def run_vendor_agent(vendor_url: str, mission: str) -> dict:
    """
    Run Nova Act against a single vendor's catalog URL.

    Opens an AgentCore Browser session, attaches Nova Act via CDP, then issues
    two sequential act() calls: one to navigate to the relevant product section,
    and one to extract matching parts as structured JSON.

    Retries once on ActError (SDK-internal failures such as the browser tool
    coroutine serialization bug). Each retry opens a fresh browser session so
    stale CDP/page state from the failed run does not carry over. Definitive
    outcomes (ActAgentFailed, ActExceededMaxStepsError) are not retried.

    Args:
        vendor_url: The vendor's catalog URL to search.
        mission:    The component specification task (natural-language string).

    Returns:
        {"vendor_url": ..., "raw": <JSON string>}  on success
        {"vendor_url": ..., "error": ...}           on unhandled exception
    """
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            logger.info(
                "Vendor sub-agent starting: %s (attempt %d/%d)",
                vendor_url, attempt, _MAX_ATTEMPTS,
            )
            async with browser_cdp_session(label=vendor_url, region=REGION) as (ws_url, ws_headers):
                with Workflow(model_id=NOVA_ACT_MODEL, nova_act_api_key=NOVA_ACT_API_KEY):
                    async with NovaAct(
                        cdp_endpoint_url=ws_url,
                        cdp_headers=ws_headers,
                        starting_page=vendor_url,
                        go_to_url_timeout=NOVA_ACT_PAGE_TIMEOUT,
                    ) as nova:
                        await nova.act(_navigation_prompt(mission))
                        # act_get is the supported API path for structured
                        # responses; act(prompt, schema=...) still works but
                        # is documented as deprecated.
                        result = await nova.act_get(
                            _extraction_prompt(mission),
                            schema=PartsList.model_json_schema(),
                        )

            if result.matches_schema:
                raw = json.dumps(result.parsed_response)
            else:
                raw = json.dumps({
                    "status": "error",
                    "reason": f"Nova Act returned unstructured response: {result.response}",
                })

            logger.info("Vendor sub-agent complete: %s", vendor_url)
            return {"vendor_url": vendor_url, "raw": raw}

        except ActAgentFailed as exc:
            # Nova Act decided the task cannot be completed (e.g. vendor does not
            # carry the requested parts). Treat as no match — do not retry.
            logger.info("Vendor %s: no matching products — %s", vendor_url, exc.message)
            return {"vendor_url": vendor_url, "raw": json.dumps({
                "status": "no_matching_parts",
                "reason": exc.message,
            })}

        except ActExceededMaxStepsError as exc:
            # Nova Act hit its step limit. Log a concise summary — the default
            # __str__ includes step timings, the full prompt, and a feedback URL.
            # Do not retry: more steps on a fresh session is unlikely to help.
            meta = exc.metadata
            detail = (
                f"{meta.num_steps_executed} steps, {meta.time_worked_s}s"
                if meta else "max steps exceeded"
            )
            logger.error(
                "Vendor %s: Nova Act exceeded max steps (%s).",
                vendor_url, detail,
            )
            return {"vendor_url": vendor_url, "error": f"Nova Act exceeded max steps ({detail})"}

        except ActError as exc:
            # SDK-internal error (e.g. browser tool raised an exception mid-await,
            # leaving a coroutine as the return_value which can't be JSON-serialized).
            # Retry with a fresh browser session; if all attempts fail, return error.
            msg = exc.message if hasattr(exc, "message") else str(exc)
            if attempt < _MAX_ATTEMPTS:
                logger.warning(
                    "Vendor %s: Nova Act error on attempt %d/%d (retrying with fresh session) — %s",
                    vendor_url, attempt, _MAX_ATTEMPTS, msg,
                )
                continue
            logger.error(
                "Vendor %s: Nova Act error after %d attempts — %s",
                vendor_url, _MAX_ATTEMPTS, msg,
            )
            return {"vendor_url": vendor_url, "error": msg}

        except Exception as exc:
            logger.exception("Vendor sub-agent failed for %s: %s", vendor_url, exc)
            return {"vendor_url": vendor_url, "error": str(exc)}

    # Unreachable — the loop always returns or continues to exhaustion.
    return {"vendor_url": vendor_url, "error": "all retry attempts exhausted"}

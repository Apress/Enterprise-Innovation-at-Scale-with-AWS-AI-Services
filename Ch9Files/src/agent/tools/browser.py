"""
browser.py — AgentCore Browser Tool wrapper.

The AgentCore Browser Tool spins up an isolated, AWS-managed Chromium
instance. Two context managers are provided:

  browser_cdp_session(label, region)
      Provisions the AgentCore browser and yields (ws_url, ws_headers) —
      the CDP WebSocket endpoint URL and SigV4 auth headers. Use this with
      Nova Act: pass ws_url as cdp_endpoint_url and ws_headers as cdp_headers.

  playwright_browser_session(label, region)
      Wraps browser_cdp_session with a full Playwright connection and yields
      a ready-to-use Playwright Page. Use this for direct Playwright automation.

      Named playwright_browser_session (not browser_session) on purpose: the
      bedrock-agentcore SDK exports its own browser_session helper from
      bedrock_agentcore.tools.browser_client, and shadowing that name here
      would confuse readers cross-referencing the AgentCore docs.

Key characteristics:
- Sessions are NOT resumable. Each context manager call opens a fresh Chromium.
- Default session duration: 15 minutes. Maximum: 8 hours.
- Billed per second of actual CPU and memory consumption — idle time and
  I/O waits don't count.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from bedrock_agentcore.tools.browser_client import BrowserClient
from playwright.async_api import Browser, Page, async_playwright

from agent.config import REGION, get_browser_id

logger = logging.getLogger(__name__)


@asynccontextmanager
async def browser_cdp_session(
    label: str = "", region: str = REGION
) -> AsyncIterator[tuple[str, dict]]:
    """
    Provision an AgentCore Browser session and yield (ws_url, ws_headers).

    ws_url is the SigV4-authenticated CDP WebSocket endpoint; ws_headers
    contains the Authorization header required for the handshake. Pass these
    directly to Nova Act:

        async with browser_cdp_session(label=vendor_url) as (ws_url, ws_headers):
            async with NovaAct(
                cdp_endpoint_url=ws_url,
                cdp_headers=ws_headers,
                ...
            ) as nova:
                ...

    The `label` is used as a prefix in log messages so overlapping sessions
    from parallel vendor sub-agents can be told apart. The AgentCore session
    is stopped automatically when the context exits.
    """
    prefix = f"[{label}] " if label else ""
    bc = BrowserClient(region=region)
    # SDK default is 3600 s (1 hr); we cap at 15 min because vendor lookups
    # finish in 1–3 min and we want orphaned sessions (after a crash that
    # bypasses the finally block) to self-terminate quickly.
    bc.start(
        identifier=get_browser_id(),
        name="web-browse-session",
        session_timeout_seconds=900,
    )
    session_id = bc.session_id
    logger.debug("%sBrowser session started: %s", prefix, session_id)

    ws_url, ws_headers = bc.generate_ws_headers()
    logger.debug("%sBrowser endpoint: %s...", prefix, ws_url[:60])

    try:
        yield ws_url, ws_headers
    finally:
        try:
            bc.stop()
        except Exception as exc:
            logger.warning("%sCould not stop browser session %s: %s", prefix, session_id, exc)


@asynccontextmanager
async def playwright_browser_session(
    label: str = "", region: str = REGION
) -> AsyncIterator[Page]:
    """
    Open an AgentCore Browser session and yield a connected Playwright Page.

    Wraps browser_cdp_session with a full Playwright connection. Use this for
    direct Playwright automation; for Nova Act use browser_cdp_session instead,
    as Nova Act manages its own Playwright connection internally.

    Usage:
        async with playwright_browser_session(label=url) as page:
            await page.goto("https://example.com")
            title = await page.title()
    """
    async with browser_cdp_session(label=label, region=region) as (ws_url, ws_headers):
        playwright = None
        browser: Browser | None = None
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.connect_over_cdp(ws_url, headers=ws_headers)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1440, "height": 900})
            await page.set_extra_http_headers(
                {
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
            )
            yield page
        finally:
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()

"""
claude_vendor_agent.py — Single-vendor Claude ReAct sub-agent.

run_vendor_agent() creates an ephemeral create_agent (LangChain ReAct loop)
scoped to one vendor URL. It opens a single AgentCore Browser session for
the duration of the agent run and exposes four browser tools that all share
the same Playwright Page:

  navigate_to(url, instructions)
      Navigate the existing browser session to url and return extracted content.
      Session state (cookies, SPA routing, auth tokens) persists between calls.

  get_page_content(instructions)
      Extract content from the current page without navigating.

  click_element(selector)
      Click a CSS selector on the current page, wait for load, return content.
      Clicking also moves the mouse to the element, triggering CSS hover menus.

  type_into(selector, text)
      Fill a text input field identified by selector.

  process_with_code_interpreter(data, task)
      Sends extracted page content to the AgentCore Code Interpreter for
      structured parsing into a JSON parts array. The Code Interpreter session
      ID is bound at tool construction (via make_code_interpreter_tool) so the
      model never sees it — all vendor sub-agents within one orchestrator run
      share the same session, so Code Interpreter state (imports, intermediate
      DataFrames) accumulates across vendors.

Page snapshots include an 'interactive' list of every button, input, select,
and ARIA widget found in the DOM, each with a ready-to-use CSS selector. This
allows the agent to pick precise selectors without guessing at markup.

The browser session is opened once in run_vendor_agent() and stays alive
for the entire agent loop. All browser tools close over the shared Page object.
The agent loop is a LangChain ReAct agent (create_agent) wrapped with
BedrockPromptCachingMiddleware so growing conversation history is served
from Bedrock's prompt cache on every turn after the first.
Nova Act is not involved — all reasoning is done by Claude via Bedrock.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import uuid
from urllib.parse import urlparse

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from playwright.async_api import Page

from agent.config import MODEL_ID, REGION
from agent.tools.browser import playwright_browser_session
from agent.tools.code_interpreter import make_code_interpreter_tool


@functools.lru_cache(maxsize=1)
def _get_vendor_llm():
    """Lazily build (and cache) the LLM used by the per-vendor ReAct agent.

    Cached to a single instance so that N parallel vendor sub-agents share one
    model client instead of constructing one per run, matching the pattern used
    in web_agent.py and tools/code_interpreter.py.
    """
    return init_chat_model(
        MODEL_ID,
        model_provider="bedrock_converse",
        region_name=REGION,
        temperature=0,
        max_tokens=2048,
    )

# ── Per-agent log labelling ───────────────────────────────────────────────────
#
# When multiple vendor sub-agents run concurrently their log lines interleave.
# _agent_label is a ContextVar so each asyncio task (one per vendor) carries
# its own value without any locking. run_vendor_agent sets it for the duration
# of its task; _LabelAdapter prepends it to every message from this module.

_agent_label: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_label", default=""
)


class _LabelAdapter(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        label = _agent_label.get("")
        return (f"[{label}] {msg}" if label else msg), kwargs


logger = _LabelAdapter(logging.getLogger(__name__), {})

# ── Browser behaviour constants ───────────────────────────────────────────────

# Timeout for page.goto() in milliseconds. Vendor catalog pages can be slow;
# 30 s gives enough headroom without waiting forever on hung servers.
_GOTO_TIMEOUT_MS: int = 30_000

# Timeout for the label-fallback click in _try_hidden_click, in milliseconds.
# (The normal click path uses _GOTO_TIMEOUT_MS instead — see the click_element
# tool below for why.)
_INTERACT_TIMEOUT_MS: int = 10_000

# Timeout for wait_for_load_state("networkidle") after navigation and interactions.
# networkidle waits for no in-flight network requests for 500 ms, which ensures
# SPA frameworks (React, Vue, Angular) have finished rendering. Pages that keep
# persistent connections (analytics pings, WebSockets) will hit this timeout and
# fall through gracefully — we proceed with whatever has already rendered.
_NETWORKIDLE_TIMEOUT_MS: int = 10_000


# ── Page snapshot helper ──────────────────────────────────────────────────────


# File extensions that Chromium hands off to a download/viewer pipeline rather
# than rendering as a page. goto() with any wait_until value aborts on these
# because there is no navigable document to commit to.
_NON_HTML_EXTENSIONS = {".pdf", ".xls", ".xlsx", ".csv", ".zip", ".doc", ".docx"}

# <input type=...> values that page.fill() cannot target. Used by type_into to
# steer the agent toward click_element when it picks the wrong tool for a
# checkbox/radio/button.
_NON_FILLABLE_INPUT_TYPES = {"checkbox", "radio", "button", "submit", "reset", "hidden", "image"}

_EXEC_CONTEXT_DESTROYED = "Execution context was destroyed"

# Short wait used at the start of _snapshot_page to anchor to the current
# document before any evaluate() calls. Vendor sites with multi-hop redirects
# (e.g. Index.shtml → /US/product/...) can still be mid-navigation when the
# outer networkidle resolves; this re-syncs to the settled document.
_ANCHOR_TIMEOUT_MS: int = 5_000

# ── In-page JavaScript snippets ──────────────────────────────────────────────
#
# Each constant is a self-contained JS function string passed to page.evaluate().
# Keeping them here avoids mixing JS into the Python control flow of _snapshot_page.

# Remove noise tags and return visible body text.
_JS_EXTRACT_TEXT = """() => {
    ['script', 'style', 'noscript', 'footer'].forEach(tag => {
        document.querySelectorAll(tag).forEach(el => el.remove());
    });
    return document.body ? document.body.innerText : '';
}"""

# Collect same-origin anchor links (http/https only), deduplicated.
_JS_COLLECT_LINKS = """() => {
    const origin = window.location.origin;
    const seen = new Set();
    return Array.from(document.querySelectorAll('a[href]'))
        .map(a => ({text: a.innerText.trim(), href: a.href}))
        .filter(l => l.href.startsWith(origin) && l.text)
        .filter(l => { if (seen.has(l.href)) return false; seen.add(l.href); return true; })
        .slice(0, 50);
}"""

# ASP.NET GridView pagination and row-action links use
# javascript:__doPostBack(...) hrefs, which are filtered out of
# the http-only links list above. Expose them separately so the
# agent can see "Page 2", "Page 3", part-number detail links, etc.
# The postback_target and postback_arg fields decode the arguments
# so the agent can recognise pagination (arg like "Page$2") vs.
# row actions (arg like "tIS43QR81024B") without parsing JS itself.
_JS_COLLECT_POSTBACK_LINKS = r"""() => {
    return Array.from(document.querySelectorAll('a[href^="javascript:__doPostBack"]'))
        .map(a => {
            const m = a.getAttribute('href').match(/__doPostBack\('([^']+)','([^']*)'\)/);
            return {
                text: a.innerText.trim(),
                href: a.getAttribute('href'),
                postback_target: m ? m[1] : '',
                postback_arg:    m ? m[2] : '',
                selector: a.id ? '#' + CSS.escape(a.id) : '',
            };
        })
        .filter(l => l.text)
        .slice(0, 30);
}"""

# Collect interactive elements (buttons, inputs, ARIA widgets) with usable
# CSS selectors. The agent passes any returned 'selector' value directly to
# click_element() or type_into() without further guessing.
_JS_COLLECT_INTERACTIVE = r"""() => {
    const seen = new Set();
    const results = [];
    const els = document.querySelectorAll(
        'button, input:not([type="hidden"]), select, textarea, ' +
        '[role="button"], [role="menuitem"], [role="tab"], ' +
        '[role="searchbox"], [role="combobox"], [role="option"], ' +
        'a[class*="btn"], a[class*="button"], a[class*="cta"], a[class*="more"]'
    );
    els.forEach(el => {
        // Styled checkboxes and radio buttons hide the native <input>
        // and make the associated <label> the actual click target.
        // Emit a label[for="id"] entry (with the label's text as the
        // name) instead of the hidden input so every filter option is
        // individually visible and immediately clickable.
        if (el.tagName === 'INPUT' && el.id &&
            (el.getAttribute('type') === 'checkbox' || el.getAttribute('type') === 'radio')) {
            const lbl = document.querySelector('label[for="' + el.id + '"]');
            if (lbl) {
                const lblName = lbl.innerText.trim().replace(/\s+/g, ' ').slice(0, 80);
                if (lblName && !seen.has(lblName)) {
                    seen.add(lblName);
                    results.push({
                        role: el.getAttribute('type'),
                        name: lblName,
                        selector: 'label[for="' + el.id + '"]',
                    });
                }
                return;  // skip standard processing for this input
            }
        }

        const name = (
            el.getAttribute('aria-label') ||
            el.textContent?.trim() ||
            el.getAttribute('placeholder') ||
            el.getAttribute('name') ||
            el.getAttribute('title') ||
            ''
        ).trim().replace(/\s+/g, ' ').slice(0, 80);
        if (!name || seen.has(name)) return;
        seen.add(name);

        let selector = '';
        if (el.id) {
            // Use attribute selector rather than #id shorthand.
            // CSS.escape() produces backslash sequences (e.g. \38 Gb
            // for IDs starting with a digit) that get double-escaped
            // through JSON → Python → Playwright and arrive malformed.
            // [id="..."] needs no escaping and survives the round-trip.
            selector = '[id="' + el.id + '"]';
        } else if (el.getAttribute('aria-label')) {
            selector = '[aria-label="' + el.getAttribute('aria-label') + '"]';
        } else if (el.getAttribute('name')) {
            selector = el.tagName.toLowerCase() + '[name="' + el.getAttribute('name') + '"]';
        } else if (el.getAttribute('placeholder')) {
            selector = '[placeholder="' + el.getAttribute('placeholder') + '"]';
        } else if (el.getAttribute('role')) {
            selector = '[role="' + el.getAttribute('role') + '"]';
        } else if (el.tagName === 'A' && el.getAttribute('href') &&
                   !el.getAttribute('href').startsWith('javascript')) {
            // Anchor-buttons (e.g. <a class="btn-round">) have no id or
            // ARIA role. Use the href as a unique, stable selector.
            selector = 'a[href="' + el.getAttribute('href').replace(/"/g, '\\"') + '"]';
        } else if (name) {
            // Last resort: Playwright :has-text() pseudo-selector.
            // Covers javascript: anchors, onclick divs, and any other
            // element with no stable id/role/href. The agent must not
            // invent selectors when this field is empty — always provide one.
            selector = el.tagName.toLowerCase() + ':has-text("' + name.replace(/"/g, '\\"') + '")';
        }

        results.push({
            role: el.getAttribute('role') || el.tagName.toLowerCase(),
            name: name,
            selector: selector,
        });
    });
    return results.slice(0, 250);
}"""


# ── Snapshot helpers ─────────────────────────────────────────────────────────


async def _snapshot_with_error(page: Page, error_msg: str) -> str:
    """Take a page snapshot and attach an error message to it."""
    snapshot = json.loads(await _snapshot_page(page))
    snapshot["error"] = error_msg
    return json.dumps(snapshot)


async def _try_hidden_click(page: Page, selector: str) -> str | None:
    """
    Attempt to click a hidden element via fallback strategies.

    Tries dispatch_event('click') first (fires the JS handler unconditionally,
    bypassing Playwright actionability checks). If that fails, tries clicking
    the associated <label for="id"> — hidden checkboxes/radios often have a
    visible styled label that IS the intended click target.

    Returns:
        Snapshot JSON string if a fallback succeeded, or None if all failed.
    """
    logger.warning(
        "click_element: selector %r is not visible — attempting dispatch_event click",
        selector,
    )

    # Strategy 1: dispatch_event('click') — fires JS unconditionally.
    try:
        await page.locator(selector).first.dispatch_event("click")
        logger.info("click_element: dispatch_event click succeeded for %r", selector)
    except Exception as dispatch_exc:
        logger.warning("click_element: dispatch_event failed for %r: %s", selector, dispatch_exc)
    else:
        try:
            await page.wait_for_load_state("networkidle", timeout=_NETWORKIDLE_TIMEOUT_MS)
        except Exception:
            pass
        return await _snapshot_page(page)

    # Strategy 2: click the associated <label for="id"> (hidden checkbox/radio).
    try:
        el_id = await page.locator(selector).first.get_attribute("id")
        if el_id:
            label_sel = f'label[for="{el_id}"]'
            if await page.locator(label_sel).count() > 0:
                await page.locator(label_sel).first.click(timeout=_INTERACT_TIMEOUT_MS)
                logger.info(
                    "click_element: clicked label[for=%r] as fallback for hidden %r",
                    el_id, selector,
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=_NETWORKIDLE_TIMEOUT_MS)
                except Exception:
                    pass
                return await _snapshot_page(page)
    except Exception as label_exc:
        logger.warning("click_element: label fallback failed for %r: %s", selector, label_exc)

    return None


async def _snapshot_page(page: Page, _instructions: str = "") -> str:
    """
    Extract a JSON snapshot of the current page state.

    Pulls visible text via document.body.innerText (noise tags removed, but nav
    is preserved so the agent can see category navigation). Also collects anchor
    links and a labelled list of interactive elements (buttons, inputs, ARIA
    widgets) each with a ready-to-use CSS selector so the agent can call
    click_element / type_into without guessing at markup.

    Anchors to the current execution context before evaluating so that
    multi-hop redirects (meta-refresh, JS redirects, server chains) do not
    cause "Execution context was destroyed" errors. If the context is destroyed
    mid-snapshot, retries once after waiting for the new document to settle.

    Args:
        page:          The active Playwright Page (session must already be open).
        _instructions: Unused; kept for call-site compatibility.

    Returns:
        JSON string with keys: url, title, content (page text), links,
        interactive (list of {role, name, selector}).
        On error: JSON string with key 'error'.
    """
    for attempt in range(2):
        try:
            # Re-anchor to whichever document is currently loaded. If a redirect
            # fired after the outer networkidle, this blocks until the new
            # document's DOM is ready before we call any evaluate().
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=_ANCHOR_TIMEOUT_MS)
            except Exception:
                pass  # Already settled; continue

            title = await page.title()
            content = await page.evaluate(_JS_EXTRACT_TEXT)

            # If JS hasn't rendered yet (networkidle timed out early due to
            # persistent connections), wait briefly and re-extract once.
            if len(content.strip()) < 100 and attempt == 0:
                await asyncio.sleep(2.0)
                content = await page.evaluate(
                    "() => document.body ? document.body.innerText : ''"
                )

            links = await page.evaluate(_JS_COLLECT_LINKS)
            postback_links = await page.evaluate(_JS_COLLECT_POSTBACK_LINKS)
            interactive = await page.evaluate(_JS_COLLECT_INTERACTIVE)

            result = {
                "url": page.url,
                "title": title,
                "content": content[:15000],
                "links": links,
                "postback_links": postback_links,
                "interactive": interactive,
            }
            logger.info(
                "Snapshot: %d chars, %d links, %d postback links, %d interactive — %s",
                len(result["content"]), len(links), len(postback_links), len(interactive), page.url,
            )
            return json.dumps(result)

        except Exception as exc:
            if _EXEC_CONTEXT_DESTROYED in str(exc) and attempt == 0:
                logger.warning(
                    "Execution context destroyed mid-snapshot for %s; "
                    "waiting for redirect to settle before retry.",
                    page.url,
                )
                try:
                    await page.wait_for_load_state("networkidle", timeout=_NETWORKIDLE_TIMEOUT_MS)
                except Exception:
                    pass
                # networkidle resolving doesn't guarantee the new document's JS
                # has finished rewriting the execution context. A short sleep
                # covers the tail of multi-hop redirect chains (e.g. .shtml →
                # /US/product/... → final URL) where networkidle fires between hops.
                await asyncio.sleep(1.0)
                continue  # retry once
            logger.error("Snapshot failed for %s: %s", page.url, exc)
            return json.dumps({"error": str(exc), "url": page.url})

    # Should not be reached, but satisfy the type checker.
    return json.dumps({"error": "snapshot failed after retry", "url": page.url})


# ── Browser tool factory ──────────────────────────────────────────────────────


def _make_browser_tools(page: Page) -> list:
    """
    Create browser interaction tools that share a single persistent Playwright Page.

    All returned tools close over `page`, so cookies, SPA routing state, and
    any session tokens established by one tool call are visible to subsequent
    calls within the same agent run.

    Args:
        page: The active Playwright Page opened by run_vendor_agent().

    Returns:
        List of LangChain tools: navigate_to, get_page_content, click_element,
        type_into.
    """

    @tool
    async def navigate_to(url: str, instructions: str) -> str:
        """
        Navigate the browser to a URL and extract page content.

        Uses the existing session — cookies, SPA state, and auth tokens set
        on previous pages remain active. Does NOT provision a new browser.

        Args:
            url:          The fully-qualified URL to navigate to.
            instructions: Ignored; kept for API compatibility.

        Returns:
            JSON string with keys: url, title, content, links, postback_links, interactive.
            On error: {error}.
        """
        # Chromium aborts navigation to non-HTML files (PDFs, spreadsheets, etc.)
        # because it routes them to a download pipeline instead of a renderer.
        # Detect these early and tell the agent to use the URL as-is rather than
        # wasting a goto() call that will always fail.
        url_lower = url.lower().split("?")[0]
        if any(url_lower.endswith(ext) for ext in _NON_HTML_EXTENSIONS):
            logger.info("navigate_to: skipping non-HTML URL %s", url)
            return json.dumps({
                "error": f"{url!r} is a non-HTML file and cannot be navigated to. "
                         "Record it directly as the datasheet_url value without navigating.",
                "url": url,
            })

        try:
            await page.goto(url, wait_until="commit", timeout=_GOTO_TIMEOUT_MS)
        except Exception as exc:
            if "ERR_ABORTED" in str(exc):
                # ERR_ABORTED on .shtml and similar server-redirect URLs can fire
                # even with wait_until="commit" when the server-side redirect chain
                # completes but Chromium marks the original request as aborted.
                # The browser may already be sitting on the redirect destination,
                # so try snapshotting whatever is currently loaded before giving up.
                snapshot_str = await _snapshot_page(page)
                snapshot = json.loads(snapshot_str)
                if snapshot.get("content", "").strip():
                    logger.info(
                        "navigate_to aborted for %s but browser landed on %s — using that",
                        url, snapshot.get("url", "?"),
                    )
                    return snapshot_str
            logger.warning("navigate_to failed for %s: %s", url, exc)
            return json.dumps({"error": str(exc), "url": url})
        try:
            await page.wait_for_load_state("networkidle", timeout=_NETWORKIDLE_TIMEOUT_MS)
        except Exception:
            pass  # Page has persistent connections; proceed with what has rendered

        # Detect redirect: if the browser landed somewhere other than the requested
        # URL, annotate the snapshot so the agent knows navigation didn't succeed.
        # Compare paths only (strips scheme, host, query, fragment) to avoid false
        # positives from http→https upgrades and trailing-slash normalization.
        final_url = page.url
        requested_path = urlparse(url).path.rstrip("/")
        final_path = urlparse(final_url).path.rstrip("/")
        snapshot_str = await _snapshot_page(page, instructions)
        if final_path != requested_path:
            logger.warning(
                "navigate_to: requested %s but landed on %s (redirect detected)", url, final_url
            )
            snapshot_data = json.loads(snapshot_str)
            snapshot_data["navigation_redirected"] = True
            snapshot_data["requested_url"] = url
            snapshot_data["error"] = (
                f"Navigation to {url!r} was redirected to {final_url!r}. "
                f"The site did not serve the requested page. "
                f"Do NOT navigate to individual product detail URLs — they all redirect. "
                f"Do NOT use click_element with a part number or any value from a product row. "
                f"Go back to the product list page, apply filters if helpful, then call "
                f"process_with_code_interpreter on the list page content to extract part data. "
                f"The list table is your data source — detail pages are not required."
            )
            return json.dumps(snapshot_data)
        return snapshot_str

    @tool
    async def get_page_content(instructions: str) -> str:
        """
        Extract content from the current page without navigating.

        Use this after an interaction (click, form submission) has changed
        the page state and you want to read the updated content. Also use
        this to re-read a long page after scrolling to reveal more content.

        Args:
            instructions: Ignored; kept for API compatibility.

        Returns:
            JSON string with keys: url, title, content, links, postback_links, interactive.
            On error: {error}.
        """
        return await _snapshot_page(page, instructions)

    @tool
    async def click_element(selector: str) -> str:
        """
        Click an element on the current page and return the updated content.

        Waits for networkidle after the click so SPA-rendered content is included
        in the returned snapshot. Falls through gracefully if the page keeps
        persistent connections.

        Args:
            selector: CSS selector for the element to click. Prefer selectors
                      from the 'interactive' list in the previous snapshot.

        Returns:
            JSON string with keys: url, title, content, links, postback_links, interactive.
            On error: {error}.
        """
        # ── Guard: reject constructed selectors ──────────────────────────
        # Comma-separated or wildcard-attribute selectors are signs the agent
        # guessed rather than copying from the interactive list.
        if "," in selector or "*=" in selector:
            logger.warning(
                "click_element: rejecting constructed selector %r — "
                "use a verbatim selector from the interactive list",
                selector,
            )
            return await _snapshot_with_error(page, (
                f"Selector {selector!r} looks constructed (contains ',' or '*='). "
                f"Only use selectors copied verbatim from the 'interactive' list. "
                f"Never build selectors yourself."
            ))

        # ── Guard: selector matches nothing ──────────────────────────────
        try:
            match_count = await page.locator(selector).count()
        except Exception:
            match_count = -1  # locator syntax error; let click() report the real error

        if match_count == 0:
            logger.warning("click_element: selector %r matched 0 elements", selector)
            return await _snapshot_with_error(page, (
                f"Selector {selector!r} matched 0 elements. "
                f"STOP trying to click this element. "
                f"If you recently applied a filter, parts visible before filtering may have been "
                f"removed from the results — do NOT search for them. "
                f"Instead: read the 'content' field of this snapshot, which contains the current "
                f"filtered product table, and call process_with_code_interpreter on that content "
                f"to extract matching parts. The content is your data source — not the detail pages."
            ))

        # ── Guard: element is disabled ───────────────────────────────────
        if match_count > 0:
            try:
                is_disabled = await page.locator(selector).first.get_attribute("disabled")
            except Exception:
                is_disabled = None
            if is_disabled is not None:
                logger.warning("click_element: selector %r is disabled — skipping click", selector)
                return await _snapshot_with_error(page, (
                    f"Selector {selector!r} is disabled and cannot be clicked. "
                    f"If it has an onclick handler, try dispatch_event or look for "
                    f"its associated label in the 'interactive' list instead."
                ))

        # ── Guard: element is hidden ─────────────────────────────────────
        # page.click() retries for the full timeout waiting for an invisible
        # element to become visible — which it never will if it is part of a
        # hidden dropdown or off-screen panel. Check before committing.
        # (match_count == -1 means the selector was syntactically invalid;
        # skip this check and let click() produce the real error message.)
        if match_count > 0:
            try:
                is_visible = await page.locator(selector).first.is_visible()
            except Exception:
                is_visible = True  # can't tell — let click() handle it

            # Try scrolling into view first (handles accordion filters,
            # sticky-header-obscured items).
            if not is_visible:
                try:
                    await page.locator(selector).first.scroll_into_view_if_needed(timeout=3000)
                    is_visible = await page.locator(selector).first.is_visible()
                except Exception:
                    pass

            if not is_visible:
                result = await _try_hidden_click(page, selector)
                if result is not None:
                    return result
                return await _snapshot_with_error(page, (
                    f"Selector {selector!r} is hidden and could not be clicked. "
                    f"It is likely inside a collapsed panel (e.g. a 'Filters' toggle). "
                    f"Look in the 'interactive' list for an expand/toggle button — "
                    f"such as one labelled 'Filters', 'Filter', or with aria-expanded='false' "
                    f"— and click that first to reveal the hidden element."
                ))

        # ── Normal click path ────────────────────────────────────────────
        # Use _GOTO_TIMEOUT_MS (not _INTERACT_TIMEOUT_MS) because page.click()
        # waits for any navigation triggered by the click to complete, not just
        # the click action itself. Slow server-side postbacks (e.g. ASP.NET
        # __doPostBack) can take 20–30 s to respond; _INTERACT_TIMEOUT_MS (10 s)
        # was too short. For non-navigating clicks (checkboxes, AJAX filters)
        # click() returns as soon as the action is done — the longer budget has
        # no cost there.
        try:
            await page.click(selector, timeout=_GOTO_TIMEOUT_MS)
        except Exception as exc:
            logger.warning("click_element failed for %r: %s", selector, exc)
            return await _snapshot_with_error(page, str(exc))
        try:
            await page.wait_for_load_state("networkidle", timeout=_NETWORKIDLE_TIMEOUT_MS)
        except Exception:
            pass
        return await _snapshot_page(page)

    @tool
    async def type_into(selector: str, text: str) -> str:
        """
        Fill a text input, search box, or dropdown (<select>) with the given text.

        For <select> elements, selects the option whose visible label matches `text`
        exactly; falls back to a value match if no label matches. For all other
        inputs, clears existing content and replaces it with `text`. Does NOT
        submit the form — follow up with click_element on the submit button if
        needed.

        Args:
            selector: CSS selector for the input or select element.
            text:     The text to type (inputs) or option label to select (dropdowns).

        Returns:
            JSON with {ok: true, current_url} on success. On error: {error}.
        """
        # Reject non-fillable input types before Playwright tries and fails.
        try:
            el_type = (await page.locator(selector).first.get_attribute("type") or "").lower()
        except Exception:
            el_type = ""
        if el_type in _NON_FILLABLE_INPUT_TYPES:
            logger.warning(
                "type_into: selector %r is type=%r — not a text input; use click_element instead",
                selector, el_type,
            )
            return json.dumps({
                "error": (
                    f"Selector {selector!r} is a {el_type!r} input and cannot be filled with text. "
                    f"Use click_element to activate it instead."
                )
            })

        # Detect <select> elements — page.fill() raises on them. Use
        # page.select_option() instead, which matches by visible label text.
        try:
            tag_name = (await page.locator(selector).first.evaluate("el => el.tagName") or "").upper()
        except Exception:
            tag_name = ""

        if tag_name == "SELECT":
            try:
                await page.select_option(selector, label=text)
                logger.info("type_into: selected option %r in select %r", text, selector)
            except Exception:
                # label match failed; try value match as fallback
                try:
                    await page.select_option(selector, value=text)
                    logger.info("type_into: selected option by value %r in select %r", text, selector)
                except Exception as exc:
                    logger.warning("type_into: select_option failed for %r value=%r: %s", selector, text, exc)
                    return json.dumps({"error": str(exc)})
            try:
                await page.wait_for_load_state("networkidle", timeout=_NETWORKIDLE_TIMEOUT_MS)
            except Exception:
                pass
            return await _snapshot_page(page)

        try:
            await page.fill(selector, text)
        except Exception as exc:
            logger.warning("type_into failed for %r: %s", selector, exc)
            return json.dumps({"error": str(exc)})
        return json.dumps({"ok": True, "current_url": page.url})

    return [navigate_to, get_page_content, click_element, type_into]


# ── System prompt ─────────────────────────────────────────────────────────────


def _build_system_prompt(vendor_url: str) -> str:
    """Return a system prompt scoped to a single vendor URL."""
    return f"""\
You are a single-vendor web research agent for Supertron LLC's hardware engineering team.

## Your Assignment
Search the following vendor's website for memory parts matching the specification in your task:
  {vendor_url}

Stay on this vendor's website only. Do not navigate to other vendors' sites.

## Browser Session
All browser tools share a single persistent Chromium session. Cookies, SPA routing
state, and any session tokens set on one page remain active for subsequent tool calls.
You do not need to re-navigate to the homepage between steps.

## Tools Available

### Browser tools (all share one persistent session)

Every browser tool that returns page content returns a JSON object with these fields:
  - url           — current page URL
  - title         — page <title>
  - content       — visible page text (up to 15 000 chars)
  - links         — list of {{text, href}} for standard anchor links (http/https only)
  - postback_links — list of ASP.NET PostBack links that would be invisible in 'links'
                    because their href is javascript:__doPostBack(...). Each entry has:
                      text            — visible link text (e.g. "2", "3", "Next", part number)
                      href            — the raw javascript: href (pass to click_element selector)
                      postback_target — the PostBack control ID (e.g. "grdvDisplayDRAM")
                      postback_arg    — the PostBack argument. "Page$2" means page 2 of a
                                        grid; "tIS43QR81024B" is a part-number row action.
                      selector        — CSS selector if the link has an id, else empty string
  - interactive   — list of {{role, name, selector}} for every button, input,
                    select, and ARIA widget on the page. The 'selector' value
                    is a ready-to-use CSS selector — pass it directly to
                    click_element() or type_into() without modification.
                    Checkbox and radio filter options each appear individually
                    with role 'checkbox' or 'radio' and a label[for=...] selector
                    that points to the visible label element — click_element works
                    on them directly without any hover or scroll step.

1. navigate_to(url, instructions)
   - Navigate to `url` within the existing browser session and extract page content.
   - Use this to move between catalog pages, search results, and part detail pages.
   - Do NOT navigate to PDF or binary file URLs (.pdf, .xls, .zip, etc.). If you
     find a datasheet link ending in .pdf, record that URL directly as datasheet_url
     in your output without calling navigate_to on it.
   - ALWAYS check the `url` field in the returned snapshot to confirm you arrived
     where you intended. If the snapshot contains `"navigation_redirected": true`,
     the site redirected you away from the requested page. In that case:
       * Do NOT try to click_element using a part number or made-up selector.
       * Return to the product list page and use its `links` or `interactive` entries
         to reach the detail page — click the part number link there rather than
         constructing a direct URL yourself.

2. get_page_content(instructions)
   - Extract content from the current page without navigating.
   - Use this after a click or form submission has changed the page, or to
     re-read content further down a long page after scrolling.

3. click_element(selector)
   - Click a CSS selector on the current page, wait for the page to settle, and
     return the new content.
   - Use selectors copied verbatim from the 'interactive' list — they are verified
     against the live DOM and will match reliably.
   - Clicking a navigation item also moves the mouse there, which triggers any
     CSS hover menus along the way. You do not need a separate hover step.

4. type_into(selector, text)
   - Fill a text input, search box, or dropdown (<select>). Use the 'selector'
     from an 'interactive' entry with role 'input', 'searchbox', 'combobox',
     or 'select'. For <select> dropdowns, pass the visible option label as text
     (e.g. type_into('[id="technology-select"]', 'DRAMs')). After selecting an
     option the tool waits for the page to settle and returns an updated snapshot.
     Follow up with click_element on a submit button for plain text inputs.
   - Do NOT call type_into on checkbox, radio, button, or submit elements —
     use click_element for those.

### Data tool

5. process_with_code_interpreter(data, task)
   - Sends extracted page content and a processing task to the Code Interpreter.
   - Use this to extract part numbers, specs, and datasheet URLs from raw page text.
   - Always include "vendor" as a field in every extracted record, using the vendor's
     brand name (e.g. "Micron", "SK Hynix", "Nanya", "ISSI").

## Selector Policy — CRITICAL
Always copy selectors verbatim from the 'interactive' list returned by the previous
snapshot. Never construct, guess, or combine selectors yourself. Specifically:
- No comma-separated selectors: 'button.ddr4, [data-tab="DDR4"]' is forbidden.
- No wildcard attribute selectors: '[class*="ddr4"]' is forbidden.
- No selectors with IDs, classes, or attributes you invented.
If no selector in the interactive list matches what you need, call get_page_content
to refresh the list, or scroll down and call get_page_content again — do not guess.

## URL Policy — CRITICAL
Never construct, guess, or modify URLs. Only call navigate_to with URLs that appear
verbatim in the 'links', 'postback_links', or 'interactive' fields of a previous
snapshot. If you cannot find the URL you need, use click_element to reveal it first.
Do not append query parameters (e.g. ?density=8Gb, ?currentPage=2) to URLs you
construct yourself — use the pagination and filter controls on the page instead.

## Workflow
1. Navigate to the vendor URL provided above.
2. Find the product section relevant to the requested component type.
   - Click navigation items to move through the site; clicking also triggers any
     hover menus, so no separate hover step is needed.
3. Find the parts catalog related to that component type, if one exists, and expand filters
   as needed to see matching parts, being careful not to exclude valid results with over-aggressive filtering.
   - Filter options appear in the 'interactive' list with role 'checkbox' or 'radio'.
     Click one filter at a time using click_element. After each click the snapshot
     returned by click_element already reflects the updated results — do NOT call
     get_page_content again before reading the new content.
   - After the filter is applied, read the 'content' field of the snapshot. That
     content IS the filtered product table — call process_with_code_interpreter on it
     immediately. Do NOT try to click on individual part numbers from the table.
     Part numbers appear as text in the table, not as clickable interactive elements.
4. If this vendor does not carry the requested component type, return the no-match
   result below immediately — do not continue browsing.
5. Otherwise, navigate to the specific product family and extract matching parts.
   - After loading a results table, check postback_links and the 'interactive' list
     for pagination controls (text like "2", "3", "Next", postback_arg like "Page$2").
     Click through every page using click_element on those controls — never navigate
     to ?currentPage=N URLs you construct yourself. Collect all results before step 6.
6. Call process_with_code_interpreter to structure the raw page content into a JSON array.

## Notes and Recommendations

Do not use search boxes to find parts. Site-wide nav search returns website sections,
not parts. Product-page search boxes are often wired to hidden suggestion dropdowns
that are never visible and cannot be clicked. Instead, navigate directly to the
relevant product category page or parts catalog and use the filters or tables already 
present there.

Do not apply attribute filters (density, speed, package) to narrow a results table
unless you are certain the filter includes all parts that could match the specification.
Filters that exclude valid results are worse than no filter. Show all results first,
then page through them, rather than filtering down to a subset.

Some vendor sites block direct navigation to product detail pages — the browser is
silently redirected to the homepage or product list. When this happens the snapshot
will include `"navigation_redirected": true`. In this case, extract what you need
from the product list page itself (part number, speed, package columns are often
visible there), and record the list page URL as `detail_url` rather than attempting
to navigate directly to each part's detail page.

Part numbers appear as text in product table rows — they are NOT interactive elements
and NOT element IDs. Never use a part number as a CSS selector, as an `id=` value,
as a `for=` attribute, or as a `name=` attribute. Only use selectors from the
'interactive' list. When the interactive list does not contain a selector for a
specific part, that part cannot be clicked — extract its data from the 'content' text.

Vendor PDF datasheets often require an active browser session with that vendor's site
to download — a direct PDF URL shared out of context will be rejected. Always capture
the HTML product detail page URL (detail_url) in addition to the PDF link so the
recipient can visit the page directly and access the datasheet through a normal session.

## Output Contract — CRITICAL
Return exactly ONE of the following:

  Parts found:
  [{{"vendor": "...", "part_number": "...", "speed": "...", "package": "...", "datasheet_url": "...", "detail_url": "..."}}, ...]

  (detail_url is the HTML product or search-result page where the datasheet link
  appears. If no dedicated detail page exists, use the catalog page URL instead.)

  No matching parts at this vendor:
  {{"vendor": "...", "status": "no_matching_parts", "reason": "brief explanation"}}

  Error:
  {{"vendor": "...", "status": "error", "reason": "what went wrong"}}

## Completion Criteria — STOP when:
- You have found and extracted matching parts at this vendor, OR
- You have confirmed this vendor does not carry the requested component type, OR
- You have attempted navigation and encountered an unrecoverable error.

Do not continue browsing once you have your result.
"""


# ── Agent entry point ─────────────────────────────────────────────────────────


async def run_vendor_agent(
    vendor_url: str,
    mission: str,
    ci_session_id: str,
) -> dict:
    """
    Run a single-vendor Claude ReAct sub-agent and return its raw output.

    Opens a single AgentCore Browser session for the duration of the agent run,
    then creates a LangChain create_agent ReAct loop with browser tools that
    share the live Playwright Page. The browser session is kept open until the
    agent loop completes, so all navigation, click, hover, and form interactions
    within one run share a single persistent Chromium context.

    Args:
        vendor_url:    The vendor's catalog URL to search.
        mission:       The component specification task (natural-language string).
        ci_session_id: Shared Code Interpreter session ID. Passing the same ID
                       across all sub-agents lets Code Interpreter state accumulate
                       across vendor runs within a single orchestrator invocation.

    Returns:
        {"vendor_url": ..., "raw": ...}   on success.
        {"vendor_url": ..., "error": ...} on unhandled exception.
    """
    # Strip "www." so labels read "issi.com", "nanya.com", etc.
    label = urlparse(vendor_url).netloc.removeprefix("www.")
    # Set the ContextVar inside the try so that any exception raised before
    # the agent loop (cold-cache LLM construction, prompt build, etc.) still
    # runs through finally and the per-task label is reset cleanly.
    _label_token = _agent_label.set(label)
    try:
        model = _get_vendor_llm()

        human_message = f"{mission}\n\nVendor URL: {vendor_url}"

        config = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 50,
        }

        logger.info("Claude vendor sub-agent starting: %s", vendor_url)
        async with playwright_browser_session(label=vendor_url) as page:
            browser_tools = _make_browser_tools(page)
            ci_tool = make_code_interpreter_tool(ci_session_id)
            # Per-vendor MemorySaver is intentional: sub-agent state is
            # ephemeral by design and must NOT survive between vendor lookups.
            # Contrast _get_vendor_llm above, which is cached so the N parallel
            # sub-agents share one model client.
            agent = create_agent(
                model=model,
                tools=browser_tools + [ci_tool],
                system_prompt=_build_system_prompt(vendor_url),
                checkpointer=MemorySaver(),
                middleware=[BedrockPromptCachingMiddleware(ttl="5m")],
            )
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=human_message)]},
                config=config,
            )

        # Per-vendor token usage breakdown, including prompt-cache stats.
        # cache_read > 0 on later turns confirms the middleware is active.
        turns = cache_read = cache_creation = input_total = output_total = 0
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                usage = getattr(msg, "usage_metadata", None) or {}
                if usage:
                    turns += 1
                    input_total += usage.get("input_tokens", 0)
                    output_total += usage.get("output_tokens", 0)
                    details = usage.get("input_token_details", {}) or {}
                    cache_read += details.get("cache_read", 0)
                    cache_creation += details.get("cache_creation", 0)
        logger.info(
            "Token usage: turns=%d input=%d output=%d cache_read=%d cache_creation=%d",
            turns, input_total, output_total, cache_read, cache_creation,
        )

        # Extract the last non-empty AIMessage as the result.
        # The isinstance guard is required: ToolMessage and HumanMessage also
        # carry a .content field and must not be mistaken for the model's reply.
        final_text = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content.strip():
                final_text = msg.content
                break
        logger.info("Claude vendor sub-agent complete: %s", vendor_url)
        return {"vendor_url": vendor_url, "raw": final_text}
    except Exception as exc:
        logger.exception("Claude vendor sub-agent failed for %s: %s", vendor_url, exc)
        return {"vendor_url": vendor_url, "error": str(exc)}
    finally:
        _agent_label.reset(_label_token)

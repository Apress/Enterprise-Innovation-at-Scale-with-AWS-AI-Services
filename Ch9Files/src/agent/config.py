"""
config.py — Constants for the Supertron Web Browsing Agent.

Centralises all configuration so model IDs, region, and example missions
are defined in exactly one place.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model & AWS ──────────────────────────────────────────────────────────────

MODEL_ID = "us.anthropic.claude-sonnet-4-6"
REGION = "us-east-1"

# Nova Act API key — required for Nova Act sub-agents.
# Obtain from https://nova.amazon.com/act and set as an environment variable or in .env file.
NOVA_ACT_API_KEY: str = os.environ.get("NOVA_ACT_API_KEY") or ""
if not NOVA_ACT_API_KEY:
    logger.warning(
        "NOVA_ACT_API_KEY is not set. Nova Act sub-agents will fail at session "
        "start. Obtain a key at https://nova.amazon.com/act and export it, or "
        "use --sub-agent claude to run the Playwright sub-agent instead."
    )

# Nova Act model ID.
# "nova-act-preview" — latest preview model (currently v1.1); most capable.
# "nova-act-latest"  — floating alias to the latest GA release (v1.0); stable/pinnable.
# "nova-act-v1.0"    — pinned to the v1.0 GA release; supported for at least 1 year.
NOVA_ACT_MODEL: str = "nova-act-preview"

# Timeout for Nova Act's initial page navigation (go_to_url_timeout), in seconds.
# Vendor catalog pages can be slow; 60 s is safer than the default 30 s.
NOVA_ACT_PAGE_TIMEOUT: int = 60

# AgentCore Browser resource ID — read from resources.json (written by
# setup_resources.py --create). Run that script once before using the agent.
_RESOURCES_FILE = Path(__file__).parent.parent / "resources.json"


def _load_browser_id() -> str:
    """Look up browser_id in resources.json. Empty string if missing.

    Intentionally silent on missing config: this runs at import time, including
    in test runners and IDE indexers where resources.json is often absent. The
    actual error is raised lazily by get_browser_id() when the agent first
    tries to open a session.
    """
    if _RESOURCES_FILE.exists():
        data = json.loads(_RESOURCES_FILE.read_text())
        if bid := data.get("browser_id"):
            return bid
    return ""


BROWSER_ID: str = _load_browser_id()


def get_browser_id() -> str:
    """Return BROWSER_ID, raising a clear error if it was not configured."""
    if not BROWSER_ID:
        raise RuntimeError(
            "browser_id not found in resources.json. Browser sessions cannot start.\n"
            "Run:  python setup_resources.py --create"
        )
    return BROWSER_ID

# ── Vendor websites ───────────────────────────────────────────────────────────
# Memory vendor catalog URLs for the multi-vendor search. The agent navigates
# these sites looking for parts matching the given specification. Add or remove
# entries here to adjust coverage.

WEBSITES = [
    "https://www.micron.com/products/memory",
    "https://product.skhynix.com/products/dram/dram.go",
    "https://www.nanya.com/en/Product/",
    "https://www.issi.com/US/Index.shtml",
    "https://www.alliancememory.com/product-overviews/#Search",
    "https://etron.com/specialty-dram/",
]

# ── Example missions ─────────────────────────────────────────────────────────
# Used in __main__ and invoke_example.py as defaults.

EXAMPLE_MISSIONS = [
    "Find all DDR4 SDRAM components matching: 8Gb density, 3200 MT/s or faster, 78-ball FBGA package. "
    "Search across available memory vendors and return a table of matching parts with vendor, "
    "part number, speed grade, operating temperature range, package, and datasheet URL.",

    "Find all LPDDR5 components matching: 16Gb density, 6400 MT/s or faster. "
    "Search across available memory vendors and return a table of matching parts with vendor, "
    "part number, speed grade, operating temperature range, package, and datasheet URL.",

    "Find all GDDR6 components matching: 8Gb density, 16 GT/s or faster. "
    "Search across available memory vendors and return a table of matching parts with vendor, "
    "part number, speed grade, package, and datasheet URL.",
]

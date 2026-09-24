"""
setup_resources.py — Create or destroy the AWS resources required by this agent.

This script provisions two long-lived AgentCore resources in your AWS account
and writes their identifiers to resources.json. That file is then read
automatically by config.py and deploy.py so you don't have to edit any source
files manually.

Resources created:
  • AgentCore Browser  — a managed Chromium pool used by the browser tools.
  • AgentCore Memory   — LangGraph checkpoint store (used when deploying; local
                         runs fall back to in-process MemorySaver without it).

Usage:
  python setup_resources.py           # Show this help message
  python setup_resources.py --create  # Create resources and write resources.json
  python setup_resources.py --destroy # Delete resources listed in resources.json

AWS credentials must be configured with the permissions in iam_caller_policy.json
before running --create or --destroy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESOURCES_FILE = Path(__file__).parent / "resources.json"

DEFAULT_REGION = "us-east-1"
BROWSER_NAME = "supertronBrowser"
MEMORY_NAME = "web_agent_memory"


# ── Create ────────────────────────────────────────────────────────────────────


def _get_or_create_browser(client, name: str) -> str:
    """Return the browserId for an existing browser, or create it.

    Paginates list_browsers so accounts with more than the default 10 browsers
    don't accidentally get a duplicate created.
    """
    paginator = client.get_paginator("list_browsers")
    for page in paginator.paginate():
        for b in page.get("browserSummaries", []):
            if b["name"] == name:
                logger.info("Found existing browser '%s': %s", name, b["browserId"])
                return b["browserId"]
    logger.info("Creating AgentCore Browser '%s'...", name)
    resp = client.create_browser(
        name=name,
        networkConfiguration={"networkMode": "PUBLIC"},
    )
    browser_id = resp["browserId"]
    logger.info("Browser created: %s", browser_id)
    return browser_id


def _get_or_create_memory(client, name: str) -> tuple[str, str]:
    """Return (memory_arn, memory_id) for an existing memory resource, or create it.

    list_memories returns arn+id but no name, so we fetch each entry by id to
    match on name. Paginates so accounts with more than the default 10 memory
    resources don't accidentally get a duplicate created.
    """
    paginator = client.get_paginator("list_memories")
    for page in paginator.paginate():
        for item in page.get("memories", []):
            mem = client.get_memory(memoryId=item["id"])["memory"]
            if mem.get("name") == name:
                logger.info("Found existing memory '%s': %s", name, mem["id"])
                return mem["arn"], mem["id"]
    logger.info("Creating AgentCore Memory '%s'...", name)
    mem = client.create_memory(
        name=name,
        description="Checkpoint store for the Supertron web browsing agent",
        eventExpiryDuration=30,
    )["memory"]
    logger.info("Memory created: %s (id: %s)", mem["arn"], mem["id"])
    return mem["arn"], mem["id"]


def create_resources(region: str, memory_id_override: str | None = None) -> None:
    """
    Create AgentCore Browser and Memory resources and write resources.json.

    Idempotent — if a browser with the given name already exists it is reused.
    If a memory resource already exists and cannot be looked up automatically,
    pass its ID via --memory-id.
    """
    control_client = boto3.client("bedrock-agentcore-control", region_name=region)

    # --- Browser ---------------------------------------------------------------
    browser_id = _get_or_create_browser(control_client, BROWSER_NAME)

    # --- Memory ----------------------------------------------------------------
    # The Memory resource is only strictly required for deployed runs (it backs
    # AgentCoreMemorySaver). Local runs fall back to in-process MemorySaver.
    if memory_id_override:
        memory_id = memory_id_override
        # Derive a plausible ARN; the actual ARN isn't needed for local use.
        account_resp = boto3.client("sts").get_caller_identity()
        account_id = account_resp["Account"]
        memory_arn = (
            f"arn:aws:bedrock-agentcore:{region}:{account_id}:memory/{memory_id}"
        )
        logger.info("Using provided memory ID: %s", memory_id)
    else:
        memory_arn, memory_id = _get_or_create_memory(control_client, MEMORY_NAME)

    # --- Write resources.json --------------------------------------------------
    resources = {
        "region": region,
        "browser_id": browser_id,
        "memory_arn": memory_arn,
        "memory_id": memory_id,
    }
    RESOURCES_FILE.write_text(json.dumps(resources, indent=2) + "\n")
    logger.info("Wrote %s", RESOURCES_FILE)

    print("\nResources created and written to resources.json:")
    print(f"  browser_id : {browser_id}")
    print(f"  memory_id  : {memory_id}")
    print(f"  memory_arn : {memory_arn}")
    print("\nYou can now run the agent:")
    print("  python run_agent.py")


# ── Destroy ───────────────────────────────────────────────────────────────────


def destroy_resources(region: str) -> None:
    """
    Delete the AgentCore resources listed in resources.json.

    Removes the resources.json file afterwards. Does not affect any AgentCore
    Runtimes that were deployed separately via deploy.py.
    """
    if not RESOURCES_FILE.exists():
        print(
            f"No {RESOURCES_FILE} found — nothing to destroy.",
            file=sys.stderr,
        )
        sys.exit(1)

    resources = json.loads(RESOURCES_FILE.read_text())
    region = resources.get("region", region)
    control_client = boto3.client("bedrock-agentcore-control", region_name=region)

    failed = []

    browser_id = resources.get("browser_id")
    if browser_id:
        logger.info("Deleting browser '%s'...", browser_id)
        try:
            control_client.delete_browser(browserId=browser_id)
            logger.info("Browser deleted.")
        except ClientError as exc:
            logger.error("Could not delete browser: %s", exc)
            failed.append("browser")

    memory_id = resources.get("memory_id")
    if memory_id:
        logger.info("Deleting memory '%s'...", memory_id)
        try:
            control_client.delete_memory(memoryId=memory_id)
            logger.info("Memory deleted.")
        except ClientError as exc:
            logger.error("Could not delete memory: %s", exc)
            failed.append("memory")

    if failed:
        print(
            f"\nFailed to delete: {', '.join(failed)}. "
            "resources.json has not been removed so you can retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    RESOURCES_FILE.unlink()
    logger.info("Removed %s", RESOURCES_FILE)
    print("\nResources destroyed and resources.json removed.")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the AgentCore Browser and Memory resources.",
    )
    parser.add_argument(
        "--destroy",
        action="store_true",
        help="Delete the resources listed in resources.json.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--memory-id",
        metavar="ID",
        help=(
            "Use with --create when a Memory resource already exists. "
            "Skips creation and records this ID in resources.json."
        ),
    )
    args = parser.parse_args()

    if not args.create and not args.destroy:
        parser.print_help()
        sys.exit(0)

    if args.create and args.destroy:
        print("Error: --create and --destroy are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.create:
            create_resources(args.region, memory_id_override=args.memory_id)
        else:
            destroy_resources(args.region)
    except (ClientError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

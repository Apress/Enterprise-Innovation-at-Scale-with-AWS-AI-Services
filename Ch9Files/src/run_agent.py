"""
run_agent.py — Unified CLI for local and deployed agent runs.

Local mode (default, no --runtime-arn):
  Runs the orchestrator in-process. Requires AWS credentials for Bedrock,
  AgentCore Browser, and AgentCore Code Interpreter.

Deployed mode (--runtime-arn <ARN>):
  Invokes the agent via the AgentCore Runtime API. Requires
  bedrock-agentcore:InvokeAgentRuntime permission on the runtime.

Usage:
  # Local
  python run_agent.py
  python run_agent.py --mission "Find all LPDDR5 parts, 16Gb, 6400 MT/s or faster"

  # Deployed
  python run_agent.py --runtime-arn arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/web-agent-XYZ
  python run_agent.py --runtime-arn <ARN> --output s3://my-bucket/results/run1

Output flag:
  --output accepts a base path (no extension). Two files are written:
    <base>.json  — full result dict (final_result, vendor_results, thread_id)
    <base>.md    — agent's synthesized markdown report only (easy to open/share)

  For local paths, parent directories are created automatically.
  For s3:// URIs, requires s3:PutObject permission on the target bucket.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

# Suppress a LangChain/Pydantic V1 compatibility warning that fires on Python 3.14+.
# Not actionable — it's an internal LangChain issue, not something we can fix.
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality", category=UserWarning)
# Nova Act's async API is in preview but required for use inside an async LangGraph graph.
# The sync API would block the event loop and serialize all parallel vendor agents.
warnings.filterwarnings("ignore", message="The async version of Nova Act", category=UserWarning)

# Load .env from the project root if present.
# This is a local-development convenience — deployed runs use runtime env vars.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on env vars being set manually

import boto3
from botocore.exceptions import NoCredentialsError

from agent.config import EXAMPLE_MISSIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# botocore logs credential refresh failures at WARNING with a full traceback before
# raising — suppress it so our own clean error message is the only output.
logging.getLogger("botocore.credentials").setLevel(logging.ERROR)
# Suppress third-party SDK INFO chatter that duplicates or obscures our own logs.
# bedrock_agentcore.tools.browser_client logs session start/stop lifecycle with
# emoji-prefixed lines (✅ Session started/stopped, "Generating websocket headers...")
# that duplicate what browser.py already logs more cleanly.
# langchain_aws.chat_models.bedrock_converse logs "Using Bedrock Converse API to
# generate response" on every model call — not useful at INFO in normal operation.
logging.getLogger("bedrock_agentcore.tools.browser_client").setLevel(logging.WARNING)
logging.getLogger("langchain_aws.chat_models.bedrock_converse").setLevel(logging.WARNING)

DEFAULT_REGION = "us-east-1"


# ── Credential error detection ────────────────────────────────────────────────


def _is_credential_error(exc: BaseException) -> bool:
    """Return True if exc (or any chained cause) is an AWS credential error."""
    # Walk the full exception chain — credential errors often surface as the
    # __cause__ or __context__ of a higher-level LangGraph or asyncio exception.
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        type_name = type(current).__name__
        if type_name in ("LoginRefreshRequired", "NoCredentialsError",
                         "CredentialRetrievalError", "TokenRetrievalError"):
            return True
        if isinstance(current, NoCredentialsError):
            return True
        current = current.__cause__ or current.__context__
    return False


def _credential_error_message() -> str:
    return (
        "\nAWS credentials are not available or have expired.\n"
        "Re-authenticate with one of:\n"
        "  aws sso login --profile <your-profile>   # IAM Identity Center\n"
        "  aws login                                 # browser-based (AWS CLI v2.30+)\n"
        "  aws configure                             # static access keys\n"
    )


# ── Deployed runtime invocation ───────────────────────────────────────────────


def invoke_agent_runtime(
    runtime_arn: str,
    payload: dict,
    session_id: str,
    region: str = DEFAULT_REGION,
) -> dict:
    """
    Invoke the deployed AgentCore Runtime and collect the response.

    AgentCore Runtime returns the entrypoint's return value over an event
    stream. For our (non-streaming) entrypoint, the stream contains a single
    JSON blob — we concatenate the chunks and decode. If the entrypoint were
    a generator, the response would be SSE-framed (`data: {...}\\n\\n` per
    event); this helper would need to be extended to parse those frames.

    Args:
        runtime_arn: Full ARN of the runtime (returned by deploy.py).
        payload:     Dict to send as the request body (mission, thread_id, etc).
        session_id:  Runtime session ID. Must be 33+ characters per the API
                     contract; reusing the same value lets a caller resume into
                     the same microVM and the same orchestrator thread.
        region:      AWS region.

    Returns:
        Parsed dict from the agent's JSON response body.
    """
    client = boto3.client("bedrock-agentcore", region_name=region)

    logger.info(
        "Invoking runtime '%s' (session %s) with payload: %s",
        runtime_arn,
        session_id,
        json.dumps(payload),
    )

    response = client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        runtimeSessionId=session_id,
        payload=json.dumps(payload).encode("utf-8"),
        qualifier="DEFAULT",
    )

    chunks: list[bytes] = []
    for chunk in response.get("response", []):
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))

    raw_body = b"".join(chunks)

    try:
        return json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Could not parse response as JSON: %s", exc)
        logger.error("Raw response: %s", raw_body[:2000])
        raise


# ── Output ────────────────────────────────────────────────────────────────────


def save_results(result: dict, output_base: str) -> None:
    """
    Write result JSON and markdown report to local files or S3.

    Two files are always written:
      <output_base>.json  — full result dict
      <output_base>.md    — agent's synthesized markdown text only

    Args:
        result:      The result dict from run_agent() or invoke_agent_runtime().
        output_base: Base path without extension. May be a local filesystem
                     path (e.g. "output/run1") or an S3 URI base
                     (e.g. "s3://my-bucket/results/run1").
    """
    json_content = json.dumps(result, indent=2, ensure_ascii=False)
    md_content = result.get("final_result") or ""

    if output_base.startswith("s3://"):
        s3_path = output_base[len("s3://"):]
        bucket, _, key_base = s3_path.partition("/")
        s3 = boto3.client("s3")

        s3.put_object(
            Bucket=bucket,
            Key=f"{key_base}.json",
            Body=json_content.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info("Saved JSON to s3://%s/%s.json", bucket, key_base)

        s3.put_object(
            Bucket=bucket,
            Key=f"{key_base}.md",
            Body=md_content.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info("Saved markdown to s3://%s/%s.md", bucket, key_base)
        print(f"\nSaved to s3://{bucket}/{key_base}.{{json,md}}")
    else:
        parent = os.path.dirname(output_base)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(f"{output_base}.json", "w", encoding="utf-8") as f:
            f.write(json_content)
        with open(f"{output_base}.md", "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Saved to %s.json and %s.md", output_base, output_base)
        print(f"\nSaved to {output_base}.json and {output_base}.md")


def print_result(result: dict) -> None:
    """Pretty-print the agent result to stdout."""
    # Reconfigure stdout to UTF-8 so Unicode/emoji in model output prints
    # correctly on Windows consoles that default to cp1252.
    if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("AGENT RESULT")
    print("=" * 60)

    if result.get("final_result"):
        print("\n## Final Result\n")
        print(result["final_result"])

    vendor_results = result.get("vendor_results", [])
    if vendor_results:
        print(f"\nVendors searched: {len(vendor_results)}")

    if result.get("thread_id"):
        print(f"Thread ID: {result['thread_id']}")

    if result.get("error"):
        print(f"\nERROR: {result['error']}")

    print("\n" + "=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Supertron multi-vendor web search agent locally or "
            "invoke a deployed AgentCore runtime."
        )
    )
    parser.add_argument(
        "--runtime-arn",
        metavar="ARN",
        help=(
            "AgentCore Runtime ARN (the full arn:aws:bedrock-agentcore:... value "
            "returned by deploy.py). When provided, invokes the deployed runtime "
            "via the AgentCore API instead of running locally."
        ),
    )
    parser.add_argument(
        "--mission",
        default=EXAMPLE_MISSIONS[0],
        help=(
            "Component specification lookup task. "
            f"Defaults to: {EXAMPLE_MISSIONS[0][:60]}..."
        ),
    )
    parser.add_argument(
        "--thread-id",
        help="Resume a previous run by specifying its thread ID.",
    )
    parser.add_argument(
        "--sub-agent",
        choices=["nova-act", "claude"],
        default="nova-act",
        help=(
            "Sub-agent implementation for vendor browsing. "
            "'nova-act' (default) uses Amazon Nova Act for browser automation. "
            "'claude' uses a LangGraph ReAct agent with Playwright and the "
            "AgentCore Code Interpreter."
        ),
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument(
        "--output",
        metavar="BASE",
        help=(
            "Base path for output files (no extension). "
            "Writes <BASE>.json (full result) and <BASE>.md (markdown report). "
            "Accepts a local path (e.g. output/run1) or an S3 URI "
            "(e.g. s3://my-bucket/results/run1)."
        ),
    )
    args = parser.parse_args()

    try:
        if args.runtime_arn:
            payload: dict = {"mission": args.mission, "sub_agent": args.sub_agent}
            if args.thread_id:
                payload["thread_id"] = args.thread_id
            # Use thread_id as the runtime session ID when provided so that
            # resuming a thread lands in the same microVM (if still alive)
            # AND the same LangGraph checkpoint. AgentCore requires
            # runtimeSessionId to be 33+ chars; UUID4 (36 chars) always
            # qualifies, but a user-supplied --thread-id may be shorter, so
            # pad short values with a UUID suffix to stay above the limit.
            session_id = args.thread_id or str(uuid.uuid4())
            if len(session_id) < 33:
                session_id = f"{session_id}-{uuid.uuid4()}"
            result = invoke_agent_runtime(
                runtime_arn=args.runtime_arn,
                payload=payload,
                session_id=session_id,
                region=args.region,
            )
        else:
            from agent.web_agent import run_agent
            result = run_agent(
                mission=args.mission,
                thread_id=args.thread_id,
                sub_agent=args.sub_agent,
            )

        print_result(result)
        if args.output:
            save_results(result, args.output)
        sys.exit(0 if not result.get("error") else 1)

    except KeyboardInterrupt:
        # Ctrl+C — asyncio.run() will cancel running tasks and their finally
        # blocks will call bc.stop() on each browser session. A second Ctrl+C
        # can bypass that; any orphaned sessions self-terminate after 15 minutes.
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

    except Exception as exc:
        if _is_credential_error(exc):
            print(_credential_error_message(), file=sys.stderr)
            sys.exit(1)
        logger.exception("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()

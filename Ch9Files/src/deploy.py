"""
deploy.py — Programmatic AgentCore Runtime deployment (boto3).

This script registers the containerised agent as an AgentCore Runtime by
calling create_agent_runtime directly. It exists for pedagogy: showing what
the high-level `agentcore deploy` CLI does under the hood. For day-to-day
deployment, the CLI is the recommended path.

What this script does:
1. Reads the AgentCore Memory ARN from resources.json (written by
   setup_resources.py --create), or creates the Memory resource if that file
   is not present.
2. Prints the manual ARM64 container build commands (the CLI does this for
   you via AWS CodeBuild).
3. Calls create_agent_runtime to register the container with AgentCore.
4. Prints the Runtime ARN and invocation example.

Prerequisites:
  - Run `python setup_resources.py --create` to provision Browser and
    Memory resources and write resources.json.
  - AWS credentials with the permissions in iam_caller_policy.json.
  - An execution role with iam_execution_role_policy.json attached and
    iam_execution_role_trust.json as its trust policy. Update
    EXECUTION_ROLE_TEMPLATE below to match the role name you create.
  - Docker (or Finch/Podman) for ARM64 image builds — the runtime requires
    linux/arm64 images. Use `docker buildx` on x86 dev machines.

Usage:
  python deploy.py --account-id 123456789012 [--region us-east-1]

Two deployment paths:
  PATH A (this script): Programmatic boto3 deployment — full control, suitable
  for CI/CD pipelines. Shows what create_agent_runtime expects.

  PATH B (CLI): The bedrock-agentcore-starter-toolkit CLI wraps this into a
  single command:  `agentcore configure -e agent/web_agent.py && agentcore deploy`
  Faster for iteration; handles ARM64 builds via CodeBuild without local Docker.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_REGION = "us-east-1"
DEFAULT_STACK_NAME = "supertron-web-agent"
AGENT_RUNTIME_NAME = "web-browsing-agent"
# Match setup_resources.py's MEMORY_NAME so the lookup-or-create fallback
# below finds the resource that setup_resources.py created.
MEMORY_RESOURCE_NAME = "web_agent_memory"

# ECR repository for the agent container image.
# account_id and region will be filled at runtime from the command line arguments.
ECR_REPO_TEMPLATE = "{account_id}.dkr.ecr.{region}.amazonaws.com/supertron-web-agent:latest"

# IAM role ARN for the AgentCore runtime execution role.
# This role must have iam_execution_role_policy.json attached and
# iam_execution_role_trust.json as its trust policy.
EXECUTION_ROLE_TEMPLATE = "arn:aws:iam::{account_id}:role/supertron-web-agent-role"

RESOURCES_FILE = Path(__file__).parent / "resources.json"


# ── Step 1: Resolve AgentCore Memory resource ─────────────────────────────────


def get_memory_arn(control_client, region: str) -> str:
    """
    Return the Memory ARN for the AgentCore runtime.

    Reads resources.json first (written by setup_resources.py --create).
    Falls back to finding-or-creating a Memory resource if resources.json is
    missing, so this script still works standalone — and persists what it
    found/created back to resources.json so subsequent deploy.py runs reuse it.
    """
    if RESOURCES_FILE.exists():
        data = json.loads(RESOURCES_FILE.read_text())
        if arn := data.get("memory_arn"):
            logger.info("Using memory ARN from resources.json: %s", arn)
            return arn
    logger.info(
        "resources.json not found or missing memory_arn — "
        "creating/finding Memory resource '%s'...",
        MEMORY_RESOURCE_NAME,
    )
    arn, memory_id = _create_or_get_memory(control_client, MEMORY_RESOURCE_NAME)
    _persist_memory_to_resources_json(arn, memory_id, region)
    return arn


def _persist_memory_to_resources_json(memory_arn: str, memory_id: str, region: str) -> None:
    """Write (or merge into) resources.json so the next run reuses this memory."""
    data: dict = {}
    if RESOURCES_FILE.exists():
        try:
            data = json.loads(RESOURCES_FILE.read_text())
        except json.JSONDecodeError:
            logger.warning("resources.json is malformed; overwriting.")
    data.update({"region": region, "memory_arn": memory_arn, "memory_id": memory_id})
    RESOURCES_FILE.write_text(json.dumps(data, indent=2) + "\n")
    logger.info("Wrote memory ARN to %s", RESOURCES_FILE)


def _create_or_get_memory(control_client, memory_name: str) -> tuple[str, str]:
    """
    Find or create an AgentCore Memory resource and return (arn, id).

    Prefer running `setup_resources.py --create` — this fallback exists only so
    `deploy.py` can run standalone. list_memories returns ids only, so we fetch
    each memory by id to match on name.
    """
    logger.info("Looking for existing memory resource '%s'...", memory_name)
    try:
        paginator = control_client.get_paginator("list_memories")
        for page in paginator.paginate():
            for item in page.get("memories", []):
                mem = control_client.get_memory(memoryId=item["id"])["memory"]
                if mem.get("name") == memory_name:
                    arn = mem["arn"]
                    logger.info("Found existing memory: %s", arn)
                    return arn, mem["id"]
    except Exception as exc:
        logger.warning("list_memories failed: %s — will attempt create.", exc)

    logger.info("Creating memory resource '%s'...", memory_name)
    mem = control_client.create_memory(
        name=memory_name,
        description="Checkpoint store for the Supertron web browsing agent",
        eventExpiryDuration=30,
    )["memory"]
    logger.info("Memory created: %s", mem["arn"])
    return mem["arn"], mem["id"]


# ── Step 2: Build and push container image ────────────────────────────────────


def build_and_push_image(account_id: str, region: str) -> str:
    """
    Print the manual ARM64 build/push commands and return the target ECR URI.

    AgentCore Runtime requires linux/arm64 images (AWS Graviton). On an x86
    dev machine, plain `docker build` produces an x86 image and the runtime
    will fail with "exec format error". Use `docker buildx --platform
    linux/arm64` (or build via AWS CodeBuild as the CLI does).

    The starter toolkit CLI handles all of this for you — these manual
    commands are shown here only to make the steps explicit.

    Returns:
        Full ECR image URI.
    """
    image_uri = ECR_REPO_TEMPLATE.format(account_id=account_id, region=region)
    logger.info(
        "NOTE: Container build step shown here as reference.\n"
        "Run these commands in your terminal (ARM64 build required):\n"
        "  aws ecr create-repository --repository-name supertron-web-agent --region %s\n"
        "  aws ecr get-login-password --region %s | docker login --username AWS "
        "--password-stdin %s.dkr.ecr.%s.amazonaws.com\n"
        "  docker buildx build --platform linux/arm64 -t supertron-web-agent .\n"
        "  docker tag supertron-web-agent:latest %s\n"
        "  docker push %s\n",
        region, region, account_id, region, image_uri, image_uri,
    )
    return image_uri


# ── Step 3: Create AgentCore Runtime ─────────────────────────────────────────


def create_agent_runtime(
    client,
    image_uri: str,
    execution_role_arn: str,
    memory_arn: str,
    region: str,
    runtime_name: str = AGENT_RUNTIME_NAME,
    nova_act_api_key: str | None = None,
) -> dict:
    """
    Register the containerised agent as an AgentCore Runtime.

    AgentCore Runtime is HTTP-based, not Lambda. The container is expected to
    serve POST /invocations and GET /ping on port 8080 — there is no
    `agentRuntimeHandler` parameter; the runtime invokes the container's
    HTTP endpoint directly. Our agent uses BedrockAgentCoreApp + @app.entrypoint
    in agent/web_agent.py to expose those routes.

    Args populate:
    - agentRuntimeArtifact.containerConfiguration.containerUri — the ARM64 image
    - networkConfiguration.networkMode — PUBLIC (the agent makes outbound calls
      to vendor websites and Bedrock)
    - roleArn — IAM role the runtime assumes to invoke Bedrock,
      Browser, Code Interpreter, and Memory at runtime
    - environmentVariables — AGENTCORE_MEMORY_ID for AgentCoreMemorySaver,
      NOVA_ACT_API_KEY for the Nova Act sub-agent (optional; production
      deployments should use IAM-based auth via Nova Act's workflow= parameter)

    Returns:
        Dict with 'agentRuntimeArn' and 'agentRuntimeId'.
    """
    logger.info("Creating AgentCore Runtime '%s'...", runtime_name)

    env_vars = {
        # AgentCoreMemorySaver expects the memory ID, not the full ARN.
        # Extract the ID portion (the segment after the last '/').
        "AGENTCORE_MEMORY_ID": memory_arn.rsplit("/", 1)[-1] if "/" in memory_arn else memory_arn,
        "AWS_DEFAULT_REGION": region,
    }
    if nova_act_api_key:
        env_vars["NOVA_ACT_API_KEY"] = nova_act_api_key

    response = client.create_agent_runtime(
        agentRuntimeName=runtime_name,
        description="Supertron hardware engineering web browsing agent",
        agentRuntimeArtifact={
            "containerConfiguration": {
                "containerUri": image_uri,
            }
        },
        networkConfiguration={"networkMode": "PUBLIC"},
        roleArn=execution_role_arn,
        environmentVariables=env_vars,
    )

    runtime_arn = response["agentRuntimeArn"]
    runtime_id = response["agentRuntimeId"]
    logger.info("Runtime created: %s", runtime_arn)

    # Poll until the runtime reaches READY state.
    _wait_for_runtime_active(client, runtime_id)

    return {"agentRuntimeArn": runtime_arn, "agentRuntimeId": runtime_id}


def _wait_for_runtime_active(client, runtime_id: str, timeout_sec: int = 300):
    """Poll the runtime status until it reaches READY or times out.

    Per the AgentCore Runtime guide, the lifecycle is:
      CREATING → READY (or CREATE_FAILED) → UPDATING → READY
    """
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = client.get_agent_runtime(agentRuntimeId=runtime_id)
        status = response.get("status", "UNKNOWN")
        logger.info("Runtime status: %s", status)
        if status == "READY":
            return
        if status in ("CREATE_FAILED", "DELETE_FAILED", "UPDATE_FAILED"):
            raise RuntimeError(
                f"Runtime entered terminal state: {status}. "
                f"failureReason: {response.get('failureReason', 'n/a')}"
            )
        time.sleep(15)
    raise TimeoutError(f"Runtime did not reach READY within {timeout_sec}s")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Deploy web browsing agent to AgentCore")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--stack-name", default=DEFAULT_STACK_NAME)
    parser.add_argument("--account-id", required=True, help="Your 12-digit AWS account ID")
    args = parser.parse_args()

    # Control plane client: used for CreateMemory, CreateAgentRuntime, etc.
    # The runtime/data plane client ("bedrock-agentcore") is used separately for
    # InvokeAgentRuntime in run_agent.py.
    control_client = boto3.client("bedrock-agentcore-control", region_name=args.region)

    # Step 1: Resolve Memory ARN from resources.json (or create one)
    memory_arn = get_memory_arn(control_client, args.region)

    # Step 2: Container image (manual instructions printed)
    image_uri = build_and_push_image(args.account_id, args.region)
    execution_role_arn = EXECUTION_ROLE_TEMPLATE.format(account_id=args.account_id)

    # Step 3: Runtime
    runtime_info = create_agent_runtime(
        client=control_client,
        image_uri=image_uri,
        execution_role_arn=execution_role_arn,
        memory_arn=memory_arn,
        region=args.region,
    )

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Runtime ARN: {runtime_info['agentRuntimeArn']}")
    print(f"  Runtime ID:  {runtime_info['agentRuntimeId']}")
    print(f"  Memory ARN:  {memory_arn}")
    print("\nTo invoke the agent:")
    print(
        f"  python run_agent.py "
        f"--runtime-arn {runtime_info['agentRuntimeArn']} "
        f"--mission \"Find all DDR4 parts matching 8Gb, 3200 MT/s or faster, 78-ball FBGA\""
    )
    print("=" * 60)


if __name__ == "__main__":
    main()

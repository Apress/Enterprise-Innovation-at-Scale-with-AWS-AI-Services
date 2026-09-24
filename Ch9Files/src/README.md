# Multi-Vendor Electronic Component Search and Summarization with Amazon Bedrock AgentCore

This directory contains the working Python agent for Chapter 9 of *Enterprise Innovation at Scale with AWS Generative AI*.

## What This Agent Does

The Supertron multi-vendor memory part search agent takes a natural-language component specification (e.g. "Find all DDR4 parts with 8Gb density, 3200 MT/s or faster, 78-ball FBGA") and searches the product catalogs of multiple pre-specified vendors in parallel, then consolidates the results into a unified comparison table.

Architecture:

- **Orchestrator** (`agent/web_agent.py`) — a LangGraph `StateGraph` that fans out to N parallel vendor sub-agents via the `Send` API, then synthesizes their results.
- **Per-vendor sub-agents** — one session per vendor URL. Two interchangeable implementations, selected via `--sub-agent`:
  - `agent/nova_vendor_agent.py` (default) — Nova Act navigates the vendor site with natural-language `act()` calls and returns structured JSON matching a Pydantic schema.
  - `agent/claude_vendor_agent.py` — a LangChain `create_agent` driving Playwright directly, with Claude choosing each click/type/navigate step. Wrapped in `BedrockPromptCachingMiddleware` so growing conversation history is served from Bedrock's prompt cache on every turn after the first. Uses the AgentCore Code Interpreter to post-process extracted page content into structured JSON.
- **AgentCore Browser Tool** — on-demand isolated Chromium sessions. Nova Act connects to the managed browser via CDP (`cdp_endpoint_url`) rather than spinning up its own. Browser egress goes through AWS network addresses, enabling Web Bot Authentication for participating WAF vendors.
- **Amazon Nova Act** — AI-driven browser automation. Issues natural-language `act()` instructions against the live browser session and returns structured data without brittle CSS selectors.
- **AgentCoreMemorySaver** — LangGraph checkpointer backed by AgentCore Memory; used on the orchestrator for crash resilience and resumable runs. Falls back to in-process `MemorySaver` when `AGENTCORE_MEMORY_ID` isn't set.
- **Claude Sonnet 4.6** (`us.anthropic.claude-sonnet-4-6`) for final synthesis of all vendor results.

## Directory Structure

```
src/
├── README.md
├── .env.example
├── requirements.txt
├── run_agent.py                  # Local-run CLI
├── setup_resources.py            # Create/destroy AgentCore Browser and Memory resources
├── iam_caller_policy.json        # Permissions for the principal that runs the agent
└── agent/
    ├── __init__.py
    ├── web_agent.py              # Orchestrator graph + entry point
    ├── nova_vendor_agent.py      # Per-vendor Nova Act sub-agent (default)
    ├── claude_vendor_agent.py    # Alternative Claude-based ReAct sub-agent
    ├── config.py                 # Model IDs, region, WEBSITES list, example missions
    ├── state.py                  # OrchestratorState, VendorLookupInput, VendorResult
    └── tools/
        ├── __init__.py
        ├── browser.py            # browser_cdp_session, playwright_browser_session
        └── code_interpreter.py   # process_with_code_interpreter tool + session helpers
```

## Prerequisites

1. **AWS account** with Bedrock model access for `us.anthropic.claude-sonnet-4-6` enabled in the region you set (default is `us-east-1`).
2. **IAM credentials** with the permissions in `iam_caller_policy.json` attached to your developer user/role.
3. **Nova Act API key** (see below).
4. **Python 3.11+**.

```bash
pip install -r requirements.txt
playwright install chromium
```

### Getting a Nova Act API key

Nova Act uses a **separate API key** from your AWS credentials. It is tied to your amazon.com consumer account (not your AWS account) and must be obtained manually:

1. Go to [nova.amazon.com/act](https://nova.amazon.com/act) and sign in with your **amazon.com** account.
2. Select **Act** in the Labs navigation pane and generate an API key.
3. Export it before running the agent:

```bash
export NOVA_ACT_API_KEY=<your-key>
```

Nova Act is generally available in the US with no charge for API-key-based usage (subject to daily limits). There is no boto3 or AWS CLI method to generate this key — the portal is the only path.

## Provision AWS Resources

Before running the agent, create the required AgentCore resources with the setup script:

```bash
python setup_resources.py --create
```

This creates two resources in your AWS account and writes their IDs to `resources.json`:

| Resource | Purpose |
|----------|---------|
| **AgentCore Browser** | Managed Chromium pool used by vendor sub-agents |
| **AgentCore Memory** | LangGraph checkpoint store for resumable runs (used when `AGENTCORE_MEMORY_ID` is set; otherwise the agent falls back to in-process `MemorySaver`) |

`agent/config.py` reads the browser ID from `resources.json` automatically — no manual editing required.

To clean up both resources when you're done:

```bash
python setup_resources.py --destroy
```

## Running the Agent

`run_agent.py` runs the orchestrator in-process from the `src/` directory:

```bash
# Default mission (DDR4 sweep with the Nova Act sub-agent)
python run_agent.py

# Custom mission
python run_agent.py --mission "Find all LPDDR5 parts with 16Gb density, 6400 MT/s or faster"

# Use the Claude/Playwright sub-agent instead
python run_agent.py --sub-agent claude
```

### Saving results to files

Pass `--output <base>` (no file extension) to write two files after the run:

| File | Contents |
|------|----------|
| `<base>.json` | Full result dict — `final_result`, `vendor_results`, `thread_id` |
| `<base>.md`   | Agent's synthesized markdown report only — easy to open and share |

Directories are created automatically. S3 URIs (`s3://bucket/prefix/name`) are also accepted and require `s3:PutObject` on the target bucket.

### Resuming a run

The thread ID is printed at the end of every run. Pass it back to resume from the last checkpoint:

```bash
python run_agent.py --thread-id <THREAD_ID>
```

In-memory checkpointing is the default and does not survive process exit. For resumption across process restarts, export `AGENTCORE_MEMORY_ID` (the value is printed by `setup_resources.py --create` and written to `resources.json`):

```bash
AGENTCORE_MEMORY_ID=<your-memory-id> python run_agent.py --thread-id <THREAD_ID>
```

## Configuring the Vendor List

The list of vendor catalog URLs is in `agent/config.py` (`WEBSITES`). Edit that list directly to add, remove, or reorder vendors. The agent will automatically spawn one sub-agent per URL.

## Deploying the Agent

AgentCore Runtime can host this agent in production. Container build, runtime registration, and IAM execution role setup track the AWS CLI release cycle and are outside the scope of this README — see the [AgentCore developer guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/) for current instructions.

## Important Notes

| Topic | Note |
|---|---|
| First-time setup | Run `python setup_resources.py --create` once before using the agent. This creates the required Browser and Memory resources and writes `resources.json`. |
| Nova Act API key | Required if you are using the Nova Act sub-agent. Obtain manually at nova.amazon.com/act using your amazon.com account. Set as `NOVA_ACT_API_KEY` env var. There is no programmatic way to generate this key. |
| Run command | Use `python run_agent.py` from the `src/` directory. |
| Vendor URL changes | It's possible vendor catalog URLs may change over time. If a sub-agent returns an error or no match unexpectedly, verify the URL in `agent/config.py` is still correct. |
| Windows console | The agent's output may include Unicode characters. If you see encoding errors on Windows, run your terminal with UTF-8 (`chcp 65001`). |
| Output files | `--output` always writes both `.json` and `.md`. The `.md` file contains only the synthesized markdown report. |
| AWS credentials | If credentials are expired, `run_agent.py` will give an error informing you to log in. Re-authenticate with `aws login` (browser-based local-dev auth in AWS CLI v2.30+), `aws sso login --profile <profile>` (IAM Identity Center), or `aws configure` (static keys). |

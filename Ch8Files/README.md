# Bedrock Agent Job Review Sample

This folder includes `review_job.py`, a minimal Python example that sends one or more job descriptions to an AWS Bedrock Agent and writes each response to a new Markdown file.

## 1) Prerequisites

- Python 3.9+
- AWS account with access to Amazon Bedrock Agents
- AWS CLI installed

Install boto3:

```bash
pip install boto3
```

## 2) Configure AWS CLI Credentials

Boto3 uses AWS's default credential chain. The easiest local setup is through AWS CLI.

### Option A: Login Directly (Easiest)

```bash
aws login
```

This will let the AWS CLI tool use an existing user log in to run commands.

### Option B: Access Keys

```bash
aws configure
```

Provide:
- AWS Access Key ID
- AWS Secret Access Key
- Default region (for example `us-east-1`)
- Output format (for example `json`)

You can get these from the "My Security Credentials" in IAM.

### Option C: AWS IAM Identity Center (SSO)

```bash
aws configure sso
aws sso login
```

After login, boto3 will use your SSO session credentials.

## 3) Required Permissions

Minimum IAM permission for this script:

- `bedrock:InvokeAgent`

Scope it to your Bedrock Agent resource when possible, for example:
- `arn:aws:bedrock:<region>:<account-id>:agent/<agent-id>`

If access is denied, confirm:
- Your identity (user/role/profile) has `bedrock:InvokeAgent`
- The policy resource includes your target agent
- Region in script matches where the agent exists

## 4) Configure Script Constants

Open `review_job.py` and set:

- `AGENT_ID` — your Bedrock Agent ID (or pass it at runtime with `--agentid`, see below)
- `AWS_REGION` — the region where your agent is deployed

The prompt is already hardcoded in the script.

## 5) Usage

Single file:

```bash
python review_job.py Test_Cases/test-case-01-noncompliant-age-bias.txt
```

Wildcard / multiple files:

```bash
python review_job.py Test_Cases/*.txt
```

Override the agent ID at the command line (no need to edit the script):

```bash
python review_job.py --agentid ABCDE12345 Test_Cases/*.txt
```

The script accepts `.txt` and `.md` files (including glob patterns) and writes output next to each input as:

- `<original-stem>.agent-output.md`

Examples:
- `test-case-01-noncompliant-age-bias.txt` -> `test-case-01-noncompliant-age-bias.agent-output.md`
- `example.md` -> `example.agent-output.md`

## 6) Common Errors

- `Set AGENT_ID in review_job.py or pass --agentid ...`
  - Set `AGENT_ID` in `review_job.py`, or pass `--agentid <id>` on the command line

- `Unable to locate credentials`
  - Run `aws login` or `aws configure` or `aws sso login`

- `AccessDeniedException`
  - Update IAM policy to include `bedrock:InvokeAgent`

- `ResourceNotFoundException` or alias errors
  - Verify `AGENT_ID`, `AGENT_ALIAS_ID`, and `AWS_REGION`

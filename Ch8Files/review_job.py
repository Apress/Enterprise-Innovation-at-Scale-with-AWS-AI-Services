#!/usr/bin/env python3
"""Bedrock Agent example: review and rewrite job descriptions."""

import argparse
import glob
import sys
import uuid
from pathlib import Path

import boto3
from botocore.config import Config

AGENT_ID = "REPLACE_WITH_AGENT_ID"
AGENT_ALIAS_ID = "TSTALIASID"
AWS_REGION = "us-east-1"
PROMPT = (
    "Review the attached job description. Return your full response in "
    "clean, well-structured Markdown with clear headings and bullet points as appropriate."
)


def collect_agent_response(response: dict) -> str:
    text_parts = []
    seen: set[str] = set()
    source_names: list[str] = []

    for event in response.get("completion", []):
        chunk = event.get("chunk")
        if not chunk:
            continue
        if "bytes" in chunk:
            text_parts.append(chunk["bytes"].decode("utf-8", errors="ignore"))
        for citation in chunk.get("attribution", {}).get("citations", []):
            for ref in citation.get("retrievedReferences", []):
                uri = ref.get("location", {}).get("s3Location", {}).get("uri", "")
                if uri:
                    doc_name = Path(uri).name
                    if doc_name not in seen:
                        seen.add(doc_name)
                        source_names.append(doc_name)

    output = "".join(text_parts).strip()
    if source_names:
        sources = "\n".join(f"- {name}" for name in source_names)
        output += f"\n\n---\n\n**Sources**\n\n{sources}"
    return output


def process_file(path: Path, client, agent_id: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Processing: {path.name}")
    content = path.read_text(encoding="utf-8")
    input_text = f"{PROMPT}\n\n---\n\nJob description:\n\n```{content}\n```"

    print("  Invoking agent...")
    response = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=str(uuid.uuid4()),
        inputText=input_text,
    )
    print("  Collecting response... (this may take a few minutes)")
    output_text = collect_agent_response(response)
    if not output_text:
        raise RuntimeError("Agent returned no text output.")

    output_path = path.with_name(f"{path.stem}.agent-output.md")
    output_path.write_text(output_text, encoding="utf-8")
    print(f"  Done -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review and rewrite job descriptions using a Bedrock Agent."
    )
    parser.add_argument(
        "--agentid",
        default=None,
        help="Override the hard-coded AGENT_ID (e.g. --agentid ABCDE12345)",
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Job description file(s) or glob pattern(s) to process",
    )
    args = parser.parse_args()

    agent_id = args.agentid or AGENT_ID
    if "REPLACE_WITH_" in agent_id:
        print("Set AGENT_ID in review_job.py or pass --agentid <id> before running.")
        return 2

    files = [Path(p) for arg in args.files for p in (glob.glob(arg) or [arg])]
    if not files:
        print("No files found to process.")
        return 2

    print(f"Using agent: {agent_id}")
    print(f"Files to process: {len(files)}\n")
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=AWS_REGION,
        config=Config(read_timeout=300),
    )

    failures = 0
    for file_path in files:
        try:
            process_file(file_path, client, agent_id)
        except Exception as exc:
            failures += 1
            print(f"  Failed: {file_path} -> {exc}")

    print(f"\nDone. Processed={len(files)} Failed={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

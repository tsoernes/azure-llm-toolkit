#!/usr/bin/env python3
"""
Manual test script to compare `max_tokens` vs `max_completion_tokens`
for Azure OpenAI chat deployments using the official `openai` library.

This script:

- Loads configuration from the `.env` file in the repo using `python-dotenv`.
- Configures an AzureOpenAI client.
- For two deployments (GPT‑4o and GPT‑5‑mini), it:
    - Calls with `max_tokens`
    - Calls with `max_completion_tokens`
- Prints whether each call succeeded or failed, with any error messages.

Usage (from repo root):

    cd /home/torstein.sornes/code/azure-llm-toolkit
    uv run --with openai --with python-dotenv tests/manual_test_max_tokens_vs_max_completion.py

Required environment variables (via `.env` or shell):

    AZURE_OPENAI_API_KEY=...
    AZURE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
    AZURE_API_VERSION=2025-03-01-preview

    # Chat deployments:
    AZURE_RERANKER_DEPLOYMENT=gpt-4o-east-US   # your GPT-4o deployment
    AZURE_CHAT_DEPLOYMENT=gpt-5-mini           # your GPT‑5-mini deployment

Notes:
- This script makes REAL Azure OpenAI API calls and will incur cost.
- It is informational: it does not assert; it just prints what works.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI


@dataclass
class AzureEnvConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment_gpt4o: str
    deployment_gpt5mini: str

    @classmethod
    def load(cls) -> "AzureEnvConfig":
        """Load Azure configuration from environment (and .env via dotenv)."""
        load_dotenv()
        missing: list[str] = []

        def env(name: str) -> str:
            val = os.getenv(name)
            if not val:
                missing.append(name)
            return val or ""

        endpoint = env("AZURE_ENDPOINT") or env("AZURE_OPENAI_ENDPOINT")
        api_key = env("AZURE_OPENAI_API_KEY") or env("OPENAI_API_KEY")
        api_version = env("AZURE_API_VERSION") or env("AZURE_OPENAI_API_VERSION")
        deployment_gpt4o = env("AZURE_RERANKER_DEPLOYMENT", "gpt-4o-east-US")
        deployment_gpt5mini = env("AZURE_CHAT_DEPLOYMENT") or env("AZURE_OPENAI_DEPLOYMENT_GPT5MINI")

        if missing:
            raise SystemExit("Missing environment variables:\n  " + "\n  ".join(sorted(set(missing))))

        if not endpoint.startswith("https://"):
            raise SystemExit(f"Invalid endpoint: {endpoint!r}")

        return cls(
            endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_gpt4o=deployment_gpt4o,
            deployment_gpt5mini=deployment_gpt5mini,
        )


def make_client(cfg: AzureEnvConfig) -> AzureOpenAI:
    """Create an AzureOpenAI client configured for your Azure resource."""
    return AzureOpenAI(
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        azure_endpoint=cfg.endpoint,
    )


def call_with_param(client: AzureOpenAI, deployment: str, param: str) -> dict[str, Any]:
    """
    Make a chat completion call with either `max_tokens` or `max_completion_tokens`.

    `param` must be "max_tokens" or "max_completion_tokens".
    """
    args: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {
                "role": "user",
                "content": "Say 'hello' in one word.",
            }
        ],
    }

    if param == "max_tokens":
        args["max_tokens"] = 16
    elif param == "max_completion_tokens":
        args["max_completion_tokens"] = 16
    else:
        raise ValueError(f"Unsupported param: {param}")

    try:
        completion = client.chat.completions.create(**args)
        return {
            "ok": True,
            "deployment": deployment,
            "param": param,
            "status": "success",
            "id": getattr(completion, "id", None),
            "model": getattr(completion, "model", None),
            "usage": getattr(completion, "usage", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "deployment": deployment,
            "param": param,
            "status": "exception",
            "error": str(exc),
        }


def print_result(result: dict[str, Any]) -> None:
    """Pretty-print a single call result."""
    deployment = result.get("deployment")
    param = result.get("param")
    header = f"[{deployment}] using {param}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"ok:     {result.get('ok')}")
    print(f"status: {result.get('status')}")
    if not result.get("ok"):
        print(f"error:  {result.get('error')}")
    else:
        print(f"model:  {result.get('model')}")
        print(f"usage:  {result.get('usage')}")
    print()


def main() -> None:
    cfg = AzureEnvConfig.load()
    client = make_client(cfg)

    results: list[dict[str, Any]] = []

    # Test GPT-4o deployment
    print(f"\n=== Testing GPT‑4o deployment: {cfg.deployment_gpt4o} ===")
    for param in ("max_tokens", "max_completion_tokens"):
        res = call_with_param(client, cfg.deployment_gpt4o, param)
        results.append(res)
        print_result(res)

    # Test GPT-5-mini deployment
    print(f"\n=== Testing GPT‑5‑mini deployment: {cfg.deployment_gpt5mini} ===")
    for param in ("max_tokens", "max_completion_tokens"):
        res = call_with_param(client, cfg.deployment_gpt5mini, param)
        results.append(res)
        print_result(res)

    print("=== JSON summary ===")
    print(json.dumps(results, default=str, indent=2))


if __name__ == "__main__":
    main()

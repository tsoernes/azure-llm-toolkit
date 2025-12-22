#!/usr/bin/env python3
"""
Test `max_tokens` vs `max_completion_tokens` against real Azure OpenAI endpoints
using the official `openai` Python library and configuration from the `.env` file.

This test is **informational**, not assertive. It will:

- Load Azure settings from `.env` via python-dotenv.
- Configure the OpenAI client for Azure.
- Call your GPT‑4o deployment twice:
    1. With `max_tokens`
    2. With `max_completion_tokens`
- Call your GPT‑5‑mini deployment twice:
    1. With `max_tokens`
    2. With `max_completion_tokens`
- Print status and any error messages for each combination so you can see which
  parameter name is accepted by which deployment.

To run:

    cd /home/torstein.sornes/code/azure-llm-toolkit
    python -m pytest tests/test_max_tokens_vs_max_completion.py -s

Required in `.env` (already mostly present in your repo, adjust as needed):

    AZURE_OPENAI_API_KEY=...
    AZURE_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
    AZURE_API_VERSION=2025-03-01-preview

    # Chat deployments:
    AZURE_RERANKER_DEPLOYMENT=gpt-4o-east-US  # GPT-4o deployment name
    AZURE_CHAT_DEPLOYMENT=gpt-5-mini          # GPT-5-mini deployment name
    AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small

NOTE: This test will make **real Azure OpenAI requests** and incur real cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import pytest
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AzureEnvConfig:
    """Azure OpenAI configuration loaded from .env / environment."""

    endpoint: str
    api_key: str
    api_version: str
    deployment_gpt4o: str
    deployment_gpt5mini: str

    @classmethod
    def load(cls) -> AzureEnvConfig:
        # Load .env from project root if present
        load_dotenv()

        missing: list[str] = []

        def env(name: str) -> str:
            val = os.getenv(name)
            if not val:
                missing.append(name)
            return val or ""

        cfg = cls(
            endpoint=env("AZURE_ENDPOINT") or env("AZURE_OPENAI_ENDPOINT"),
            api_key=env("AZURE_OPENAI_API_KEY") or env("OPENAI_API_KEY"),
            api_version=env("AZURE_API_VERSION") or env("AZURE_OPENAI_API_VERSION"),
            deployment_gpt4o=env("AZURE_RERANKER_DEPLOYMENT") or "gpt-4o-east-US",
            deployment_gpt5mini=env("AZURE_CHAT_DEPLOYMENT") or env("AZURE_OPENAI_DEPLOYMENT_GPT5MINI"),
        )

        # skip guard removed: always load config defaults to run tests

        if not cfg.endpoint.startswith("https://"):
            pytest.skip(f"Invalid AZURE endpoint: {cfg.endpoint!r}")

        return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_azure_client(cfg: AzureEnvConfig) -> AzureOpenAI:
    """
    Create an AzureOpenAI client from the env config.

    Uses the `azure` (Azure OpenAI) mode of the official SDK.
    """
    client = AzureOpenAI(
        api_key=cfg.api_key,
        api_version=cfg.api_version,
        azure_endpoint=cfg.endpoint,
    )
    return client


@dataclass
class CallResult:
    deployment: str
    param_name: str
    status: str
    ok: bool
    error_message: str | None
    raw: Any | None


def chat_with_param(
    client: AzureOpenAI,
    deployment: str,
    param_name: str,
) -> CallResult:
    """
    Make a chat completion using the SDK, injecting either `max_tokens` or
    `max_completion_tokens` into the request.

    param_name must be either "max_tokens" or "max_completion_tokens".
    """
    # Base arguments
    args: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "user", "content": "Say 'hello' in one word."},
        ],
    }

    if param_name == "max_tokens":
        args["max_tokens"] = 16
    elif param_name == "max_completion_tokens":
        args["max_completion_tokens"] = 16
    else:
        raise ValueError(f"Unsupported parameter name: {param_name!r}")

    try:
        completion = client.chat.completions.create(**args)
        # Successful call; capture a compact summary
        status = "success"
        ok = True
        error_message = None
        raw = {
            "id": getattr(completion, "id", None),
            "model": getattr(completion, "model", None),
            "usage": getattr(completion, "usage", None),
        }
    except Exception as exc:  # noqa: BLE001
        status = "exception"
        ok = False
        error_message = str(exc)
        raw = None

    return CallResult(
        deployment=deployment,
        param_name=param_name,
        status=status,
        ok=ok,
        error_message=error_message,
        raw=raw,
    )


def print_result(result: CallResult) -> None:
    """Pretty-print the result (works best with pytest -s)."""
    header = f"[{result.deployment}] using {result.param_name}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print(f"Status: {result.status}, ok={result.ok}")
    if result.error_message:
        print(f"Error:  {result.error_message}")
    if result.raw is not None:
        try:
            snippet = json.dumps(result.raw, default=str, indent=2)
            print("Response snippet:")
            print(snippet[:1000])
        except Exception:
            print("Response raw:", result.raw)
    print()


# ---------------------------------------------------------------------------
# Tests (informational)
# ---------------------------------------------------------------------------


@pytest.mark.azure
def test_gpt4o_max_tokens_vs_max_completion_tokens() -> None:
    """
    Exercise both `max_tokens` and `max_completion_tokens` on the GPT‑4o deployment.

    This is **informational**: it prints the behavior for each parameter name,
    rather than asserting, because Azure behavior may vary by API version and
    deployment.

    Typical expectation (to be confirmed by running this):

    - gpt-4o-* often still accepts `max_tokens`.
    - `max_completion_tokens` may or may not be wired for that deployment yet.
    """
    cfg = AzureEnvConfig.load()
    client = make_azure_client(cfg)

    print(f"\n=== Testing GPT‑4o deployment: {cfg.deployment_gpt4o} ===\n")

    for param in ("max_tokens", "max_completion_tokens"):
        result = chat_with_param(client, cfg.deployment_gpt4o, param)
        print_result(result)

    # No hard assertion; inspect output and then tighten behavior in your client.


@pytest.mark.azure
def test_gpt5mini_max_tokens_vs_max_completion_tokens() -> None:
    """
    Exercise both `max_tokens` and `max_completion_tokens` on the GPT‑5‑mini deployment.

    This is especially interesting because newer models (5‑series, reasoning, etc.)
    tend to deprecate `max_tokens` in favor of `max_completion_tokens`.

    Run with:

        pytest tests/test_max_tokens_vs_max_completion.py::test_gpt5mini_max_tokens_vs_max_completion_tokens -s

    and inspect the printed output.
    """
    cfg = AzureEnvConfig.load()
    client = make_azure_client(cfg)

    print(f"\n=== Testing GPT‑5‑mini deployment: {cfg.deployment_gpt5mini} ===\n")

    for param in ("max_tokens", "max_completion_tokens"):
        result = chat_with_param(client, cfg.deployment_gpt5mini, param)
        print_result(result)

    # Again, no hard assertions; once you see how your deployments behave,
    # you can codify that logic into your client and optionally change this
    # test to assert on the expected behavior.

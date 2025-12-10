#!/usr/bin/env python
"""
OpenTelemetry + Jaeger demo for azure-llm-toolkit.

This example shows how to:

1. Start a Jaeger all-in-one container (already done from the host):
   podman run --rm -d --name jaeger-azure-llm-toolkit \
       -p 16686:16686 -p 4317:4317 -p 4318:4318 \
       docker.io/jaegertracing/all-in-one:1.57

   Jaeger UI will be available at:
   http://localhost:16686

2. Configure OpenTelemetry tracing in Python to export spans to Jaeger
   via the OTLP gRPC exporter on port 4317.

3. Use azure-llm-toolkit (AzureLLMClient and reranker) in a way that
   produces spans visible in Jaeger.

Prerequisites
-------------
Install the required packages in your environment:

    pip install \
        opentelemetry-sdk \
        opentelemetry-exporter-otlp-proto-grpc \
        opentelemetry-api

    # And azure-llm-toolkit itself (if not installed):
    pip install -e .

You also need to have your Azure OpenAI environment variables set,
or otherwise configure AzureConfig manually:

    export AZURE_OPENAI_API_KEY="your-key"
    export AZURE_ENDPOINT="https://your-resource.openai.azure.com"
    export AZURE_CHAT_DEPLOYMENT="gpt-4o"
    export AZURE_EMBEDDING_DEPLOYMENT="text-embedding-3-large"

Then run:

    python examples/otel_jaeger_demo.py --mode real

If you don't have Azure credentials configured, you can still run in
dummy mode (no network calls) to exercise tracing:

    python examples/otel_jaeger_demo.py --mode dummy

In dummy mode, the script monkeypatches the underlying client calls
to avoid real API calls, but still emits OTEL spans.

"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, List

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from azure_llm_toolkit import (
    AzureConfig,
    AzureLLMClient,
    LogprobReranker,
    RerankerConfig,
)
from azure_llm_toolkit.opentelemetry_integration import (
    init_tracer_provider,
    get_tracer,
    start_span,
    attach_llm_result_attributes,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def run_real_demo() -> None:
    """
    Run a small real Azure demo that emits spans to Jaeger.

    This assumes AzureConfig can be constructed from environment variables
    (see module docstring for details).
    """
    logger.info("Running real Azure demo with tracing enabled.")

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=True, enable_cache=True)

    # Simple chat completion
    tracer = get_tracer("examples.otel_jaeger_demo")

    messages = [{"role": "user", "content": "Explain what a large language model is."}]
    system_prompt = "You are a helpful AI assistant."

    async with start_span(
        tracer,
        "demo.chat_completion",
        {"demo.step": "chat"},
    ) as span:
        response = await client.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
        )
        usage = response.usage
        attach_llm_result_attributes(
            span=span,
            operation="chat_completion",
            model=response.model,
            usage=usage,
            cost=None,
            success=True,
            error_type=None,
        )
        logger.info("Chat response: %s", response.content)

    # Simple embedding + reranking demo
    docs = [
        "Machine learning enables computers to learn from data.",
        "Cooking recipes often involve ingredients and steps.",
        "Neural networks are a class of machine learning models.",
    ]
    query = "What is machine learning?"

    # Embedding demo
    async with start_span(
        tracer,
        "demo.embed_texts",
        {"demo.step": "embed"},
    ) as span:
        embedding_result = await client.embed_texts(docs)
        usage = embedding_result.usage
        attach_llm_result_attributes(
            span=span,
            operation="embed_texts",
            model=embedding_result.model,
            usage=usage,
            cost=None,
            success=True,
            error_type=None,
        )
        logger.info("Embedded %d documents with model %s", len(docs), embedding_result.model)

    # Reranker demo
    reranker = LogprobReranker(
        client=client.client,  # underlying AsyncAzureOpenAI
        params=RerankerConfig(
            deployment=config.chat_deployment,
        ),
    )

    async with start_span(
        tracer,
        "demo.rerank",
        {"demo.step": "rerank"},
    ) as span:
        results = await reranker.rerank(query, docs, top_k=3)
        # We don't currently have direct usage/cost here, but we can annotate metadata
        try:
            span.set_attribute("rerank.num_docs", len(docs))
            span.set_attribute("rerank.top_k", 3)
        except Exception:
            pass

        logger.info("Reranked results:")
        for idx, doc, score in results:
            logger.info("  idx=%d score=%.3f doc=%s", idx, score, doc)

    logger.info("Real demo completed. Check Jaeger UI at http://localhost:16686")


async def run_dummy_demo() -> None:
    """
    Run a dummy demo that avoids real Azure calls but still emits spans.

    This monkeypatches the underlying `.client` attribute to avoid network.
    The goal is to exercise OTEL wiring in a safe environment.
    """
    logger.info("Running dummy demo with tracing enabled (no real Azure calls).")

    config = AzureConfig()
    client = AzureLLMClient(config=config, enable_rate_limiting=False, enable_cache=False)

    # Monkeypatch chat and embedding calls on the underlying AsyncAzureOpenAI client.
    # NOTE: This assumes the current client implementation has `client.chat.completions.create`
    # and `client.embeddings.create`. If the internals change, adjust accordingly.

    async def fake_chat_create(*args: Any, **kwargs: Any) -> Any:
        class Usage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15
            prompt_tokens_details = type("PTD", (), {"cached_tokens": 0})()

        class Choice:
            def __init__(self) -> None:
                self.message = type("Msg", (), {"content": "This is a dummy chat response."})
                self.finish_reason = "stop"
                self.logprobs = None

        class Resp:
            def __init__(self) -> None:
                self.usage = Usage()
                self.choices = [Choice()]
                self.model = kwargs.get("model", config.chat_deployment)

        return Resp()

    async def fake_embed_create(*args: Any, **kwargs: Any) -> Any:
        class Usage:
            prompt_tokens = 12
            completion_tokens = 0
            total_tokens = 12
            prompt_tokens_details = type("PTD", (), {"cached_tokens": 0})()

        class DataItem:
            def __init__(self, dim: int = 5) -> None:
                self.embedding = [0.1 * i for i in range(dim)]

        class Resp:
            def __init__(self) -> None:
                self.usage = Usage()
                self.data = [DataItem()]
                self.model = kwargs.get("model", config.embedding_deployment)

        return Resp()

    # Apply monkeypatch-style overrides
    # client.client: AsyncAzureOpenAI
    client.client.chat.completions.create = fake_chat_create  # type: ignore[attr-defined]
    client.client.embeddings.create = fake_embed_create  # type: ignore[attr-defined]

    tracer = get_tracer("examples.otel_jaeger_demo_dummy")

    # Chat
    async with start_span(
        tracer,
        "demo_dummy.chat_completion",
        {"llm.operation": "chat_completion", "llm.model": config.chat_deployment},
    ) as span:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello from dummy mode."}],
            system_prompt="You are a dummy test assistant.",
        )
        usage = response.usage
        attach_llm_result_attributes(
            span=span,
            operation="chat_completion",
            model=config.chat_deployment,
            usage=usage,
            cost=None,
            success=True,
            error_type=None,
        )
        logger.info("Dummy chat response: %s", response.content)

    # Embed
    async with start_span(
        tracer,
        "demo_dummy.embed_text",
        {"llm.operation": "embed_text", "llm.model": config.embedding_deployment},
    ) as span:
        emb = await client.embed_text("Dummy text for embedding.")
        # For the dummy embed, we don't have usage from embed_text directly,
        # but the monkeypatched underlying call does; it's used internally.
        attach_llm_result_attributes(
            span=span,
            operation="embed_text",
            model=config.embedding_deployment,
            usage=None,
            cost=None,
            success=True,
            error_type=None,
        )
        logger.info("Dummy embedding length: %d", len(emb))

    logger.info("Dummy demo completed. Check Jaeger UI at http://localhost:16686")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenTelemetry + Jaeger demo for azure-llm-toolkit.")
    parser.add_argument(
        "--mode",
        choices=["real", "dummy"],
        default="dummy",
        help="Run in 'real' Azure mode or 'dummy' mode (no network calls).",
    )
    parser.add_argument(
        "--service-name",
        default="azure-llm-toolkit-demo",
        help="Service name to use in OTEL resource attributes.",
    )
    parser.add_argument(
        "--otel-endpoint",
        default="http://localhost:4317",
        help="OTLP gRPC endpoint for OTEL exporter (e.g., Jaeger all-in-one).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize OpenTelemetry tracer provider to send spans to Jaeger/OTLP.
    exporter = OTLPSpanExporter(endpoint=args.otel_endpoint, insecure=True)
    init_tracer_provider(service_name=args.service_name, use_batch_span_processor=True, exporter=exporter)

    if args.mode == "real":
        asyncio.run(run_real_demo())
    else:
        asyncio.run(run_dummy_demo())


if __name__ == "__main__":
    main()

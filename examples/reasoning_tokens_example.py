"""
Example: Reasoning Token Tracking

Demonstrates how to track and log reasoning tokens from reasoning models (o1, GPT-5).
Reasoning tokens represent the "thinking" tokens used by the model during reasoning.

Requirements:
- Set AZURE_OPENAI_API_KEY environment variable
- Set AZURE_ENDPOINT environment variable
- Deploy a reasoning model (e.g., gpt-5-mini, o1-preview)
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, AzureLLMClient
from azure_llm_toolkit.metrics import MetricsCollector, create_collector_with_prometheus

# Load environment variables
load_dotenv()

# Configure logging to see reasoning token details
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


async def basic_reasoning_example():
    """Basic example showing reasoning token tracking."""
    print("\n=== Basic Reasoning Token Example ===\n")

    # Create client with a reasoning model
    config = AzureConfig(
        chat_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"),
        timeout_seconds=None,  # Infinite timeout for reasoning models
    )
    client = AzureLLMClient(config=config)

    # Make a chat completion that requires reasoning
    result = await client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": "Solve this step by step: If a train travels 120 miles in 2 hours, "
                "then speeds up and travels 180 miles in the next 2 hours, "
                "what is the average speed for the entire journey?",
            }
        ],
        reasoning_effort="high",  # Request high reasoning effort
        max_tokens=500,
    )

    # Print results with reasoning token breakdown
    print(f"Response: {result.content}\n")
    print("=" * 60)
    print("Token Usage:")
    print(f"  Prompt tokens:      {result.usage.prompt_tokens}")
    print(f"  Completion tokens:  {result.usage.completion_tokens}")
    print(f"  Reasoning tokens:   {result.usage.reasoning_tokens}")
    print(f"  Cached tokens:      {result.usage.cached_prompt_tokens}")
    print(f"  Total tokens:       {result.usage.total_tokens}")
    print("=" * 60)

    if result.usage.reasoning_tokens > 0:
        reasoning_pct = (result.usage.reasoning_tokens / result.usage.completion_tokens) * 100
        print(f"\nReasoning tokens are {reasoning_pct:.1f}% of completion tokens")
        print("This shows how much 'thinking' the model did before responding.")


async def metrics_tracking_example():
    """Example with metrics collection for reasoning tokens."""
    print("\n\n=== Metrics Tracking Example ===\n")

    # Create client with metrics collector
    metrics_collector = MetricsCollector()
    client = AzureLLMClient(
        config=AzureConfig(chat_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"), timeout_seconds=None),
        metrics_collector=metrics_collector,
    )

    # Make multiple requests with different complexity
    test_prompts = [
        "What is 2 + 2?",
        "Explain the steps to solve a quadratic equation.",
        "A farmer has chickens and rabbits. There are 35 heads and 94 legs. "
        "How many chickens and how many rabbits does the farmer have? "
        "Show your step-by-step reasoning.",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nRequest {i}: {prompt[:50]}...")
        result = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}], reasoning_effort="medium", max_tokens=300
        )
        print(f"  Reasoning tokens: {result.usage.reasoning_tokens}")
        print(f"  Completion tokens: {result.usage.completion_tokens}")

    # Show aggregated metrics
    print("\n" + "=" * 60)
    print("Aggregated Metrics:")
    agg = metrics_collector.get_aggregated()
    print(f"  Total requests:         {agg.total_requests}")
    print(f"  Total tokens (input):   {agg.total_tokens_input}")
    print(f"  Total tokens (output):  {agg.total_tokens_output}")
    print(f"  Total tokens (cached):  {agg.total_tokens_cached}")
    print(f"  Total tokens (reasoning): {agg.total_tokens_reasoning}")
    print(f"  Total cost:             {agg.total_cost:.4f} kr")
    print("=" * 60)

    if agg.total_tokens_output > 0:
        reasoning_ratio = agg.total_tokens_reasoning / agg.total_tokens_output
        print(f"\nReasoning ratio: {reasoning_ratio:.2%} of output tokens were reasoning tokens")


async def cost_analysis_example():
    """Example showing cost analysis with reasoning tokens."""
    print("\n\n=== Cost Analysis Example ===\n")

    from azure_llm_toolkit import CostTracker

    cost_tracker = CostTracker()
    client = AzureLLMClient(
        config=AzureConfig(chat_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"), timeout_seconds=None),
        cost_tracker=cost_tracker,
    )

    # Make a reasoning-intensive request
    result = await client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": "Design a algorithm to solve the Traveling Salesman Problem "
                "for 5 cities. Explain your approach step by step with "
                "time complexity analysis.",
            }
        ],
        reasoning_effort="high",
        max_tokens=800,
    )

    print(f"Response length: {len(result.content)} characters")
    print(f"\nToken breakdown:")
    print(f"  Prompt:     {result.usage.prompt_tokens}")
    print(f"  Completion: {result.usage.completion_tokens}")
    print(f"  Reasoning:  {result.usage.reasoning_tokens}")
    print(f"  Total:      {result.usage.total_tokens}")

    # Show cost breakdown
    summary = cost_tracker.get_summary()
    print(f"\nCost Summary:")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Total cost:       {summary['total_cost']:.4f} {summary['currency']}")
    print(f"  Total tokens:     {summary['total_tokens']}")

    # Show per-operation details
    print("\nPer-operation details:")
    for op in cost_tracker.get_operations():
        print(f"  Category: {op['category']}, Model: {op['model']}")
        print(f"    Cost: {op['amount']:.4f} {op['currency']}")
        if "reasoning_tokens" in op.get("metadata", {}):
            reasoning = op["metadata"]["reasoning_tokens"]
            print(f"    Reasoning tokens: {reasoning}")
            print(f"    This operation used reasoning extensively!")


async def comparison_example():
    """Compare reasoning token usage across different reasoning efforts."""
    print("\n\n=== Reasoning Effort Comparison ===\n")

    client = AzureLLMClient(
        config=AzureConfig(chat_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"), timeout_seconds=None)
    )

    # Same problem with different reasoning efforts
    problem = (
        "You have 100 apples. You give 20% to Alice, 30% of what remains to Bob, "
        "and then eat 10 apples yourself. How many apples do you have left?"
    )

    efforts = ["low", "medium", "high"]
    results = []

    for effort in efforts:
        print(f"\nTesting reasoning_effort='{effort}'...")
        result = await client.chat_completion(
            messages=[{"role": "user", "content": problem}], reasoning_effort=effort, max_tokens=300
        )

        results.append(
            {
                "effort": effort,
                "reasoning_tokens": result.usage.reasoning_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
                "content_length": len(result.content),
            }
        )

        print(f"  Reasoning tokens: {result.usage.reasoning_tokens}")
        print(f"  Completion tokens: {result.usage.completion_tokens}")
        print(f"  Response length: {len(result.content)} chars")

    # Summary comparison
    print("\n" + "=" * 60)
    print("Comparison Summary:")
    print("=" * 60)
    print(f"{'Effort':<10} {'Reasoning':<12} {'Completion':<12} {'Total':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['effort']:<10} {r['reasoning_tokens']:<12} {r['completion_tokens']:<12} {r['total_tokens']:<10}")
    print("=" * 60)

    print("\nObservations:")
    print("- Higher reasoning effort typically uses more reasoning tokens")
    print("- Reasoning tokens show model's 'thinking' before answering")
    print("- Consider cost vs quality tradeoff when choosing reasoning effort")


async def prometheus_export_example():
    """Example with Prometheus metrics export including reasoning tokens."""
    print("\n\n=== Prometheus Metrics Example ===\n")

    try:
        # Create collector with Prometheus export
        collector = create_collector_with_prometheus(namespace="reasoning_demo")

        client = AzureLLMClient(
            config=AzureConfig(chat_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-5-mini"), timeout_seconds=None),
            metrics_collector=collector,
        )

        # Make some requests
        await client.chat_completion(
            messages=[{"role": "user", "content": "Calculate 15% of 240"}], reasoning_effort="medium"
        )

        print("Metrics are now available for Prometheus scraping!")
        print("Reasoning tokens are tracked as token_type='reasoning'")
        print("\nExample Prometheus queries:")
        print("  - reasoning_demo_tokens_total{token_type='reasoning'}")
        print("  - rate(reasoning_demo_tokens_total{token_type='reasoning'}[5m])")
        print("  - reasoning_demo_tokens_total{token_type='reasoning'} / ")
        print("    reasoning_demo_tokens_total{token_type='output'}")

    except ImportError:
        print("Prometheus client not installed. Install with:")
        print("  pip install prometheus-client")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Reasoning Token Tracking Examples")
    print("=" * 60)

    # Check if API credentials are set
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("\nError: AZURE_OPENAI_API_KEY not set!")
        print("Set your environment variables in .env file:")
        print("  AZURE_OPENAI_API_KEY=your-key")
        print("  AZURE_ENDPOINT=your-endpoint")
        print("  AZURE_CHAT_DEPLOYMENT=gpt-5-mini  # or o1-preview")
        return

    try:
        # Run examples
        await basic_reasoning_example()
        await metrics_tracking_example()
        await cost_analysis_example()
        await comparison_example()
        await prometheus_export_example()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Valid Azure OpenAI credentials")
        print("  2. A reasoning model deployment (gpt-5-mini, o1-preview)")
        print("  3. Sufficient quota for reasoning models")


if __name__ == "__main__":
    asyncio.run(main())

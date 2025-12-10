"""
Simple demonstration of the logprob-based reranker API.

This script shows the basic usage patterns without requiring live Azure credentials.
For a full working example with real API calls, see reranker_example.py
"""

from azure_llm_toolkit.reranker import (
    LogprobReranker,
    RerankerConfig,
    RerankResult,
    create_reranker,
)


def demonstrate_config():
    """Show various configuration options."""
    print("=" * 80)
    print("Reranker Configuration Examples")
    print("=" * 80)

    # Default configuration
    config_default = RerankerConfig()
    print("\n1. Default Configuration:")
    print(f"   Model: {config_default.model} (default)")
    print(f"   Bins: {config_default.bins}")
    print(f"   Temperature: {config_default.temperature}")
    print(f"   Top logprobs: {config_default.top_logprobs}")

    # Custom 5-level scale
    config_5level = RerankerConfig(
        model="gpt-4o",
        bins=["0", "1", "2", "3", "4"],
        temperature=0.1,
    )
    print("\n2. Custom 5-Level Scale:")
    print(f"   Model: {config_5level.model}")
    print(f"   Bins: {config_5level.bins}")
    print(f"   Temperature: {config_5level.temperature}")

    # Custom 3-level scale with labels
    config_labeled = RerankerConfig(
        model="gpt-4o-mini",
        bins=["low", "medium", "high"],
        temperature=0.2,
        top_logprobs=3,
    )
    print("\n3. Labeled 3-Level Scale:")
    print(f"   Model: {config_labeled.model}")
    print(f"   Bins: {config_labeled.bins}")
    print(f"   Top logprobs: {config_labeled.top_logprobs}")


def demonstrate_result_structure():
    """Show the RerankResult dataclass structure."""
    print("\n" + "=" * 80)
    print("RerankResult Structure")
    print("=" * 80)

    # Create example results
    result1 = RerankResult(
        index=0,
        document="Machine learning is a subset of artificial intelligence.",
        score=0.92,
        bin_probabilities={"8": 0.3, "9": 0.5, "10": 0.2},
    )

    result2 = RerankResult(
        index=1,
        document="Python is a programming language.",
        score=0.45,
    )

    print("\nResult with bin probabilities:")
    print(f"  Index: {result1.index}")
    print(f"  Score: {result1.score:.3f}")
    print(f"  Document: {result1.document[:50]}...")
    print(f"  Bin probs: {result1.bin_probabilities}")

    print("\nResult without bin probabilities:")
    print(f"  Index: {result2.index}")
    print(f"  Score: {result2.score:.3f}")
    print(f"  Document: {result2.document[:50]}...")

    # Convert to dict
    print("\nAs dictionary:")
    print(f"  {result1.to_dict()}")


def demonstrate_api_patterns():
    """Show common API usage patterns."""
    print("\n" + "=" * 80)
    print("API Usage Patterns")
    print("=" * 80)

    print("\n1. Basic Usage:")
    print("""
    from azure_llm_toolkit import AzureLLMClient, AzureConfig
    from azure_llm_toolkit.reranker import LogprobReranker

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Uses gpt-4o-east-US by default
    reranker = LogprobReranker(client=client)

    # Rerank documents
    results = await reranker.rerank(query, documents, top_k=5)
    """)

    print("\n2. Using create_reranker helper:")
    print("""
    from azure_llm_toolkit.reranker import create_reranker

    reranker = create_reranker(
        client=client,
        model="gpt-4o",
        temperature=0.1,
        bins=["0", "1", "2", "3", "4"],
    )
    """)

    print("\n3. Score individual documents:")
    print("""
    # Score a single document
    score = await reranker.score(query, document)

    # Score with bin probabilities
    score, bin_probs = await reranker.score(
        query,
        document,
        include_bin_probs=True
    )
    """)

    print("\n4. Rerank with bin probabilities:")
    print("""
    results = await reranker.rerank(
        query,
        documents,
        top_k=10,
        include_bin_probs=True,
    )

    for result in results:
        print(f"Score: {result.score}")
        print(f"Bins: {result.bin_probabilities}")
    """)


def demonstrate_rag_integration():
    """Show RAG pipeline integration pattern."""
    print("\n" + "=" * 80)
    print("RAG Pipeline Integration Pattern")
    print("=" * 80)

    print("""
    # Typical RAG pipeline with reranking:

    async def rag_with_reranking(query: str):
        # Step 1: Initial retrieval from vector database
        candidates = await vector_db.similarity_search(
            query=query,
            top_k=20,  # Retrieve more candidates
        )

        # Step 2: Rerank for better relevance
        reranker = LogprobReranker(client=llm_client)
        reranked = await reranker.rerank(
            query=query,
            documents=candidates,
            top_k=5,  # Keep top 5 after reranking
        )

        # Step 3: Use top documents as context
        context = "\\n\\n".join([r.document for r in reranked[:3]])

        # Step 4: Generate final answer
        response = await llm_client.chat_completion(
            messages=[{
                "role": "user",
                "content": f"Context:\\n{context}\\n\\nQuestion: {query}"
            }],
            system_prompt="Answer based on the provided context.",
        )

        return response.content, reranked
    """)


def demonstrate_benefits():
    """Explain key benefits and use cases."""
    print("\n" + "=" * 80)
    print("Key Benefits")
    print("=" * 80)

    benefits = [
        ("Zero-shot", "No training or fine-tuning required"),
        ("Calibrated", "Probabilistic scores in [0.0, 1.0] range"),
        ("Cost-effective", "Only 1 token per document (max_tokens=1)"),
        ("Model-agnostic", "Works with any Azure OpenAI model supporting logprobs"),
        ("Transparent", "Bin probabilities show model's confidence"),
        ("Integrated", "Works seamlessly with AzureLLMClient features"),
    ]

    for title, description in benefits:
        print(f"\n• {title}")
        print(f"  {description}")

    print("\n" + "=" * 80)
    print("Common Use Cases")
    print("=" * 80)

    use_cases = [
        "RAG pipelines - Rerank vector search results",
        "Document ranking - Sort by semantic relevance",
        "Search improvement - Better than keyword matching",
        "Question answering - Find most relevant passages",
        "Content recommendation - Rank items by user interest",
        "Semantic deduplication - Identify similar documents",
    ]

    for i, use_case in enumerate(use_cases, 1):
        print(f"\n{i}. {use_case}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("Logprob-Based Reranker - API Demonstration")
    print("=" * 80)
    print("\nThis script demonstrates the reranker API without requiring")
    print("live Azure OpenAI credentials. For working examples with real")
    print("API calls, see reranker_example.py")

    demonstrate_config()
    demonstrate_result_structure()
    demonstrate_api_patterns()
    demonstrate_rag_integration()
    demonstrate_benefits()

    print("\n" + "=" * 80)
    print("For more information:")
    print("  - Full examples: examples/reranker_example.py")
    print("  - Unit tests: tests/test_reranker.py")
    print("  - Documentation: README.md")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

"""
Example demonstrating the logprob-based reranker functionality.

This example shows how to use the LogprobReranker to score and rerank documents
based on their relevance to a query using Azure OpenAI's chat completions API
with log probabilities.
"""

import asyncio
import os
from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig, AzureLLMClient
from azure_llm_toolkit.reranker import LogprobReranker, RerankerConfig, create_reranker


async def basic_reranking():
    """Basic reranking example with default configuration."""
    print("=" * 80)
    print("Basic Reranking Example")
    print("=" * 80)

    # Load environment variables
    load_dotenv()

    # Create client
    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Create reranker (uses chat deployment from config)
    reranker = LogprobReranker(client=client)

    # Sample query and documents
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.",
        "The weather forecast predicts rain tomorrow with a 60% chance of precipitation.",
        "Supervised learning is a type of machine learning where the model is trained on labeled data.",
    ]

    print(f"\nQuery: {query}")
    print(f"\nReranking {len(documents)} documents...\n")

    # Rerank documents
    results = await reranker.rerank(query, documents, top_k=3)

    print("Top 3 Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Original Index: {result.index}")
        print(f"   Document: {result.document[:100]}...")


async def custom_configuration():
    """Example using custom reranker configuration."""
    print("\n" + "=" * 80)
    print("Custom Configuration Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    # Custom configuration with 5 bins for faster scoring
    reranker_config = RerankerConfig(
        model=config.chat_deployment,
        bins=["0", "1", "2", "3", "4"],  # 5-level relevance scale
        temperature=0.1,  # Lower temperature for more deterministic scores
        top_logprobs=3,  # Fewer logprobs to analyze
    )

    reranker = LogprobReranker(client=client, config=reranker_config)

    query = "How do neural networks work?"
    documents = [
        "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers.",
        "Cloud computing provides on-demand access to computing resources over the internet.",
        "Backpropagation is an algorithm used to train neural networks by computing gradients of the loss function.",
        "Convolutional neural networks (CNNs) are specialized for processing grid-like data such as images.",
    ]

    print(f"\nQuery: {query}")
    print(f"Configuration: {len(reranker_config.bins)} bins, temperature={reranker_config.temperature}")
    print(f"\nReranking {len(documents)} documents...\n")

    results = await reranker.rerank(query, documents)

    print("Results:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.4f} (Index: {result.index})")
        print(f"   {result.document[:120]}...")


async def with_bin_probabilities():
    """Example showing bin probability distributions."""
    print("\n" + "=" * 80)
    print("Bin Probabilities Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    reranker = create_reranker(client=client, temperature=0.2)

    query = "Explain quantum computing"
    documents = [
        "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations.",
        "Coffee is a brewed drink prepared from roasted coffee beans.",
        "Classical computers use bits that are either 0 or 1, while quantum computers use qubits that can be in superposition.",
    ]

    print(f"\nQuery: {query}")
    print(f"\nScoring with bin probability distributions...\n")

    results = await reranker.rerank(query, documents, include_bin_probs=True)

    print("Results with Bin Probabilities:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Document: {result.document[:80]}...")
        if result.bin_probabilities:
            print(f"   Bin Probabilities:")
            # Show top 5 bins by probability
            sorted_bins = sorted(result.bin_probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
            for bin_tok, prob in sorted_bins:
                print(f"      Bin '{bin_tok}': {prob:.4f}")


async def single_document_scoring():
    """Example scoring individual documents."""
    print("\n" + "=" * 80)
    print("Single Document Scoring Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    reranker = LogprobReranker(client=client)

    query = "What is natural language processing?"
    documents = [
        "Natural language processing (NLP) is a field of AI focused on enabling computers to understand, interpret, and generate human language.",
        "Basketball is a team sport played on a rectangular court.",
        "NLP techniques include tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis.",
    ]

    print(f"\nQuery: {query}\n")

    for i, doc in enumerate(documents, 1):
        score = await reranker.score(query, doc)
        print(f"{i}. Score: {score:.4f}")
        print(f"   Document: {doc[:80]}...")
        print()


async def compare_reranking():
    """Compare original order vs reranked order."""
    print("\n" + "=" * 80)
    print("Comparison: Original vs Reranked Order")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    reranker = LogprobReranker(client=client)

    query = "What are transformers in deep learning?"
    documents = [
        "The Transformer architecture was introduced in the 'Attention is All You Need' paper and revolutionized NLP.",
        "Electric transformers are devices that transfer electrical energy between circuits.",
        "Attention mechanisms allow models to focus on relevant parts of the input when making predictions.",
        "The Great Wall of China is one of the most famous landmarks in the world.",
        "BERT and GPT are transformer-based language models that achieve state-of-the-art results on many NLP tasks.",
        "Gardening is a relaxing hobby that involves growing and maintaining plants.",
    ]

    print(f"\nQuery: {query}\n")

    print("Original Order:")
    print("-" * 80)
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc[:80]}...")

    results = await reranker.rerank(query, documents)

    print("\n\nReranked Order (by relevance):")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.4f} (was #{result.index + 1})")
        print(f"   {result.document[:80]}...")
        print()


async def rag_pipeline_example():
    """Example showing integration with RAG pipeline."""
    print("\n" + "=" * 80)
    print("RAG Pipeline Integration Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClient(config=config)

    reranker = create_reranker(client=client, model=config.chat_deployment)

    # Simulate retrieved documents from a vector database
    query = "How does gradient descent work in neural networks?"

    # In a real RAG system, these would come from vector similarity search
    retrieved_docs = [
        "Gradient descent is an optimization algorithm that iteratively adjusts model parameters to minimize a loss function.",
        "Neural networks consist of layers of interconnected neurons that process information.",
        "The learning rate determines the step size during gradient descent optimization.",
        "Stochastic gradient descent uses random mini-batches of data instead of the full dataset.",
        "Backpropagation computes gradients by applying the chain rule to propagate errors backward through the network.",
        "Overfitting occurs when a model performs well on training data but poorly on unseen data.",
        "Momentum helps gradient descent escape local minima by adding a fraction of the previous update.",
        "Adam optimizer combines ideas from RMSprop and momentum for adaptive learning rates.",
    ]

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieved_docs)} documents from vector search")
    print("\nApplying reranking for better relevance...\n")

    # Rerank top 5 most relevant
    reranked = await reranker.rerank(query, retrieved_docs, top_k=5)

    print("Top 5 Reranked Documents:")
    print("-" * 80)
    for i, result in enumerate(reranked, 1):
        print(f"\n{i}. Relevance Score: {result.score:.4f}")
        print(f"   {result.document}")

    # Use top documents for context generation
    context_docs = [result.document for result in reranked[:3]]
    context = "\n\n".join(context_docs)

    print("\n\nGenerating answer with top 3 documents as context...")
    print("-" * 80)

    # Generate final answer using the reranked context
    response = await client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nProvide a comprehensive answer based on the context.",
            }
        ],
        system_prompt="You are a helpful AI assistant. Answer questions based on the provided context.",
    )

    print(f"\nAnswer: {response.content}")
    print(f"\nTokens used: {response.usage.total_tokens}")


async def main():
    """Run all examples."""
    examples = [
        ("Basic Reranking", basic_reranking),
        ("Custom Configuration", custom_configuration),
        ("Bin Probabilities", with_bin_probabilities),
        ("Single Document Scoring", single_document_scoring),
        ("Original vs Reranked", compare_reranking),
        ("RAG Pipeline Integration", rag_pipeline_example),
    ]

    print("\n" + "=" * 80)
    print("Logprob Reranker Examples")
    print("=" * 80)
    print("\nThese examples demonstrate various ways to use the logprob-based reranker")
    print("for semantic relevance scoring in Azure OpenAI applications.\n")

    for name, example_func in examples:
        try:
            await example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

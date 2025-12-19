"""
Example demonstrating the synchronous client wrapper.

This example shows how to use AzureLLMClientSync for non-async codebases
and legacy applications that don't support async/await.
"""

from dotenv import load_dotenv

from azure_llm_toolkit import AzureConfig
from azure_llm_toolkit.sync_client import AzureLLMClientSync


def basic_sync_usage():
    """Basic synchronous usage example."""
    print("=" * 80)
    print("Basic Synchronous Client Example")
    print("=" * 80)

    load_dotenv()

    # Create synchronous client
    config = AzureConfig()
    client = AzureLLMClientSync(config=config)

    print("\nExample 1: Embedding text")
    print("-" * 40)

    text = "Azure OpenAI provides powerful language models"
    embedding = client.embed_text(text)

    print(f"Text: {text}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    print("\nExample 2: Chat completion")
    print("-" * 40)

    messages = [{"role": "user", "content": "What is machine learning in one sentence?"}]

    response = client.chat_completion(
        messages=messages,
        system_prompt="You are a helpful AI assistant.",
        max_tokens=100,
    )

    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")


def context_manager_usage():
    """Using sync client with context manager."""
    print("\n" + "=" * 80)
    print("Context Manager Usage")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()

    # Automatic cleanup with context manager
    with AzureLLMClientSync(config=config) as client:
        print("\nClient created with context manager")

        # Embed texts
        texts = [
            "Python is a programming language",
            "Machine learning is a subset of AI",
            "Natural language processing enables text understanding",
        ]

        result = client.embed_texts(texts)

        print(f"\nEmbedded {len(texts)} texts")
        print(f"Total tokens: {result.usage.total_tokens}")
        print(f"Embedding shape: {len(result.embeddings)} x {len(result.embeddings[0])}")

    print("\nClient automatically closed after context manager exit")


def cost_estimation():
    """Cost estimation with sync client."""
    print("\n" + "=" * 80)
    print("Cost Estimation Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClientSync(config=config)

    # Estimate embedding cost
    texts_to_embed = [
        "Document 1 with some content",
        "Document 2 with more content",
        "Document 3 with even more content",
    ]

    embedding_cost = client.estimate_embedding_cost(texts_to_embed)

    print("\nEmbedding Cost Estimation:")
    print(f"Number of texts: {len(texts_to_embed)}")
    print(f"Estimated cost: {embedding_cost:.4f} {client.cost_estimator.currency}")

    # Estimate chat cost
    messages = [{"role": "user", "content": "Explain quantum computing in detail"}]

    chat_cost = client.estimate_chat_cost(
        messages=messages,
        estimated_output_tokens=500,
    )

    print("\nChat Completion Cost Estimation:")
    print(f"Estimated output tokens: 500")
    print(f"Estimated cost: {chat_cost:.4f} {client.cost_estimator.currency}")


def token_counting():
    """Token counting examples."""
    print("\n" + "=" * 80)
    print("Token Counting Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClientSync(config=config)

    # Count tokens in text
    text = "The quick brown fox jumps over the lazy dog"
    token_count = client.count_tokens(text)

    print(f"\nText: {text}")
    print(f"Token count: {token_count}")

    # Count tokens in messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "AI is the simulation of human intelligence."},
        {"role": "user", "content": "Can you give examples?"},
    ]

    message_tokens = client.count_message_tokens(messages)

    print(f"\nMessages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    print(f"Total tokens: {message_tokens}")


def batch_embeddings():
    """Batch embedding example."""
    print("\n" + "=" * 80)
    print("Batch Embeddings Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClientSync(config=config)

    # Create a batch of texts
    documents = [
        "Azure OpenAI provides powerful AI models",
        "Machine learning enables computers to learn from data",
        "Natural language processing is a branch of AI",
        "Deep learning uses neural networks",
        "Computer vision enables machines to interpret images",
    ]

    print(f"\nEmbedding {len(documents)} documents...")

    result = client.embed_texts(documents, batch_size=3)

    print(f"Successfully embedded {len(result.embeddings)} documents")
    print(f"Model: {result.model}")
    print(f"Total tokens: {result.usage.total_tokens}")
    print(f"Embedding dimension: {len(result.embeddings[0])}")


def multi_turn_conversation():
    """Multi-turn conversation example."""
    print("\n" + "=" * 80)
    print("Multi-Turn Conversation Example")
    print("=" * 80)

    load_dotenv()

    config = AzureConfig()
    client = AzureLLMClientSync(config=config)

    # Conversation history
    messages = []

    # Turn 1
    user_msg_1 = "What is Python?"
    messages.append({"role": "user", "content": user_msg_1})

    response_1 = client.chat_completion(
        messages=messages,
        system_prompt="You are a helpful programming tutor.",
        max_tokens=100,
    )

    messages.append({"role": "assistant", "content": response_1.content})

    print(f"\nUser: {user_msg_1}")
    print(f"Assistant: {response_1.content}")

    # Turn 2
    user_msg_2 = "What are its main use cases?"
    messages.append({"role": "user", "content": user_msg_2})

    response_2 = client.chat_completion(
        messages=messages,
        max_tokens=150,
    )

    messages.append({"role": "assistant", "content": response_2.content})

    print(f"\nUser: {user_msg_2}")
    print(f"Assistant: {response_2.content}")

    # Turn 3
    user_msg_3 = "How do I get started learning it?"
    messages.append({"role": "user", "content": user_msg_3})

    response_3 = client.chat_completion(
        messages=messages,
        max_tokens=150,
    )

    print(f"\nUser: {user_msg_3}")
    print(f"Assistant: {response_3.content}")


# Query rewriting example removed.
# This project no longer provides a built-in `rewrite_query` API.
# If you previously relied on query rewriting, implement your own rewriting
# logic or use an external service integrated into your retrieval pipeline.


def legacy_framework_integration():
    """Example showing integration with legacy frameworks."""
    print("\n" + "=" * 80)
    print("Legacy Framework Integration Example")
    print("=" * 80)

    load_dotenv()

    # Simulate a legacy class that doesn't support async
    class LegacyDocumentProcessor:
        def __init__(self):
            self.client = AzureLLMClientSync(
                config=AzureConfig(),
                enable_rate_limiting=True,
                enable_cache=True,
            )

        def process_document(self, text: str) -> dict:
            """Process a document synchronously."""
            # Embed the document
            embedding = self.client.embed_text(text)

            # Summarize the document
            messages = [{"role": "user", "content": f"Summarize this text in one sentence:\n\n{text}"}]

            summary_response = self.client.chat_completion(
                messages=messages,
                max_tokens=100,
            )

            return {
                "original_text": text,
                "embedding": embedding[:5],  # First 5 values
                "summary": summary_response.content,
                "tokens_used": summary_response.usage.total_tokens,
            }

        def cleanup(self):
            """Cleanup resources."""
            self.client.close()

    # Use the legacy processor
    processor = LegacyDocumentProcessor()

    document = (
        "Azure OpenAI Service provides REST API access to OpenAI's powerful language models. "
        "These models can be easily adapted to your specific task including content generation, "
        "summarization, semantic search, and natural language to code translation."
    )

    print("\nProcessing document with legacy processor...")
    result = processor.process_document(document)

    print(f"Document length: {len(document)} characters")
    print(f"Embedding (first 5): {result['embedding']}")
    print(f"Summary: {result['summary']}")
    print(f"Tokens used: {result['tokens_used']}")

    processor.cleanup()
    print("\nProcessor cleaned up")


def main():
    """Run all examples."""
    examples = [
        ("Basic Usage", basic_sync_usage),
        ("Context Manager", context_manager_usage),
        ("Cost Estimation", cost_estimation),
        ("Token Counting", token_counting),
        ("Batch Embeddings", batch_embeddings),
        ("Multi-Turn Conversation", multi_turn_conversation),
        ("Legacy Framework Integration", legacy_framework_integration),
    ]

    print("\n" + "=" * 80)
    print("Synchronous Client Examples")
    print("=" * 80)
    print("\nThese examples demonstrate using AzureLLMClientSync for")
    print("non-async codebases and legacy applications.\n")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

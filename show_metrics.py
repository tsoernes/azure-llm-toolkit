#!/usr/bin/env python3
"""
Show Azure OpenAI metrics from Prometheus.

Displays requests per minute (RPM) and tokens per minute (TPM)
for both chat and embedding endpoints.
"""

import requests
from collections import defaultdict


def query_prometheus(query):
    """Query Prometheus and return results."""
    try:
        resp = requests.get("http://localhost:9090/api/v1/query", params={"query": query}, timeout=5)
        data = resp.json()
        if data.get("status") == "success":
            return data["data"]["result"]
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
    return []


def main():
    print("═" * 70)
    print("  📊 AZURE OPENAI METRICS - Requests & Tokens Per Minute")
    print("═" * 70)
    print()

    # Get total requests
    results = query_prometheus("azure_llm_requests_total")
    if results:
        by_operation = defaultdict(int)

        for r in results:
            op = r["metric"].get("operation", "unknown")
            count = int(float(r["value"][1]))
            by_operation[op] += count

        print("📊 TOTAL REQUESTS (All Time):")
        for op, count in sorted(by_operation.items()):
            print(f"  {op:20s}: {count:5d} requests")
        print(f"  {'TOTAL':20s}: {sum(by_operation.values()):5d} requests")
        print()
    else:
        print("No request data available\n")

    # Get rate (requests per minute)
    print("📈 REQUESTS PER MINUTE (Current Rate):")
    rate_results = query_prometheus("rate(azure_llm_requests_total[1m])*60")
    if rate_results:
        by_operation = defaultdict(float)
        for r in rate_results:
            op = r["metric"].get("operation", "unknown")
            rpm = float(r["value"][1])
            by_operation[op] += rpm

        for op, rpm in sorted(by_operation.items()):
            print(f"  {op:20s}: {rpm:7.2f} req/min")
        print(f"  {'TOTAL':20s}: {sum(by_operation.values()):7.2f} req/min")
    else:
        print("  Waiting for data (need at least 1 minute of history)...")
    print()

    # Get tokens
    print("🔢 TOKEN USAGE (All Time):")
    token_results = query_prometheus("azure_llm_tokens_total")
    if token_results:
        input_by_model = defaultdict(int)
        output_by_model = defaultdict(int)

        for r in token_results:
            token_type = r["metric"].get("type", "")
            model = r["metric"].get("model", "unknown")
            count = int(float(r["value"][1]))

            if token_type == "input":
                input_by_model[model] += count
            elif token_type == "output":
                output_by_model[model] += count

        # Categorize by endpoint type
        chat_models = [m for m in input_by_model.keys() if "embedding" not in m.lower()]
        embed_models = [m for m in input_by_model.keys() if "embedding" in m.lower()]

        print("  💬 CHAT ENDPOINT:")
        chat_input = sum(input_by_model.get(m, 0) for m in chat_models)
        chat_output = sum(output_by_model.get(m, 0) for m in chat_models)
        print(f"     Input tokens:  {chat_input:8,}")
        print(f"     Output tokens: {chat_output:8,}")
        print(f"     TOTAL:         {chat_input + chat_output:8,}")

        if embed_models:
            print()
            print("  🔤 EMBEDDING ENDPOINT:")
            embed_input = sum(input_by_model.get(m, 0) for m in embed_models)
            print(f"     Input tokens:  {embed_input:8,}")

        print()
        total_tokens = sum(input_by_model.values()) + sum(output_by_model.values())
        print(f"  📊 GRAND TOTAL:    {total_tokens:8,} tokens")
    else:
        print("  No token data available")
    print()

    # Get token rate (tokens per minute)
    print("⚡ TOKENS PER MINUTE (Current Rate):")
    token_rate_results = query_prometheus("rate(azure_llm_tokens_total[1m])*60")
    if token_rate_results:
        input_by_model = defaultdict(float)
        output_by_model = defaultdict(float)

        for r in token_rate_results:
            token_type = r["metric"].get("type", "")
            model = r["metric"].get("model", "unknown")
            tpm = float(r["value"][1])

            if token_type == "input":
                input_by_model[model] += tpm
            elif token_type == "output":
                output_by_model[model] += tpm

        # Categorize
        chat_models = [m for m in input_by_model.keys() if "embedding" not in m.lower()]
        embed_models = [m for m in input_by_model.keys() if "embedding" in m.lower()]

        print("  💬 CHAT ENDPOINT:")
        chat_input_tpm = sum(input_by_model.get(m, 0) for m in chat_models)
        chat_output_tpm = sum(output_by_model.get(m, 0) for m in chat_models)
        print(f"     Input TPM:  {chat_input_tpm:10.0f} tokens/min")
        print(f"     Output TPM: {chat_output_tpm:10.0f} tokens/min")
        print(f"     TOTAL:      {chat_input_tpm + chat_output_tpm:10.0f} tokens/min")

        if embed_models:
            print()
            print("  🔤 EMBEDDING ENDPOINT:")
            embed_input_tpm = sum(input_by_model.get(m, 0) for m in embed_models)
            print(f"     Input TPM:  {embed_input_tpm:10.0f} tokens/min")
    else:
        print("  Waiting for data (need at least 1 minute of history)...")

    print()

    # Get cost
    print("💰 COST:")
    cost_results = query_prometheus("azure_llm_cost_dollars_total")
    if cost_results:
        total_cost = sum(float(r["value"][1]) for r in cost_results)
        print(f"  Total cost: ${total_cost:.6f} USD")
    else:
        print("  No cost data available")

    print()
    print("═" * 70)
    print()
    print("🔗 View more at:")
    print("  Dashboard:   http://localhost:8765/")
    print("  Prometheus:  http://localhost:9090/")
    print("  Metrics:     http://localhost:8765/metrics")
    print()


if __name__ == "__main__":
    main()

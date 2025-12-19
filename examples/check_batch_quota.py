"""
Utility to check Azure OpenAI batch quota and provide recommendations.

This script helps users understand their current batch quota limits and
provides guidance on whether batch processing is suitable for their workload.
"""

import asyncio
import os
from typing import Any

from azure_llm_toolkit import AzureConfig


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def get_quota_info(subscription_type: str, model: str) -> dict[str, Any]:
    """
    Get quota information based on subscription type and model.

    Returns estimated enqueued token limits based on Azure documentation.
    Note: Actual limits may vary - check Azure Portal for exact values.
    """
    # Global Batch default quotas (enqueued tokens)
    quotas = {
        "enterprise": {
            "gpt-4o": 5_000_000_000,
            "gpt-4o-mini": 15_000_000_000,
            "gpt-4": 150_000_000,
            "gpt-35-turbo": 10_000_000_000,
        },
        "default": {
            "gpt-4o": 200_000_000,
            "gpt-4o-mini": 1_000_000_000,
            "gpt-4": 30_000_000,
            "gpt-35-turbo": 1_000_000_000,
        },
        "credit_card": {
            "gpt-4o": 50_000_000,
            "gpt-4o-mini": 50_000_000,
            "gpt-4": 5_000_000,
            "gpt-35-turbo": 100_000_000,
        },
    }

    model_key = model.lower()
    if "gpt-4o-mini" in model_key or "gpt-4o-mini" in model_key:
        model_key = "gpt-4o-mini"
    elif "gpt-4o" in model_key:
        model_key = "gpt-4o"
    elif "gpt-4" in model_key:
        model_key = "gpt-4"
    elif "gpt-35" in model_key or "gpt-3.5" in model_key:
        model_key = "gpt-35-turbo"
    else:
        return {
            "limit": "Unknown",
            "note": "Model not in quota table. Check Azure Portal.",
        }

    limit = quotas.get(subscription_type, quotas["default"]).get(model_key, "Unknown")

    return {
        "limit": limit,
        "formatted": format_number(limit) if isinstance(limit, int) else limit,
        "model_normalized": model_key,
    }


def calculate_job_requirements(num_texts: int, avg_tokens_per_text: int) -> dict[str, Any]:
    """Calculate requirements for a batch job."""
    total_tokens = num_texts * avg_tokens_per_text

    # Estimate batch processing time (rough estimate)
    # Assume ~10,000 tokens/second processing rate
    estimated_seconds = total_tokens / 10_000
    estimated_minutes = estimated_seconds / 60

    return {
        "total_tokens": total_tokens,
        "formatted_tokens": format_number(total_tokens),
        "estimated_minutes": round(estimated_minutes, 1),
        "estimated_hours": round(estimated_minutes / 60, 2),
    }


def provide_recommendations(quota_limit: int, job_tokens: int) -> list[str]:
    """Provide recommendations based on quota and job size."""
    recommendations = []

    usage_percent = (job_tokens / quota_limit) * 100 if quota_limit > 0 else 0

    if usage_percent < 50:
        recommendations.append("✅ Your job fits comfortably within quota limits")
        recommendations.append("   You can likely run multiple jobs concurrently")
    elif usage_percent < 90:
        recommendations.append("⚠️  Your job uses a significant portion of quota")
        recommendations.append("   Consider breaking into smaller batches for flexibility")
    else:
        recommendations.append("❌ Your job exceeds or nearly exceeds quota limits")
        recommendations.append("   You MUST split into smaller jobs or request quota increase")
        recommendations.append("   Go to Azure Portal → Quotas → Request quota increase")

    if job_tokens > 50_000_000:  # 50M tokens
        recommendations.append("\n💡 For very large jobs:")
        recommendations.append("   - Split into multiple files (max 100K requests per file)")
        recommendations.append("   - Set file expiration to increase file limit from 500 to 10,000")
        recommendations.append("   - Monitor job status regularly")

    return recommendations


async def main():
    """Main function to check quota and provide guidance."""
    print("\n" + "=" * 70)
    print("Azure OpenAI Batch Quota Checker")
    print("=" * 70)

    # Load config
    config = AzureConfig()

    print(f"\n📋 Configuration:")
    print(f"   Endpoint: {config.endpoint}")
    print(f"   Embedding Model: {config.embedding_deployment}")

    # Get subscription type
    print(f"\n📊 Subscription Type:")
    print(f"   1. Enterprise / MCA-E")
    print(f"   2. Default / Pay-as-you-go")
    print(f"   3. Monthly credit card")

    try:
        sub_choice = input("\nSelect your subscription type (1-3) [default: 2]: ").strip() or "2"
        sub_type_map = {
            "1": "enterprise",
            "2": "default",
            "3": "credit_card",
        }
        sub_type = sub_type_map.get(sub_choice, "default")
    except (EOFError, KeyboardInterrupt):
        sub_type = "default"
        print("2")

    # Get model info
    model = config.embedding_deployment
    quota_info = get_quota_info(sub_type, model)

    print(f"\n📈 Quota Information:")
    print(f"   Subscription Type: {sub_type.replace('_', ' ').title()}")
    print(f"   Model: {model}")
    print(f"   Normalized Model: {quota_info.get('model_normalized', 'Unknown')}")
    print(f"   Enqueued Token Limit: {quota_info['formatted']}")

    if isinstance(quota_info["limit"], str):
        print(f"\n{quota_info.get('note', '')}")
        print("\nℹ️  Check Azure Portal for exact quota limits:")
        print("   Azure OpenAI Resource → Quotas → View/Manage Quotas")
        return

    # Get job requirements
    print(f"\n💼 Batch Job Planning:")

    try:
        num_texts = int(input("   Number of texts to embed: ").strip() or "1000")
        avg_tokens = int(input("   Average tokens per text [default: 100]: ").strip() or "100")
    except (ValueError, EOFError, KeyboardInterrupt):
        num_texts = 1000
        avg_tokens = 100
        print(f"   Using defaults: {num_texts} texts, {avg_tokens} tokens/text")

    job_reqs = calculate_job_requirements(num_texts, avg_tokens)

    print(f"\n📊 Job Requirements:")
    print(f"   Total texts: {num_texts:,}")
    print(f"   Avg tokens/text: {avg_tokens}")
    print(f"   Total tokens: {job_reqs['formatted_tokens']} ({job_reqs['total_tokens']:,})")
    print(f"   Estimated processing time: {job_reqs['estimated_minutes']} min ({job_reqs['estimated_hours']} hrs)")

    # Calculate quota usage
    quota_limit = quota_info["limit"]
    usage_percent = (job_reqs["total_tokens"] / quota_limit) * 100

    print(f"\n📉 Quota Usage:")
    print(f"   Your job: {job_reqs['formatted_tokens']}")
    print(f"   Quota limit: {quota_info['formatted']}")
    print(f"   Usage: {usage_percent:.1f}%")

    # Provide recommendations
    print(f"\n💡 Recommendations:")
    recommendations = provide_recommendations(quota_limit, job_reqs["total_tokens"])
    for rec in recommendations:
        print(rec)

    # Additional notes
    print(f"\n📝 Important Notes:")
    print(f"   • Batch API has 50% cost discount vs standard API")
    print(f"   • Rate limits (TPM/RPM) still apply to batch jobs")
    print(f"   • Default 50K token queue limit for initial job submissions")
    print(f"   • Jobs take up to 24 hours (typically much faster)")
    print(f"   • Maximum 100,000 requests per input file")
    print(f"   • Maximum 200 MB input file size")

    print(f"\n🔗 Useful Links:")
    print(f"   Azure Portal: https://portal.azure.com")
    print(f"   Quota Management: Your Resource → Quotas → Request quota increase")
    print(f"   Documentation: docs/BATCH_API_NOTES.md")

    print(f"\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

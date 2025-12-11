#!/usr/bin/env python3
"""
Live Prometheus Dashboard with REAL Azure OpenAI API calls.

This example demonstrates real-time metrics collection from actual Azure OpenAI
API calls, exposed via Prometheus metrics and a web dashboard.

Requirements:
    - Azure OpenAI API credentials configured
    - pip install aiohttp prometheus-client

Environment variables needed:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_DEPLOYMENT (chat model)
    - AZURE_OPENAI_EMBEDDING_DEPLOYMENT (embedding model, optional)

Run this script and then:
- View dashboard at: http://localhost:8765/
- View metrics at: http://localhost:8765/metrics
- Configure Prometheus to scrape: http://localhost:8765/metrics
"""

import asyncio
import logging
import random
import time
from datetime import datetime

import json
from typing import Any

import aiohttp
from aiohttp import web
from prometheus_client import Counter, Gauge, Histogram, generate_latest

from azure_llm_toolkit import AzureLLMClient, AzureConfig
from azure_llm_toolkit.metrics import create_collector_with_prometheus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prometheus metrics
request_counter = Counter("azure_llm_requests_total", "Total number of LLM requests", ["operation", "model", "status"])

request_duration = Histogram(
    "azure_llm_request_duration_seconds",
    "Request duration in seconds",
    ["operation", "model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

token_counter = Counter("azure_llm_tokens_total", "Total tokens processed", ["type", "model"])

cost_counter = Counter("azure_llm_cost_dollars_total", "Total cost in USD", ["model"])

active_requests = Gauge("azure_llm_active_requests", "Number of requests currently being processed")


# Statistics for dashboard
stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_tokens_input": 0,
    "total_tokens_output": 0,
    "total_tokens_cached": 0,
    "total_cost": 0.0,
    "operations": {},
    "models": {},
    "errors": {},
    "avg_duration": 0.0,
    "min_duration": float("inf"),
    "max_duration": 0.0,
    # Per-endpoint aggregates
    "chat_requests": 0,
    "embed_requests": 0,
    "chat_tokens_input": 0,
    "chat_tokens_output": 0,
    "embed_tokens_input": 0,
    # Prometheus-derived rates (RPM/TPM)
    "chat_rpm": 0.0,
    "embed_rpm": 0.0,
    "chat_tpm_input": 0.0,
    "chat_tpm_output": 0.0,
    "embed_tpm_input": 0.0,
}


# Global client
client = None
_prometheus_session: aiohttp.ClientSession | None = None


async def make_real_chat_request(prompt: str, model: str = None):
    """Make a real chat completion request."""
    global stats

    stats["total_requests"] += 1
    operation = "chat_completion"
    stats["operations"][operation] = stats["operations"].get(operation, 0) + 1

    active_requests.inc()
    start_time = time.time()

    try:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}], max_tokens=100, model=model
        )

        duration = time.time() - start_time
        model_used = response.model

        # Extract usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        cached_tokens = usage.cached_prompt_tokens

        # Update per-endpoint token stats
        stats["chat_tokens_input"] += input_tokens
        stats["chat_tokens_output"] += output_tokens

        # Calculate cost (simplified - you may want to use cost_estimator)
        if "gpt-4" in model_used:
            cost = input_tokens * 0.00003 + output_tokens * 0.00006
        elif "gpt-35" in model_used or "gpt-3.5" in model_used:
            cost = input_tokens * 0.000015 + output_tokens * 0.00002
        else:
            cost = input_tokens * 0.00001 + output_tokens * 0.00002

        # Update Prometheus metrics
        request_counter.labels(operation=operation, model=model_used, status="success").inc()
        request_duration.labels(operation=operation, model=model_used).observe(duration)
        token_counter.labels(type="input", model=model_used).inc(input_tokens)
        token_counter.labels(type="output", model=model_used).inc(output_tokens)
        if cached_tokens > 0:
            token_counter.labels(type="cached", model=model_used).inc(cached_tokens)
        cost_counter.labels(model=model_used).inc(cost)

        # Update stats
        stats["successful_requests"] += 1
        stats["models"][model_used] = stats["models"].get(model_used, 0) + 1
        stats["total_tokens_input"] += input_tokens
        stats["total_tokens_output"] += output_tokens
        stats["total_tokens_cached"] += cached_tokens
        stats["total_cost"] += cost
        stats["chat_requests"] += 1

        # Update duration stats
        stats["avg_duration"] = (stats["avg_duration"] * (stats["successful_requests"] - 1) + duration) / stats[
            "successful_requests"
        ]
        stats["min_duration"] = min(stats["min_duration"], duration)
        stats["max_duration"] = max(stats["max_duration"], duration)

        answer = response.content
        logger.info(
            f"✅ {operation} ({model_used}): {input_tokens}+{output_tokens} tokens, "
            f"${cost:.6f}, {duration:.2f}s - {answer[:50]}..."
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        error_type = type(e).__name__

        request_counter.labels(operation=operation, model=model or "unknown", status="error").inc()
        stats["failed_requests"] += 1
        stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1

        logger.error(f"❌ {operation}: {error_type} - {str(e)[:100]}")

    finally:
        active_requests.dec()


async def make_real_embedding_request(text: str, deployment: str = None):
    """Make a real embedding request."""
    global stats

    stats["total_requests"] += 1
    operation = "embed_text"
    stats["operations"][operation] = stats["operations"].get(operation, 0) + 1

    active_requests.inc()
    start_time = time.time()

    try:
        embedding = await client.embed_text(
            text=text,
            model=deployment or client.config.embedding_deployment,
        )

        duration = time.time() - start_time
        model_used = deployment or client.config.embedding_deployment

        # Estimate tokens (rough approximation)
        input_tokens = int(len(text.split()) * 1.3)

        # Calculate cost
        cost = input_tokens * 0.0001 / 1000  # Ada-002 pricing

        # Update Prometheus metrics
        request_counter.labels(operation=operation, model=model_used, status="success").inc()
        request_duration.labels(operation=operation, model=model_used).observe(duration)
        token_counter.labels(type="input", model=model_used).inc(input_tokens)
        cost_counter.labels(model=model_used).inc(cost)

        # Update stats
        stats["successful_requests"] += 1
        stats["models"][model_used] = stats["models"].get(model_used, 0) + 1
        stats["total_tokens_input"] += input_tokens
        stats["total_cost"] += cost
        stats["embed_requests"] += 1
        stats["embed_tokens_input"] += input_tokens

        # Update duration stats
        stats["avg_duration"] = (stats["avg_duration"] * (stats["successful_requests"] - 1) + duration) / stats[
            "successful_requests"
        ]
        stats["min_duration"] = min(stats["min_duration"], duration)
        stats["max_duration"] = max(stats["max_duration"], duration)

        logger.info(
            f"✅ {operation} ({model_used}): {int(input_tokens)} tokens, "
            f"${cost:.6f}, {duration:.2f}s - {len(embedding)} dimensions"
        )

        return embedding

    except Exception as e:
        duration = time.time() - start_time
        error_type = type(e).__name__

        request_counter.labels(operation=operation, model=deployment or "unknown", status="error").inc()
        stats["failed_requests"] += 1
        stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1

        logger.error(f"❌ {operation}: {error_type} - {str(e)[:100]}")

    finally:
        active_requests.dec()


async def _query_prometheus(query: str) -> list[dict[str, Any]]:
    """
    Query local Prometheus HTTP API.

    Returns a list of result objects or an empty list on error.
    """
    global _prometheus_session

    if _prometheus_session is None:
        _prometheus_session = aiohttp.ClientSession()

    try:
        async with _prometheus_session.get(
            "http://localhost:9090/api/v1/query",
            params={"query": query},
            timeout=3,
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    except Exception:
        return []

    if data.get("status") != "success":
        return []

    return data.get("data", {}).get("result", [])


async def _refresh_rpm_tpm() -> None:
    """
    Periodically refresh RPM/TPM metrics from Prometheus and store in stats.

    This does not block request handling; it's a background task.
    """
    while True:
        try:
            # Requests per minute by operation
            results = await _query_prometheus("rate(azure_llm_requests_total[1m]) * 60")
            chat_rpm = 0.0
            embed_rpm = 0.0
            for r in results:
                op = r.get("metric", {}).get("operation", "")
                val = float(r.get("value", [0, "0"])[1])
                if op == "chat_completion":
                    chat_rpm += val
                elif op == "embed_text":
                    embed_rpm += val

            stats["chat_rpm"] = chat_rpm
            stats["embed_rpm"] = embed_rpm

            # Tokens per minute by type/model
            token_results = await _query_prometheus("rate(azure_llm_tokens_total[1m]) * 60")
            chat_in = 0.0
            chat_out = 0.0
            embed_in = 0.0
            for r in token_results:
                metric = r.get("metric", {})
                token_type = metric.get("type", "")
                model = metric.get("model", "").lower()
                val = float(r.get("value", [0, "0"])[1])

                is_embedding_model = "embedding" in model

                if token_type == "input":
                    if is_embedding_model:
                        embed_in += val
                    else:
                        chat_in += val
                elif token_type == "output" and not is_embedding_model:
                    chat_out += val

            stats["chat_tpm_input"] = chat_in
            stats["chat_tpm_output"] = chat_out
            stats["embed_tpm_input"] = embed_in

        except Exception:
            # On any error, keep previous values and try again later
            pass

        # Refresh every ~10 seconds
        await asyncio.sleep(10)


async def background_requests():
    """Generate background requests continuously with real API calls."""

    # Sample prompts for variety
    chat_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of async programming?",
        "Describe the Python programming language.",
        "What is machine learning?",
        "Explain the difference between AI and ML.",
        "What is a REST API?",
        "Describe cloud computing.",
        "What is Docker?",
    ]

    embedding_texts = [
        "Azure OpenAI provides powerful language models.",
        "Prometheus is a monitoring and alerting toolkit.",
        "Python is a versatile programming language.",
        "Machine learning enables computers to learn from data.",
        "Cloud computing provides on-demand resources.",
    ]

    while True:
        try:
            # Randomly choose between chat and embedding
            if random.random() > 0.3:  # 70% chat, 30% embeddings
                prompt = random.choice(chat_prompts)
                await make_real_chat_request(prompt)
            else:
                text = random.choice(embedding_texts)
                await make_real_embedding_request(text)

            # Random delay between requests (to avoid rate limits)
            await asyncio.sleep(random.uniform(2.0, 5.0))

        except Exception as e:
            logger.error(f"Error in background requests: {e}")
            await asyncio.sleep(5)


async def metrics_endpoint(request):
    """Prometheus metrics endpoint."""
    metrics_data = generate_latest()
    return web.Response(body=metrics_data, content_type="text/plain; version=0.0.4")


async def dashboard_endpoint(request):
    """HTML dashboard showing real-time metrics."""

    success_rate = (stats["successful_requests"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Azure LLM Toolkit - Live Prometheus Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            padding: 30px;
        }}
        h1 {{
            color: #667eea;
            margin-top: 0;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
            border-bottom: 2px solid #764ba2;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-grid-small {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            text-transform: uppercase;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-card .label {{
            font-size: 12px;
            opacity: 0.8;
        }}
        .success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .error {{ background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }}
        .cost {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .tokens {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .info {{
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #4caf50;
        }}
        .button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
            text-decoration: none;
            display: inline-block;
        }}
        .button:hover {{
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
            margin-top: 20px;
            text-align: center;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }}
        .status-online {{ background: #38ef7d; }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        .live-badge {{
            display: inline-block;
            background: #4caf50;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span class="status-indicator status-online"></span>
            Azure LLM Toolkit - Live Prometheus Dashboard
            <span class="live-badge">🔴 LIVE DATA</span>
        </h1>

        <div class="info">
            <strong>📊 Real-Time Metrics from Azure OpenAI API</strong><br>
            This dashboard shows REAL metrics from actual Azure OpenAI API calls.
            Auto-refreshes every 5 seconds. Prometheus metrics: <a href="/metrics">/metrics</a>
        </div>

        <h2>📈 Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Total Requests</h3>
                <div class="value">{stats["total_requests"]}</div>
                <div class="label">Real API calls</div>
            </div>
            <div class="metric-card success">
                <h3>Successful</h3>
                <div class="value">{stats["successful_requests"]}</div>
                <div class="label">{success_rate:.1f}% success rate</div>
            </div>
            <div class="metric-card error">
                <h3>Failed</h3>
                <div class="value">{stats["failed_requests"]}</div>
                <div class="label">{100 - success_rate:.1f}% error rate</div>
            </div>
            <div class="metric-card cost">
                <h3>Total Cost</h3>
                <div class="value">${stats["total_cost"]:.4f}</div>
                <div class="label">Real USD spent</div>
            </div>
        </div>

        <h2>📈 Requests & Tokens per Minute</h2>
        <div class="metric-grid-small">
            <div class="metric-card">
                <h3>Chat Requests</h3>
                <div class="value">{stats["chat_requests"]}</div>
                <div class="label">
                    Total chat calls · ~{stats["chat_rpm"]:.2f} RPM (1m window)
                </div>
            </div>
            <div class="metric-card">
                <h3>Embedding Requests</h3>
                <div class="value">{stats["embed_requests"]}</div>
                <div class="label">
                    Total embedding calls · ~{stats["embed_rpm"]:.2f} RPM (1m window)
                </div>
            </div>
            <div class="metric-card tokens">
                <h3>Chat Tokens</h3>
                <div class="value">{stats["chat_tokens_input"] + stats["chat_tokens_output"]:,}</div>
                <div class="label">
                    Input+output tokens · in: ~{stats["chat_tpm_input"]:.0f} TPM,
                    out: ~{stats["chat_tpm_output"]:.0f} TPM
                </div>
            </div>
            <div class="metric-card tokens">
                <h3>Embedding Tokens</h3>
                <div class="value">{stats["embed_tokens_input"]:,}</div>
                <div class="label">
                    Input tokens · ~{stats["embed_tpm_input"]:.0f} TPM
                </div>
            </div>
        </div>

        <h2>🔢 Token Usage (Real API Data)</h2>
        <div class="metric-grid">
            <div class="metric-card tokens">
                <h3>Input Tokens</h3>
                <div class="value">{stats["total_tokens_input"]:,}</div>
                <div class="label">Prompt tokens</div>
            </div>
            <div class="metric-card tokens">
                <h3>Output Tokens</h3>
                <div class="value">{stats["total_tokens_output"]:,}</div>
                <div class="label">Completion tokens</div>
            </div>
            <div class="metric-card tokens">
                <h3>Cached Tokens</h3>
                <div class="value">{stats["total_tokens_cached"]:,}</div>
                <div class="label">From cache</div>
            </div>
            <div class="metric-card tokens">
                <h3>Total Tokens</h3>
                <div class="value">{stats["total_tokens_input"] + stats["total_tokens_output"]:,}</div>
                <div class="label">All tokens</div>
            </div>
        </div>

        <h2>⏱️ Performance (Real API Latency)</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Average Duration</h3>
                <div class="value">{stats["avg_duration"]:.3f}s</div>
                <div class="label">Mean latency</div>
            </div>
            <div class="metric-card success">
                <h3>Min Duration</h3>
                <div class="value">{stats["min_duration"] if stats["min_duration"] != float("inf") else 0:.3f}s</div>
                <div class="label">Fastest request</div>
            </div>
            <div class="metric-card error">
                <h3>Max Duration</h3>
                <div class="value">{stats["max_duration"]:.3f}s</div>
                <div class="label">Slowest request</div>
            </div>
        </div>
"""

    if stats["operations"]:
        html += """
        <h2>📋 Requests by Operation</h2>
        <table>
            <thead>
                <tr>
                    <th>Operation</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
        for op, count in sorted(stats["operations"].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0
            html += f"""
                <tr>
                    <td>{op}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""

    if stats["models"]:
        html += """
        <h2>🤖 Requests by Model (Real Azure Models)</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
        for model, count in sorted(stats["models"].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0
            html += f"""
                <tr>
                    <td>{model}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""

    if stats["errors"]:
        html += """
        <h2>❌ Errors by Type</h2>
        <table>
            <thead>
                <tr>
                    <th>Error Type</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
"""
        for error_type, count in sorted(stats["errors"].items(), key=lambda x: x[1], reverse=True):
            html += f"""
                <tr>
                    <td>{error_type}</td>
                    <td>{count}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""

    html += f"""
        <h2>🔗 Actions</h2>
        <a href="/metrics" class="button">📊 View Prometheus Metrics</a>
        <a href="/" class="button">🔄 Refresh Dashboard</a>

        <div class="timestamp">
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            | Auto-refresh in 5 seconds | Active requests: {int(active_requests._value._value)}
        </div>
    </div>
</body>
</html>
"""

    return web.Response(text=html, content_type="text/html")


async def init_app():
    """Initialize the web application."""
    global client

    # Create Azure LLM client
    logger.info("Creating Azure LLM client...")
    try:
        config = AzureConfig()
        client = AzureLLMClient(config=config)
        logger.info(f"✅ Client created successfully")
        logger.info(f"   Chat deployment: {config.chat_deployment}")
        logger.info(f"   Embedding deployment: {config.embedding_deployment}")
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        logger.error("Make sure you have set:")
        logger.error("  - AZURE_OPENAI_API_KEY")
        logger.error("  - AZURE_OPENAI_ENDPOINT")
        logger.error("  - AZURE_OPENAI_DEPLOYMENT")
        raise

    # Test connection with a simple request
    logger.info("Testing API connection...")
    try:
        await make_real_chat_request("Hello! Say 'hi' back.", model=config.chat_deployment)
        logger.info("✅ API connection test successful")
    except Exception as e:
        logger.error(f"API connection test failed: {e}")
        raise

    # Create web app
    app = web.Application()
    app.router.add_get("/", dashboard_endpoint)
    app.router.add_get("/metrics", metrics_endpoint)

    # Start background RPM/TPM refresher
    app["rpm_tpm_task"] = asyncio.create_task(_refresh_rpm_tpm())

    return app


async def cleanup(app):
    """Cleanup resources."""
    global client, _prometheus_session
    if client:
        await client.close()
    if _prometheus_session is not None:
        await _prometheus_session.close()
    # Cancel background task if present
    task = app.get("rpm_tpm_task")
    if task is not None:
        task.cancel()


async def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Azure LLM Toolkit - Live Prometheus Dashboard")
    logger.info("=" * 60)
    logger.info("")
    logger.info("⚠️  This uses REAL Azure OpenAI API calls")
    logger.info("⚠️  Real costs will be incurred!")
    logger.info("")

    # Initialize app
    app = await init_app()
    app.on_cleanup.append(cleanup)

    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8765)

    logger.info("")
    logger.info("🚀 Server started!")
    logger.info("")
    logger.info("📊 Dashboard:    http://localhost:8765/")
    logger.info("📈 Metrics:      http://localhost:8765/metrics")
    logger.info("")
    logger.info("🔴 Making REAL API calls every 2-5 seconds")
    logger.info("")
    logger.info("Configure Prometheus to scrape:")
    logger.info("  scrape_configs:")
    logger.info("    - job_name: 'azure-llm-toolkit'")
    logger.info("      static_configs:")
    logger.info("        - targets: ['localhost:8765']")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    await site.start()

    # Start background request generator
    asyncio.create_task(background_requests())

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

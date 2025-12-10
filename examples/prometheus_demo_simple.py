#!/usr/bin/env python3
"""
Simple Prometheus Dashboard Demo for Azure LLM Toolkit.

This demo shows Prometheus metrics without requiring Azure credentials.
It simulates LLM operations to generate realistic metrics.

Run this script and then:
- View metrics at: http://localhost:8765/metrics (Prometheus format)
- View dashboard at: http://localhost:8765/ (HTML dashboard)

Requirements:
    pip install prometheus-client aiohttp
"""

import asyncio
import logging
import random
import time
from datetime import datetime

from aiohttp import web
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Prometheus metrics
request_counter = Counter("azure_llm_requests_total", "Total number of LLM requests", ["operation", "model", "status"])

request_duration = Histogram(
    "azure_llm_request_duration_seconds", "Request duration in seconds", ["operation", "model"]
)

token_counter = Counter(
    "azure_llm_tokens_total",
    "Total tokens processed",
    ["type", "model"],  # type: input, output, cached
)

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
}


async def simulate_llm_request(operation: str, model: str):
    """Simulate an LLM request with realistic metrics."""
    global stats

    stats["total_requests"] += 1
    stats["operations"][operation] = stats["operations"].get(operation, 0) + 1
    stats["models"][model] = stats["models"].get(model, 0) + 1

    active_requests.inc()

    try:
        # Simulate request duration
        duration = random.uniform(0.5, 3.0)
        start_time = time.time()

        await asyncio.sleep(duration)

        # Simulate success/failure (95% success rate)
        success = random.random() > 0.05

        if success:
            # Simulate tokens
            if operation == "chat_completion":
                input_tokens = random.randint(50, 500)
                output_tokens = random.randint(20, 200)
                cached_tokens = random.randint(0, 50)
            elif operation == "embed_text":
                input_tokens = random.randint(10, 100)
                output_tokens = 0
                cached_tokens = 0
            else:
                input_tokens = random.randint(20, 150)
                output_tokens = random.randint(10, 100)
                cached_tokens = 0

            # Calculate cost (simplified)
            if "gpt-4" in model:
                cost = input_tokens * 0.00003 + output_tokens * 0.00006
            elif "gpt-35" in model:
                cost = input_tokens * 0.000015 + output_tokens * 0.00002
            else:
                cost = input_tokens * 0.00001 + output_tokens * 0.00002

            # Update metrics
            request_counter.labels(operation=operation, model=model, status="success").inc()
            request_duration.labels(operation=operation, model=model).observe(duration)
            token_counter.labels(type="input", model=model).inc(input_tokens)
            token_counter.labels(type="output", model=model).inc(output_tokens)
            token_counter.labels(type="cached", model=model).inc(cached_tokens)
            cost_counter.labels(model=model).inc(cost)

            # Update stats
            stats["successful_requests"] += 1
            stats["total_tokens_input"] += input_tokens
            stats["total_tokens_output"] += output_tokens
            stats["total_tokens_cached"] += cached_tokens
            stats["total_cost"] += cost

            logger.info(
                f"✅ {operation} ({model}): {input_tokens}+{output_tokens} tokens, ${cost:.6f}, {duration:.2f}s"
            )
        else:
            error_type = random.choice(["rate_limit", "timeout", "api_error"])
            request_counter.labels(operation=operation, model=model, status="error").inc()
            stats["failed_requests"] += 1
            stats["errors"][error_type] = stats["errors"].get(error_type, 0) + 1
            logger.warning(f"❌ {operation} ({model}): {error_type}")

    finally:
        active_requests.dec()


async def background_requests():
    """Generate background requests continuously."""
    operations = ["chat_completion", "embed_text", "chat_completion", "embed_text", "chat_completion"]
    models = ["gpt-4", "gpt-4", "gpt-35-turbo", "gpt-35-turbo", "text-embedding-ada-002"]

    while True:
        operation = random.choice(operations)
        model = random.choice(models)

        asyncio.create_task(simulate_llm_request(operation, model))

        # Random delay between requests
        await asyncio.sleep(random.uniform(0.5, 2.0))


async def metrics_endpoint(request):
    """Prometheus metrics endpoint."""
    metrics_data = generate_latest()
    return web.Response(body=metrics_data, content_type="text/plain; version=0.0.4")


async def dashboard_endpoint(request):
    """HTML dashboard showing metrics."""

    success_rate = (stats["successful_requests"] / stats["total_requests"] * 100) if stats["total_requests"] > 0 else 0

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Azure LLM Toolkit - Prometheus Demo</title>
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
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #2196f3;
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
        .demo-badge {{
            display: inline-block;
            background: #ff9800;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span class="status-indicator status-online"></span>
            Azure LLM Toolkit - Prometheus Metrics Dashboard
            <span class="demo-badge">DEMO MODE</span>
        </h1>

        <div class="info">
            <strong>📊 Live Metrics Dashboard (Simulated Data)</strong><br>
            This dashboard shows simulated LLM metrics and automatically refreshes every 5 seconds.
            Prometheus metrics are available at <a href="/metrics">/metrics</a>
        </div>

        <h2>📈 Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Total Requests</h3>
                <div class="value">{stats["total_requests"]}</div>
                <div class="label">All operations</div>
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
                <div class="label">USD spent</div>
            </div>
        </div>

        <h2>🔢 Token Usage</h2>
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
        <h2>🤖 Requests by Model</h2>
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
    app = web.Application()
    app.router.add_get("/", dashboard_endpoint)
    app.router.add_get("/metrics", metrics_endpoint)
    return app


async def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Azure LLM Toolkit - Prometheus Demo (Simulated)")
    logger.info("=" * 60)

    # Initialize app
    app = await init_app()

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

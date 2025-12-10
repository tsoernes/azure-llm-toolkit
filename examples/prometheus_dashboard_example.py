#!/usr/bin/env python3
"""
Prometheus Dashboard Example for Azure LLM Toolkit.

This example demonstrates:
1. Setting up Prometheus metrics collection
2. Exposing metrics via HTTP endpoint
3. Making LLM requests that generate metrics
4. Viewing metrics in a web dashboard

Run this script and then:
- View metrics at: http://localhost:8000/metrics (Prometheus format)
- View dashboard at: http://localhost:8000/ (HTML dashboard)

Requirements:
    pip install azure-llm-toolkit prometheus-client aiohttp
"""

import asyncio
import logging
from datetime import datetime

from aiohttp import web
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from azure_llm_toolkit import AzureLLMClient, AzureConfig
from azure_llm_toolkit.metrics import create_collector_with_prometheus, PROMETHEUS_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global metrics collector
metrics_collector = None
client = None


async def make_sample_requests():
    """Make some sample LLM requests to generate metrics."""
    global client

    if client is None:
        logger.error("Client not initialized")
        return

    logger.info("Making sample LLM requests to generate metrics...")

    try:
        # Request 1: Simple question
        response1 = await client.chat_completion(messages=[{"role": "user", "content": "What is 2+2?"}], max_tokens=50)
        logger.info(f"Response 1: {response1.choices[0].message.content[:50]}")

        # Request 2: Another question
        response2 = await client.chat_completion(
            messages=[{"role": "user", "content": "Name three primary colors."}], max_tokens=50
        )
        logger.info(f"Response 2: {response2.choices[0].message.content[:50]}")

        # Request 3: Embedding
        embedding = await client.embed_text(
            text="Azure LLM Toolkit is great for monitoring!",
            deployment=None,  # Use default
        )
        logger.info(f"Generated embedding with {len(embedding)} dimensions")

        logger.info("✅ Sample requests completed!")

    except Exception as e:
        logger.error(f"Error making requests: {e}")


async def metrics_endpoint(request):
    """Prometheus metrics endpoint."""
    metrics_data = generate_latest()
    return web.Response(body=metrics_data, content_type=CONTENT_TYPE_LATEST)


async def dashboard_endpoint(request):
    """HTML dashboard showing metrics."""

    # Get aggregated metrics
    aggregated = metrics_collector.aggregate() if metrics_collector else None

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Azure LLM Toolkit - Prometheus Dashboard</title>
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
        }}
        .status-online {{ background: #38ef7d; }}
        .status-offline {{ background: #f45c43; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>
            <span class="status-indicator status-online"></span>
            Azure LLM Toolkit - Prometheus Metrics Dashboard
        </h1>

        <div class="info">
            <strong>📊 Live Metrics Dashboard</strong><br>
            This dashboard automatically refreshes every 5 seconds.
            Prometheus metrics are available at <a href="/metrics">/metrics</a>
        </div>
"""

    if aggregated:
        success_rate = (
            (aggregated.successful_requests / aggregated.total_requests * 100) if aggregated.total_requests > 0 else 0
        )

        html += f"""
        <h2>📈 Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Total Requests</h3>
                <div class="value">{aggregated.total_requests}</div>
                <div class="label">All operations</div>
            </div>
            <div class="metric-card success">
                <h3>Successful</h3>
                <div class="value">{aggregated.successful_requests}</div>
                <div class="label">{success_rate:.1f}% success rate</div>
            </div>
            <div class="metric-card error">
                <h3>Failed</h3>
                <div class="value">{aggregated.failed_requests}</div>
                <div class="label">{100 - success_rate:.1f}% error rate</div>
            </div>
            <div class="metric-card cost">
                <h3>Total Cost</h3>
                <div class="value">${aggregated.total_cost:.4f}</div>
                <div class="label">USD spent</div>
            </div>
        </div>

        <h2>🔢 Token Usage</h2>
        <div class="metric-grid">
            <div class="metric-card tokens">
                <h3>Input Tokens</h3>
                <div class="value">{aggregated.total_tokens_input:,}</div>
                <div class="label">Prompt tokens</div>
            </div>
            <div class="metric-card tokens">
                <h3>Output Tokens</h3>
                <div class="value">{aggregated.total_tokens_output:,}</div>
                <div class="label">Completion tokens</div>
            </div>
            <div class="metric-card tokens">
                <h3>Cached Tokens</h3>
                <div class="value">{aggregated.total_tokens_cached:,}</div>
                <div class="label">From cache</div>
            </div>
            <div class="metric-card tokens">
                <h3>Total Tokens</h3>
                <div class="value">{aggregated.total_tokens_input + aggregated.total_tokens_output:,}</div>
                <div class="label">All tokens</div>
            </div>
        </div>

        <h2>⏱️ Performance</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Average Duration</h3>
                <div class="value">{aggregated.avg_duration_seconds:.3f}s</div>
                <div class="label">Mean latency</div>
            </div>
            <div class="metric-card success">
                <h3>Min Duration</h3>
                <div class="value">{aggregated.min_duration_seconds:.3f}s</div>
                <div class="label">Fastest request</div>
            </div>
            <div class="metric-card error">
                <h3>Max Duration</h3>
                <div class="value">{aggregated.max_duration_seconds:.3f}s</div>
                <div class="label">Slowest request</div>
            </div>
        </div>
"""

        if aggregated.requests_by_operation:
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
            for op, count in sorted(aggregated.requests_by_operation.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / aggregated.total_requests * 100) if aggregated.total_requests > 0 else 0
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

        if aggregated.requests_by_model:
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
            for model, count in sorted(aggregated.requests_by_model.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / aggregated.total_requests * 100) if aggregated.total_requests > 0 else 0
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

        if aggregated.errors_by_type:
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
            for error_type, count in sorted(aggregated.errors_by_type.items(), key=lambda x: x[1], reverse=True):
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
    else:
        html += """
        <div class="info">
            <strong>ℹ️ No metrics collected yet</strong><br>
            Make some LLM requests to see metrics appear here.
        </div>
"""

    html += f"""
        <h2>🔗 Actions</h2>
        <a href="/trigger" class="button">🚀 Trigger Sample Requests</a>
        <a href="/metrics" class="button">📊 View Prometheus Metrics</a>
        <a href="/" class="button">🔄 Refresh Dashboard</a>

        <div class="timestamp">
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            | Auto-refresh in 5 seconds
        </div>
    </div>
</body>
</html>
"""

    return web.Response(text=html, content_type="text/html")


async def trigger_endpoint(request):
    """Trigger sample requests."""
    asyncio.create_task(make_sample_requests())

    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Triggered</title>
    <meta http-equiv="refresh" content="2;url=/">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .message {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            text-align: center;
        }
        h1 { color: #667eea; margin: 0 0 20px 0; }
        p { color: #666; }
    </style>
</head>
<body>
    <div class="message">
        <h1>🚀 Sample Requests Triggered!</h1>
        <p>Making LLM requests to generate metrics...</p>
        <p>Redirecting to dashboard in 2 seconds...</p>
    </div>
</body>
</html>
"""
    return web.Response(text=html, content_type="text/html")


async def init_app():
    """Initialize the web application."""
    global metrics_collector, client

    # Check if Prometheus is available
    if not PROMETHEUS_AVAILABLE:
        logger.error("Prometheus client not available. Install with: pip install prometheus-client")
        raise RuntimeError("Prometheus client required")

    # Create metrics collector with Prometheus
    logger.info("Creating metrics collector with Prometheus...")
    metrics_collector = create_collector_with_prometheus(namespace="azure_llm")

    # Create Azure LLM client with metrics
    logger.info("Creating Azure LLM client with metrics collection...")
    try:
        config = AzureConfig()
        client = AzureLLMClient(config=config, metrics_collector=metrics_collector)
        logger.info("✅ Client created successfully")
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        logger.info("Make sure you have set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
        raise

    # Create web app
    app = web.Application()
    app.router.add_get("/", dashboard_endpoint)
    app.router.add_get("/metrics", metrics_endpoint)
    app.router.add_get("/trigger", trigger_endpoint)

    return app


async def cleanup(app):
    """Cleanup resources."""
    global client
    if client:
        await client.close()


async def main():
    """Main function."""
    logger.info("=" * 60)
    logger.info("Azure LLM Toolkit - Prometheus Dashboard Example")
    logger.info("=" * 60)

    # Initialize app
    app = await init_app()
    app.on_cleanup.append(cleanup)

    # Start server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8888)

    logger.info("")
    logger.info("🚀 Server started!")
    logger.info("")
    logger.info("📊 Dashboard:    http://localhost:8888/")
    logger.info("📈 Metrics:      http://localhost:8888/metrics")
    logger.info("🚀 Trigger Test: http://localhost:8888/trigger")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("")

    await site.start()

    # Run initial requests after a short delay
    await asyncio.sleep(2)
    logger.info("Running initial sample requests...")
    await make_sample_requests()

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down...")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

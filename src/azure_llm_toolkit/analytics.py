"""Cost analytics and reporting for Azure LLM operations.

This module provides detailed cost analysis, trend detection, anomaly detection,
and reporting capabilities for LLM usage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CostBreakdown:
    """Cost breakdown by various dimensions."""

    total_cost: float
    by_model: dict[str, float] = field(default_factory=dict)
    by_operation: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, float] = field(default_factory=dict)
    by_day: dict[str, float] = field(default_factory=dict)
    by_hour: dict[int, float] = field(default_factory=dict)


@dataclass
class UsageStats:
    """Usage statistics."""

    total_requests: int
    total_tokens_input: int
    total_tokens_output: int
    total_tokens_cached: int
    total_tokens: int
    avg_tokens_per_request: float
    requests_by_model: dict[str, int] = field(default_factory=dict)
    requests_by_operation: dict[str, int] = field(default_factory=dict)


@dataclass
class CostTrend:
    """Cost trend analysis."""

    period: str  # "daily", "weekly", "monthly"
    average_cost: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    change_percentage: float
    projection_next_period: float


@dataclass
class Anomaly:
    """Cost or usage anomaly."""

    timestamp: datetime
    metric: str  # "cost", "tokens", "requests"
    value: float
    expected_value: float
    deviation_percentage: float
    severity: str  # "low", "medium", "high"
    description: str


@dataclass
class CostReport:
    """Comprehensive cost report."""

    period_start: datetime
    period_end: datetime
    breakdown: CostBreakdown
    usage: UsageStats
    trends: list[CostTrend]
    anomalies: list[Anomaly]
    recommendations: list[str]


class CostAnalytics:
    """
    Analyze costs and usage patterns.

    Provides insights into spending, trends, and optimization opportunities.

    Example:
        >>> from azure_llm_toolkit import InMemoryCostTracker, CostAnalytics
        >>>
        >>> tracker = InMemoryCostTracker()
        >>> # ... record some costs
        >>>
        >>> analytics = CostAnalytics(tracker)
        >>> report = analytics.generate_report(days=30)
        >>> print(f"Total cost: {report.breakdown.total_cost:.2f}")
        >>> print(f"Top model: {max(report.breakdown.by_model, key=report.breakdown.by_model.get)}")
    """

    def __init__(self, cost_tracker: Any):
        """
        Initialize cost analytics.

        Args:
            cost_tracker: Cost tracker instance with get_entries() method
        """
        self.cost_tracker = cost_tracker

    def get_breakdown(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> CostBreakdown:
        """
        Get cost breakdown for a time period.

        Args:
            start_date: Start of period (default: beginning of time)
            end_date: End of period (default: now)

        Returns:
            CostBreakdown with spending by various dimensions
        """
        entries = self.cost_tracker.get_entries()

        # Filter by date
        if start_date or end_date:
            filtered = []
            for entry in entries:
                # Assume entries have a timestamp or use current time
                entry_time = entry.get("timestamp", datetime.now())
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                if start_date and entry_time < start_date:
                    continue
                if end_date and entry_time > end_date:
                    continue
                filtered.append(entry)
            entries = filtered

        breakdown = CostBreakdown(total_cost=0.0)

        for entry in entries:
            amount = entry.get("amount", 0.0)
            model = entry.get("model", "unknown")
            category = entry.get("category", "unknown")

            breakdown.total_cost += amount

            # By model
            breakdown.by_model[model] = breakdown.by_model.get(model, 0.0) + amount

            # By category
            breakdown.by_category[category] = breakdown.by_category.get(category, 0.0) + amount

            # By operation (if available in metadata)
            metadata = entry.get("metadata", {})
            operation = metadata.get("operation", "unknown")
            breakdown.by_operation[operation] = breakdown.by_operation.get(operation, 0.0) + amount

            # By day
            entry_time = entry.get("timestamp", datetime.now())
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            day_key = entry_time.strftime("%Y-%m-%d")
            breakdown.by_day[day_key] = breakdown.by_day.get(day_key, 0.0) + amount

            # By hour
            hour = entry_time.hour
            breakdown.by_hour[hour] = breakdown.by_hour.get(hour, 0.0) + amount

        return breakdown

    def get_usage_stats(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> UsageStats:
        """
        Get usage statistics for a time period.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            UsageStats with request and token counts
        """
        entries = self.cost_tracker.get_entries()

        # Filter by date
        if start_date or end_date:
            filtered = []
            for entry in entries:
                entry_time = entry.get("timestamp", datetime.now())
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                if start_date and entry_time < start_date:
                    continue
                if end_date and entry_time > end_date:
                    continue
                filtered.append(entry)
            entries = filtered

        total_requests = len(entries)
        total_tokens_input = 0
        total_tokens_output = 0
        total_tokens_cached = 0
        requests_by_model: dict[str, int] = {}
        requests_by_operation: dict[str, int] = {}

        for entry in entries:
            total_tokens_input += entry.get("tokens_input", 0)
            total_tokens_output += entry.get("tokens_output", 0)
            total_tokens_cached += entry.get("tokens_cached_input", 0)

            model = entry.get("model", "unknown")
            requests_by_model[model] = requests_by_model.get(model, 0) + 1

            metadata = entry.get("metadata", {})
            operation = metadata.get("operation", "unknown")
            requests_by_operation[operation] = requests_by_operation.get(operation, 0) + 1

        total_tokens = total_tokens_input + total_tokens_output + total_tokens_cached
        avg_tokens = total_tokens / total_requests if total_requests > 0 else 0.0

        return UsageStats(
            total_requests=total_requests,
            total_tokens_input=total_tokens_input,
            total_tokens_output=total_tokens_output,
            total_tokens_cached=total_tokens_cached,
            total_tokens=total_tokens,
            avg_tokens_per_request=avg_tokens,
            requests_by_model=requests_by_model,
            requests_by_operation=requests_by_operation,
        )

    def analyze_trends(
        self,
        days: int = 30,
        period: str = "daily",
    ) -> list[CostTrend]:
        """
        Analyze cost trends over time.

        Args:
            days: Number of days to analyze
            period: Trend period ("daily", "weekly", "monthly")

        Returns:
            List of CostTrend objects
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        breakdown = self.get_breakdown(start_date, end_date)

        if not breakdown.by_day:
            return []

        # Sort days
        sorted_days = sorted(breakdown.by_day.items())
        if len(sorted_days) < 2:
            return []

        # Calculate average cost
        total_cost = sum(cost for _, cost in sorted_days)
        avg_cost = total_cost / len(sorted_days)

        # Calculate trend
        first_half_days = sorted_days[: len(sorted_days) // 2]
        second_half_days = sorted_days[len(sorted_days) // 2 :]

        first_half_avg = sum(cost for _, cost in first_half_days) / len(first_half_days) if first_half_days else 0
        second_half_avg = sum(cost for _, cost in second_half_days) / len(second_half_days) if second_half_days else 0

        if first_half_avg == 0:
            change_pct = 0.0
            direction = "stable"
        else:
            change_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
            if abs(change_pct) < 5:
                direction = "stable"
            elif change_pct > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

        # Simple projection
        projection = second_half_avg * (1 + (change_pct / 100))

        trend = CostTrend(
            period=period,
            average_cost=avg_cost,
            trend_direction=direction,
            change_percentage=change_pct,
            projection_next_period=projection,
        )

        return [trend]

    def detect_anomalies(
        self,
        days: int = 30,
        sensitivity: float = 2.0,
    ) -> list[Anomaly]:
        """
        Detect cost and usage anomalies.

        Args:
            days: Number of days to analyze
            sensitivity: Standard deviation multiplier for anomaly detection

        Returns:
            List of detected anomalies
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        breakdown = self.get_breakdown(start_date, end_date)

        if not breakdown.by_day or len(breakdown.by_day) < 7:
            return []

        anomalies = []

        # Calculate daily cost statistics
        daily_costs = list(breakdown.by_day.values())
        avg_cost = sum(daily_costs) / len(daily_costs)

        # Calculate standard deviation
        variance = sum((x - avg_cost) ** 2 for x in daily_costs) / len(daily_costs)
        std_dev = variance**0.5

        threshold = avg_cost + (sensitivity * std_dev)

        # Check each day for anomalies
        for day_str, cost in breakdown.by_day.items():
            if cost > threshold:
                deviation_pct = ((cost - avg_cost) / avg_cost) * 100 if avg_cost > 0 else 0

                severity = "low"
                if deviation_pct > 100:
                    severity = "high"
                elif deviation_pct > 50:
                    severity = "medium"

                anomaly = Anomaly(
                    timestamp=datetime.fromisoformat(day_str),
                    metric="cost",
                    value=cost,
                    expected_value=avg_cost,
                    deviation_percentage=deviation_pct,
                    severity=severity,
                    description=f"Daily cost ({cost:.2f}) exceeded expected ({avg_cost:.2f}) by {deviation_pct:.1f}%",
                )
                anomalies.append(anomaly)

        return anomalies

    def get_recommendations(
        self,
        breakdown: CostBreakdown,
        usage: UsageStats,
    ) -> list[str]:
        """
        Generate cost optimization recommendations.

        Args:
            breakdown: Cost breakdown
            usage: Usage statistics

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Check for high-cost models
        if breakdown.by_model:
            total = breakdown.total_cost
            for model, cost in breakdown.by_model.items():
                if cost > total * 0.5:  # More than 50% of total
                    recommendations.append(
                        f"Model '{model}' accounts for {(cost / total) * 100:.1f}% of costs. "
                        f"Consider using a smaller/cheaper model for some workloads."
                    )

        # Check cache usage
        if usage.total_tokens_cached < usage.total_tokens_input * 0.1:  # Less than 10% cached
            recommendations.append(
                "Low cache hit rate detected. Enable prompt caching to reduce costs by up to 90% for repeated prompts."
            )

        # Check for optimization opportunities
        if usage.avg_tokens_per_request > 10000:
            recommendations.append(
                f"Average tokens per request is high ({usage.avg_tokens_per_request:.0f}). "
                f"Consider breaking down prompts or using summarization to reduce token usage."
            )

        # Peak hour analysis
        if breakdown.by_hour:
            # Determine peak hour safely using dict.items() so the key function and types
            # are unambiguous for type checkers and runtime.
            try:
                peak_hour, peak_cost = max(breakdown.by_hour.items(), key=lambda kv: kv[1])
            except ValueError:
                peak_hour, peak_cost = None, 0.0
            if peak_hour is not None and peak_cost > breakdown.total_cost * 0.3:  # More than 30% in one hour
                recommendations.append(
                    f"High usage detected during hour {peak_hour}:00. "
                    f"Consider load balancing or batch processing during off-peak hours."
                )

        if not recommendations:
            recommendations.append("No major optimization opportunities detected. Current usage looks efficient.")

        return recommendations

    def generate_report(
        self,
        days: int = 30,
        include_trends: bool = True,
        include_anomalies: bool = True,
    ) -> CostReport:
        """
        Generate comprehensive cost report.

        Args:
            days: Number of days to analyze
            include_trends: Whether to include trend analysis
            include_anomalies: Whether to detect anomalies

        Returns:
            CostReport with complete analysis

        Example:
            >>> analytics = CostAnalytics(tracker)
            >>> report = analytics.generate_report(days=30)
            >>>
            >>> print(f"Period: {report.period_start} to {report.period_end}")
            >>> print(f"Total cost: {report.breakdown.total_cost:.2f}")
            >>> print(f"\\nTop 3 models by cost:")
            >>> for model, cost in sorted(report.breakdown.by_model.items(), key=lambda x: x[1], reverse=True)[:3]:
            >>>     print(f"  {model}: {cost:.2f}")
            >>>
            >>> if report.anomalies:
            >>>     print(f"\\n⚠️  {len(report.anomalies)} anomalies detected")
            >>>
            >>> print(f"\\nRecommendations:")
            >>> for rec in report.recommendations:
            >>>     print(f"  • {rec}")
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get breakdown and usage
        breakdown = self.get_breakdown(start_date, end_date)
        usage = self.get_usage_stats(start_date, end_date)

        # Analyze trends
        trends = []
        if include_trends:
            trends = self.analyze_trends(days=days)

        # Detect anomalies
        anomalies = []
        if include_anomalies:
            anomalies = self.detect_anomalies(days=days)

        # Generate recommendations
        recommendations = self.get_recommendations(breakdown, usage)

        return CostReport(
            period_start=start_date,
            period_end=end_date,
            breakdown=breakdown,
            usage=usage,
            trends=trends,
            anomalies=anomalies,
            recommendations=recommendations,
        )

    def export_report_text(self, report: CostReport) -> str:
        """
        Export report as formatted text.

        Args:
            report: Cost report to export

        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("COST ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Cost: {report.breakdown.total_cost:.2f}")
        lines.append(f"Total Requests: {report.usage.total_requests:,}")
        lines.append(f"Total Tokens: {report.usage.total_tokens:,}")
        lines.append(f"  - Input: {report.usage.total_tokens_input:,}")
        lines.append(f"  - Output: {report.usage.total_tokens_output:,}")
        lines.append(f"  - Cached: {report.usage.total_tokens_cached:,}")
        lines.append(f"Average Tokens/Request: {report.usage.avg_tokens_per_request:.1f}")
        lines.append("")

        # Cost by model
        if report.breakdown.by_model:
            lines.append("COST BY MODEL")
            lines.append("-" * 80)
            for model, cost in sorted(report.breakdown.by_model.items(), key=lambda x: x[1], reverse=True):
                pct = (cost / report.breakdown.total_cost) * 100 if report.breakdown.total_cost > 0 else 0
                lines.append(f"  {model:30s} {cost:12.2f}  ({pct:5.1f}%)")
            lines.append("")

        # Cost by category
        if report.breakdown.by_category:
            lines.append("COST BY CATEGORY")
            lines.append("-" * 80)
            for category, cost in sorted(report.breakdown.by_category.items(), key=lambda x: x[1], reverse=True):
                pct = (cost / report.breakdown.total_cost) * 100 if report.breakdown.total_cost > 0 else 0
                lines.append(f"  {category:30s} {cost:12.2f}  ({pct:5.1f}%)")
            lines.append("")

        # Trends
        if report.trends:
            lines.append("TRENDS")
            lines.append("-" * 80)
            for trend in report.trends:
                lines.append(f"  Direction: {trend.trend_direction.upper()}")
                lines.append(f"  Change: {trend.change_percentage:+.1f}%")
                lines.append(f"  Average Cost: {trend.average_cost:.2f}")
                lines.append(f"  Projected Next Period: {trend.projection_next_period:.2f}")
            lines.append("")

        # Anomalies
        if report.anomalies:
            lines.append("ANOMALIES DETECTED")
            lines.append("-" * 80)
            for anomaly in report.anomalies:
                lines.append(f"  [{anomaly.severity.upper()}] {anomaly.timestamp.strftime('%Y-%m-%d')}")
                lines.append(f"    {anomaly.description}")
            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 80)
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


__all__ = [
    "CostAnalytics",
    "CostReport",
    "CostBreakdown",
    "UsageStats",
    "CostTrend",
    "Anomaly",
]

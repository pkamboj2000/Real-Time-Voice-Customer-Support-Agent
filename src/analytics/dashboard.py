"""
Simple analytics dashboard data endpoint.

Provides pre-computed analytics views that could feed a frontend
dashboard (React, Grafana, whatever). For the portfolio this just
returns JSON; in production you'd probably push to a proper BI tool.
"""

from typing import Dict, List

from src.analytics.metrics import MetricsCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardService:
    """
    Provides dashboard-ready analytics from the metrics collector.
    """

    def __init__(self, metrics: MetricsCollector):
        self._metrics = metrics

    def get_overview(self) -> Dict:
        """
        High-level overview suitable for a summary dashboard card.
        """
        summary = self._metrics.get_summary()

        return {
            "total_calls": summary["total_sessions"],
            "total_interactions": summary["total_requests"],
            "resolution_rate": summary["session_metrics"].get("resolution_rate", 0),
            "escalation_rate": summary["session_metrics"].get("escalation_rate", 0),
            "avg_response_time_ms": summary["request_metrics"].get("avg_latency_ms", 0),
            "avg_turns_per_call": summary["session_metrics"].get("avg_turns_per_session", 0),
        }

    def get_latency_breakdown(self) -> Dict:
        """Latency analysis for performance monitoring."""
        summary = self._metrics.get_summary()

        return {
            "average_ms": summary["request_metrics"].get("avg_latency_ms", 0),
            "min_ms": summary["request_metrics"].get("min_latency_ms", 0),
            "max_ms": summary["request_metrics"].get("max_latency_ms", 0),
            "percentiles": summary.get("latency_percentiles", {}),
        }

    def get_intent_breakdown(self) -> Dict:
        """Intent distribution for understanding what callers need."""
        summary = self._metrics.get_summary()
        distribution = summary.get("intent_distribution", {})

        total = sum(distribution.values()) if distribution else 1
        return {
            "distribution": distribution,
            "percentages": {
                intent: round(count / total * 100, 1)
                for intent, count in distribution.items()
            },
        }

    def get_full_dashboard(self) -> Dict:
        """Everything at once — for a single API call to populate the whole dashboard."""
        return {
            "overview": self.get_overview(),
            "latency": self.get_latency_breakdown(),
            "intents": self.get_intent_breakdown(),
        }

"""
Metrics collection and analytics for the voice support agent.

Tracks request-level and session-level metrics in memory. In a
production deployment you'd ship these to Prometheus, Datadog,
or similar. The in-memory store is fine for the dashboard endpoint
and the portfolio demo.

Metrics we track:
  - per-request: latency, intent, escalation status
  - per-session: total turns, avg latency, resolution, escalation
  - aggregate: resolution rate, avg latency, escalation rate, intent distribution
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RequestMetric:
    """Single request (one turn in a conversation)."""
    session_id: str
    intent: str
    latency_ms: float
    escalated: bool
    resolved: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionMetric:
    """Aggregate metrics for a completed call session."""
    session_id: str
    turn_count: int
    avg_latency_ms: float
    resolved: bool
    escalated: bool
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Thread-safe metrics collector. Stores everything in memory
    and provides aggregated summaries for the dashboard.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: List[RequestMetric] = []
        self._sessions: List[SessionMetric] = []

    def record_request(
        self,
        session_id: str,
        intent: str,
        latency_ms: float,
        escalated: bool = False,
        resolved: bool = True,
    ) -> None:
        """Record metrics for a single request/turn."""
        metric = RequestMetric(
            session_id=session_id,
            intent=intent,
            latency_ms=latency_ms,
            escalated=escalated,
            resolved=resolved,
        )
        with self._lock:
            self._requests.append(metric)

    def record_session_complete(
        self,
        session_id: str,
        turn_count: int,
        avg_latency: float,
        resolved: bool,
        escalated: bool,
    ) -> None:
        """Record aggregate metrics when a call session ends."""
        metric = SessionMetric(
            session_id=session_id,
            turn_count=turn_count,
            avg_latency_ms=avg_latency,
            resolved=resolved,
            escalated=escalated,
        )
        with self._lock:
            self._sessions.append(metric)

        logger.info(
            "metrics.session_recorded",
            session=session_id,
            turns=turn_count,
            resolved=resolved,
        )

    def get_summary(self) -> Dict:
        """
        Compute aggregate metrics across all recorded data.
        This is what the /metrics endpoint returns.
        """
        with self._lock:
            requests = list(self._requests)
            sessions = list(self._sessions)

        summary = {
            "total_requests": len(requests),
            "total_sessions": len(sessions),
            "request_metrics": self._compute_request_metrics(requests),
            "session_metrics": self._compute_session_metrics(sessions),
            "intent_distribution": self._compute_intent_distribution(requests),
            "latency_percentiles": self._compute_latency_percentiles(requests),
        }

        return summary

    @staticmethod
    def _compute_request_metrics(requests: List[RequestMetric]) -> Dict:
        """Aggregate request-level numbers."""
        if not requests:
            return {
                "avg_latency_ms": 0,
                "escalation_rate": 0,
                "resolution_rate": 0,
            }

        latencies = [r.latency_ms for r in requests]
        escalated_count = sum(1 for r in requests if r.escalated)
        resolved_count = sum(1 for r in requests if r.resolved)

        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "escalation_rate": round(escalated_count / len(requests), 4),
            "resolution_rate": round(resolved_count / len(requests), 4),
        }

    @staticmethod
    def _compute_session_metrics(sessions: List[SessionMetric]) -> Dict:
        """Aggregate session-level numbers."""
        if not sessions:
            return {
                "avg_turns_per_session": 0,
                "avg_session_latency_ms": 0,
                "resolution_rate": 0,
                "escalation_rate": 0,
            }

        turn_counts = [s.turn_count for s in sessions]
        latencies = [s.avg_latency_ms for s in sessions]
        resolved = sum(1 for s in sessions if s.resolved)
        escalated = sum(1 for s in sessions if s.escalated)

        return {
            "avg_turns_per_session": round(sum(turn_counts) / len(turn_counts), 1),
            "avg_session_latency_ms": round(sum(latencies) / len(latencies), 2),
            "resolution_rate": round(resolved / len(sessions), 4),
            "escalation_rate": round(escalated / len(sessions), 4),
        }

    @staticmethod
    def _compute_intent_distribution(requests: List[RequestMetric]) -> Dict[str, int]:
        """Count requests by intent category."""
        dist: Dict[str, int] = defaultdict(int)
        for r in requests:
            dist[r.intent] += 1
        return dict(sorted(dist.items(), key=lambda x: -x[1]))

    @staticmethod
    def _compute_latency_percentiles(requests: List[RequestMetric]) -> Dict:
        """Calculate p50, p90, p95, p99 latency."""
        if not requests:
            return {"p50": 0, "p90": 0, "p95": 0, "p99": 0}

        latencies = sorted(r.latency_ms for r in requests)
        n = len(latencies)

        def percentile(p):
            idx = int(n * p / 100)
            idx = min(idx, n - 1)
            return round(latencies[idx], 2)

        return {
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        }

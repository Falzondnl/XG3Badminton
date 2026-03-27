"""
observability_agent.py
======================
ObservabilityAgent — Metrics, latency tracking, and alerting for live pricing.

Tracks:
  - Per-rally pricing latency (p50, p95, p99)
  - QA gate pass/fail counts
  - Match-level event totals
  - Feed gap and ghost/suspend events

Publishes structured metrics to structlog every 10 points.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

_METRICS_PUBLISH_EVERY_N_POINTS = 10
_LATENCY_WINDOW = 100  # Rolling window for latency percentiles


@dataclass
class MatchMetrics:
    """Accumulated metrics for a match."""
    match_id: str
    points_processed: int = 0
    points_ghost: int = 0
    points_suspended: int = 0
    qa_violations: int = 0
    sharp_alerts: int = 0
    cascade_triggers: int = 0
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))

    @property
    def p50_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        data = sorted(self.latencies_ms)
        return data[int(len(data) * 0.50)]

    @property
    def p95_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        data = sorted(self.latencies_ms)
        return data[int(len(data) * 0.95)]

    @property
    def p99_ms(self) -> Optional[float]:
        if not self.latencies_ms:
            return None
        data = sorted(self.latencies_ms)
        return data[min(len(data) - 1, int(len(data) * 0.99))]


class ObservabilityAgent:
    """
    Collects and publishes live pricing metrics.

    Called after each rally completes pricing.
    """

    def __init__(self, match_id: str) -> None:
        self._metrics = MatchMetrics(match_id=match_id)

    def record_rally(
        self,
        latency_ms: float,
        is_ghost: bool = False,
        is_suspended: bool = False,
        qa_violations: int = 0,
        sharp_alert: bool = False,
        cascade_triggered: bool = False,
    ) -> None:
        """Record metrics for a single rally."""
        m = self._metrics
        m.points_processed += 1
        m.latencies_ms.append(latency_ms)

        if is_ghost:
            m.points_ghost += 1
        if is_suspended:
            m.points_suspended += 1
        if qa_violations > 0:
            m.qa_violations += qa_violations
        if sharp_alert:
            m.sharp_alerts += 1
        if cascade_triggered:
            m.cascade_triggers += 1

        # SLA check: p99 < 200ms target
        if latency_ms > 200.0:
            logger.warning(
                "live_latency_breach",
                match_id=m.match_id,
                latency_ms=round(latency_ms, 2),
                sla_threshold_ms=200,
                points_processed=m.points_processed,
            )

        # Publish periodic metrics
        if m.points_processed % _METRICS_PUBLISH_EVERY_N_POINTS == 0:
            self._publish_metrics()

    def _publish_metrics(self) -> None:
        m = self._metrics
        logger.info(
            "live_pricing_metrics",
            match_id=m.match_id,
            points_processed=m.points_processed,
            p50_ms=round(m.p50_ms, 2) if m.p50_ms else None,
            p95_ms=round(m.p95_ms, 2) if m.p95_ms else None,
            p99_ms=round(m.p99_ms, 2) if m.p99_ms else None,
            ghost_pct=round(m.points_ghost / max(1, m.points_processed), 4),
            qa_violations=m.qa_violations,
            sharp_alerts=m.sharp_alerts,
            cascade_triggers=m.cascade_triggers,
        )

    def get_metrics(self) -> Dict:
        m = self._metrics
        return {
            "match_id": m.match_id,
            "points_processed": m.points_processed,
            "p50_ms": round(m.p50_ms, 2) if m.p50_ms else None,
            "p95_ms": round(m.p95_ms, 2) if m.p95_ms else None,
            "p99_ms": round(m.p99_ms, 2) if m.p99_ms else None,
            "points_ghost": m.points_ghost,
            "points_suspended": m.points_suspended,
            "qa_violations": m.qa_violations,
            "sharp_alerts": m.sharp_alerts,
            "cascade_triggers": m.cascade_triggers,
        }

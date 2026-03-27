"""
monitoring_supervisor.py
========================
MonitoringSupervisorAgent — health, alerting, and observability.

Responsibilities:
  - Aggregate health metrics from all supervisors + feeds
  - QA gate continuous monitoring (H1, H7, H10 spot checks)
  - Latency tracking: p50/p95/p99 per-rally reprice latency
  - Alert routing (structured log + callback)
  - Operational dashboard data

Architecture:
  - Supervised by BadmintonOrchestratorAgent
  - Reads from FeedHealthMonitor + TradingControlManager
  - Publishes to structured log (structlog) — no external deps required
  - Alert thresholds from config

ZERO business logic — observability layer only.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional

import structlog

from config.badminton_config import (
    LIVE_GHOST_TRIGGER_S,
    LIVE_SUSPEND_TRIGGER_S,
    OVERROUND_MIN,
    OVERROUND_MAX,
)

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    FEED = "feed"
    PRICING = "pricing"
    QA_GATE = "qa_gate"
    LATENCY = "latency"
    SETTLEMENT = "settlement"
    TRADING = "trading"


@dataclass
class Alert:
    """Single monitoring alert."""
    severity: AlertSeverity
    category: AlertCategory
    title: str
    detail: str
    match_id: Optional[str] = None
    market_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyBucket:
    """Rolling latency tracker for a single operation."""
    operation: str
    window: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float) -> None:
        self.window.append(latency_ms)

    @property
    def p50(self) -> Optional[float]:
        if not self.window:
            return None
        s = sorted(self.window)
        return round(s[len(s) // 2], 2)

    @property
    def p95(self) -> Optional[float]:
        if not self.window:
            return None
        s = sorted(self.window)
        idx = int(len(s) * 0.95)
        return round(s[min(idx, len(s) - 1)], 2)

    @property
    def p99(self) -> Optional[float]:
        if not self.window:
            return None
        s = sorted(self.window)
        idx = int(len(s) * 0.99)
        return round(s[min(idx, len(s) - 1)], 2)

    @property
    def mean(self) -> Optional[float]:
        if not self.window:
            return None
        return round(sum(self.window) / len(self.window), 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "n_samples": len(self.window),
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "mean_ms": self.mean,
        }


@dataclass
class FeedHealthSummary:
    """Aggregated feed health snapshot."""
    feed_name: str
    status: str  # "healthy" | "degraded" | "unhealthy" | "down"
    gap_s: float
    error_rate: float
    messages_per_minute: float
    last_event_at: float


@dataclass
class MatchMonitorRecord:
    """Per-match monitoring record."""
    match_id: str
    discipline: str
    registered_at: float
    live_started_at: Optional[float] = None
    last_price_at: Optional[float] = None
    last_score_update_at: Optional[float] = None
    total_price_updates: int = 0
    total_score_updates: int = 0
    h1_failures: int = 0
    h7_failures: int = 0
    h10_failures: int = 0
    qa_gate_passes: int = 0


class MonitoringSupervisorAgent:
    """
    Observability supervisor for the full badminton platform.

    Collects metrics, validates QA gates spot-check style, and routes
    alerts to structured log + optional callback.
    """

    # Latency targets from V1 Plan (CLAUDE.md §14)
    P50_TARGET_MS = 50.0
    P95_TARGET_MS = 100.0
    P99_TARGET_MS = 200.0

    def __init__(
        self,
        alert_callback: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self._alert_callback = alert_callback
        self._alerts: Deque[Alert] = deque(maxlen=500)

        # Latency buckets
        self._latency: Dict[str, LatencyBucket] = {
            "per_rally_reprice": LatencyBucket("per_rally_reprice"),
            "pre_match_price": LatencyBucket("pre_match_price"),
            "sgp_price": LatencyBucket("sgp_price"),
            "settlement": LatencyBucket("settlement"),
            "feed_ingestion": LatencyBucket("feed_ingestion"),
        }

        # Per-match records
        self._matches: Dict[str, MatchMonitorRecord] = {}

        # Feed health snapshots
        self._feed_health: Dict[str, FeedHealthSummary] = {}

        # Counters
        self._total_alerts: Dict[AlertSeverity, int] = {s: 0 for s in AlertSeverity}
        self._total_qa_spot_checks = 0
        self._total_qa_failures = 0

        logger.info("monitoring_supervisor_initialised")

    # ------------------------------------------------------------------
    # Match lifecycle
    # ------------------------------------------------------------------

    def on_match_registered(self, match_id: str, discipline: str) -> None:
        """Record match registration."""
        self._matches[match_id] = MatchMonitorRecord(
            match_id=match_id,
            discipline=discipline,
            registered_at=time.time(),
        )

    def on_match_live_started(self, match_id: str) -> None:
        """Record live phase start."""
        if match_id in self._matches:
            self._matches[match_id].live_started_at = time.time()

    def on_match_removed(self, match_id: str) -> None:
        """Remove match monitoring record."""
        self._matches.pop(match_id, None)

    # ------------------------------------------------------------------
    # Metrics ingestion
    # ------------------------------------------------------------------

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """
        Record operation latency.

        Operations: "per_rally_reprice", "pre_match_price", "sgp_price",
                    "settlement", "feed_ingestion"
        """
        if operation not in self._latency:
            self._latency[operation] = LatencyBucket(operation)

        self._latency[operation].record(latency_ms)

        # Alert on p99 breaches
        if operation == "per_rally_reprice" and latency_ms > self.P99_TARGET_MS:
            self._emit_alert(Alert(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.LATENCY,
                title="Per-rally reprice p99 breach",
                detail=f"Latency {latency_ms:.1f}ms > {self.P99_TARGET_MS}ms target",
                metadata={"operation": operation, "latency_ms": latency_ms},
            ))

    def record_price_update(
        self,
        match_id: str,
        latency_ms: float,
        h1_passed: Optional[bool] = None,
        h7_passed: Optional[bool] = None,
        h10_passed: Optional[bool] = None,
    ) -> None:
        """Record a pricing cycle with optional QA gate results."""
        self.record_latency("per_rally_reprice", latency_ms)

        record = self._matches.get(match_id)
        if record:
            record.total_price_updates += 1
            record.last_price_at = time.time()

        self._spot_check_qa(
            match_id=match_id,
            h1_passed=h1_passed,
            h7_passed=h7_passed,
            h10_passed=h10_passed,
        )

    def record_score_update(self, match_id: str, latency_ms: float) -> None:
        """Record score update ingestion."""
        self.record_latency("feed_ingestion", latency_ms)

        record = self._matches.get(match_id)
        if record:
            record.total_score_updates += 1
            record.last_score_update_at = time.time()

    def update_feed_health(
        self,
        feed_name: str,
        status: str,
        gap_s: float,
        error_rate: float,
        messages_per_minute: float,
        last_event_at: float,
    ) -> None:
        """Update feed health snapshot from FeedHealthMonitor."""
        prev = self._feed_health.get(feed_name)
        summary = FeedHealthSummary(
            feed_name=feed_name,
            status=status,
            gap_s=gap_s,
            error_rate=error_rate,
            messages_per_minute=messages_per_minute,
            last_event_at=last_event_at,
        )
        self._feed_health[feed_name] = summary

        # Alert on status degradation
        if prev and prev.status != status:
            severity = AlertSeverity.WARNING
            if status in ("unhealthy", "down"):
                severity = AlertSeverity.CRITICAL if status == "down" else AlertSeverity.ERROR

            self._emit_alert(Alert(
                severity=severity,
                category=AlertCategory.FEED,
                title=f"Feed {feed_name!r} status changed: {prev.status} → {status}",
                detail=f"gap_s={gap_s:.1f}, error_rate={error_rate:.3f}",
                metadata={"feed_name": feed_name, "gap_s": gap_s, "error_rate": error_rate},
            ))

        # Alert on ghost mode
        if gap_s >= LIVE_GHOST_TRIGGER_S and status == "degraded":
            self._emit_alert(Alert(
                severity=AlertSeverity.WARNING,
                category=AlertCategory.FEED,
                title=f"Feed {feed_name!r} in ghost mode",
                detail=f"Gap {gap_s:.1f}s >= {LIVE_GHOST_TRIGGER_S}s threshold",
                metadata={"feed_name": feed_name, "gap_s": gap_s},
            ))

    # ------------------------------------------------------------------
    # QA gate spot checks
    # ------------------------------------------------------------------

    def _spot_check_qa(
        self,
        match_id: str,
        h1_passed: Optional[bool],
        h7_passed: Optional[bool],
        h10_passed: Optional[bool],
    ) -> None:
        """Emit alerts if QA gates fail."""
        self._total_qa_spot_checks += 1
        record = self._matches.get(match_id)

        if h1_passed is False:
            self._total_qa_failures += 1
            if record:
                record.h1_failures += 1
            self._emit_alert(Alert(
                severity=AlertSeverity.ERROR,
                category=AlertCategory.QA_GATE,
                title="H1 overround gate failure",
                detail=(
                    f"Overround outside [{OVERROUND_MIN:.1%}, {OVERROUND_MAX:.1%}] "
                    f"for match {match_id!r}"
                ),
                match_id=match_id,
            ))

        if h7_passed is False:
            self._total_qa_failures += 1
            if record:
                record.h7_failures += 1
            self._emit_alert(Alert(
                severity=AlertSeverity.ERROR,
                category=AlertCategory.QA_GATE,
                title="H7 arbitrage-free gate failure",
                detail=f"Market not arbitrage-free for match {match_id!r}",
                match_id=match_id,
            ))

        if h10_passed is False:
            self._total_qa_failures += 1
            if record:
                record.h10_failures += 1
            self._emit_alert(Alert(
                severity=AlertSeverity.ERROR,
                category=AlertCategory.QA_GATE,
                title="H10 minimum odds gate failure",
                detail=f"Odds below 1.01 for match {match_id!r}",
                match_id=match_id,
            ))

        if h1_passed and h7_passed and h10_passed:
            if record:
                record.qa_gate_passes += 1

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------

    def _emit_alert(self, alert: Alert) -> None:
        """Store alert and route to logger + optional callback."""
        self._alerts.append(alert)
        self._total_alerts[alert.severity] += 1

        log_fn = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }[alert.severity]

        log_fn(
            "monitoring_alert",
            severity=alert.severity.value,
            category=alert.category.value,
            title=alert.title,
            detail=alert.detail,
            match_id=alert.match_id,
            market_id=alert.market_id,
            **alert.metadata,
        )

        if self._alert_callback is not None:
            try:
                self._alert_callback(alert)
            except Exception as exc:
                logger.error("alert_callback_error", error=str(exc))

    def emit_info(self, category: AlertCategory, title: str, detail: str = "") -> None:
        """Emit an informational alert."""
        self._emit_alert(Alert(
            severity=AlertSeverity.INFO,
            category=category,
            title=title,
            detail=detail,
        ))

    def emit_warning(
        self,
        category: AlertCategory,
        title: str,
        detail: str = "",
        match_id: Optional[str] = None,
    ) -> None:
        """Emit a warning alert."""
        self._emit_alert(Alert(
            severity=AlertSeverity.WARNING,
            category=category,
            title=title,
            detail=detail,
            match_id=match_id,
        ))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_latency_report(self) -> Dict[str, Any]:
        """Return latency metrics for all operations."""
        report = {}
        for op, bucket in self._latency.items():
            d = bucket.to_dict()
            if op == "per_rally_reprice":
                d["p50_target_ms"] = self.P50_TARGET_MS
                d["p95_target_ms"] = self.P95_TARGET_MS
                d["p99_target_ms"] = self.P99_TARGET_MS
                d["p50_ok"] = (bucket.p50 or 0) <= self.P50_TARGET_MS
                d["p95_ok"] = (bucket.p95 or 0) <= self.P95_TARGET_MS
                d["p99_ok"] = (bucket.p99 or 0) <= self.P99_TARGET_MS
            report[op] = d
        return report

    def get_feed_health_report(self) -> Dict[str, Any]:
        """Return all feed health summaries."""
        return {
            name: {
                "status": s.status,
                "gap_s": round(s.gap_s, 1),
                "error_rate": round(s.error_rate, 4),
                "messages_per_minute": round(s.messages_per_minute, 1),
                "last_event_age_s": round(time.time() - s.last_event_at, 1),
            }
            for name, s in self._feed_health.items()
        }

    def get_qa_summary(self) -> Dict[str, Any]:
        """Return QA gate aggregate statistics."""
        return {
            "total_spot_checks": self._total_qa_spot_checks,
            "total_failures": self._total_qa_failures,
            "failure_rate": (
                round(self._total_qa_failures / self._total_qa_spot_checks, 4)
                if self._total_qa_spot_checks > 0
                else 0.0
            ),
            "per_match": {
                match_id: {
                    "h1_failures": r.h1_failures,
                    "h7_failures": r.h7_failures,
                    "h10_failures": r.h10_failures,
                    "qa_gate_passes": r.qa_gate_passes,
                }
                for match_id, r in self._matches.items()
            },
        }

    def get_alert_summary(self) -> Dict[str, Any]:
        """Return alert count by severity + recent alerts."""
        recent = list(self._alerts)[-20:]
        return {
            "counts": {s.value: c for s, c in self._total_alerts.items()},
            "recent_alerts": [
                {
                    "severity": a.severity.value,
                    "category": a.category.value,
                    "title": a.title,
                    "match_id": a.match_id,
                    "timestamp": a.timestamp,
                }
                for a in reversed(recent)
            ],
        }

    def get_full_dashboard(self) -> Dict[str, Any]:
        """Complete monitoring dashboard payload."""
        active_matches = len(self._matches)
        return {
            "timestamp": time.time(),
            "active_matches": active_matches,
            "feeds": self.get_feed_health_report(),
            "latency": self.get_latency_report(),
            "qa": self.get_qa_summary(),
            "alerts": self.get_alert_summary(),
        }

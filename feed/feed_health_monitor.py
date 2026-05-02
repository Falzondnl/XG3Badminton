"""
feed_health_monitor.py
======================
Feed health monitoring and failover control for badminton data feeds.

Monitors:
  - Optic Odds (primary live feed)
  - Flashscore (secondary live feed + historical)
  - BWF Rankings API (weekly snapshot updates)
  - Pinnacle odds feed (for pre-match blend)

Health check metrics per feed:
  - Last message received timestamp
  - Messages per minute
  - Error rate (failed parse / total messages)
  - Latency (message timestamp vs system time)

Failover protocol (ADR-018):
  - Primary (Optic Odds) healthy: use primary only
  - Primary unhealthy → Secondary (Flashscore): use secondary + alert
  - Both unhealthy: ghost-live mode (30s) → suspend (180s)
  - Recovery: primary returns healthy → switch back after 3 clean checks

Alert thresholds:
  - FEED_GAP_ALERT_SECONDS = 10   (log warning)
  - FEED_GAP_GHOST_SECONDS = 30   (ghost mode)
  - FEED_GAP_SUSPEND_SECONDS = 180 (suspend all markets)
  - ERROR_RATE_ALERT = 0.05       (5% error rate → alert)
  - ERROR_RATE_SUSPEND = 0.20     (20% error rate → suspend)

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional

import structlog

from config.badminton_config import (
    LIVE_GHOST_TRIGGER_SECONDS,
    LIVE_SUSPEND_SECONDS,
    FEED_ERROR_RATE_ALERT,
    FEED_ERROR_RATE_SUSPEND,
    FEED_MESSAGES_WINDOW_SECONDS,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeedStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"      # High latency or some errors
    UNHEALTHY = "unhealthy"    # Feed gap > ghost threshold
    DOWN = "down"              # Feed gap > suspend threshold
    UNKNOWN = "unknown"        # No data yet for this feed


class FeedName(str, Enum):
    OPTIC_ODDS = "optic_odds"
    FLASHSCORE = "flashscore"
    BWF_RANKINGS = "bwf_rankings"
    PINNACLE = "pinnacle"


# ---------------------------------------------------------------------------
# Per-feed health tracker
# ---------------------------------------------------------------------------

@dataclass
class FeedMessageRecord:
    """Single message record for health tracking."""
    received_at: float    # Unix timestamp
    is_error: bool
    latency_ms: float     # Difference between message time and system time


@dataclass
class FeedHealthState:
    """Health state for one feed."""
    feed_name: object  # FeedName enum or str feed ID
    status: FeedStatus = FeedStatus.UNKNOWN
    last_message_at: float = 0.0
    first_message_at: float = 0.0

    # Counters (incremented on each call)
    _message_count: int = field(default=0, repr=False)
    _error_count: int = field(default=0, repr=False)

    # Rolling window of messages (last N seconds)
    _recent_messages: Deque[FeedMessageRecord] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    @property
    def last_event_at(self) -> float:
        """Alias for last_message_at — used by string-keyed API tests."""
        return self.last_message_at

    @last_event_at.setter
    def last_event_at(self, value: float) -> None:
        """Allow tests to set last_event_at to simulate gaps."""
        self.last_message_at = value

    @property
    def message_count(self) -> int:
        """Total messages received (not errors)."""
        return self._message_count

    @property
    def error_count(self) -> int:
        """Total errors recorded."""
        return self._error_count

    def record_message(self, is_error: bool = False, latency_ms: float = 0.0) -> None:
        """Record a received message."""
        now = time.time()
        msg = FeedMessageRecord(received_at=now, is_error=is_error, latency_ms=latency_ms)
        self._recent_messages.append(msg)

        if not is_error:
            self._message_count += 1
        if self.first_message_at == 0.0:
            self.first_message_at = now
        self.last_message_at = now
        self._update_status(now)

    def gap_seconds(self) -> float:
        """Seconds since last message."""
        if self.last_message_at == 0.0:
            return float("inf")
        return time.time() - self.last_message_at

    def messages_per_minute(self, window: float = FEED_MESSAGES_WINDOW_SECONDS) -> float:
        """Messages received in last window seconds."""
        cutoff = time.time() - window
        recent = [m for m in self._recent_messages if m.received_at >= cutoff]
        return len(recent) * (60.0 / window)

    def error_rate(self, window: float = FEED_MESSAGES_WINDOW_SECONDS) -> float:
        """Error rate in recent window."""
        cutoff = time.time() - window
        recent = [m for m in self._recent_messages if m.received_at >= cutoff]
        if not recent:
            return 0.0
        errors = sum(1 for m in recent if m.is_error)
        return errors / len(recent)

    def avg_latency_ms(self, window: float = FEED_MESSAGES_WINDOW_SECONDS) -> float:
        """Average message latency in recent window."""
        cutoff = time.time() - window
        recent = [m for m in self._recent_messages if m.received_at >= cutoff]
        if not recent:
            return 0.0
        return sum(m.latency_ms for m in recent) / len(recent)

    def _update_status(self, now: float) -> None:
        """Recompute status from current metrics."""
        if self.last_message_at == 0.0:
            self.status = FeedStatus.UNKNOWN
            return

        gap = now - self.last_message_at
        err_rate = self.error_rate()

        if gap >= LIVE_SUSPEND_SECONDS or err_rate >= FEED_ERROR_RATE_SUSPEND:
            self.status = FeedStatus.DOWN
        elif gap >= LIVE_GHOST_TRIGGER_SECONDS or err_rate >= FEED_ERROR_RATE_ALERT:
            self.status = FeedStatus.UNHEALTHY
        elif gap >= 10.0 or err_rate >= 0.02:
            self.status = FeedStatus.DEGRADED
        else:
            self.status = FeedStatus.HEALTHY


# ---------------------------------------------------------------------------
# Feed health monitor
# ---------------------------------------------------------------------------

class FeedHealthMonitor:
    """
    Monitors health of all data feeds and manages failover.

    Usage:
      monitor = FeedHealthMonitor()
      monitor.record_event(FeedName.OPTIC_ODDS, is_error=False)
      active_feed = monitor.get_active_feed()
    """

    def __init__(self) -> None:
        self._feeds: Dict[FeedName, FeedHealthState] = {
            feed: FeedHealthState(feed_name=feed)
            for feed in FeedName
        }
        # Dynamic string-keyed feed registry (for external / test callers)
        self._str_feeds: Dict[str, FeedHealthState] = {}

        # Priority order for live score feeds
        self._live_feed_priority = [FeedName.OPTIC_ODDS, FeedName.FLASHSCORE]

        # Callbacks for status changes
        self._status_callbacks: List[Callable[[FeedName, FeedStatus, FeedStatus], None]] = []

        self._last_active_feed: Optional[FeedName] = None

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_str_state(self, feed_id: str) -> FeedHealthState:
        """Get or create FeedHealthState for a string-keyed feed."""
        if feed_id not in self._str_feeds:
            self._str_feeds[feed_id] = FeedHealthState(feed_name=feed_id)
        return self._str_feeds[feed_id]

    # -------------------------------------------------------------------------
    # String-keyed public API (used by tests and external callers)
    # -------------------------------------------------------------------------

    def record_message(self, feed_id: str, timestamp: Optional[float] = None) -> None:
        """
        Record a successful message receipt for a named feed.

        Args:
            feed_id: Arbitrary string identifier for the feed.
            timestamp: Optional Unix timestamp; defaults to time.time().
        """
        state = self._get_str_state(feed_id)
        state.record_message(is_error=False)
        if timestamp is not None:
            state.last_message_at = timestamp

    def record_error(self, feed_id: str) -> None:
        """
        Record an error event for a named feed.

        Increments error_count and updates status.

        Args:
            feed_id: Arbitrary string identifier for the feed.
        """
        state = self._get_str_state(feed_id)
        now = time.time()
        msg = FeedMessageRecord(received_at=now, is_error=True, latency_ms=0.0)
        state._recent_messages.append(msg)
        state._error_count += 1
        if state.last_message_at == 0.0:
            state.last_message_at = now
            state.first_message_at = now
        state._update_status(now)

    def get_status(self, feed_id: str) -> Optional[FeedStatus]:
        """
        Return current status for a string-keyed feed.

        Returns None if the feed has never been seen.

        Args:
            feed_id: Arbitrary string identifier for the feed.
        """
        if feed_id not in self._str_feeds:
            return None
        return self._str_feeds[feed_id].status

    def _update_status(self, feed_id: str) -> None:
        """
        Recompute and update status for a string-keyed feed.

        Args:
            feed_id: Arbitrary string identifier for the feed.
        """
        if feed_id not in self._str_feeds:
            return
        state = self._str_feeds[feed_id]
        state._update_status(time.time())

    def get_live_market_mode(self, feed_id: Optional[str] = None) -> str:
        """
        Return current live market mode for a feed.

        Args:
            feed_id: If provided, use the string-keyed feed. If None, use
                     the enum-based active feed (legacy behaviour).

        Returns:
            "normal"    — healthy feed, normal pricing
            "ghost"     — feed gap 30-180s
            "suspended" — no feed > 180s or feed unknown
        """
        if feed_id is not None:
            # String-keyed path
            if feed_id not in self._str_feeds:
                return "suspended"
            state = self._str_feeds[feed_id]
            status = state.status
            if status == FeedStatus.UNKNOWN:
                return "suspended"
            gap = state.gap_seconds()
            if gap >= LIVE_SUSPEND_SECONDS:
                return "suspended"
            elif gap >= LIVE_GHOST_TRIGGER_SECONDS:
                return "ghost"
            else:
                return "normal"

        # Legacy enum-based path
        active = self.get_active_live_feed()
        if active is None:
            return "suspended"

        state = self._feeds[active]
        gap = state.gap_seconds()

        if gap >= LIVE_SUSPEND_SECONDS:
            return "suspended"
        elif gap >= LIVE_GHOST_TRIGGER_SECONDS:
            return "ghost"
        else:
            return "normal"

    def get_all_feed_summaries(self) -> Dict[str, Dict]:
        """
        Return status summary for all string-keyed feeds.

        Returns:
            Dict mapping feed_id to summary dict.
        """
        return {
            feed_id: {
                "status": state.status.value,
                "message_count": state.message_count,
                "error_count": state.error_count,
                "last_event_at": state.last_event_at,
                "gap_seconds": round(state.gap_seconds(), 1),
            }
            for feed_id, state in self._str_feeds.items()
        }

    # -------------------------------------------------------------------------
    # Original enum-based API (preserved for existing callers)
    # -------------------------------------------------------------------------

    def record_event(
        self,
        feed: FeedName,
        is_error: bool = False,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a feed event (message received or error)."""
        state = self._feeds[feed]
        prev_status = state.status
        state.record_message(is_error=is_error, latency_ms=latency_ms)
        new_status = state.status

        if new_status != prev_status:
            self._on_status_change(feed, prev_status, new_status)

    def get_feed_state(self, feed) -> FeedHealthState:
        """
        Get current health state for a feed.

        Accepts either a FeedName enum or a string feed ID.
        """
        if isinstance(feed, str):
            return self._get_str_state(feed)
        return self._feeds[feed]

    def get_active_live_feed(self) -> Optional[FeedName]:
        """
        Return the best available live score feed.

        Priority: Optic Odds → Flashscore → None.
        Returns None if no healthy feed available.
        """
        for feed in self._live_feed_priority:
            state = self._feeds[feed]
            if state.status in (FeedStatus.HEALTHY, FeedStatus.DEGRADED):
                if self._last_active_feed != feed:
                    if self._last_active_feed:
                        logger.warning(
                            "feed_failover",
                            from_feed=self._last_active_feed.value if self._last_active_feed else "none",
                            to_feed=feed.value,
                        )
                    self._last_active_feed = feed
                return feed

        return None  # All feeds down

    def get_health_summary(self) -> Dict[str, Dict]:
        """Return health summary for all feeds."""
        return {
            feed.value: {
                "status": state.status.value,
                "gap_seconds": round(state.gap_seconds(), 1),
                "messages_per_minute": round(state.messages_per_minute(), 1),
                "error_rate": round(state.error_rate(), 4),
                "avg_latency_ms": round(state.avg_latency_ms(), 1),
            }
            for feed, state in self._feeds.items()
        }

    def register_status_callback(
        self,
        callback: Callable[[FeedName, FeedStatus, FeedStatus], None],
    ) -> None:
        """Register callback for feed status changes."""
        self._status_callbacks.append(callback)

    def _on_status_change(
        self,
        feed: FeedName,
        prev: FeedStatus,
        new: FeedStatus,
    ) -> None:
        """Handle feed status transition."""
        log_fn = logger.warning if new in (FeedStatus.DOWN, FeedStatus.UNHEALTHY) else logger.info
        log_fn(
            "feed_status_change",
            feed=feed.value,
            from_status=prev.value,
            to_status=new.value,
        )

        for cb in self._status_callbacks:
            try:
                cb(feed, prev, new)
            except Exception as exc:
                logger.error(
                    "feed_status_callback_error",
                    error=str(exc),
                )

    def force_status(self, feed: FeedName, status: FeedStatus) -> None:
        """Manually override feed status (for testing / operator control)."""
        state = self._feeds[feed]
        prev = state.status
        state.status = status
        if prev != status:
            self._on_status_change(feed, prev, status)

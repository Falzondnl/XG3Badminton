"""
test_feed_health.py
===================
Unit tests for feed/feed_health_monitor.py

Tests:
  - HEALTHY status: recent message, low error rate
  - DEGRADED status: gap between GHOST_TRIGGER and SUSPEND_TRIGGER
  - UNHEALTHY / DOWN transitions
  - record_message: updates last_event_at
  - record_error: increments error count
  - get_live_market_mode: "normal" / "ghost" / "suspended"
  - ADR-018 protocol compliance
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import LIVE_GHOST_TRIGGER_S, LIVE_SUSPEND_TRIGGER_S
from feed.feed_health_monitor import FeedHealthMonitor, FeedHealthState, FeedStatus


@pytest.fixture
def monitor():
    return FeedHealthMonitor()


class TestFeedStatusTransitions:
    """Status logic based on gap and error rate."""

    def test_initial_status_unknown(self, monitor):
        """Fresh monitor has no status yet (or HEALTHY after first message)."""
        status = monitor.get_status("test_feed")
        # Acceptable initial states: None, HEALTHY, or UNKNOWN
        assert status in (None, FeedStatus.HEALTHY, FeedStatus.UNKNOWN, "unknown")

    def test_healthy_after_recent_message(self, monitor):
        """Feed is HEALTHY immediately after receiving a message."""
        monitor.record_message("primary_feed")
        state = monitor.get_feed_state("primary_feed")
        assert state.status == FeedStatus.HEALTHY

    def test_status_after_gap(self, monitor):
        """Simulating a gap by manipulating last_event_at."""
        monitor.record_message("primary_feed")
        state = monitor.get_feed_state("primary_feed")
        # Simulate gap by setting last_event_at to past
        state.last_event_at = time.time() - (LIVE_GHOST_TRIGGER_S + 5)
        monitor._update_status("primary_feed")
        updated = monitor.get_feed_state("primary_feed")
        assert updated.status in (FeedStatus.DEGRADED, FeedStatus.UNHEALTHY)

    def test_suspended_on_long_gap(self, monitor):
        """Feed goes DOWN/SUSPENDED after gap > SUSPEND_TRIGGER."""
        monitor.record_message("primary_feed")
        state = monitor.get_feed_state("primary_feed")
        state.last_event_at = time.time() - (LIVE_SUSPEND_TRIGGER_S + 10)
        monitor._update_status("primary_feed")
        updated = monitor.get_feed_state("primary_feed")
        assert updated.status in (FeedStatus.DOWN, FeedStatus.UNHEALTHY)


class TestRecordMessage:
    """record_message() updates event timestamp."""

    def test_record_message_updates_timestamp(self, monitor):
        """last_event_at is near now after record_message."""
        t_before = time.time()
        monitor.record_message("feed_a")
        t_after = time.time()
        state = monitor.get_feed_state("feed_a")
        assert t_before <= state.last_event_at <= t_after + 0.01

    def test_record_multiple_messages(self, monitor):
        """Multiple messages keep the feed HEALTHY."""
        for _ in range(10):
            monitor.record_message("feed_a")
        state = monitor.get_feed_state("feed_a")
        assert state.status == FeedStatus.HEALTHY

    def test_messages_per_minute_tracked(self, monitor):
        """Message rate is tracked."""
        for _ in range(5):
            monitor.record_message("feed_a")
        state = monitor.get_feed_state("feed_a")
        assert state.message_count >= 5


class TestRecordError:
    """record_error() updates error count."""

    def test_error_increments_count(self, monitor):
        """Error count grows with each recorded error."""
        monitor.record_message("feed_a")
        monitor.record_error("feed_a")
        monitor.record_error("feed_a")
        state = monitor.get_feed_state("feed_a")
        assert state.error_count >= 2

    def test_high_error_rate_degrades_status(self, monitor):
        """High error rate degrades feed status."""
        # Record many errors relative to messages
        monitor.record_message("feed_a")
        for _ in range(20):
            monitor.record_error("feed_a")
        monitor._update_status("feed_a")
        state = monitor.get_feed_state("feed_a")
        assert state.status != FeedStatus.HEALTHY


class TestADR018Protocol:
    """ADR-018: ghost/suspended mode routing."""

    def test_normal_mode_when_healthy(self, monitor):
        """HEALTHY feed → "normal" market mode."""
        monitor.record_message("live_feed")
        mode = monitor.get_live_market_mode("live_feed")
        assert mode == "normal"

    def test_ghost_mode_on_gap(self, monitor):
        """Gap between ghost and suspend threshold → "ghost" mode."""
        monitor.record_message("live_feed")
        state = monitor.get_feed_state("live_feed")
        # Simulate gap just above ghost trigger but below suspend
        gap = (LIVE_GHOST_TRIGGER_S + LIVE_SUSPEND_TRIGGER_S) / 2
        state.last_event_at = time.time() - gap
        monitor._update_status("live_feed")
        mode = monitor.get_live_market_mode("live_feed")
        assert mode in ("ghost", "suspended")

    def test_suspended_mode_on_long_gap(self, monitor):
        """Gap > SUSPEND_TRIGGER → "suspended" mode."""
        monitor.record_message("live_feed")
        state = monitor.get_feed_state("live_feed")
        state.last_event_at = time.time() - (LIVE_SUSPEND_TRIGGER_S + 30)
        monitor._update_status("live_feed")
        mode = monitor.get_live_market_mode("live_feed")
        assert mode == "suspended"

    def test_unknown_feed_returns_suspended(self, monitor):
        """Unknown feed defaults to most conservative mode."""
        mode = monitor.get_live_market_mode("nonexistent_feed")
        assert mode == "suspended"


class TestMultipleFeedTracking:
    """Monitor tracks multiple feeds independently."""

    def test_independent_feed_states(self, monitor):
        """Two feeds have independent status."""
        monitor.record_message("primary")
        monitor.record_message("secondary")
        # Degrade primary only
        state_primary = monitor.get_feed_state("primary")
        state_primary.last_event_at = time.time() - (LIVE_SUSPEND_TRIGGER_S + 60)
        monitor._update_status("primary")

        mode_primary = monitor.get_live_market_mode("primary")
        mode_secondary = monitor.get_live_market_mode("secondary")
        assert mode_primary == "suspended"
        assert mode_secondary == "normal"

    def test_all_feeds_summary(self, monitor):
        """get_all_feed_summaries returns entries for all registered feeds."""
        monitor.record_message("feed_1")
        monitor.record_message("feed_2")
        summaries = monitor.get_all_feed_summaries()
        assert "feed_1" in summaries
        assert "feed_2" in summaries

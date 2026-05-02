"""
test_orchestrator.py
====================
Tests for BadmintonOrchestratorAgent — match lifecycle, registration,
queries, and operational metrics.

All external dependencies (FeedHealthMonitor, GradingService,
TradingControlManager, BadmintonTradingSupervisor) are fully mocked
so that tests run without real data, models, or infrastructure.

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Module-level patches — mock heavy external dependencies so orchestrator
# imports cleanly without real infrastructure.
# ---------------------------------------------------------------------------

_feed_monitor_mock = MagicMock()
_feed_monitor_mock.return_value.get_health_summary.return_value = {
    "optic_odds": "healthy",
    "flashscore": "healthy",
}
_feed_monitor_mock.return_value.get_live_market_mode.return_value = "normal"
_feed_monitor_mock.return_value.record_event = MagicMock()

_grading_mock = MagicMock()
_trading_ctrl_mock = MagicMock()
_trading_sup_mock = MagicMock()


@pytest.fixture(autouse=True)
def _patch_orchestrator_deps():
    """Patch all heavy imports used by orchestrator module."""
    with (
        patch("agents.orchestrator.FeedHealthMonitor", _feed_monitor_mock),
        patch("agents.orchestrator.GradingService", _grading_mock),
        patch("agents.orchestrator.TradingControlManager", _trading_ctrl_mock),
        patch("agents.orchestrator.BadmintonTradingSupervisor", _trading_sup_mock),
    ):
        yield


# ---------------------------------------------------------------------------
# Imports (after patching fixture is declared — pytest applies autouse)
# ---------------------------------------------------------------------------

from config.badminton_config import Discipline, TournamentTier
from agents.orchestrator import (
    BadmintonOrchestratorAgent,
    MatchLifecycleState,
    ActiveMatchRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def orchestrator() -> BadmintonOrchestratorAgent:
    """Return a fresh orchestrator instance with mocked dependencies."""
    return BadmintonOrchestratorAgent()


@pytest.fixture
def registered_match(orchestrator: BadmintonOrchestratorAgent) -> ActiveMatchRecord:
    """Register a standard MS match and return the record."""
    return orchestrator.register_match(
        match_id="test_match_001",
        entity_a_id="player_alpha",
        entity_b_id="player_beta",
        discipline=Discipline.MS,
        tier=TournamentTier.SUPER_1000,
        tournament_id="all_england_2025",
    )


# ---------------------------------------------------------------------------
# MatchLifecycleState enum completeness
# ---------------------------------------------------------------------------

class TestMatchLifecycleStateEnum:
    """Verify all expected lifecycle states are present in the enum."""

    EXPECTED_STATES = {
        "SCHEDULED",
        "PRE_MATCH",
        "LIVE",
        "SUSPENDED",
        "RESULTED",
        "SETTLED",
        "ABANDONED",
    }

    def test_all_expected_states_present(self) -> None:
        actual_names = {s.name for s in MatchLifecycleState}
        for expected in self.EXPECTED_STATES:
            assert expected in actual_names, (
                f"MatchLifecycleState is missing expected state: {expected}"
            )

    def test_no_unexpected_states(self) -> None:
        actual_names = {s.name for s in MatchLifecycleState}
        assert actual_names == self.EXPECTED_STATES, (
            f"MatchLifecycleState has unexpected states: "
            f"{actual_names - self.EXPECTED_STATES}"
        )

    def test_states_are_string_valued(self) -> None:
        for state in MatchLifecycleState:
            assert isinstance(state.value, str)


# ---------------------------------------------------------------------------
# Orchestrator instantiation
# ---------------------------------------------------------------------------

class TestOrchestratorInstantiation:
    """Orchestrator must instantiate cleanly without external deps."""

    def test_instantiates_without_error(self, orchestrator: BadmintonOrchestratorAgent) -> None:
        assert orchestrator is not None

    def test_no_active_matches_on_init(self, orchestrator: BadmintonOrchestratorAgent) -> None:
        matches = orchestrator.get_active_matches()
        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_get_operational_metrics_returns_dict(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        metrics = orchestrator.get_operational_metrics()
        assert isinstance(metrics, dict)
        assert "active_matches" in metrics
        assert metrics["active_matches"] == 0

    def test_get_feed_health_returns_dict(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        health = orchestrator.get_feed_health()
        assert isinstance(health, dict)


# ---------------------------------------------------------------------------
# Match registration
# ---------------------------------------------------------------------------

class TestRegisterMatch:
    """Tests for register_match() lifecycle transitions."""

    def test_register_creates_record(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        record = orchestrator.register_match(
            match_id="reg_001",
            entity_a_id="p_a",
            entity_b_id="p_b",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            tournament_id="indonesia_open_2025",
        )
        assert isinstance(record, ActiveMatchRecord)
        assert record.match_id == "reg_001"
        assert record.entity_a_id == "p_a"
        assert record.entity_b_id == "p_b"
        assert record.discipline == Discipline.MS
        assert record.tier == TournamentTier.SUPER_500

    def test_register_initial_lifecycle_state_is_pre_match(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        record = orchestrator.register_match(
            match_id="reg_002",
            entity_a_id="p_a",
            entity_b_id="p_b",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_300,
            tournament_id="thailand_open_2025",
        )
        assert record.lifecycle_state == MatchLifecycleState.PRE_MATCH

    def test_duplicate_match_id_returns_existing(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        """Registering same match_id twice must NOT raise; returns existing record."""
        orchestrator.register_match(
            match_id="dup_001",
            entity_a_id="p_a",
            entity_b_id="p_b",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_1000,
            tournament_id="t1",
        )
        # Second call with same match_id
        record2 = orchestrator.register_match(
            match_id="dup_001",
            entity_a_id="p_x",
            entity_b_id="p_y",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_750,
            tournament_id="t2",
        )
        # Should return the original record, not a new one
        assert record2.entity_a_id == "p_a"
        assert record2.discipline == Discipline.MS

    def test_register_all_disciplines(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        """All five BWF disciplines can be registered."""
        for i, disc in enumerate(Discipline):
            record = orchestrator.register_match(
                match_id=f"disc_{i}",
                entity_a_id=f"a_{i}",
                entity_b_id=f"b_{i}",
                discipline=disc,
                tier=TournamentTier.SUPER_500,
                tournament_id=f"t_{i}",
            )
            assert record.discipline == disc


# ---------------------------------------------------------------------------
# get_active_match / get_active_matches
# ---------------------------------------------------------------------------

class TestGetMatch:
    """Tests for match retrieval methods."""

    def test_get_active_match_returns_none_for_unknown(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        result = orchestrator.get_active_match("nonexistent_match_id")
        assert result is None

    def test_get_active_match_returns_registered(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        result = orchestrator.get_active_match("test_match_001")
        assert result is not None
        assert result.match_id == "test_match_001"

    def test_list_active_matches_includes_registered(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        matches = orchestrator.get_active_matches()
        assert len(matches) >= 1
        match_ids = [m.match_id for m in matches]
        assert "test_match_001" in match_ids

    def test_list_active_matches_filter_by_discipline(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        orchestrator.register_match(
            match_id="ms_01", entity_a_id="a", entity_b_id="b",
            discipline=Discipline.MS, tier=TournamentTier.SUPER_500,
            tournament_id="t",
        )
        orchestrator.register_match(
            match_id="ws_01", entity_a_id="c", entity_b_id="d",
            discipline=Discipline.WS, tier=TournamentTier.SUPER_500,
            tournament_id="t",
        )
        ms_matches = orchestrator.get_active_matches(discipline=Discipline.MS)
        assert all(m.discipline == Discipline.MS for m in ms_matches)
        assert any(m.match_id == "ms_01" for m in ms_matches)


# ---------------------------------------------------------------------------
# get_match_prices / _get_record_or_raise
# ---------------------------------------------------------------------------

class TestGetMatchPrices:
    """Test price retrieval raises for unknown matches."""

    def test_raises_for_unknown_match(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        """Calling _get_record_or_raise for unknown match raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            orchestrator._get_record_or_raise("no_such_match")


# ---------------------------------------------------------------------------
# Lifecycle transitions
# ---------------------------------------------------------------------------

class TestLifecycleTransitions:
    """Test match lifecycle state transitions."""

    def test_transition_to_live(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        orchestrator.transition_to_live("test_match_001")
        record = orchestrator.get_active_match("test_match_001")
        assert record is not None
        assert record.lifecycle_state == MatchLifecycleState.LIVE
        assert record.live_active is True

    def test_transition_to_resulted(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        orchestrator.transition_to_live("test_match_001")
        orchestrator.transition_to_resulted("test_match_001")
        record = orchestrator.get_active_match("test_match_001")
        assert record is not None
        assert record.lifecycle_state == MatchLifecycleState.RESULTED
        assert record.live_active is False

    def test_suspend_match(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        orchestrator.suspend_match("test_match_001", reason="shuttle_broken")
        record = orchestrator.get_active_match("test_match_001")
        assert record is not None
        assert record.lifecycle_state == MatchLifecycleState.SUSPENDED

    def test_resume_match_from_suspended(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        orchestrator.suspend_match("test_match_001")
        orchestrator.resume_match("test_match_001")
        record = orchestrator.get_active_match("test_match_001")
        assert record is not None
        assert record.lifecycle_state == MatchLifecycleState.LIVE

    def test_settled_match_removed_from_active(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        orchestrator.transition_to_live("test_match_001")
        orchestrator.transition_to_resulted("test_match_001")
        orchestrator.transition_to_settled("test_match_001", n_markets=10)
        # Settled match is removed from active registry
        assert orchestrator.get_active_match("test_match_001") is None


# ---------------------------------------------------------------------------
# on_feed_event (score update)
# ---------------------------------------------------------------------------

class TestOnScoreUpdate:
    """Test that on_feed_event with score_update is callable."""

    def test_on_feed_event_callable_with_score_update(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        from feed.feed_health_monitor import FeedName
        # Should not raise
        orchestrator.on_feed_event(
            feed=FeedName.OPTIC_ODDS,
            event_type="score_update",
            payload={
                "match_id": "test_match_001",
                "winner": "A",
                "score_a": 1,
                "score_b": 0,
                "game_number": 1,
                "server": "A",
            },
        )

    def test_on_feed_event_unknown_match_does_not_raise(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        from feed.feed_health_monitor import FeedName
        # Unknown match_id should be silently ignored
        orchestrator.on_feed_event(
            feed=FeedName.OPTIC_ODDS,
            event_type="score_update",
            payload={"match_id": "unknown_match_id"},
        )


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------

class TestOperationalMetrics:
    """Test get_operational_metrics returns expected structure."""

    def test_metrics_keys(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        metrics = orchestrator.get_operational_metrics()
        expected_keys = {
            "active_matches",
            "matches_processed",
            "markets_settled",
            "errors_count",
            "feed_health",
            "live_mode",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_metrics_reflect_registration(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        metrics = orchestrator.get_operational_metrics()
        assert metrics["active_matches"] == 1

    def test_metrics_reflect_settlement(
        self, orchestrator: BadmintonOrchestratorAgent, registered_match: ActiveMatchRecord
    ) -> None:
        orchestrator.transition_to_live("test_match_001")
        orchestrator.transition_to_resulted("test_match_001")
        orchestrator.transition_to_settled("test_match_001", n_markets=5)
        metrics = orchestrator.get_operational_metrics()
        assert metrics["matches_processed"] >= 1
        assert metrics["markets_settled"] >= 5
        assert metrics["active_matches"] == 0


# ---------------------------------------------------------------------------
# Feed health
# ---------------------------------------------------------------------------

class TestFeedHealth:
    """Test get_feed_health returns dict from FeedHealthMonitor."""

    def test_feed_health_returns_dict(
        self, orchestrator: BadmintonOrchestratorAgent
    ) -> None:
        health = orchestrator.get_feed_health()
        assert isinstance(health, dict)

"""
test_trading_control.py
=======================
Unit tests for markets/market_trading_control.py

Tests:
  - MarketControl state transitions (ACTIVE → SUSPENDED → LOCKED)
  - Click scale limits
  - Liability tracking and auto-suspension threshold
  - TradingControlManager: per-match control management
  - Bet recording (valid + rejected)
  - Manual suspend/resume
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from markets.market_trading_control import (
    TradingControlManager,
    MarketControl,
    MarketState,
    BetRecord,
    LiabilityPosition,
)


@pytest.fixture
def manager():
    return TradingControlManager(match_id="trading_test_001")


class TestMarketControlStates:
    """MarketControl state machine."""

    def test_initial_state_active(self, manager):
        """New market is ACTIVE by default."""
        manager.add_market("match_winner", outcomes=["player_a", "player_b"])
        ctrl = manager.get_market("match_winner")
        assert ctrl.state == MarketState.ACTIVE

    def test_suspend_market(self, manager):
        """Suspending an active market sets state to SUSPENDED."""
        manager.add_market("correct_score", outcomes=["A_2-0", "A_2-1", "B_2-0"])
        manager.suspend_market("correct_score", reason="feed_gap")
        ctrl = manager.get_market("correct_score")
        assert ctrl.state == MarketState.SUSPENDED

    def test_resume_market(self, manager):
        """Resuming a suspended market returns to ACTIVE."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.suspend_market("match_winner")
        manager.resume_market("match_winner")
        ctrl = manager.get_market("match_winner")
        assert ctrl.state == MarketState.ACTIVE

    def test_cannot_resume_locked_market(self, manager):
        """LOCKED markets cannot be resumed."""
        manager.add_market("total_games", outcomes=["over", "under"])
        manager.lock_market("total_games")
        with pytest.raises((RuntimeError, ValueError)):
            manager.resume_market("total_games")

    def test_suspend_all(self, manager):
        """suspend_all suspends every market."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.add_market("correct_score", outcomes=["A_2-0", "A_2-1"])
        manager.suspend_all(reason="match_suspended")
        for market_id in ["match_winner", "correct_score"]:
            ctrl = manager.get_market(market_id)
            assert ctrl.state == MarketState.SUSPENDED

    def test_unknown_market_raises(self, manager):
        """Accessing non-existent market raises KeyError."""
        with pytest.raises(KeyError):
            manager.get_market("nonexistent_market")


class TestClickScaling:
    """Click scale limits on bet acceptance."""

    def test_bet_within_click_scale_accepted(self, manager):
        """Bet at or below click scale is accepted."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        result = manager.record_bet(
            market_id="match_winner",
            outcome="a",
            stake=10.0,
            odds=1.85,
        )
        assert result.accepted

    def test_bet_exceeds_click_scale_rejected(self, manager):
        """Bet above click scale is rejected."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.set_click_scale("match_winner", max_stake=5.0)
        result = manager.record_bet(
            market_id="match_winner",
            outcome="a",
            stake=50.0,
            odds=1.85,
        )
        assert not result.accepted
        assert result.rejection_reason is not None

    def test_click_scale_reduced_on_momentum(self, manager):
        """Click scale can be reduced (e.g. during momentum)."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.set_click_scale("match_winner", max_stake=25.0)
        ctrl = manager.get_market("match_winner")
        assert ctrl.max_stake_per_bet == 25.0

    def test_bet_on_suspended_market_rejected(self, manager):
        """Bets rejected when market is suspended."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.suspend_market("match_winner")
        result = manager.record_bet(
            market_id="match_winner",
            outcome="a",
            stake=10.0,
            odds=1.85,
        )
        assert not result.accepted
        assert "suspended" in result.rejection_reason.lower()


class TestLiabilityTracking:
    """Liability position tracking and auto-suspension."""

    def test_liability_accumulates(self, manager):
        """Liability grows with each accepted bet."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.record_bet("match_winner", "a", stake=10.0, odds=2.00)
        pos = manager.get_liability("match_winner")
        assert pos["a"] > 0.0

    def test_auto_suspend_on_liability_threshold(self, manager):
        """Market auto-suspends when liability threshold exceeded."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.set_liability_threshold("match_winner", max_liability=50.0)

        # Place bets to exceed threshold
        for _ in range(10):
            manager.record_bet("match_winner", "a", stake=10.0, odds=2.00)

        ctrl = manager.get_market("match_winner")
        # Should be auto-suspended once liability exceeded
        assert ctrl.state == MarketState.SUSPENDED

    def test_liability_per_outcome(self, manager):
        """Liability tracked separately per outcome."""
        manager.add_market("correct_score", outcomes=["A_2-0", "A_2-1", "B_2-0"])
        manager.record_bet("correct_score", "A_2-0", stake=20.0, odds=3.0)
        manager.record_bet("correct_score", "B_2-0", stake=15.0, odds=4.0)

        pos = manager.get_liability("correct_score")
        assert pos["A_2-0"] > pos["B_2-0"] or pos.get("B_2-0", 0) > 0

    def test_total_bets_tracked(self, manager):
        """Total number of accepted bets is tracked."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.record_bet("match_winner", "a", stake=10.0, odds=1.85)
        manager.record_bet("match_winner", "b", stake=5.0, odds=2.10)
        stats = manager.get_market_stats("match_winner")
        assert stats["total_bets"] >= 2


class TestTradingControlManager:
    """TradingControlManager orchestration."""

    def test_add_multiple_markets(self, manager):
        """Multiple markets can be added and retrieved."""
        for mid in ["match_winner", "correct_score", "total_games"]:
            manager.add_market(mid, outcomes=["o1", "o2"])

        for mid in ["match_winner", "correct_score", "total_games"]:
            assert manager.get_market(mid).market_id == mid

    def test_list_active_markets(self, manager):
        """list_active_markets returns only ACTIVE markets."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.add_market("correct_score", outcomes=["A_2-0"])
        manager.suspend_market("correct_score")

        active = manager.list_active_markets()
        assert "match_winner" in active
        assert "correct_score" not in active

    def test_mark_resulted(self, manager):
        """Marking a market as resulted sets state to RESULTED."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.mark_resulted("match_winner", winning_outcome="a")
        ctrl = manager.get_market("match_winner")
        assert ctrl.state == MarketState.RESULTED
        assert ctrl.winning_outcome == "a"

    def test_operational_summary(self, manager):
        """get_operational_summary returns dict with key metrics."""
        manager.add_market("match_winner", outcomes=["a", "b"])
        manager.add_market("correct_score", outcomes=["A_2-0", "B_2-0"])
        summary = manager.get_operational_summary()
        assert "total_markets" in summary
        assert summary["total_markets"] >= 2
        assert "active_markets" in summary
        assert "suspended_markets" in summary

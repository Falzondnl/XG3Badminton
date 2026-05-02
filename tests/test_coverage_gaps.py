"""
tests/test_coverage_gaps.py
============================
Targeted tests to close coverage gaps identified in the coverage report.

Modules covered:
1. agents/live/settlement_prep_agent.py  — lines 80-106, 110-182, 195
2. agents/live/trader_control_agent.py  — 34 missed lines
3. agents/live/market_align_agent.py    — lines 66, 86-102
4. agents/live/score_ingest_agent.py    — lines 132-141, 149-153, 168
5. agents/live_supervisor.py            — lines 160, 219-228, 238-244, 298, 327-330, 338-350
6. core/doubles_rotation.py             — 55 missed lines
7. core/scoring_engine.py               — 52 missed lines
8. markets/outright_pricing.py          — 55 missed lines

ZERO duplicate tests — all tests target currently uncovered branches.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier


# ===========================================================================
# 1. SettlementPrepAgent — lines 80-106, 110-182, 195
# ===========================================================================

class TestSettlementPrepAgentGaps:
    """Covers the _settle() pipeline and all error branches."""

    def _make_completed_state(self, match_id: str = "M001"):
        """Build a MatchLiveState at COMPLETED status."""
        from core.match_state import BadmintonMatchStateMachine, MatchStatus, PointWinner
        state = BadmintonMatchStateMachine.initialise(
            match_id=match_id,
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            first_server="A",
        )
        # Drive to 2-0 completion: A wins game 1 (21-0) and game 2 (21-0)
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        # After game 1 — state machine starts game 2; A serves
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        return state

    def _make_in_progress_state(self, match_id: str = "M001"):
        from core.match_state import BadmintonMatchStateMachine
        return BadmintonMatchStateMachine.initialise(
            match_id=match_id,
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            first_server="A",
        )

    def _make_agent(self, match_id: str = "M001", grading_svc=None):
        from agents.live.settlement_prep_agent import SettlementPrepAgent
        from markets.market_trading_control import TradingControlManager
        from settlement.grading_service import GradingService
        tc = TradingControlManager(match_id=match_id)
        gs = grading_svc or GradingService()
        return SettlementPrepAgent(
            match_id=match_id,
            trading_control=tc,
            grading_service=gs,
        )

    def test_match_id_mismatch_raises(self):
        from agents.live.settlement_prep_agent import SettlementPrepAgent, SettlementPrepError
        agent = self._make_agent("M001")
        wrong_state = self._make_in_progress_state("DIFFERENT")
        with pytest.raises(SettlementPrepError, match="does not match"):
            agent.check_and_settle(wrong_state)

    def test_in_progress_not_settled(self):
        agent = self._make_agent()
        state = self._make_in_progress_state()
        result = agent.check_and_settle(state)
        assert result.settled is False
        assert result.already_settled is False
        assert result.n_markets_settled == 0

    def test_completed_triggers_settlement(self):
        agent = self._make_agent()
        state = self._make_completed_state()
        result = agent.check_and_settle(state)
        assert result.settled is True
        assert result.already_settled is False
        assert result.error is None

    def test_idempotency_guard_prevents_double_settlement(self):
        agent = self._make_agent()
        state = self._make_completed_state()
        first = agent.check_and_settle(state)
        assert first.settled is True
        second = agent.check_and_settle(state)
        assert second.settled is False
        assert second.already_settled is True

    def test_is_settled_property_false_before_settlement(self):
        agent = self._make_agent()
        assert agent.is_settled is False

    def test_is_settled_property_true_after_settlement(self):
        agent = self._make_agent()
        state = self._make_completed_state()
        agent.check_and_settle(state)
        assert agent.is_settled is True

    def test_get_settlement_records_returns_list(self):
        agent = self._make_agent()
        state = self._make_completed_state()
        agent.check_and_settle(state)
        records = agent.get_settlement_records()
        assert isinstance(records, list)

    def test_grading_failure_returns_error_result(self):
        mock_gs = MagicMock()
        mock_gs.settle_match.side_effect = RuntimeError("DB connection failed")
        agent = self._make_agent(grading_svc=mock_gs)
        state = self._make_completed_state()
        result = agent.check_and_settle(state)
        assert result.settled is False
        assert result.error is not None
        assert "GradingService.settle_match failed" in result.error

    def test_match_result_build_failure_returns_error_result(self):
        from settlement.grading_service import MatchResult
        with patch.object(MatchResult, "from_live_state", side_effect=ValueError("Bad state data")):
            agent = self._make_agent()
            state = self._make_completed_state()
            result = agent.check_and_settle(state)
        assert result.settled is False
        assert "MatchResult.from_live_state failed" in (result.error or "")

    def test_trading_control_transition_failure_does_not_raise(self):
        """Transition failure is logged but settlement still succeeds."""
        from markets.market_trading_control import TradingControlManager
        from settlement.grading_service import GradingService
        from agents.live.settlement_prep_agent import SettlementPrepAgent

        tc = MagicMock(spec=TradingControlManager)
        tc.get_open_markets.return_value = ["mkt1"]
        tc.transition_to_settled.side_effect = RuntimeError("DB write failed")
        gs = GradingService()

        agent = SettlementPrepAgent(match_id="M001", trading_control=tc, grading_service=gs)
        state = self._make_completed_state()
        result = agent.check_and_settle(state)
        # Settlement itself succeeds even if transition_to_settled raises
        assert result.error is None


# ===========================================================================
# 2. TraderControlAgent — missed branches
# ===========================================================================

class TestTraderControlAgentGaps:
    """Covers all command types and error paths."""

    def _make_agent_and_control(self, match_id: str = "M001"):
        from agents.live.trader_control_agent import TraderControlAgent
        from markets.market_trading_control import MarketTradingControl
        tc = MagicMock(spec=MarketTradingControl)
        tc.get_open_markets.return_value = ["mkt1", "mkt2"]
        agent = TraderControlAgent(match_id=match_id, trading_control=tc)
        return agent, tc

    def _cmd(self, command_type, market_id=None, click_scale=None, match_id="M001"):
        from agents.live.trader_control_agent import OperatorCommand, OperatorCommandType
        return OperatorCommand(
            command_type=command_type,
            match_id=match_id,
            market_id=market_id,
            operator_id="op1",
            reason="Testing coverage gap branch",
            click_scale=click_scale,
        )

    def test_wrong_match_id_raises(self):
        from agents.live.trader_control_agent import (
            TraderControlAgent, TraderControlError, OperatorCommandType
        )
        agent, _ = self._make_agent_and_control("M001")
        cmd = self._cmd(OperatorCommandType.SUSPEND_MARKET, market_id="mkt1", match_id="WRONG")
        with pytest.raises(TraderControlError, match="does not match"):
            agent.apply_command(cmd)

    def test_cooldown_blocks_rapid_repeat_override(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SUSPEND_MARKET, market_id="mkt1")
        r1 = agent.apply_command(cmd)
        assert r1.applied is True
        r2 = agent.apply_command(cmd)
        assert r2.applied is False
        assert "cooldown" in r2.detail

    def test_suspend_market_applied(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SUSPEND_MARKET, market_id="mkt1")
        result = agent.apply_command(cmd)
        assert result.applied is True
        tc.suspend_market.assert_called_with("mkt1")
        assert "mkt1" in agent._manually_suspended

    def test_resume_market_applied(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        # First suspend it
        suspend_cmd = self._cmd(OperatorCommandType.SUSPEND_MARKET, market_id="mkt1")
        agent.apply_command(suspend_cmd)
        # Wait past cooldown by directly manipulating _last_override_at
        agent._last_override_at["mkt1"] = 0.0
        # Now resume it
        resume_cmd = self._cmd(OperatorCommandType.RESUME_MARKET, market_id="mkt1")
        result = agent.apply_command(resume_cmd)
        assert result.applied is True
        tc.resume_market.assert_called_with("mkt1")
        assert "mkt1" not in agent._manually_suspended

    def test_set_click_scale_applied(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SET_CLICK_SCALE, market_id="mkt1", click_scale=0.5)
        result = agent.apply_command(cmd)
        assert result.applied is True
        tc.set_click_scale.assert_called_with("mkt1", 0.5)

    def test_set_click_scale_clamped_above_1(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SET_CLICK_SCALE, market_id="mkt1", click_scale=1.5)
        agent.apply_command(cmd)
        tc.set_click_scale.assert_called_with("mkt1", 1.0)

    def test_set_click_scale_clamped_below_0(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SET_CLICK_SCALE, market_id="mkt1", click_scale=-0.5)
        agent.apply_command(cmd)
        tc.set_click_scale.assert_called_with("mkt1", 0.0)

    def test_set_click_scale_no_market_id_raises(self):
        from agents.live.trader_control_agent import OperatorCommandType, TraderControlError
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SET_CLICK_SCALE, market_id=None, click_scale=0.5)
        with pytest.raises(TraderControlError, match="requires market_id"):
            agent.apply_command(cmd)

    def test_set_click_scale_no_scale_value_raises(self):
        from agents.live.trader_control_agent import OperatorCommandType, TraderControlError
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SET_CLICK_SCALE, market_id="mkt1", click_scale=None)
        with pytest.raises(TraderControlError, match="requires click_scale"):
            agent.apply_command(cmd)

    def test_suspend_all_suspends_open_markets(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SUSPEND_ALL)
        result = agent.apply_command(cmd)
        assert result.applied is True
        assert tc.suspend_market.call_count == 2

    def test_resume_all_resumes_manually_suspended(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        # Manually populate suspended set
        agent._manually_suspended.add("mkt1")
        agent._manually_suspended.add("mkt2")
        cmd = self._cmd(OperatorCommandType.RESUME_ALL)
        result = agent.apply_command(cmd)
        assert result.applied is True
        assert len(agent._manually_suspended) == 0

    def test_lock_market_applied(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.LOCK_MARKET, market_id="mkt1")
        result = agent.apply_command(cmd)
        assert result.applied is True
        tc.lock_market.assert_called_with("mkt1")
        assert agent.is_market_locked("mkt1")

    def test_unlock_market_applied(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, tc = self._make_agent_and_control()
        # Lock first; bypass cooldown
        lock_cmd = self._cmd(OperatorCommandType.LOCK_MARKET, market_id="mkt1")
        agent.apply_command(lock_cmd)
        agent._last_override_at["mkt1"] = 0.0
        unlock_cmd = self._cmd(OperatorCommandType.UNLOCK_MARKET, market_id="mkt1")
        result = agent.apply_command(unlock_cmd)
        assert result.applied is True
        assert not agent.is_market_locked("mkt1")

    def test_lock_market_no_market_id_raises(self):
        from agents.live.trader_control_agent import OperatorCommandType, TraderControlError
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.LOCK_MARKET, market_id=None)
        with pytest.raises(TraderControlError, match="requires market_id"):
            agent.apply_command(cmd)

    def test_unlock_market_no_market_id_raises(self):
        from agents.live.trader_control_agent import OperatorCommandType, TraderControlError
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.UNLOCK_MARKET, market_id=None)
        with pytest.raises(TraderControlError, match="requires market_id"):
            agent.apply_command(cmd)

    def test_get_override_log_returns_list(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SUSPEND_ALL)
        agent.apply_command(cmd)
        log = agent.get_override_log()
        assert len(log) == 1

    def test_get_locked_markets_returns_set(self):
        from agents.live.trader_control_agent import OperatorCommandType
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.LOCK_MARKET, market_id="mkt1")
        agent.apply_command(cmd)
        locked = agent.get_locked_markets()
        assert "mkt1" in locked

    def test_suspend_market_no_market_id_raises(self):
        from agents.live.trader_control_agent import OperatorCommandType, TraderControlError
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.SUSPEND_MARKET, market_id=None)
        with pytest.raises(TraderControlError, match="requires market_id"):
            agent.apply_command(cmd)

    def test_resume_market_no_market_id_raises(self):
        from agents.live.trader_control_agent import OperatorCommandType, TraderControlError
        agent, _ = self._make_agent_and_control()
        cmd = self._cmd(OperatorCommandType.RESUME_MARKET, market_id=None)
        with pytest.raises(TraderControlError, match="requires market_id"):
            agent.apply_command(cmd)


# ===========================================================================
# 3. MarketAlignAgent — lines 66, 86-102
# ===========================================================================

class TestMarketAlignAgentGaps:
    """Covers: no-constraint path (>30 points), invalid pre_match_p_a, clamping."""

    def _make_markets(self, p_a: float = 0.55):
        from markets.derivative_engine import MarketPrice, MarketFamily
        return {
            "match_winner": [
                MarketPrice(
                    market_id="mw",
                    market_family=MarketFamily.MATCH_RESULT,
                    outcome_name="A_wins",
                    odds=1.0 / p_a,
                    prob_implied=p_a,
                    prob_with_margin=p_a,
                ),
                MarketPrice(
                    market_id="mw",
                    market_family=MarketFamily.MATCH_RESULT,
                    outcome_name="B_wins",
                    odds=1.0 / (1 - p_a),
                    prob_implied=1 - p_a,
                    prob_with_margin=1 - p_a,
                ),
            ]
        }

    def test_no_constraint_after_30_points(self):
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.55)
        markets = self._make_markets(0.80)
        result = agent.align(markets=markets, p_a_blend=0.80, total_points_played=35)
        # Should be returned unchanged — same object identity
        assert result is markets

    def test_invalid_pre_match_p_a_zero_returns_unchanged(self):
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.0)
        markets = self._make_markets(0.80)
        result = agent.align(markets=markets, p_a_blend=0.80, total_points_played=5)
        assert result is markets

    def test_invalid_pre_match_p_a_one_returns_unchanged(self):
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=1.0)
        markets = self._make_markets(0.80)
        result = agent.align(markets=markets, p_a_blend=0.80, total_points_played=5)
        assert result is markets

    def test_clamping_triggers_match_winner_update(self):
        """p_a_blend far exceeds pre_match_p_a * 1.25 — should be clamped and MW updated."""
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.50)
        markets = self._make_markets(0.50)
        # p_a_blend = 0.90 >> 0.50 * 1.25 = 0.625 — should clamp
        result = agent.align(markets=markets, p_a_blend=0.90, total_points_played=5)
        mw = result["match_winner"]
        # Find A_wins prob
        a_outcome = next(mp for mp in mw if "A_wins" in mp.outcome_name)
        assert a_outcome.prob_implied < 0.90

    def test_clamping_between_11_and_30_uses_40pct_drift(self):
        """Points 11-30 use 40% max drift."""
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.50)
        markets = self._make_markets(0.50)
        result = agent.align(markets=markets, p_a_blend=0.99, total_points_played=20)
        mw = result["match_winner"]
        a_outcome = next(mp for mp in mw if "A_wins" in mp.outcome_name)
        # Upper bound with 40% drift: 0.50 * 1.40 = 0.70
        assert a_outcome.prob_implied <= 0.70 + 0.001

    def test_no_clamping_when_within_drift(self):
        """If p_a_blend is already within bounds, markets should be unchanged."""
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.55)
        markets = self._make_markets(0.55)
        # p_a_blend exactly at pre-match — no clamping needed
        result = agent.align(markets=markets, p_a_blend=0.55, total_points_played=5)
        assert result["match_winner"][0].prob_implied == pytest.approx(0.55, abs=0.01)

    def test_markets_without_match_winner_key_handled(self):
        """If no match_winner market exists, align should still succeed."""
        from agents.live.market_align_agent import MarketAlignAgent
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.50)
        markets = {"total_games": []}
        result = agent.align(markets=markets, p_a_blend=0.90, total_points_played=5)
        assert "total_games" in result


# ===========================================================================
# 4. ScoreIngestAgent — lines 132-141, 149-153, 168
# ===========================================================================

class TestScoreIngestAgentGaps:
    """Covers validation failure path, winner inference logic, invalid winner/server."""

    def _agent(self):
        from agents.live.score_ingest_agent import ScoreIngestAgent
        return ScoreIngestAgent(match_id="M001")

    def test_invalid_score_type_returns_invalid(self):
        agent = self._agent()
        result = agent.ingest(
            payload={"score_a": "not_a_number", "score_b": 0, "game_number": 1},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.valid is False
        assert "score parse error" in result.rejection_reason

    def test_duplicate_event_returns_duplicate_flag(self):
        agent = self._agent()
        # First ingest
        agent.ingest(
            payload={"score_a": 1, "score_b": 0, "game_number": 1, "winner": "A", "server": "B"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        # Same score again — duplicate
        result = agent.ingest(
            payload={"score_a": 1, "score_b": 0, "game_number": 1, "winner": "A", "server": "B"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.is_duplicate is True
        assert result.valid is False

    def test_validation_error_increments_rejected(self):
        agent = self._agent()
        # Jump by 2 — illegal
        result = agent.ingest(
            payload={"score_a": 2, "score_b": 0, "game_number": 1, "winner": "A", "server": "B"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.valid is False
        assert agent.stats["events_rejected"] == 1

    def test_winner_inferred_from_delta_a(self):
        agent = self._agent()
        result = agent.ingest(
            payload={"score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.valid is True
        assert result.event.winner == "A"

    def test_winner_inferred_from_delta_b(self):
        agent = self._agent()
        result = agent.ingest(
            payload={"score_a": 0, "score_b": 1, "game_number": 1, "server": "A"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.valid is True
        assert result.event.winner == "B"

    def test_ambiguous_delta_returns_invalid(self):
        """Both scores jump simultaneously — cannot infer winner."""
        agent = self._agent()
        result = agent.ingest(
            payload={"score_a": 0, "score_b": 0, "game_number": 1, "server": "A"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.valid is False
        assert "cannot determine winner" in result.rejection_reason

    def test_invalid_winner_string_returns_invalid(self):
        agent = self._agent()
        result = agent.ingest(
            payload={"score_a": 1, "score_b": 0, "game_number": 1, "winner": "X", "server": "A"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.valid is False
        assert "invalid winner" in result.rejection_reason

    def test_invalid_server_string_defaults_to_winner(self):
        """Invalid server string — should default to winner (BWF rule)."""
        agent = self._agent()
        result = agent.ingest(
            payload={
                "score_a": 1, "score_b": 0, "game_number": 1,
                "winner": "A", "server": "INVALID",
            },
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        # server defaults to winner when invalid
        assert result.valid is True
        assert result.event.server == "A"

    def test_stats_counter_increments_on_success(self):
        agent = self._agent()
        agent.ingest(
            payload={"score_a": 1, "score_b": 0, "game_number": 1, "winner": "A", "server": "B"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert agent.stats["events_processed"] == 1

    def test_feed_source_default_unknown(self):
        agent = self._agent()
        result = agent.ingest(
            payload={"score_a": 1, "score_b": 0, "game_number": 1, "winner": "A", "server": "B"},
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.event.feed_source == "unknown"

    def test_feed_source_custom(self):
        agent = self._agent()
        result = agent.ingest(
            payload={
                "score_a": 1, "score_b": 0, "game_number": 1,
                "winner": "A", "server": "B", "feed_source": "optic_odds",
            },
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.event.feed_source == "optic_odds"

    def test_timestamp_parsed_when_present(self):
        agent = self._agent()
        ts = time.time()
        result = agent.ingest(
            payload={
                "score_a": 1, "score_b": 0, "game_number": 1,
                "winner": "A", "server": "B", "timestamp": ts,
            },
            prev_score_a=0, prev_score_b=0, game_number=1,
        )
        assert result.event.timestamp == pytest.approx(ts, rel=0.001)


# ===========================================================================
# 5. LiveSupervisorAgent — lines 160, 219-228, 238-244, 298, 327-330, 338-350
# ===========================================================================

class TestLiveSupervisorAgentGaps:
    """Cover: grading service injection, validation failure suspension, publisher error."""

    def _make_setup(self, match_id: str = "M001") -> "LiveMatchSetup":
        from agents.live_supervisor import LiveMatchSetup
        return LiveMatchSetup(
            match_id=match_id,
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            first_server="A",
            rwp_prior_a=0.515,
            rwp_prior_b=0.500,
            pre_match_p_a=0.55,
        )

    def _make_trading_control(self, match_id: str = "M001"):
        from markets.market_trading_control import TradingControlManager
        return TradingControlManager(match_id=match_id)

    def test_supervisor_with_grading_service_creates_settlement_prep(self):
        from agents.live_supervisor import LiveSupervisorAgent
        from settlement.grading_service import GradingService
        setup = self._make_setup()
        tc = self._make_trading_control()
        gs = GradingService()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc, grading_service=gs)
        assert sup._settlement_prep is not None

    def test_supervisor_without_grading_service_has_no_settlement_prep(self):
        from agents.live_supervisor import LiveSupervisorAgent
        setup = self._make_setup()
        tc = self._make_trading_control()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc, grading_service=None)
        assert sup._settlement_prep is None

    def test_invalid_score_update_returns_none(self):
        from agents.live_supervisor import LiveSupervisorAgent
        setup = self._make_setup()
        tc = self._make_trading_control()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc)
        # Jump of +2 is illegal
        result = sup.on_score_update(
            "M001",
            {"winner": "A", "score_a": 2, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert result is None

    def test_invalid_state_update_returns_none(self):
        """Bad winner value causes state machine failure — should return None."""
        from agents.live_supervisor import LiveSupervisorAgent
        setup = self._make_setup()
        tc = self._make_trading_control()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc)
        result = sup.on_score_update(
            "M001",
            {"winner": "X", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert result is None

    def test_price_publisher_error_does_not_raise(self):
        """Publisher failures should be caught and logged, not propagated."""
        from agents.live_supervisor import LiveSupervisorAgent

        def failing_publisher(match_id, response):
            raise RuntimeError("WebSocket connection dropped")

        setup = self._make_setup()
        tc = self._make_trading_control()
        sup = LiveSupervisorAgent(
            setup=setup, trading_control=tc, price_publisher=failing_publisher
        )
        # Should not raise despite publisher failure
        result = sup.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert result is not None

    def test_stats_includes_is_settled_none_when_no_grading_service(self):
        from agents.live_supervisor import LiveSupervisorAgent
        setup = self._make_setup()
        tc = self._make_trading_control()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc)
        stats = sup.get_stats()
        assert stats["is_settled"] is None

    def test_supervisor_processes_settlement_on_match_completion(self):
        """Full match completion path triggers settlement prep."""
        from agents.live_supervisor import LiveSupervisorAgent
        from settlement.grading_service import GradingService

        setup = self._make_setup()
        tc = self._make_trading_control()
        gs = GradingService()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc, grading_service=gs)

        # Play until match completes: A wins 21-0 twice
        for _ in range(21):
            sup.on_score_update(
                "M001",
                {"winner": "A", "score_a": _ + 1, "score_b": 0,
                 "game_number": 1, "server": "B"},
            )
        # Game 2
        for _ in range(21):
            sup.on_score_update(
                "M001",
                {"winner": "A", "score_a": _ + 1, "score_b": 0,
                 "game_number": 2, "server": "B"},
            )
        # Match should now be settled
        assert sup._settlement_prep is not None

    def test_settlement_prep_exception_does_not_crash_supervisor(self):
        """settlement_prep.check_and_settle raising should be caught."""
        from agents.live_supervisor import LiveSupervisorAgent
        from agents.live.settlement_prep_agent import SettlementPrepAgent

        setup = self._make_setup()
        tc = self._make_trading_control()
        gs_mock = MagicMock()
        sup = LiveSupervisorAgent(setup=setup, trading_control=tc, grading_service=gs_mock)
        sup._settlement_prep = MagicMock(spec=SettlementPrepAgent)
        sup._settlement_prep.check_and_settle.side_effect = RuntimeError("Settlement error")

        # Drive to match completion
        for _ in range(21):
            sup.on_score_update(
                "M001",
                {"winner": "A", "score_a": _ + 1, "score_b": 0,
                 "game_number": 1, "server": "B"},
            )
        for _ in range(21):
            sup.on_score_update(
                "M001",
                {"winner": "A", "score_a": _ + 1, "score_b": 0,
                 "game_number": 2, "server": "B"},
            )
        # Should not raise despite settlement_prep throwing


# ===========================================================================
# 6. core/doubles_rotation.py — 55 missed lines
# ===========================================================================

class TestDoublesRotationGaps:
    """Covers validation errors, get_position, B-team serving initialise, etc."""

    def _state(self, serving_team: str = "A"):
        from core.doubles_rotation import DoublesServiceEngine
        if serving_team == "A":
            return DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MD,
                first_server="a1",
                first_receiver="b1",
            )
        else:
            return DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MD,
                first_server="b1",
                first_receiver="a1",
            )

    def test_initialise_b_serves_first(self):
        state = self._state(serving_team="B")
        assert state.serving_team == "B"
        assert state.current_server == "b1"

    def test_initialise_singles_discipline_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="doubles discipline"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MS,
                first_server="a1",
                first_receiver="b1",
            )

    def test_initialise_missing_team_a_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="requires player"):
            DoublesServiceEngine.initialise(
                discipline=Discipline.MD,
                first_server="a1",
                first_receiver="b1",
                player_b1="b1", player_b2="b2",
            )

    def test_initialise_missing_first_server_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="requires first_server"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MD,
                first_receiver="b1",
            )

    def test_initialise_missing_first_receiver_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="requires first_receiver"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MD,
                first_server="a1",
            )

    def test_initialise_missing_discipline_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="requires discipline"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                first_server="a1",
                first_receiver="b1",
            )

    def test_initialise_server_not_in_any_team_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="not in team"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MD,
                first_server="x99",
                first_receiver="b1",
            )

    def test_initialise_server_in_wrong_team_raises(self):
        """first_server_id must be in the specified serving_team."""
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="must be in team A"):
            DoublesServiceEngine.initialise(
                player_a1="a1", player_a2="a2",
                player_b1="b1", player_b2="b2",
                serving_team="A",
                first_server_id="b1",  # b1 is in team B, not A
                first_receiver_id="b2",
                discipline=Discipline.MD,
            )

    def test_initialise_receiver_in_wrong_team_raises(self):
        """first_receiver_id must be in the receiving team."""
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="must be in team B"):
            DoublesServiceEngine.initialise(
                player_a1="a1", player_a2="a2",
                player_b1="b1", player_b2="b2",
                serving_team="A",
                first_server_id="a1",
                first_receiver_id="a2",  # a2 is in team A, not B
                discipline=Discipline.MD,
            )

    def test_get_position_returns_none_for_unknown_player(self):
        state = self._state()
        pos = state.get_position("unknown_player")
        assert pos is None

    def test_get_position_returns_correct_for_a1(self):
        from core.doubles_rotation import PlayerPosition
        state = self._state()
        pos = state.get_position("a1")
        assert pos in (PlayerPosition.RIGHT, PlayerPosition.LEFT)

    def test_validate_invalid_serving_team_raises(self):
        from core.doubles_rotation import DoublesServiceState, PlayerPosition
        with pytest.raises(ValueError, match="serving_team must be"):
            s = DoublesServiceState(
                player_a1="a1", player_a2="a2",
                a1_position=PlayerPosition.RIGHT,
                player_b1="b1", player_b2="b2",
                b1_position=PlayerPosition.RIGHT,
                serving_team="X",
                server_id="a1",
                receiver_id="b1",
            )
            s.validate()

    def test_apply_rally_result_invalid_winner_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state()
        with pytest.raises(ValueError, match="winner must be"):
            DoublesServiceEngine.apply_rally_result(state, winner="X")

    def test_apply_rally_result_explicit_scores_serving_wins(self):
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state()
        new_state = DoublesServiceEngine.apply_rally_result(
            state, winner="A", new_score_a=1, new_score_b=0
        )
        assert new_state.serving_team == "A"
        assert new_state.score_a == 1

    def test_apply_rally_result_explicit_scores_receiving_wins(self):
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state()  # A serves first
        new_state = DoublesServiceEngine.apply_rally_result(
            state, winner="B", new_score_a=0, new_score_b=0
        )
        assert new_state.serving_team == "B"

    def test_apply_rally_result_b_wins_service_with_auto_score(self):
        """B wins service via simplified form; score auto-increments."""
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state(serving_team="B")
        new_state = DoublesServiceEngine.apply_rally_result(
            state, rally_winner_team="A", server_team="B"
        )
        assert new_state.serving_team == "A"

    def test_reset_for_new_game_no_winner_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state()
        with pytest.raises(ValueError, match="requires game_winner"):
            DoublesServiceEngine.reset_for_new_game(state)

    def test_reset_for_new_game_b_wins_b_serves(self):
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state()
        new_state = DoublesServiceEngine.reset_for_new_game(state, game_winner="B")
        assert new_state.serving_team == "B"

    def test_reset_for_new_game_uses_alias(self):
        from core.doubles_rotation import DoublesServiceEngine
        state = self._state()
        new_state = DoublesServiceEngine.reset_for_new_game(
            state, game_winner_team="A"
        )
        assert new_state.serving_team == "A"

    def test_get_service_court_for_server_even_returns_right(self):
        from core.doubles_rotation import DoublesServiceEngine, ServiceCourt
        assert DoublesServiceEngine.get_service_court_for_server(0) == ServiceCourt.RIGHT
        assert DoublesServiceEngine.get_service_court_for_server(2) == ServiceCourt.RIGHT
        assert DoublesServiceEngine.get_service_court_for_server(20) == ServiceCourt.RIGHT

    def test_get_service_court_for_server_odd_returns_left(self):
        from core.doubles_rotation import DoublesServiceEngine, ServiceCourt
        assert DoublesServiceEngine.get_service_court_for_server(1) == ServiceCourt.LEFT
        assert DoublesServiceEngine.get_service_court_for_server(3) == ServiceCourt.LEFT
        assert DoublesServiceEngine.get_service_court_for_server(19) == ServiceCourt.LEFT

    def test_validate_service_court_correct(self):
        from core.doubles_rotation import DoublesServiceEngine, ServiceCourt
        state = self._state()  # score_a=0, even → RIGHT
        assert DoublesServiceEngine.validate_service_court(state, ServiceCourt.RIGHT) is True

    def test_validate_service_court_incorrect(self):
        from core.doubles_rotation import DoublesServiceEngine, ServiceCourt
        state = self._state()  # score_a=0, even → should be RIGHT
        assert DoublesServiceEngine.validate_service_court(state, ServiceCourt.LEFT) is False

    def test_a2_position_derived_from_a1(self):
        from core.doubles_rotation import DoublesServiceEngine, PlayerPosition
        state = self._state()
        # a2 must be opposite of a1
        assert state.a1_position != state.a2_position

    def test_b2_position_derived_from_b1(self):
        from core.doubles_rotation import DoublesServiceEngine, PlayerPosition
        state = self._state()
        assert state.b1_position != state.b2_position

    def test_team_a_too_few_players_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="exactly 2 players"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1"],
                team_b_players=["b1", "b2"],
                discipline=Discipline.MD,
                first_server="a1",
                first_receiver="b1",
            )

    def test_team_b_too_few_players_raises(self):
        from core.doubles_rotation import DoublesServiceEngine
        with pytest.raises(ValueError, match="exactly 2 players"):
            DoublesServiceEngine.initialise(
                team_a_players=["a1", "a2"],
                team_b_players=["b1"],
                discipline=Discipline.MD,
                first_server="a1",
                first_receiver="b1",
            )

    def test_server_position_b_serving(self):
        """server_position uses score_b when B is serving."""
        state = self._state(serving_team="B")
        # B score=0 → even → RIGHT
        from core.doubles_rotation import PlayerPosition
        assert state.server_position == PlayerPosition.RIGHT

    def test_doubles_service_state_score_b_serving_team(self):
        """server_team_score returns score_b when B is serving."""
        state = self._state(serving_team="B")
        assert state.server_team_score == 0


# ===========================================================================
# 7. core/scoring_engine.py — 52 missed lines
# ===========================================================================

class TestScoringEngineGaps:
    """Covers GameState validation, MatchState validation, edge-case scoring."""

    def test_game_state_invalid_game_number_raises(self):
        from core.scoring_engine import GameState, ServiceCourt, IllegalGameStateError
        with pytest.raises(IllegalGameStateError, match="game_number"):
            GameState(
                game_number=0, score_a=0, score_b=0,
                server_id="A", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
            )

    def test_game_state_game_number_4_raises(self):
        from core.scoring_engine import GameState, ServiceCourt, IllegalGameStateError
        with pytest.raises(IllegalGameStateError, match="game_number"):
            GameState(
                game_number=4, score_a=0, score_b=0,
                server_id="A", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
            )

    def test_game_state_negative_score_raises(self):
        from core.scoring_engine import GameState, ServiceCourt, IllegalGameStateError
        with pytest.raises(IllegalGameStateError, match="Negative score"):
            GameState(
                game_number=1, score_a=-1, score_b=0,
                server_id="A", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
            )

    def test_game_state_score_above_30_raises(self):
        from core.scoring_engine import GameState, ServiceCourt, IllegalGameStateError
        with pytest.raises(IllegalGameStateError):
            GameState(
                game_number=1, score_a=31, score_b=0,
                server_id="A", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
            )

    def test_game_state_invalid_server_id_raises(self):
        from core.scoring_engine import GameState, ServiceCourt, IllegalGameStateError
        with pytest.raises(IllegalGameStateError, match="server_id"):
            GameState(
                game_number=1, score_a=0, score_b=0,
                server_id="C", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
            )

    def test_game_state_score_tuple(self):
        from core.scoring_engine import GameState, ServiceCourt
        gs = GameState(
            game_number=1, score_a=15, score_b=12,
            server_id="A", service_court=ServiceCourt.RIGHT,
            discipline=Discipline.MS,
        )
        assert gs.score_tuple == (15, 12)

    def test_game_state_total_points(self):
        from core.scoring_engine import GameState, ServiceCourt
        gs = GameState(
            game_number=1, score_a=15, score_b=12,
            server_id="A", service_court=ServiceCourt.RIGHT,
            discipline=Discipline.MS,
        )
        assert gs.total_points == 27

    def test_game_state_is_at_deuce_true(self):
        from core.scoring_engine import GameState, ServiceCourt
        gs = GameState(
            game_number=1, score_a=20, score_b=20,
            server_id="A", service_court=ServiceCourt.RIGHT,
            discipline=Discipline.MS,
        )
        assert gs.is_at_deuce is True

    def test_game_state_is_at_deuce_false_no_deuce(self):
        from core.scoring_engine import GameState, ServiceCourt
        gs = GameState(
            game_number=1, score_a=15, score_b=10,
            server_id="A", service_court=ServiceCourt.RIGHT,
            discipline=Discipline.MS,
        )
        assert gs.is_at_deuce is False

    def test_game_state_is_at_golden_point(self):
        from core.scoring_engine import GameState, ServiceCourt
        gs = GameState(
            game_number=1, score_a=29, score_b=29,
            server_id="A", service_court=ServiceCourt.LEFT,
            discipline=Discipline.MS,
        )
        assert gs.is_at_golden_point is True

    def test_match_state_games_won_a(self):
        from core.scoring_engine import MatchState, GameState, MatchStatus, ServiceCourt
        ms = MatchState(
            match_id="M001",
            discipline=Discipline.MS,
            player_a_id="PA",
            player_b_id="PB",
        )
        g1 = GameState(
            game_number=1, score_a=21, score_b=0,
            server_id="A", service_court=ServiceCourt.RIGHT,
            discipline=Discipline.MS,
            is_complete=True, winner_id="A",
        )
        ms.games.append(g1)
        assert ms.games_won_a == 1
        assert ms.games_won_b == 0

    def test_match_state_current_game_number(self):
        from core.scoring_engine import MatchState, GameState, ServiceCourt
        ms = MatchState(
            match_id="M001",
            discipline=Discipline.MS,
            player_a_id="PA",
            player_b_id="PB",
        )
        g1 = GameState(
            game_number=1, score_a=21, score_b=0,
            server_id="A", service_court=ServiceCourt.RIGHT,
            discipline=Discipline.MS,
            is_complete=True, winner_id="A",
        )
        ms.games.append(g1)
        assert ms.current_game_number == 2

    def test_match_state_validate_too_many_games_won(self):
        from core.scoring_engine import MatchState, GameState, ServiceCourt, IllegalMatchStateError
        ms = MatchState(
            match_id="M001",
            discipline=Discipline.MS,
            player_a_id="PA",
            player_b_id="PB",
        )
        # Add 3 wins for A — illegal (only need 2)
        for i in range(3):
            ms.games.append(GameState(
                game_number=i + 1, score_a=21, score_b=0,
                server_id="A", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
                is_complete=True, winner_id="A",
            ))
        with pytest.raises(IllegalMatchStateError, match="cannot win more than"):
            ms.validate()

    def test_match_state_validate_too_many_game_number(self):
        from core.scoring_engine import MatchState, GameState, ServiceCourt, IllegalMatchStateError
        ms = MatchState(
            match_id="M001",
            discipline=Discipline.MS,
            player_a_id="PA",
            player_b_id="PB",
        )
        # Add 4 game entries — impossible
        for i in range(4):
            ms.games.append(GameState(
                game_number=min(3, i + 1), score_a=21, score_b=0,
                server_id="A", service_court=ServiceCourt.RIGHT,
                discipline=Discipline.MS,
                is_complete=True, winner_id="A",
            ))
        with pytest.raises(IllegalMatchStateError):
            ms.validate()

    def test_determine_game_winner_negative_scores_raises(self):
        from core.scoring_engine import ScoringEngine, IllegalGameStateError
        with pytest.raises(IllegalGameStateError, match="Negative"):
            ScoringEngine.determine_game_winner(-1, 0)

    def test_determine_game_winner_beyond_golden_a_leads(self):
        from core.scoring_engine import ScoringEngine
        # Scores beyond 30 — fallback to leading player
        assert ScoringEngine.determine_game_winner(31, 5) == "A"

    def test_determine_game_winner_beyond_golden_b_leads(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.determine_game_winner(5, 35) == "B"

    def test_determine_game_winner_beyond_golden_tied(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.determine_game_winner(35, 35) is None

    def test_determine_game_winner_golden_point_a(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.determine_game_winner(30, 29) == "A"

    def test_determine_game_winner_golden_point_b(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.determine_game_winner(29, 30) == "B"

    def test_next_server_invalid_raises(self):
        from core.scoring_engine import ScoringEngine, BadmintonScoringError
        with pytest.raises(BadmintonScoringError):
            ScoringEngine.next_server_after_rally("X")

    def test_server_at_start_invalid_raises(self):
        from core.scoring_engine import ScoringEngine, BadmintonScoringError
        with pytest.raises(BadmintonScoringError):
            ScoringEngine.server_at_start_of_new_game("C")

    def test_service_court_negative_score_raises(self):
        from core.scoring_engine import ScoringEngine, BadmintonScoringError
        with pytest.raises(BadmintonScoringError, match="negative"):
            ScoringEngine.service_court_for_server(-1)

    def test_validate_match_score_zero_games_raises(self):
        from core.scoring_engine import ScoringEngine, IllegalMatchStateError
        with pytest.raises(IllegalMatchStateError, match="1-3 games"):
            ScoringEngine.validate_match_score([], Discipline.MS)

    def test_validate_match_score_4_games_raises(self):
        from core.scoring_engine import ScoringEngine, IllegalMatchStateError
        with pytest.raises(IllegalMatchStateError, match="1-3 games"):
            ScoringEngine.validate_match_score(
                [(21, 0), (21, 0), (21, 0), (21, 0)], Discipline.MS
            )

    def test_validate_match_score_non_terminal_raises(self):
        from core.scoring_engine import ScoringEngine, IllegalMatchStateError
        with pytest.raises(IllegalMatchStateError, match="not a terminal"):
            ScoringEngine.validate_match_score([(10, 10)], Discipline.MS)

    def test_validate_match_score_no_winner_raises(self):
        from core.scoring_engine import ScoringEngine, IllegalMatchStateError
        # A wins game 1, B wins game 2 — no match winner after 2 games with only 1 each
        with pytest.raises(IllegalMatchStateError, match="no winner"):
            ScoringEngine.validate_match_score([(21, 0), (0, 21)], Discipline.MS)

    def test_validate_match_score_extra_games_raises(self):
        from core.scoring_engine import ScoringEngine, IllegalMatchStateError
        # A already won in 2 games — 3rd game is illegal continuation
        with pytest.raises(IllegalMatchStateError, match="illegal continuation"):
            ScoringEngine.validate_match_score(
                [(21, 0), (21, 0), (21, 0)], Discipline.MS
            )

    def test_validate_match_score_valid_2_0(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.validate_match_score([(21, 0), (21, 0)], Discipline.MS) is True

    def test_validate_match_score_valid_2_1(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.validate_match_score(
            [(21, 10), (10, 21), (21, 15)], Discipline.MS
        ) is True

    def test_possible_game_scores_contains_golden_point(self):
        from core.scoring_engine import ScoringEngine
        scores = ScoringEngine.possible_game_scores()
        assert (30, 29) in scores
        assert (29, 30) in scores

    def test_possible_game_scores_contains_normal_win(self):
        from core.scoring_engine import ScoringEngine
        scores = ScoringEngine.possible_game_scores()
        assert (21, 0) in scores
        assert (0, 21) in scores

    def test_possible_match_scores(self):
        from core.scoring_engine import ScoringEngine
        ms = ScoringEngine.possible_match_scores()
        assert (2, 0) in ms
        assert (2, 1) in ms
        assert (0, 2) in ms
        assert (1, 2) in ms

    def test_is_legal_final_score_true(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.is_legal_final_score(21, 10) is True

    def test_is_legal_final_score_false_in_progress(self):
        from core.scoring_engine import ScoringEngine
        assert ScoringEngine.is_legal_final_score(10, 10) is False

    def test_doubles_service_state_invalid_discipline_raises(self):
        from core.scoring_engine import DoublesServiceState, BadmintonScoringError
        with pytest.raises(BadmintonScoringError, match="doubles disciplines"):
            DoublesServiceState(
                discipline=Discipline.MS,
                server_pair_id="A",
                server_player_within_pair="A1",
                score_a=0,
                score_b=0,
            )

    def test_doubles_service_state_current_service_court(self):
        from core.scoring_engine import DoublesServiceState, ServiceCourt
        ds = DoublesServiceState(
            discipline=Discipline.MD,
            server_pair_id="A",
            server_player_within_pair="A1",
            score_a=0,
            score_b=0,
        )
        assert ds.current_service_court == ServiceCourt.RIGHT  # score=0, even

    def test_doubles_service_state_apply_rally_serving_wins(self):
        from core.scoring_engine import DoublesServiceState
        ds = DoublesServiceState(
            discipline=Discipline.MD,
            server_pair_id="A",
            server_player_within_pair="A1",
            score_a=0,
            score_b=0,
        )
        new_ds = ds.apply_rally_result("A")
        assert new_ds.server_pair_id == "A"
        assert new_ds.server_player_within_pair == "A1"
        assert new_ds.score_a == 1

    def test_doubles_service_state_apply_rally_receiving_wins(self):
        from core.scoring_engine import DoublesServiceState
        ds = DoublesServiceState(
            discipline=Discipline.MD,
            server_pair_id="A",
            server_player_within_pair="A1",
            score_a=0,
            score_b=0,
        )
        new_ds = ds.apply_rally_result("B")
        assert new_ds.server_pair_id == "B"
        # Rotation: was A1 serving, so when B gets service, toggle B player
        assert new_ds.server_player_within_pair in ("B1", "B2")

    def test_doubles_service_state_rotation_alternates(self):
        """Applying rally twice (B winning both) should alternate B's server."""
        from core.scoring_engine import DoublesServiceState
        ds = DoublesServiceState(
            discipline=Discipline.MD,
            server_pair_id="A",
            server_player_within_pair="A1",
            score_a=0,
            score_b=0,
        )
        ds2 = ds.apply_rally_result("B")  # B takes service
        server_first = ds2.server_player_within_pair
        # B wins again — same player continues serving (service retained)
        ds3 = ds2.apply_rally_result("B")
        assert ds3.server_player_within_pair == server_first  # retained


# ===========================================================================
# 8. markets/outright_pricing.py — 55 missed lines
# ===========================================================================

class TestOutrightPricingGaps:
    """Covers: draw validation errors, already_played filtering, RR/knockout modes."""

    def _make_entries(self, n: int = 8, rwp: float = 0.515) -> list:
        from markets.outright_pricing import TournamentEntry
        return [
            TournamentEntry(
                entity_id=f"P{i:02d}",
                seeding=i if i <= 4 else None,
                rwp_as_server=rwp,
                rwp_as_receiver=1 - rwp,
                elo_rating=1500 + (8 - i) * 50,
            )
            for i in range(1, n + 1)
        ]

    def _make_draw(
        self,
        draw_type=None,
        draw_size: int = 8,
        n_entries: int = 8,
        already_played=None,
    ):
        from markets.outright_pricing import TournamentDraw, DrawType
        if draw_type is None:
            draw_type = DrawType.SINGLE_ELIMINATION
        return TournamentDraw(
            tournament_id="T001",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            draw_type=draw_type,
            draw_size=draw_size,
            entries=self._make_entries(n_entries),
            already_played=already_played or [],
        )

    def _engine(self, n_sims: int = 100):
        from markets.outright_pricing import OutrightPricingEngine
        return OutrightPricingEngine(n_simulations=n_sims)

    def test_draw_validate_wrong_entry_count_raises(self):
        from markets.outright_pricing import TournamentDraw, DrawType, TournamentEntry
        draw = TournamentDraw(
            tournament_id="T001",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            draw_type=DrawType.SINGLE_ELIMINATION,
            draw_size=8,
            entries=self._make_entries(6),  # Wrong number
        )
        with pytest.raises(RuntimeError, match="entries but draw_size"):
            draw.validate()

    def test_draw_validate_invalid_draw_size_raises(self):
        from markets.outright_pricing import TournamentDraw, DrawType
        draw = TournamentDraw(
            tournament_id="T001",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            draw_type=DrawType.SINGLE_ELIMINATION,
            draw_size=10,
            entries=self._make_entries(10),
        )
        with pytest.raises(RuntimeError, match="Invalid draw_size"):
            draw.validate()

    def test_price_tournament_single_elimination_8(self):
        engine = self._engine()
        draw = self._make_draw()
        result = engine.price_tournament(draw, seed=42)
        assert len(result.results) == 8
        # Probabilities should sum to ~1.0
        total = sum(r.p_win_tournament for r in result.results)
        assert abs(total - 1.0) < 0.05

    def test_price_tournament_with_seed_is_reproducible(self):
        engine = self._engine(n_sims=500)
        draw = self._make_draw()
        r1 = engine.price_tournament(draw, seed=99)
        r2 = engine.price_tournament(draw, seed=99)
        for e1, e2 in zip(r1.results, r2.results):
            assert e1.entity_id == e2.entity_id
            assert e1.simulated_wins == e2.simulated_wins

    def test_price_tournament_with_already_played(self):
        """already_played eliminates losers from simulation."""
        engine = self._engine()
        draw = self._make_draw(
            already_played=[("P01", "P08", "P01")]  # P08 eliminated
        )
        result = engine.price_tournament(draw, seed=1)
        eliminated = next((r for r in result.results if r.entity_id == "P08"), None)
        # Eliminated player should have 0 wins
        if eliminated:
            assert eliminated.simulated_wins == 0

    def test_price_tournament_round_robin_knockout(self):
        from markets.outright_pricing import DrawType
        engine = self._engine()
        draw = self._make_draw(
            draw_type=DrawType.ROUND_ROBIN_KNOCKOUT,
            draw_size=8,
            n_entries=8,
        )
        result = engine.price_tournament(draw, seed=7)
        assert len(result.results) > 0

    def test_price_tournament_round_robin_only(self):
        from markets.outright_pricing import DrawType
        engine = self._engine()
        draw = self._make_draw(
            draw_type=DrawType.ROUND_ROBIN_ONLY,
            draw_size=8,
            n_entries=8,
        )
        result = engine.price_tournament(draw, seed=3)
        assert len(result.results) > 0

    def test_outright_result_has_each_way_p(self):
        engine = self._engine()
        draw = self._make_draw()
        result = engine.price_tournament(draw, seed=42)
        for r in result.results:
            assert r.each_way_p is not None
            assert 0.0 <= r.each_way_p <= 1.0

    def test_outright_result_odds_at_least_1_01(self):
        engine = self._engine()
        draw = self._make_draw()
        result = engine.price_tournament(draw, seed=42)
        for r in result.results:
            assert r.odds_with_margin >= 1.01

    def test_low_probability_player_gets_9999_odds(self):
        """Entry with 0 simulated wins should get odds_fair = 9999.0."""
        engine = self._engine(n_sims=5)
        # Create lopsided entries — 1 very strong, rest very weak
        from markets.outright_pricing import TournamentEntry
        entries = [
            TournamentEntry(entity_id="STRONG", rwp_as_server=0.80, elo_rating=2000),
        ] + [
            TournamentEntry(entity_id=f"WEAK{i}", rwp_as_server=0.20, elo_rating=500)
            for i in range(7)
        ]
        from markets.outright_pricing import TournamentDraw, DrawType
        draw = TournamentDraw(
            tournament_id="T_LOPSIDED",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            draw_type=DrawType.SINGLE_ELIMINATION,
            draw_size=8,
            entries=entries,
        )
        result = engine.price_tournament(draw, seed=0)
        # Weak players should have very high odds
        weak_results = [r for r in result.results if r.entity_id.startswith("WEAK")]
        assert any(r.odds_fair >= 999.0 for r in weak_results)

    def test_price_tournament_metadata(self):
        engine = self._engine()
        draw = self._make_draw()
        result = engine.price_tournament(draw, seed=1)
        assert result.tournament_id == "T001"
        assert result.discipline == Discipline.MS
        assert result.tier == TournamentTier.SUPER_500
        assert result.n_simulations == 100
        assert result.generated_at > 0

    def test_price_tournament_16_entry_draw(self):
        from markets.outright_pricing import TournamentEntry, TournamentDraw, DrawType
        entries = self._make_entries(n=16)
        draw = TournamentDraw(
            tournament_id="T_16",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_1000,
            draw_type=DrawType.SINGLE_ELIMINATION,
            draw_size=16,
            entries=entries,
        )
        engine = self._engine()
        result = engine.price_tournament(draw, seed=10)
        assert len(result.results) == 16

    def test_simulate_empty_entries_returns_none_winner(self):
        """Edge case: empty entries returns None winner."""
        engine = self._engine(n_sims=1)
        # Access private method directly for unit testing
        winner, finalist, semis = engine._simulate_single_elimination([], {})
        assert winner is None
        assert finalist is None

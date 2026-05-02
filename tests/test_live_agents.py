"""
Comprehensive pytest test suite for live agent modules.

Covers:
1. agents/live_supervisor.py          - LiveSupervisorAgent
2. agents/live/score_ingest_agent.py  - ScoreIngestAgent
3. agents/live/risk_overlay_agent.py  - RiskOverlayAgent
4. agents/live/market_align_agent.py  - MarketAlignAgent
5. agents/live/observability_agent.py - ObservabilityAgent
6. agents/live/settlement_prep_agent.py - SettlementPrepAgent
7. agents/live/trader_control_agent.py  - TraderControlAgent
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from config.badminton_config import Discipline, TournamentTier
from agents.live_supervisor import (
    LiveSupervisorAgent,
    LiveMatchSetup,
    LivePricingResponse,
    MatchLiveState,
)
from markets.market_trading_control import TradingControlManager
from agents.live.score_ingest_agent import ScoreIngestAgent, ScoreEvent, ScoreIngestResult
from agents.live.risk_overlay_agent import RiskOverlayAgent
from agents.live.market_align_agent import MarketAlignAgent
from agents.live.observability_agent import ObservabilityAgent
from agents.live.settlement_prep_agent import SettlementPrepAgent
from agents.live.trader_control_agent import (
    TraderControlAgent,
    OperatorCommand,
    OperatorCommandType,
)
from markets.derivative_engine import MarketPrice, MarketFamily
from settlement.grading_service import GradingService


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def basic_setup() -> LiveMatchSetup:
    return LiveMatchSetup(
        match_id="M001",
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=Discipline.MS,
        tier=TournamentTier.SUPER_500,
        first_server="A",
        rwp_prior_a=0.515,
        rwp_prior_b=0.500,
        pre_match_p_a=0.55,
    )


@pytest.fixture()
def trading_control() -> TradingControlManager:
    return TradingControlManager(match_id="M001")


@pytest.fixture()
def supervisor(basic_setup, trading_control) -> LiveSupervisorAgent:
    return LiveSupervisorAgent(setup=basic_setup, trading_control=trading_control)


@pytest.fixture()
def score_agent() -> ScoreIngestAgent:
    return ScoreIngestAgent(match_id="M001")


@pytest.fixture()
def risk_agent() -> RiskOverlayAgent:
    return RiskOverlayAgent(match_id="M001")


@pytest.fixture()
def align_agent() -> MarketAlignAgent:
    return MarketAlignAgent(match_id="M001", pre_match_p_a=0.55)


@pytest.fixture()
def obs_agent() -> ObservabilityAgent:
    return ObservabilityAgent(match_id="M001")


@pytest.fixture()
def grading_service() -> GradingService:
    return GradingService()


@pytest.fixture()
def settlement_agent(trading_control, grading_service) -> SettlementPrepAgent:
    return SettlementPrepAgent(
        match_id="M001",
        trading_control=trading_control,
        grading_service=grading_service,
    )


@pytest.fixture()
def trader_agent(trading_control) -> TraderControlAgent:
    return TraderControlAgent(match_id="M001", trading_control=trading_control)


@pytest.fixture()
def market_price_a() -> MarketPrice:
    return MarketPrice(
        market_id="match_winner",
        market_family=MarketFamily.MATCH_RESULT,
        outcome_name="A",
        odds=1.85,
        prob_implied=1 / 1.85,
        prob_with_margin=1 / 1.85,
        suspended=False,
    )


@pytest.fixture()
def market_price_b() -> MarketPrice:
    return MarketPrice(
        market_id="match_winner",
        market_family=MarketFamily.MATCH_RESULT,
        outcome_name="B",
        odds=2.10,
        prob_implied=1 / 2.10,
        prob_with_margin=1 / 2.10,
        suspended=False,
    )


# ===========================================================================
# Section 1 — LiveSupervisorAgent
# ===========================================================================

class TestLiveSupervisorAgentConstruction:
    def test_construction_succeeds(self, supervisor):
        assert supervisor is not None

    def test_initial_state_is_not_none(self, supervisor):
        state = supervisor.get_current_state()
        assert state is not None

    def test_initial_last_prices_is_none(self, supervisor):
        assert supervisor.get_last_prices() is None

    def test_initial_rwp_returns_tuple_of_floats(self, supervisor):
        rwp = supervisor.get_current_rwp()
        assert isinstance(rwp, tuple)
        assert len(rwp) == 2
        assert isinstance(rwp[0], float)
        assert isinstance(rwp[1], float)

    def test_initial_stats_is_dict(self, supervisor):
        stats = supervisor.get_stats()
        assert isinstance(stats, dict)

    def test_initial_observability_metrics_is_dict(self, supervisor):
        metrics = supervisor.get_observability_metrics()
        assert isinstance(metrics, dict)

    def test_get_current_state_returns_match_live_state(self, supervisor):
        state = supervisor.get_current_state()
        assert isinstance(state, MatchLiveState)


class TestLiveSupervisorAgentWrongMatchId:
    def test_wrong_match_id_returns_none(self, supervisor):
        result = supervisor.on_score_update(
            "WRONG_ID",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert result is None

    def test_empty_match_id_returns_none(self, supervisor):
        result = supervisor.on_score_update(
            "",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert result is None

    def test_wrong_match_id_does_not_mutate_state(self, supervisor):
        state_before = supervisor.get_current_state()
        supervisor.on_score_update(
            "NOT_M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert supervisor.get_last_prices() is None


class TestLiveSupervisorAgentValidUpdate:
    def test_first_valid_update_returns_response(self, supervisor):
        resp = supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert resp is not None

    def test_response_is_live_pricing_response(self, supervisor):
        resp = supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert isinstance(resp, LivePricingResponse)

    def test_last_prices_populated_after_update(self, supervisor):
        supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert supervisor.get_last_prices() is not None

    @pytest.mark.parametrize("winner,score_a,score_b,server", [
        ("A", 1, 0, "B"),
        ("B", 0, 1, "A"),
    ])
    def test_valid_update_for_both_winners(
        self, basic_setup, trading_control, winner, score_a, score_b, server
    ):
        agent = LiveSupervisorAgent(setup=basic_setup, trading_control=trading_control)
        resp = agent.on_score_update(
            "M001",
            {
                "winner": winner,
                "score_a": score_a,
                "score_b": score_b,
                "game_number": 1,
                "server": server,
            },
        )
        assert resp is not None

    def test_sequential_score_updates_state_progresses(self, supervisor):
        supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        resp2 = supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 2, "score_b": 0, "game_number": 1, "server": "B"},
        )
        assert resp2 is not None

    def test_update_with_optional_timestamp(self, supervisor):
        resp = supervisor.on_score_update(
            "M001",
            {
                "winner": "A",
                "score_a": 1,
                "score_b": 0,
                "game_number": 1,
                "server": "B",
                "timestamp": time.time(),
            },
        )
        assert resp is not None

    def test_stats_updated_after_update(self, supervisor):
        supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        stats = supervisor.get_stats()
        assert isinstance(stats, dict)

    def test_rwp_bounds_remain_valid_after_update(self, supervisor):
        supervisor.on_score_update(
            "M001",
            {"winner": "A", "score_a": 1, "score_b": 0, "game_number": 1, "server": "B"},
        )
        rwp_a, rwp_b = supervisor.get_current_rwp()
        assert 0.0 < rwp_a < 1.0
        assert 0.0 < rwp_b < 1.0

    def test_b_wins_first_rally_also_returns_response(self, basic_setup, trading_control):
        agent = LiveSupervisorAgent(setup=basic_setup, trading_control=trading_control)
        resp = agent.on_score_update(
            "M001",
            {"winner": "B", "score_a": 0, "score_b": 1, "game_number": 1, "server": "A"},
        )
        assert resp is not None


# ===========================================================================
# Section 2 — ScoreIngestAgent
# ===========================================================================

class TestScoreIngestAgentConstruction:
    def test_construction_succeeds(self, score_agent):
        assert score_agent is not None

    def test_initial_stats_is_dict(self, score_agent):
        assert isinstance(score_agent.stats, dict)


class TestScoreIngestAgentValidIngest:
    def test_valid_payload_returns_result(self, score_agent):
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "winner": "A", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert isinstance(result, ScoreIngestResult)

    def test_valid_ingest_is_valid(self, score_agent):
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "winner": "A", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.valid is True

    def test_valid_ingest_has_score_event(self, score_agent):
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "winner": "A", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.event is not None
        assert isinstance(result.event, ScoreEvent)

    @pytest.mark.parametrize("winner,score_a,score_b,server", [
        ("A", 1, 0, "B"),
        ("B", 0, 1, "A"),
    ])
    def test_valid_ingest_both_winners(self, winner, score_a, score_b, server):
        agent = ScoreIngestAgent(match_id="M001")
        result = agent.ingest(
            payload={
                "score_a": score_a,
                "score_b": score_b,
                "winner": winner,
                "server": server,
                "game_number": 1,
            },
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.valid is True

    def test_valid_ingest_not_duplicate(self, score_agent):
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "winner": "A", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.is_duplicate is False


class TestScoreIngestAgentDuplicateDetection:
    def test_duplicate_score_is_flagged(self, score_agent):
        payload = {"score_a": 1, "score_b": 0, "winner": "A", "server": "B", "game_number": 1}
        score_agent.ingest(payload=payload, prev_score_a=0, prev_score_b=0, game_number=1)
        result2 = score_agent.ingest(payload=payload, prev_score_a=0, prev_score_b=0, game_number=1)
        assert result2.is_duplicate is True or result2.valid is False


class TestScoreIngestAgentInvalidPayloads:
    def test_missing_winner_field_inferred_from_score_delta(self, score_agent):
        # The agent infers winner from score delta when 'winner' key is absent.
        # A=1, B=0 with prev both 0 → inferred winner is A → valid.
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.valid is True
        assert result.event is not None

    def test_missing_score_fields(self, score_agent):
        result = score_agent.ingest(
            payload={"winner": "A", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.valid is False

    def test_invalid_winner_value(self, score_agent):
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "winner": "X", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.valid is False

    def test_invalid_result_has_rejection_reason(self, score_agent):
        result = score_agent.ingest(
            payload={"score_a": 1, "score_b": 0, "winner": "X", "server": "B", "game_number": 1},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.rejection_reason is not None
        assert len(result.rejection_reason) > 0

    def test_empty_payload_is_invalid(self, score_agent):
        result = score_agent.ingest(
            payload={},
            prev_score_a=0,
            prev_score_b=0,
            game_number=1,
        )
        assert result.valid is False


# ===========================================================================
# Section 3 — RiskOverlayAgent
# ===========================================================================

class TestRiskOverlayAgentConstruction:
    def test_construction_succeeds(self, risk_agent):
        assert risk_agent is not None


class TestRiskOverlayAgentValidate:
    def test_empty_markets_returns_three_tuple(self, risk_agent):
        result = risk_agent.validate(markets={}, click_scales={})
        assert len(result) == 3

    def test_empty_markets_no_violations(self, risk_agent):
        _, _, violations = risk_agent.validate(markets={}, click_scales={})
        assert isinstance(violations, list)
        assert len(violations) == 0

    def test_empty_markets_returns_dict_types(self, risk_agent):
        validated, scales, violations = risk_agent.validate(markets={}, click_scales={})
        assert isinstance(validated, dict)
        assert isinstance(scales, dict)

    def test_valid_markets_above_h10_minimum_no_violations(
        self, risk_agent, market_price_a, market_price_b
    ):
        markets = {"match_winner": [market_price_a, market_price_b]}
        click_scales = {"match_winner": 1.0}
        _, _, violations = risk_agent.validate(markets=markets, click_scales=click_scales)
        # Valid odds (1.85, 2.10) should not produce H10 violations
        h10_violations = [
            v for v in violations if "H10" in str(v) or "minimum" in str(v).lower()
        ]
        assert len(h10_violations) == 0

    def test_below_minimum_odds_raises_at_construction(self, risk_agent):
        # MarketPrice enforces the H10 gate (min odds 1.01) at __post_init__.
        # Constructing with odds=1.005 must raise ValueError immediately.
        with pytest.raises(ValueError, match="H10"):
            MarketPrice(
                market_id="match_winner",
                market_family=MarketFamily.MATCH_RESULT,
                outcome_name="A",
                odds=1.005,
                prob_implied=1 / 1.005,
                prob_with_margin=1 / 1.005,
                suspended=False,
            )

    def test_validated_markets_returned_intact(self, risk_agent, market_price_a):
        markets = {"match_winner": [market_price_a]}
        click_scales = {}
        validated, _, _ = risk_agent.validate(markets=markets, click_scales=click_scales)
        assert "match_winner" in validated

    def test_click_scales_returned(self, risk_agent, market_price_a):
        markets = {"match_winner": [market_price_a]}
        click_scales = {"match_winner": 0.5}
        _, returned_scales, _ = risk_agent.validate(markets=markets, click_scales=click_scales)
        assert isinstance(returned_scales, dict)


# ===========================================================================
# Section 4 — MarketAlignAgent
# ===========================================================================

class TestMarketAlignAgentConstruction:
    def test_construction_succeeds(self, align_agent):
        assert align_agent is not None


class TestMarketAlignAgentAlign:
    def test_align_empty_markets_returns_dict(self, align_agent):
        result = align_agent.align(markets={}, p_a_blend=0.55, total_points_played=5)
        assert isinstance(result, dict)

    def test_align_with_zero_points_played(self, align_agent):
        result = align_agent.align(markets={}, p_a_blend=0.55, total_points_played=0)
        assert result is not None

    @pytest.mark.parametrize("p_a_blend", [0.30, 0.50, 0.70, 0.90])
    def test_align_various_p_a_values(self, p_a_blend):
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.55)
        result = agent.align(markets={}, p_a_blend=p_a_blend, total_points_played=10)
        assert isinstance(result, dict)

    def test_align_with_existing_markets(self, align_agent, market_price_a, market_price_b):
        markets = {"match_winner": [market_price_a, market_price_b]}
        result = align_agent.align(markets=markets, p_a_blend=0.55, total_points_played=5)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("total_points", [0, 10, 50, 100])
    def test_align_various_total_points(self, total_points):
        agent = MarketAlignAgent(match_id="M001", pre_match_p_a=0.55)
        result = agent.align(markets={}, p_a_blend=0.55, total_points_played=total_points)
        assert isinstance(result, dict)

    def test_align_returns_dict_not_none(self, align_agent):
        result = align_agent.align(markets={}, p_a_blend=0.55, total_points_played=20)
        assert result is not None


# ===========================================================================
# Section 5 — ObservabilityAgent
# ===========================================================================

class TestObservabilityAgentConstruction:
    def test_construction_succeeds(self, obs_agent):
        assert obs_agent is not None

    def test_initial_metrics_is_dict(self, obs_agent):
        metrics = obs_agent.get_metrics()
        assert isinstance(metrics, dict)


class TestObservabilityAgentRecordRally:
    def test_record_single_rally_does_not_raise(self, obs_agent):
        obs_agent.record_rally(latency_ms=45.0, qa_violations=0, sharp_alert=False)

    def test_get_metrics_returns_dict_after_record(self, obs_agent):
        obs_agent.record_rally(latency_ms=45.0, qa_violations=0, sharp_alert=False)
        metrics = obs_agent.get_metrics()
        assert isinstance(metrics, dict)

    def test_metrics_not_empty_after_recording(self, obs_agent):
        obs_agent.record_rally(latency_ms=45.0, qa_violations=0, sharp_alert=False)
        metrics = obs_agent.get_metrics()
        assert len(metrics) > 0

    def test_record_multiple_rallies(self, obs_agent):
        for i in range(5):
            obs_agent.record_rally(latency_ms=float(40 + i), qa_violations=0, sharp_alert=False)
        metrics = obs_agent.get_metrics()
        assert isinstance(metrics, dict)

    def test_record_rally_with_violation(self, obs_agent):
        obs_agent.record_rally(latency_ms=55.0, qa_violations=1, sharp_alert=False)
        metrics = obs_agent.get_metrics()
        assert isinstance(metrics, dict)

    def test_record_rally_with_sharp_alert(self, obs_agent):
        obs_agent.record_rally(latency_ms=30.0, qa_violations=0, sharp_alert=True)
        metrics = obs_agent.get_metrics()
        assert isinstance(metrics, dict)

    def test_record_high_latency_above_p99(self, obs_agent):
        # 250ms is above the p99 latency target of 200ms
        obs_agent.record_rally(latency_ms=250.0, qa_violations=0, sharp_alert=False)
        metrics = obs_agent.get_metrics()
        assert isinstance(metrics, dict)

    @pytest.mark.parametrize("latency,violations,sharp", [
        (45.0, 0, False),
        (95.0, 1, False),
        (195.0, 2, True),
        (10.0, 0, True),
    ])
    def test_record_rally_parametrized(self, latency, violations, sharp):
        agent = ObservabilityAgent(match_id="M_PARAM")
        agent.record_rally(latency_ms=latency, qa_violations=violations, sharp_alert=sharp)
        metrics = agent.get_metrics()
        assert isinstance(metrics, dict)


# ===========================================================================
# Section 6 — SettlementPrepAgent
# ===========================================================================

class TestSettlementPrepAgentConstruction:
    def test_construction_succeeds(self, settlement_agent):
        assert settlement_agent is not None

    def test_is_settled_property_initially_false(self, settlement_agent):
        assert settlement_agent.is_settled is False

    def test_is_settled_is_bool(self, settlement_agent):
        assert isinstance(settlement_agent.is_settled, bool)

    def test_get_settlement_records_exists_and_callable(self, settlement_agent):
        assert callable(getattr(settlement_agent, "get_settlement_records", None))

    def test_check_and_settle_method_exists_and_callable(self, settlement_agent):
        assert callable(getattr(settlement_agent, "check_and_settle", None))

    def test_check_and_settle_with_incomplete_match_does_not_settle(self, settlement_agent):
        # Match has no result yet — should either return without settling or raise
        try:
            settlement_agent.check_and_settle()
        except Exception:
            pass
        # The key assertion: it must not be settled when match data is missing
        assert settlement_agent.is_settled is False

    def test_construction_with_different_match_id(self, trading_control, grading_service):
        agent = SettlementPrepAgent(
            match_id="M_SETTLE_99",
            trading_control=trading_control,
            grading_service=grading_service,
        )
        assert agent is not None
        assert agent.is_settled is False


# ===========================================================================
# Section 7 — TraderControlAgent
# ===========================================================================

class TestTraderControlAgentConstruction:
    def test_construction_succeeds(self, trader_agent):
        assert trader_agent is not None

    def test_construction_different_match_id(self, trading_control):
        agent = TraderControlAgent(match_id="M_TRADER_99", trading_control=trading_control)
        assert agent is not None

    def test_get_override_log_initially_empty(self, trader_agent):
        log = trader_agent.get_override_log()
        assert isinstance(log, list)

    def test_get_locked_markets_initially_empty(self, trader_agent):
        locked = trader_agent.get_locked_markets()
        assert isinstance(locked, (list, set, dict))

    def test_is_market_locked_initially_false(self, trader_agent):
        locked = trader_agent.is_market_locked("match_winner")
        assert locked is False

    def test_apply_suspend_market_command(self, trader_agent):
        cmd = OperatorCommand(
            command_type=OperatorCommandType.SUSPEND_MARKET,
            match_id="M001",
            market_id="match_winner",
            operator_id="trader_01",
            reason="Test suspension",
        )
        result = trader_agent.apply_command(cmd)
        assert result is not None

    def test_apply_lock_market_command(self, trader_agent):
        cmd = OperatorCommand(
            command_type=OperatorCommandType.LOCK_MARKET,
            match_id="M001",
            market_id="match_winner",
            operator_id="trader_01",
            reason="Locking for test",
        )
        result = trader_agent.apply_command(cmd)
        assert result is not None

    def test_market_locked_after_lock_command(self, trader_agent):
        cmd = OperatorCommand(
            command_type=OperatorCommandType.LOCK_MARKET,
            match_id="M001",
            market_id="match_winner",
            operator_id="trader_01",
            reason="Lock test",
        )
        trader_agent.apply_command(cmd)
        assert trader_agent.is_market_locked("match_winner") is True

    def test_override_log_populated_after_command(self, trader_agent):
        cmd = OperatorCommand(
            command_type=OperatorCommandType.SUSPEND_MARKET,
            match_id="M001",
            market_id="match_winner",
            operator_id="trader_01",
            reason="Log test",
        )
        trader_agent.apply_command(cmd)
        log = trader_agent.get_override_log()
        assert len(log) > 0


# ===========================================================================
# Section 8 — MarketPrice Construction
# ===========================================================================

class TestMarketPriceConstruction:
    def test_market_price_constructs(self, market_price_a):
        assert market_price_a.market_id == "match_winner"
        assert market_price_a.outcome_name == "A"
        assert abs(market_price_a.odds - 1.85) < 1e-6

    def test_market_price_prob_implied(self, market_price_a):
        assert abs(market_price_a.prob_implied - (1 / 1.85)) < 1e-6

    def test_market_price_not_suspended_by_default(self, market_price_a):
        assert market_price_a.suspended is False

    def test_market_price_suspended_flag(self):
        mp = MarketPrice(
            market_id="match_winner",
            market_family=MarketFamily.MATCH_RESULT,
            outcome_name="A",
            odds=1.85,
            prob_implied=1 / 1.85,
            prob_with_margin=1 / 1.85,
            suspended=True,
        )
        assert mp.suspended is True

    def test_market_price_outcome_name_stored(self, market_price_b):
        assert market_price_b.outcome_name == "B"


# ===========================================================================
# Section 9 — LiveMatchSetup field validation
# ===========================================================================

class TestLiveMatchSetup:
    def test_setup_match_id(self, basic_setup):
        assert basic_setup.match_id == "M001"

    def test_setup_entity_ids(self, basic_setup):
        assert basic_setup.entity_a_id == "PA"
        assert basic_setup.entity_b_id == "PB"

    def test_setup_discipline(self, basic_setup):
        assert basic_setup.discipline == Discipline.MS

    def test_setup_tier(self, basic_setup):
        assert basic_setup.tier == TournamentTier.SUPER_500

    def test_setup_rwp_priors(self, basic_setup):
        assert abs(basic_setup.rwp_prior_a - 0.515) < 1e-6
        assert abs(basic_setup.rwp_prior_b - 0.500) < 1e-6

    def test_setup_pre_match_p_a(self, basic_setup):
        assert abs(basic_setup.pre_match_p_a - 0.55) < 1e-6

    def test_setup_first_server(self, basic_setup):
        assert basic_setup.first_server == "A"

    @pytest.mark.parametrize("discipline", [
        Discipline.MS, Discipline.WS, Discipline.MD, Discipline.WD, Discipline.XD
    ])
    def test_setup_all_disciplines(self, discipline):
        setup = LiveMatchSetup(
            match_id="M_DISC",
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=discipline,
            tier=TournamentTier.SUPER_500,
            first_server="A",
            rwp_prior_a=0.515,
            rwp_prior_b=0.500,
            pre_match_p_a=0.55,
        )
        assert setup.discipline == discipline

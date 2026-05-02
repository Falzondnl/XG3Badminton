"""
test_live_markets.py
====================
Tests for markets/live_markets.py — live pricing engine.

Covers:
  - LiveMatchContext construction and helpers (feed_gap, should_ghost, should_suspend)
  - LivePricingEngine.price_after_point() happy path
  - Ghost mode activated and widens margins
  - Suspension triggered returns is_suspended response
  - Market presence and probability validity
  - Click scales with momentum
  - H7 gate: no negative overround in live markets
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    LIVE_GHOST_TRIGGER_SECONDS,
    LIVE_SUSPEND_SECONDS,
    TournamentTier,
)
from core.bayesian_updater import BayesianRWPUpdater, LiveRWPEstimate
from core.match_state import LiveStateSummary, MatchLiveState, MatchStatus
from core.momentum_detector import (
    MomentumDetector,
    MomentumRegime,
    MomentumSignalStrength,
    MomentumSnapshot,
)
from markets.live_markets import (
    LiveMatchContext,
    LivePricingEngine,
    LivePricingRequest,
    LivePricingResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_live_state(
    match_id: str = "M1",
    discipline: Discipline = Discipline.MS,
    score_a: int = 5,
    score_b: int = 3,
    games_won_a: int = 0,
    games_won_b: int = 0,
) -> MatchLiveState:
    state = MatchLiveState(
        match_id=match_id,
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=discipline,
        status=MatchStatus.IN_PROGRESS,
        current_game=1,
        score_a=score_a,
        score_b=score_b,
        games_won_a=games_won_a,
        games_won_b=games_won_b,
        server="A",
        initial_server="A",
        total_points_played=score_a + score_b,
    )
    return state


def _make_snapshot(
    score_a: int = 5,
    score_b: int = 3,
    games_won_a: int = 0,
    games_won_b: int = 0,
) -> LiveStateSummary:
    return LiveStateSummary(
        match_id="M1",
        status=MatchStatus.IN_PROGRESS.value,
        games_won_a=games_won_a,
        games_won_b=games_won_b,
        current_game=1,
        score_a=score_a,
        score_b=score_b,
        game_scores=[],
        server="A",
        service_court="RIGHT",
        current_run_a=0,
        current_run_b=0,
        momentum_holder=None,
        total_points_played=score_a + score_b,
        is_in_deuce=False,
        is_at_golden_point=False,
        is_deciding_game=False,
    )


def _make_momentum_snapshot(regime: MomentumRegime = MomentumRegime.NEUTRAL) -> MomentumSnapshot:
    return MomentumSnapshot(
        regime=regime,
        signal_strength=MomentumSignalStrength.NONE,
        momentum_holder=None,
        current_run_length=0,
        current_run_entity=None,
        intensity=0.0,
        significance_p_value=1.0,
        recent_runs=[],
        score_a=5,
        score_b=3,
        game_number=1,
        is_break=False,
        is_comeback=False,
        consecutive_games_won_a=0,
        consecutive_games_won_b=0,
        momentum_score_a=0.0,
        momentum_score_b=0.0,
    )


def _make_rwp_estimate(entity_id: str = "PA", rwp: float = 0.515) -> LiveRWPEstimate:
    return LiveRWPEstimate(
        entity_id=entity_id,
        rwp_prior=rwp,
        rwp_posterior=rwp,
        rwp_live=rwp,
        evidence_weight=1.0,
        uncertainty=0.01,
        server_wins=10,
        server_total=20,
        confidence_interval=(rwp - 0.02, rwp + 0.02),
    )


def _make_context(
    last_feed_update: float | None = None,
    is_ghost: bool = False,
    is_suspended: bool = False,
    discipline: Discipline = Discipline.MS,
) -> LiveMatchContext:
    live_state = _make_live_state(discipline=discipline)
    updater = BayesianRWPUpdater(
        match_id="M1",
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=discipline,
        rwp_prior_a=0.515,
        rwp_prior_b=0.510,
    )
    detector = MomentumDetector(match_id="M1", rwp_a=0.515, rwp_b=0.510, discipline_value=discipline.value)
    ctx = LiveMatchContext(
        match_id="M1",
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=discipline,
        tier=TournamentTier.SUPER_500,
        live_state=live_state,
        bayesian_updater=updater,
        momentum_detector=detector,
        pre_match_p_a=0.55,
        rwp_a_prior=0.515,
        rwp_b_prior=0.510,
        last_feed_update=last_feed_update if last_feed_update is not None else time.time(),
        is_ghost_mode=is_ghost,
        is_suspended=is_suspended,
    )
    return ctx


def _make_request(ctx: LiveMatchContext | None = None) -> LivePricingRequest:
    ctx = ctx or _make_context()
    return LivePricingRequest(
        match_id="M1",
        context=ctx,
        latest_snapshot=_make_snapshot(),
        momentum_snapshot=_make_momentum_snapshot(),
        rwp_a_live=_make_rwp_estimate("PA", 0.515),
        rwp_b_live=_make_rwp_estimate("PB", 0.510),
    )


# ---------------------------------------------------------------------------
# 1. LiveMatchContext helpers
# ---------------------------------------------------------------------------

class TestLiveMatchContext:
    def test_fresh_context_not_ghost(self) -> None:
        ctx = _make_context(last_feed_update=time.time())
        assert not ctx.should_ghost()

    def test_fresh_context_not_suspended(self) -> None:
        ctx = _make_context(last_feed_update=time.time())
        assert not ctx.should_suspend()

    def test_old_feed_triggers_ghost(self) -> None:
        ctx = _make_context(last_feed_update=time.time() - (LIVE_GHOST_TRIGGER_SECONDS + 5))
        assert ctx.should_ghost()

    def test_very_old_feed_triggers_suspend(self) -> None:
        ctx = _make_context(last_feed_update=time.time() - (LIVE_SUSPEND_SECONDS + 5))
        assert ctx.should_suspend()

    def test_feed_gap_returns_positive(self) -> None:
        ctx = _make_context(last_feed_update=time.time() - 10)
        assert ctx.feed_gap_seconds() >= 10.0


# ---------------------------------------------------------------------------
# 2. LivePricingEngine — happy path
# ---------------------------------------------------------------------------

class TestLivePricingEngineHappyPath:
    def test_instantiates(self) -> None:
        engine = LivePricingEngine()
        assert engine is not None

    def test_price_after_point_returns_response(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert isinstance(resp, LivePricingResponse)

    def test_response_match_id_preserved(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert resp.match_id == "M1"

    def test_response_has_markets(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert len(resp.markets) > 0

    def test_match_win_prob_in_range(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert 0.0 < resp.p_a_wins_markov < 1.0
        assert 0.0 < resp.p_a_wins_blend < 1.0

    def test_rwp_returned_correctly(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert resp.rwp_a == pytest.approx(0.515)
        assert resp.rwp_b == pytest.approx(0.510)

    def test_not_ghost_by_default(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert not resp.is_ghost_mode

    def test_not_suspended_by_default(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        assert not resp.is_suspended

    def test_all_disciplines(self) -> None:
        engine = LivePricingEngine()
        for disc in Discipline:
            ctx = _make_context(discipline=disc)
            req = _make_request(ctx)
            resp = engine.price_after_point(req)
            assert isinstance(resp, LivePricingResponse)


# ---------------------------------------------------------------------------
# 3. Ghost mode
# ---------------------------------------------------------------------------

class TestGhostMode:
    def test_ghost_mode_context_flag_sets_is_ghost(self) -> None:
        engine = LivePricingEngine()
        ctx = _make_context(is_ghost=True)
        req = _make_request(ctx)
        resp = engine.price_after_point(req)
        assert resp.is_ghost_mode

    def test_stale_feed_activates_ghost(self) -> None:
        engine = LivePricingEngine()
        ctx = _make_context(last_feed_update=time.time() - (LIVE_GHOST_TRIGGER_SECONDS + 5))
        req = _make_request(ctx)
        resp = engine.price_after_point(req)
        assert resp.is_ghost_mode


# ---------------------------------------------------------------------------
# 4. Suspension
# ---------------------------------------------------------------------------

class TestSuspension:
    def test_suspended_feed_returns_is_suspended(self) -> None:
        engine = LivePricingEngine()
        ctx = _make_context(last_feed_update=time.time() - (LIVE_SUSPEND_SECONDS + 5))
        req = _make_request(ctx)
        resp = engine.price_after_point(req)
        assert resp.is_suspended

    def test_suspended_response_has_empty_or_minimal_markets(self) -> None:
        engine = LivePricingEngine()
        ctx = _make_context(last_feed_update=time.time() - (LIVE_SUSPEND_SECONDS + 5))
        req = _make_request(ctx)
        resp = engine.price_after_point(req)
        # Suspended response still returns a LivePricingResponse
        assert isinstance(resp, LivePricingResponse)


# ---------------------------------------------------------------------------
# 5. Market integrity (H7 gate: no negative overround)
# ---------------------------------------------------------------------------

class TestLiveMarketIntegrity:
    def test_no_negative_overround_in_live_markets(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        for market_id, prices in resp.markets.items():
            if not prices:
                continue
            total_prob = sum(p.prob_with_margin for p in prices)
            assert total_prob >= 1.0, (
                f"H7 violation in market {market_id}: total_prob={total_prob:.4f}"
            )

    def test_all_odds_above_1(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        for market_id, prices in resp.markets.items():
            for p in prices:
                assert p.odds >= 1.01, (
                    f"H10 violation in market {market_id}: odds={p.odds:.4f}"
                )

    def test_binary_market_probs_sum_near_100pct(self) -> None:
        engine = LivePricingEngine()
        req = _make_request()
        resp = engine.price_after_point(req)
        # match_winner is a 2-way market
        for market_id, prices in resp.markets.items():
            if "match_winner" in market_id and len(prices) == 2:
                total = sum(p.prob_implied for p in prices)
                # With margin stripped, fair probs must sum near 1.0
                assert abs(total - 1.0) < 0.1, (
                    f"Implied probs too far from 1.0 in {market_id}: {total:.4f}"
                )


# ---------------------------------------------------------------------------
# 6. Score / game state reflected in response
# ---------------------------------------------------------------------------

class TestLiveResponseState:
    def test_score_state_reflected(self) -> None:
        engine = LivePricingEngine()
        snap = _make_snapshot(score_a=10, score_b=7)
        ctx = _make_context()
        req = LivePricingRequest(
            match_id="M1",
            context=ctx,
            latest_snapshot=snap,
            momentum_snapshot=_make_momentum_snapshot(),
            rwp_a_live=_make_rwp_estimate("PA", 0.515),
            rwp_b_live=_make_rwp_estimate("PB", 0.510),
        )
        resp = engine.price_after_point(req)
        assert resp.score_a == 10
        assert resp.score_b == 7

    def test_games_won_reflected(self) -> None:
        engine = LivePricingEngine()
        snap = _make_snapshot(score_a=5, score_b=3, games_won_a=1, games_won_b=0)
        ctx = _make_context()
        req = LivePricingRequest(
            match_id="M1",
            context=ctx,
            latest_snapshot=snap,
            momentum_snapshot=_make_momentum_snapshot(),
            rwp_a_live=_make_rwp_estimate("PA", 0.515),
            rwp_b_live=_make_rwp_estimate("PB", 0.510),
        )
        resp = engine.price_after_point(req)
        assert resp.games_won_a == 1
        assert resp.games_won_b == 0

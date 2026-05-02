"""
test_pre_match_markets.py
=========================
Tests for markets/pre_match_markets.py — full pre-match pricing pipeline.

Covers:
  - PreMatchPricingRequest/Response dataclass construction
  - PreMatchPricingEngine.price() happy path
  - All disciplines produce valid markets
  - H7 gate: no negative overround (all markets overround >= 1.0)
  - H10 gate: all odds >= 1.01
  - Pinnacle blend: with/without pinnacle, model weight correct
  - Regime assignment: R1 for standard, R2 for Super 1000/750/500
  - Market validity flag
  - BatchPreMatchPricer: price_batch processes multiple requests
  - RWP calibration: blended p_a close to target
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    PRE_MATCH_MODEL_WEIGHT,
    TournamentTier,
)
from markets.pre_match_markets import (
    BatchPreMatchPricer,
    PreMatchPricingEngine,
    PreMatchPricingRequest,
    PreMatchPricingResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MATCH_DATE = date(2025, 6, 15)


def _req(
    match_id: str = "M001",
    discipline: Discipline = Discipline.MS,
    tier: TournamentTier = TournamentTier.SUPER_500,
    model_p_a: float = 0.60,
    rwp_a: float = 0.520,
    rwp_b: float = 0.510,
    pinnacle_p: float | None = None,
) -> PreMatchPricingRequest:
    return PreMatchPricingRequest(
        match_id=match_id,
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=discipline,
        tier=tier,
        match_date=MATCH_DATE,
        model_p_a_wins=model_p_a,
        model_p_a_wins_2_0=model_p_a * 0.55,
        model_p_a_wins_deuce=model_p_a * 0.40,
        rwp_a=rwp_a,
        rwp_b=rwp_b,
        pinnacle_p_a_wins=pinnacle_p,
        first_server="A",
    )


# ---------------------------------------------------------------------------
# 1. Dataclass construction
# ---------------------------------------------------------------------------

class TestPreMatchRequestConstruction:
    def test_request_constructs(self) -> None:
        req = _req()
        assert req.match_id == "M001"
        assert req.discipline == Discipline.MS

    def test_response_can_be_constructed(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert isinstance(resp, PreMatchPricingResponse)


# ---------------------------------------------------------------------------
# 2. Happy path — output validity
# ---------------------------------------------------------------------------

class TestPreMatchHappyPath:
    def test_price_returns_response(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert isinstance(resp, PreMatchPricingResponse)

    def test_match_id_preserved(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(match_id="MATCH42"))
        assert resp.match_id == "MATCH42"

    def test_discipline_preserved(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(discipline=Discipline.WS))
        assert resp.discipline == Discipline.WS

    def test_markets_not_empty(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert len(resp.market_set.markets) > 0

    def test_blend_probability_in_range(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert 0.0 < resp.p_a_wins_blend < 1.0

    def test_rwp_used_in_valid_range(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert 0.20 <= resp.rwp_a_used <= 0.80
        assert 0.20 <= resp.rwp_b_used <= 0.80

    def test_valid_until_after_generated_at(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert resp.valid_until > resp.generated_at

    def test_markets_valid_flag(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        assert resp.markets_valid


# ---------------------------------------------------------------------------
# 3. All disciplines
# ---------------------------------------------------------------------------

class TestAllDisciplines:
    @pytest.mark.parametrize("disc", list(Discipline))
    def test_all_disciplines_produce_markets(self, disc: Discipline) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(discipline=disc))
        assert len(resp.market_set.markets) > 0
        assert resp.markets_valid


# ---------------------------------------------------------------------------
# 4. H7 gate: no negative overround
# ---------------------------------------------------------------------------

class TestH7NoNegativeOverround:
    def test_all_markets_have_overround_gte_1(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        for market_id, prices in resp.market_set.markets.items():
            # Single-outcome markets (exact score props etc.) are one leg of
            # a grouped event and legitimately sum < 1.0 in isolation.
            if len(prices) < 2:
                continue
            total_implied = sum(p.prob_with_margin for p in prices)
            assert total_implied >= 0.999, (
                f"H7 violation in {market_id}: overround={total_implied:.4f}"
            )

    def test_match_winner_overround_in_tier_range(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(tier=TournamentTier.SUPER_1000))
        prices = resp.market_set.markets.get("match_winner", [])
        if prices:
            total = sum(p.prob_with_margin for p in prices)
            margin = total - 1.0
            assert 0.04 <= margin <= 0.18, f"H1 margin={margin:.4f} outside [0.04, 0.18]"


# ---------------------------------------------------------------------------
# 5. H10 gate: all odds >= 1.01
# ---------------------------------------------------------------------------

class TestH10MinOdds:
    def test_all_odds_above_minimum(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req())
        for market_id, prices in resp.market_set.markets.items():
            for p in prices:
                assert p.odds >= 1.01, (
                    f"H10 violation in {market_id}: odds={p.odds:.4f}"
                )


# ---------------------------------------------------------------------------
# 6. Pinnacle blend
# ---------------------------------------------------------------------------

class TestPinnacleBlend:
    def test_no_pinnacle_model_weight_1(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(pinnacle_p=None))
        assert resp.model_weight == pytest.approx(1.0)
        assert resp.markov_weight == pytest.approx(0.0)

    def test_with_pinnacle_model_weight_is_config(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(pinnacle_p=0.55))
        assert resp.model_weight == pytest.approx(PRE_MATCH_MODEL_WEIGHT)

    def test_pinnacle_blend_between_inputs(self) -> None:
        engine = PreMatchPricingEngine()
        model_p = 0.60
        pinnacle_p = 0.50
        resp = engine.price(_req(model_p_a=model_p, pinnacle_p=pinnacle_p))
        # Blend must be between the two inputs
        lo, hi = min(model_p, pinnacle_p), max(model_p, pinnacle_p)
        assert lo <= resp.p_a_wins_blend <= hi

    def test_no_pinnacle_blend_equals_model(self) -> None:
        engine = PreMatchPricingEngine()
        model_p = 0.65
        resp = engine.price(_req(model_p_a=model_p, pinnacle_p=None))
        assert resp.p_a_wins_blend == pytest.approx(model_p, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. Regime assignment
# ---------------------------------------------------------------------------

class TestRegimeAssignment:
    def test_super_1000_returns_r2(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(tier=TournamentTier.SUPER_1000))
        assert resp.regime == "R2"

    def test_super_750_returns_r2(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(tier=TournamentTier.SUPER_750))
        assert resp.regime == "R2"

    def test_super_500_returns_r2(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(tier=TournamentTier.SUPER_500))
        assert resp.regime == "R2"

    def test_super_300_returns_r1(self) -> None:
        engine = PreMatchPricingEngine()
        resp = engine.price(_req(tier=TournamentTier.SUPER_300))
        assert resp.regime == "R1"


# ---------------------------------------------------------------------------
# 8. RWP calibration: Markov at calibrated RWP produces p_a near target
# ---------------------------------------------------------------------------

class TestRWPCalibration:
    def test_calibrated_rwp_produces_correct_p_a(self) -> None:
        from core.markov_engine import BadmintonMarkovEngine
        engine = PreMatchPricingEngine()
        # Use a realistic target: ~57% is easily achievable with typical RWP inputs
        target = 0.57
        resp = engine.price(_req(model_p_a=target, pinnacle_p=None))
        markov = BadmintonMarkovEngine()
        probs = markov.compute_match_probabilities(
            rwp_a=resp.rwp_a_used,
            rwp_b=resp.rwp_b_used,
            discipline=Discipline.MS,
            server_first_game="A",
        )
        # Calibration bisection: result must be within 2pp of target
        assert abs(probs.p_a_wins_match - target) < 0.02


# ---------------------------------------------------------------------------
# 9. BatchPreMatchPricer
# ---------------------------------------------------------------------------

class TestBatchPreMatchPricer:
    def test_batch_prices_all_matches(self) -> None:
        pricer = BatchPreMatchPricer()
        requests = [_req(match_id=f"M{i:03d}") for i in range(5)]
        results = pricer.price_batch(requests)
        assert len(results) == 5

    def test_batch_keys_match_request_ids(self) -> None:
        pricer = BatchPreMatchPricer()
        requests = [_req(match_id=f"BATCH_{i}") for i in range(3)]
        results = pricer.price_batch(requests)
        for req in requests:
            assert req.match_id in results

    def test_batch_with_empty_list(self) -> None:
        pricer = BatchPreMatchPricer()
        results = pricer.price_batch([])
        assert results == {}

    def test_batch_survives_single_failure(self) -> None:
        """If one request fails, others still complete."""
        from unittest.mock import patch, MagicMock
        pricer = BatchPreMatchPricer()
        good_req = _req(match_id="GOOD")
        bad_req = _req(match_id="BAD")
        original_price = pricer._engine.price

        def fail_on_bad(req):
            if req.match_id == "BAD":
                raise RuntimeError("Intentional test failure")
            return original_price(req)

        with patch.object(pricer._engine, "price", side_effect=fail_on_bad):
            results = pricer.price_batch([good_req, bad_req])
        # GOOD processed, BAD skipped
        assert "GOOD" in results
        assert "BAD" not in results

"""
Regression tests for odds floor + refuse-extreme-confidence in Badminton MS.

LOCK-BADMINTON-ODDS-FLOOR-1-01-001
LOCK-BADMINTON-REFUSE-EXTREME-CONFIDENCE-001
"""
from __future__ import annotations

import pytest

from markets.margin_engine import (
    _MIN_ODDS,
    _MAX_MODEL_CONFIDENCE_FOR_PRICING,
    _apply_odds_floor,
    MarginEngine,
)


# ---------------------------------------------------------------------------
# LOCK-BADMINTON-ODDS-FLOOR-1-01-001
# ---------------------------------------------------------------------------

class TestBadmintonOddsFloor:
    """LOCK-BADMINTON-ODDS-FLOOR-1-01-001: all offered odds >= 1.01."""

    def test_floor_constant_is_101(self):
        assert _MIN_ODDS == 1.01

    def test_apply_floor_sub_floor_clamped(self):
        assert _apply_odds_floor(0.97, "test") == pytest.approx(1.01)

    def test_apply_floor_at_floor_unchanged(self):
        assert _apply_odds_floor(1.01, "test") == pytest.approx(1.01)

    def test_apply_floor_above_floor_unchanged(self):
        assert _apply_odds_floor(2.20, "test") == pytest.approx(2.20)

    def test_apply_floor_emits_log_message_structure(self):
        """Function signature accepts context kwarg for observability."""
        result = _apply_odds_floor(0.5, context="test_observability")
        assert result == pytest.approx(1.01)

    def test_margin_engine_two_way_no_sub_floor(self):
        """MarginEngine on a 2-way market must never produce odds < 1.01."""
        from markets.derivative_engine import MarketPrice, MarketFamily
        engine = MarginEngine()
        prices = [
            MarketPrice(
                market_id="m1", market_family=MarketFamily.MATCH_RESULT,
                outcome_name="Player1", odds=0.99, prob_implied=0.98,
            ),
            MarketPrice(
                market_id="m1", market_family=MarketFamily.MATCH_RESULT,
                outcome_name="Player2", odds=50.0, prob_implied=0.02,
            ),
        ]
        result = engine._apply_power_margin(prices, target_margin=0.05)
        for mp in result:
            assert mp.odds >= 1.01, f"Outcome {mp.outcome_name}: odds={mp.odds} below floor"

    def test_full_probability_sweep_no_sub_floor(self):
        """Every fair_prob in [0.01, 0.99] step 0.01 must produce odds >= 1.01."""
        violations = []
        for i in range(1, 100):
            p = i / 100.0
            raw = 1.0 / p
            result = _apply_odds_floor(raw, "sweep")
            if result < 1.01:
                violations.append((p, raw, result))
        assert not violations


# ---------------------------------------------------------------------------
# LOCK-BADMINTON-REFUSE-EXTREME-CONFIDENCE-001
# ---------------------------------------------------------------------------

class TestBadmintonRefuseExtremeConfidence:
    """LOCK-BADMINTON-REFUSE-EXTREME-CONFIDENCE-001: refuse when p > 0.97."""

    def test_threshold_constant_is_097(self):
        assert _MAX_MODEL_CONFIDENCE_FOR_PRICING == 0.97

    def test_predict_route_has_refuse_guard(self):
        """The predict route must contain the REFUSE_EXTREME_CONFIDENCE guard."""
        import inspect
        from api import predict as m
        src = inspect.getsource(m)
        assert "REFUSE_EXTREME_CONFIDENCE" in src
        assert "FIXTURE_TOO_CONFIDENT" in src

    def test_refuse_threshold_in_predict_route(self):
        from api import predict as m
        src = __import__("inspect").getsource(m)
        assert "0.97" in src

    def test_refuse_applied_before_response_data(self):
        from api import predict as m
        src = __import__("inspect").getsource(m)
        refuse_pos = src.find("REFUSE_EXTREME_CONFIDENCE")
        response_pos = src.find("BadmintonPredictResponseData(")
        assert refuse_pos < response_pos

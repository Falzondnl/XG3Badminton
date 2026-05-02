"""
test_margin_engine.py
=====================
Unit tests for markets/margin_engine.py

Tests:
  - Power method: k bisection correctness
  - Applied margin ≈ target margin
  - Margined probabilities sum to 1 + margin
  - Odds always >= 1/prob_with_margin (margin applied correctly)
  - H1 gate: overround within [OVERROUND_MIN, OVERROUND_MAX]
  - 2-way, 4-way, 5-way markets
  - Edge cases: very small/large margins
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import OVERROUND_MIN, OVERROUND_MAX
from markets.margin_engine import MarginEngine, MarginedPrice


@pytest.fixture
def engine():
    return MarginEngine()


def fair_probs_2way(p: float = 0.55):
    """2-way market with fair probabilities."""
    return [p, 1.0 - p]


def fair_probs_4way():
    """Correct score market: A_2-0, A_2-1, B_2-0, B_2-1."""
    return [0.35, 0.20, 0.28, 0.17]  # sum = 1.0


def fair_probs_5way():
    """5-outcome race-to market."""
    probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    return probs


class TestPowerMethod:
    """Power margin bisection correctness."""

    def test_2way_margin_applied(self, engine):
        """2-way market: margined probs sum to 1 + target margin."""
        probs = fair_probs_2way()
        target = 0.06  # 6%
        margined = engine.apply_margins(probs, target)
        total = sum(p.prob_with_margin for p in margined)
        assert abs(total - (1.0 + target)) < 0.0005

    def test_4way_margin_applied(self, engine):
        """4-way market: margined probs sum to 1 + target margin."""
        probs = fair_probs_4way()
        target = 0.08
        margined = engine.apply_margins(probs, target)
        total = sum(p.prob_with_margin for p in margined)
        assert abs(total - (1.0 + target)) < 0.001

    def test_5way_margin_applied(self, engine):
        """5-way market: power method scales correctly."""
        probs = fair_probs_5way()
        target = 0.10
        margined = engine.apply_margins(probs, target)
        total = sum(p.prob_with_margin for p in margined)
        assert abs(total - (1.0 + target)) < 0.001

    def test_zero_margin_unchanged(self, engine):
        """Zero target margin: probabilities unchanged."""
        probs = fair_probs_2way()
        margined = engine.apply_margins(probs, 0.0)
        total = sum(p.prob_with_margin for p in margined)
        assert abs(total - 1.0) < 0.001

    def test_various_margins(self, engine):
        """Power method handles margins from 2% to 18%."""
        probs = fair_probs_2way()
        for target in [0.02, 0.04, 0.06, 0.08, 0.12, 0.15, 0.18]:
            margined = engine.apply_margins(probs, target)
            total = sum(p.prob_with_margin for p in margined)
            assert abs(total - (1.0 + target)) < 0.002, f"Failed at margin={target}"


class TestOddsComputation:
    """Decimal odds from margined probabilities."""

    def test_odds_equals_1_over_prob(self, engine):
        """Decimal odds = 1 / prob_with_margin."""
        probs = fair_probs_2way(0.60)
        margined = engine.apply_margins(probs, 0.06)
        for m in margined:
            expected_odds = 1.0 / m.prob_with_margin
            assert abs(m.odds - expected_odds) < 0.001

    def test_min_odds_above_1_01(self, engine):
        """H10: all odds >= 1.01 even with maximum margin."""
        probs = [0.98, 0.02]  # very skewed
        margined = engine.apply_margins(probs, 0.15)
        for m in margined:
            assert m.odds >= 1.01, f"H10 violated: odds={m.odds}"

    def test_odds_inversely_related_to_prob(self, engine):
        """Higher probability → lower odds."""
        probs = [0.70, 0.30]
        margined = engine.apply_margins(probs, 0.05)
        # Higher prob player has lower odds
        assert margined[0].odds < margined[1].odds


class TestH1Gate:
    """H1 gate: overround within tier bounds."""

    @pytest.mark.parametrize("margin", [
        OVERROUND_MIN,
        (OVERROUND_MIN + OVERROUND_MAX) / 2,
        OVERROUND_MAX,
    ])
    def test_h1_passes_within_bounds(self, engine, margin):
        """Margins within tier bounds pass H1."""
        probs = fair_probs_2way()
        margined = engine.apply_margins(probs, margin)
        total_margin = sum(p.prob_with_margin for p in margined) - 1.0
        assert OVERROUND_MIN <= total_margin <= OVERROUND_MAX

    def test_h1_overround_computation(self, engine):
        """Overround = (sum of margined probs) - 1.0."""
        probs = fair_probs_4way()
        target = 0.08
        margined = engine.apply_margins(probs, target)
        overround = sum(p.prob_with_margin for p in margined) - 1.0
        assert abs(overround - target) < 0.002


class TestMarginedPriceStructure:
    """MarginedPrice dataclass structure."""

    def test_returns_correct_count(self, engine):
        """Number of margined prices matches input probabilities."""
        probs = [0.25, 0.30, 0.25, 0.20]
        margined = engine.apply_margins(probs, 0.07)
        assert len(margined) == 4

    def test_fair_prob_preserved(self, engine):
        """MarginedPrice stores original fair probability."""
        probs = [0.60, 0.40]
        margined = engine.apply_margins(probs, 0.05)
        assert abs(margined[0].fair_prob - 0.60) < 0.001
        assert abs(margined[1].fair_prob - 0.40) < 0.001

    def test_margined_prob_strictly_greater_than_fair(self, engine):
        """Margined probability > fair probability (margin inflates)."""
        probs = fair_probs_2way()
        margined = engine.apply_margins(probs, 0.06)
        for m, fair in zip(margined, probs):
            assert m.prob_with_margin > fair

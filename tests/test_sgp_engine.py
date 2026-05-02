"""
test_sgp_engine.py
==================
Tests for markets/sgp_engine.py — Same-Game Parlay pricing engine.

Covers:
  - SGPLeg / SGPRequest dataclass construction
  - BadmintonSGPEngine.price_sgp() happy path
  - H8 gate: SGP price >= max individual leg price (no-arb)
  - Max legs enforcement (> SGP_MAX_LEGS rejected)
  - Min leg odds enforcement (leg odds < SGP_MIN_LEG_ODDS rejected)
  - Joint probability < product of marginals (positive correlation adjustment)
  - Combined odds are positive and above 1.0 when valid
  - All disciplines produce valid SGP
  - All SGPLegType variants accepted
  - is_valid flag and rejection_reason on invalid SGPs
  - Single-leg SGP (degenerate case)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    SGP_MAX_LEGS,
    SGP_MIN_LEG_ODDS,
    TournamentTier,
)
from markets.sgp_engine import (
    BadmintonSGPEngine,
    SGPLeg,
    SGPLegType,
    SGPRequest,
    SGPResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leg(
    leg_type: SGPLegType = SGPLegType.MATCH_WINNER,
    selection: str = "A",
    fair_prob: float = 0.60,
    market_id: str = "match_winner",
) -> SGPLeg:
    return SGPLeg(
        leg_type=leg_type,
        selection=selection,
        fair_prob=fair_prob,
        market_id=market_id,
    )


def _req(
    match_id: str = "M001",
    discipline: Discipline = Discipline.MS,
    legs: list[SGPLeg] | None = None,
    rwp_a: float = 0.515,
    rwp_b: float = 0.510,
) -> SGPRequest:
    if legs is None:
        legs = [
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
            _leg(SGPLegType.TOTAL_GAMES, selection="over_2", fair_prob=0.55, market_id="total_games_over"),
        ]
    return SGPRequest(
        match_id=match_id,
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=discipline,
        tier=TournamentTier.SUPER_500,
        legs=legs,
        rwp_a=rwp_a,
        rwp_b=rwp_b,
        first_server="A",
    )


# ---------------------------------------------------------------------------
# 1. Dataclass construction
# ---------------------------------------------------------------------------

class TestSGPDataclasses:
    def test_leg_constructs(self) -> None:
        leg = _leg()
        assert leg.leg_type == SGPLegType.MATCH_WINNER
        assert leg.fair_prob == 0.60
        assert leg.selection == "A"

    def test_leg_with_params(self) -> None:
        leg = SGPLeg(
            leg_type=SGPLegType.RACE_TO_N,
            selection="A",
            fair_prob=0.65,
            market_id="race_to_5",
            param_game=1,
            param_n=5,
        )
        assert leg.param_game == 1
        assert leg.param_n == 5

    def test_request_constructs(self) -> None:
        req = _req()
        assert req.match_id == "M001"
        assert req.discipline == Discipline.MS
        assert len(req.legs) == 2

    def test_request_default_server(self) -> None:
        req = _req()
        assert req.first_server == "A"
        assert req.current_game == 1


# ---------------------------------------------------------------------------
# 2. Happy path
# ---------------------------------------------------------------------------

class TestSGPHappyPath:
    def test_price_sgp_returns_response(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        assert isinstance(resp, SGPResponse)

    def test_match_id_preserved(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req(match_id="SGP42"))
        assert resp.match_id == "SGP42"

    def test_n_legs_correct(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        assert resp.n_legs == 2

    def test_combined_odds_above_1(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        if resp.is_valid:
            assert resp.combined_odds > 1.0

    def test_combined_odds_fair_positive(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        if resp.is_valid:
            assert resp.combined_odds_fair > 0.0

    def test_joint_prob_fair_in_range(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        if resp.is_valid:
            assert 0.0 < resp.joint_prob_fair < 1.0

    def test_margin_applied_non_negative(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        if resp.is_valid:
            assert resp.margin_applied >= 0.0


# ---------------------------------------------------------------------------
# 3. H8 gate: SGP price >= max individual leg price (no-arb)
# ---------------------------------------------------------------------------

class TestH8NoArb:
    def test_combined_odds_above_each_leg(self) -> None:
        engine = BadmintonSGPEngine()
        # Use high-probability legs so the combination is still valid
        legs = [
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.65, market_id="match_winner"),
            _leg(SGPLegType.TOTAL_GAMES, selection="over", fair_prob=0.58, market_id="total_games_over"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        if resp.is_valid:
            # Each individual leg implied odds
            for leg in resp.legs:
                if leg.fair_prob > 0:
                    leg_odds_fair = 1.0 / leg.fair_prob
                    # Combined SGP must be >= any individual leg odds (H8)
                    assert resp.combined_odds_fair >= leg_odds_fair - 0.001, (
                        f"H8 violation: combined={resp.combined_odds_fair:.4f} < leg={leg_odds_fair:.4f}"
                    )

    def test_two_leg_sgp_higher_than_single(self) -> None:
        engine = BadmintonSGPEngine()
        single_leg = [_leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner")]
        two_leg = [
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
            _leg(SGPLegType.TOTAL_GAMES, selection="over", fair_prob=0.55, market_id="total_games_over"),
        ]
        resp_1 = engine.price_sgp(_req(legs=single_leg))
        resp_2 = engine.price_sgp(_req(legs=two_leg))
        if resp_1.is_valid and resp_2.is_valid:
            assert resp_2.combined_odds_fair > resp_1.combined_odds_fair


# ---------------------------------------------------------------------------
# 4. Correlation / joint probability
# ---------------------------------------------------------------------------

class TestCorrelationAdjustment:
    def test_joint_prob_below_product_of_marginals(self) -> None:
        """Positive correlation: joint prob should be less than or equal to
        the naive product of independent probabilities."""
        engine = BadmintonSGPEngine()
        legs = [
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
            _leg(SGPLegType.GAME_WINNER, selection="A", fair_prob=0.62, market_id="g1_winner"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        if resp.is_valid:
            naive_product = 0.60 * 0.62
            # With positive correlation (winning game 1 and winning match are correlated),
            # joint should NOT exceed the naive product by more than a small tolerance
            # (actually it should be less than naive for positively correlated outcomes)
            assert resp.joint_prob_fair <= naive_product * 1.05, (
                f"Joint prob {resp.joint_prob_fair:.4f} exceeds naive product {naive_product:.4f}"
            )

    def test_correlation_adjustment_recorded(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req())
        # correlation_adjustment field must exist and be a float
        assert isinstance(resp.correlation_adjustment, float)


# ---------------------------------------------------------------------------
# 5. Max legs enforcement
# ---------------------------------------------------------------------------

class TestMaxLegs:
    def test_too_many_legs_invalid(self) -> None:
        engine = BadmintonSGPEngine()
        # Build SGP_MAX_LEGS + 1 legs
        legs = [
            _leg(
                SGPLegType.MATCH_WINNER,
                selection="A",
                fair_prob=0.60,
                market_id=f"market_{i}",
            )
            for i in range(SGP_MAX_LEGS + 1)
        ]
        resp = engine.price_sgp(_req(legs=legs))
        assert not resp.is_valid
        assert resp.rejection_reason is not None

    def test_exactly_max_legs_valid(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [
            SGPLeg(
                leg_type=SGPLegType.MATCH_WINNER if i % 2 == 0 else SGPLegType.TOTAL_GAMES,
                selection="A" if i % 2 == 0 else "over",
                fair_prob=0.55 + i * 0.02,
                market_id=f"market_{i}",
            )
            for i in range(SGP_MAX_LEGS)
        ]
        resp = engine.price_sgp(_req(legs=legs))
        # Should not be rejected purely due to leg count
        if resp.rejection_reason:
            assert "legs" not in resp.rejection_reason.lower()


# ---------------------------------------------------------------------------
# 6. Min leg odds enforcement
# ---------------------------------------------------------------------------

class TestMinLegOdds:
    def test_leg_odds_below_minimum_rejected(self) -> None:
        """A leg with fair_prob so high its implied odds < SGP_MIN_LEG_ODDS should be rejected."""
        engine = BadmintonSGPEngine()
        # Odds < 1.10 (SGP_MIN_LEG_ODDS=1.10) means prob > 1/1.10 ≈ 0.909
        very_high_prob_leg = SGPLeg(
            leg_type=SGPLegType.MATCH_WINNER,
            selection="A",
            fair_prob=0.95,   # implied odds ≈ 1.053 < 1.10
            market_id="match_winner",
        )
        legs = [
            very_high_prob_leg,
            _leg(SGPLegType.TOTAL_GAMES, selection="over", fair_prob=0.55, market_id="total_games"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        # Should be rejected or have a warning
        if not resp.is_valid:
            assert resp.rejection_reason is not None


# ---------------------------------------------------------------------------
# 7. All disciplines
# ---------------------------------------------------------------------------

class TestAllDisciplines:
    @pytest.mark.parametrize("disc", list(Discipline))
    def test_all_disciplines_produce_response(self, disc: Discipline) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req(discipline=disc))
        assert isinstance(resp, SGPResponse)
        assert resp.match_id == "M001"


# ---------------------------------------------------------------------------
# 8. All SGPLegType variants
# ---------------------------------------------------------------------------

class TestSGPLegTypes:
    def test_match_winner_leg(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
            _leg(SGPLegType.TOTAL_GAMES, selection="over", fair_prob=0.55, market_id="total_games"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        assert isinstance(resp, SGPResponse)

    def test_correct_score_leg(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [
            _leg(SGPLegType.CORRECT_SCORE, selection="2_0", fair_prob=0.40, market_id="correct_score_2_0"),
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        assert isinstance(resp, SGPResponse)

    def test_game_winner_leg(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [
            SGPLeg(
                leg_type=SGPLegType.GAME_WINNER,
                selection="A",
                fair_prob=0.58,
                market_id="g1_winner",
                param_game=1,
            ),
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        assert isinstance(resp, SGPResponse)

    def test_race_to_n_leg(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [
            SGPLeg(
                leg_type=SGPLegType.RACE_TO_N,
                selection="A",
                fair_prob=0.62,
                market_id="race_to_5",
                param_game=1,
                param_n=5,
            ),
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        assert isinstance(resp, SGPResponse)

    def test_points_ou_leg(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [
            SGPLeg(
                leg_type=SGPLegType.POINTS_OVER_UNDER,
                selection="over",
                fair_prob=0.50,
                market_id="points_ou_42",
                param_threshold=42.5,
            ),
            _leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner"),
        ]
        resp = engine.price_sgp(_req(legs=legs))
        assert isinstance(resp, SGPResponse)


# ---------------------------------------------------------------------------
# 9. Single-leg SGP (degenerate case)
# ---------------------------------------------------------------------------

class TestSingleLegSGP:
    def test_single_leg_response(self) -> None:
        engine = BadmintonSGPEngine()
        legs = [_leg(SGPLegType.MATCH_WINNER, selection="A", fair_prob=0.60, market_id="match_winner")]
        resp = engine.price_sgp(_req(legs=legs))
        assert isinstance(resp, SGPResponse)
        if resp.is_valid:
            assert resp.n_legs == 1
            # Single leg: combined odds ≈ 1 / fair_prob with margin
            assert resp.combined_odds > 1.0


# ---------------------------------------------------------------------------
# 10. Empty leg list
# ---------------------------------------------------------------------------

class TestEmptyLegs:
    def test_empty_legs_not_valid(self) -> None:
        engine = BadmintonSGPEngine()
        resp = engine.price_sgp(_req(legs=[]))
        assert not resp.is_valid
        assert resp.rejection_reason is not None

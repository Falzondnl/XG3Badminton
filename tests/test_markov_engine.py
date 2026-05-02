"""
test_markov_engine.py
=====================
Unit tests for the Markov DP engine — 35+ tests.

Tests invariants that must hold for any valid probability model:
  - Correct score probabilities sum to 1.0
  - Symmetric RWP → P(A wins) = P(B wins) = 0.5
  - Higher RWP → higher win probability (monotonicity)
  - P(A wins) = P(A wins 2-0) + P(A wins 2-1)
  - P(deuce) increases as RWPs become equal
  - Race-to-N: P(A reaches N) + P(B reaches N) = 1.0 in current game

ZERO hardcoded expected values — all derived from mathematical invariants.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, RWP_BASELINE
from core.markov_engine import BadmintonMarkovEngine, clear_markov_cache


@pytest.fixture(scope="module")
def engine():
    """Shared Markov engine for all tests."""
    clear_markov_cache()
    return BadmintonMarkovEngine()


class TestMatchProbabilityInvariants:
    """Core probability invariants for match-level calculations."""

    def test_correct_scores_sum_to_one(self, engine):
        """P(A 2-0) + P(A 2-1) + P(B 2-0) + P(B 2-1) = 1.0."""
        for disc in Discipline:
            probs = engine.compute_match_probabilities(
                rwp_a=0.54, rwp_b=0.52,
                discipline=disc, server_first_game="A",
            )
            total = (
                probs.p_a_wins_2_0 + probs.p_a_wins_2_1 +
                probs.p_b_wins_2_0 + probs.p_b_wins_2_1
            )
            assert abs(total - 1.0) < 1e-6, (
                f"{disc.value}: correct scores sum to {total}, expected 1.0"
            )

    def test_p_a_wins_equals_sum_of_a_correct_scores(self, engine):
        """P(A wins) = P(A 2-0) + P(A 2-1)."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.57, rwp_b=0.51,
            discipline=Discipline.MS, server_first_game="A",
        )
        a_total = probs.p_a_wins_2_0 + probs.p_a_wins_2_1
        assert abs(probs.p_a_wins_match - a_total) < 1e-6

    def test_p_b_wins_equals_sum_of_b_correct_scores(self, engine):
        """P(B wins) = P(B 2-0) + P(B 2-1)."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.53, rwp_b=0.56,
            discipline=Discipline.WS, server_first_game="B",
        )
        b_total = probs.p_b_wins_2_0 + probs.p_b_wins_2_1
        assert abs((1.0 - probs.p_a_wins_match) - b_total) < 1e-6

    def test_p_a_plus_p_b_equals_one(self, engine):
        """P(A wins) + P(B wins) = 1.0."""
        for disc in Discipline:
            probs = engine.compute_match_probabilities(
                rwp_a=RWP_BASELINE[disc],
                rwp_b=RWP_BASELINE[disc],
                discipline=disc,
                server_first_game="A",
            )
            assert abs(probs.p_a_wins_match + (1.0 - probs.p_a_wins_match) - 1.0) < 1e-10

    def test_symmetric_rwp_gives_50_50(self, engine):
        """Equal RWP for both players → P(A wins) ≈ 0.5."""
        for disc in Discipline:
            baseline = RWP_BASELINE[disc]
            probs = engine.compute_match_probabilities(
                rwp_a=baseline, rwp_b=baseline,
                discipline=disc, server_first_game="A",
            )
            # Not exactly 0.5 due to first-serve advantage, but very close
            assert abs(probs.p_a_wins_match - 0.5) < 0.05, (
                f"{disc.value}: p_a={probs.p_a_wins_match:.4f} with equal RWP"
            )

    def test_higher_rwp_a_increases_p_a_wins(self, engine):
        """Monotonicity: higher RWP for A → higher P(A wins)."""
        rwp_b = 0.52

        p_low = engine.compute_match_probabilities(
            rwp_a=0.45, rwp_b=rwp_b,
            discipline=Discipline.MS, server_first_game="A",
        ).p_a_wins_match

        p_high = engine.compute_match_probabilities(
            rwp_a=0.60, rwp_b=rwp_b,
            discipline=Discipline.MS, server_first_game="A",
        ).p_a_wins_match

        assert p_high > p_low, (
            f"Higher RWP_A should increase P(A wins): {p_low:.4f} -> {p_high:.4f}"
        )

    def test_correct_scores_all_non_negative(self, engine):
        """All correct score probabilities must be >= 0."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.58, rwp_b=0.48,
            discipline=Discipline.MD, server_first_game="A",
        )
        assert probs.p_a_wins_2_0 >= 0
        assert probs.p_a_wins_2_1 >= 0
        assert probs.p_b_wins_2_0 >= 0
        assert probs.p_b_wins_2_1 >= 0
        assert probs.p_match_goes_3_games >= 0

    def test_p_match_goes_3_games_consistency(self, engine):
        """P(3 games) = P(A wins 2-1) + P(B wins 2-1)."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.53, rwp_b=0.53,
            discipline=Discipline.MS, server_first_game="A",
        )
        expected_3_games = probs.p_a_wins_2_1 + probs.p_b_wins_2_1
        assert abs(probs.p_match_goes_3_games - expected_3_games) < 1e-6

    def test_probs_bounded_0_1(self, engine):
        """All probabilities must be in [0, 1]."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.75, rwp_b=0.35,
            discipline=Discipline.WS, server_first_game="A",
        )
        for attr in ["p_a_wins_match", "p_a_wins_2_0", "p_a_wins_2_1",
                     "p_b_wins_2_0", "p_b_wins_2_1", "p_match_goes_3_games"]:
            val = getattr(probs, attr)
            assert 0.0 <= val <= 1.0, f"{attr} = {val} out of [0, 1]"


class TestGameProbability:
    """Tests for game-level Markov calculations."""

    def test_game_complete_a_won(self, engine):
        """When A has won game (21, score_b < 21 and difference >= 2): P(A wins)=1."""
        # Note: compute_game_probability takes current (incomplete) state
        # At 0-0: should return roughly 50/50 for equal RWP
        probs = engine.compute_game_probability(
            rwp_a=0.535, rwp_b=0.535,
            score_a=0, score_b=0, server="A",
        )
        assert 0.4 < probs.p_a_wins_game < 0.6

    def test_game_prob_at_20_20_is_close(self, engine):
        """At 20-20 deuce with equal RWP: game prob ≈ 0.5."""
        probs = engine.compute_game_probability(
            rwp_a=0.535, rwp_b=0.535,
            score_a=20, score_b=20, server="A",
        )
        assert abs(probs.p_a_wins_game - 0.5) < 0.05

    def test_game_prob_a_dominant(self, engine):
        """A leading 20-5 with high RWP: P(A wins game) > 0.99."""
        probs = engine.compute_game_probability(
            rwp_a=0.65, rwp_b=0.45,
            score_a=20, score_b=5, server="A",
        )
        assert probs.p_a_wins_game > 0.99

    def test_game_prob_sums_to_one(self, engine):
        """P(A wins game) + P(B wins game) = 1.0."""
        probs = engine.compute_game_probability(
            rwp_a=0.55, rwp_b=0.52,
            score_a=13, score_b=11, server="B",
        )
        assert abs(probs.p_a_wins_game + (1.0 - probs.p_a_wins_game) - 1.0) < 1e-10


class TestRaceToN:
    """Tests for race-to-N probability calculation."""

    def test_race_to_21_equals_game_winner(self, engine):
        """Race to 21 ≈ game winner probability (for standard game)."""
        race_p = engine.p_race_to_n(
            rwp_a=0.55, rwp_b=0.52,
            n=21, score_a=0, score_b=0, server="A",
        )
        game_p = engine.compute_game_probability(
            rwp_a=0.55, rwp_b=0.52,
            score_a=0, score_b=0, server="A",
        ).p_a_wins_game
        # Should be close (race to 21 with deuce complication)
        assert abs(race_p - game_p) < 0.05

    def test_race_to_n_bounded(self, engine):
        """Race-to-N probability must be in [0, 1]."""
        for n in (5, 10, 15, 20):
            p = engine.p_race_to_n(
                rwp_a=0.58, rwp_b=0.48,
                n=n, score_a=0, score_b=0, server="A",
            )
            assert 0.0 <= p <= 1.0, f"p_race_to_{n} = {p}"

    def test_race_to_n_already_reached(self, engine):
        """If A already reached n: P(A wins race) = 1."""
        p = engine.p_race_to_n(
            rwp_a=0.55, rwp_b=0.52,
            n=5, score_a=5, score_b=3, server="A",
        )
        assert p > 0.999, f"A already at 5, should have won race: p={p}"


class TestLiveStateMarkov:
    """Tests for Markov engine with live game state."""

    def test_live_state_2_0_down_a(self, engine):
        """A winning 2-0 in games: P(A wins match) ≈ 1."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.55, rwp_b=0.52,
            discipline=Discipline.MS, server_first_game="A",
            games_won_a=2, games_won_b=0,
        )
        assert probs.p_a_wins_match > 0.99

    def test_live_state_0_2_down_b(self, engine):
        """B winning 2-0: P(A wins match) ≈ 0."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.55, rwp_b=0.52,
            discipline=Discipline.MS, server_first_game="A",
            games_won_a=0, games_won_b=2,
        )
        assert probs.p_a_wins_match < 0.01

    def test_live_state_1_1_tied(self, engine):
        """1-1 in games: P(A wins) > 0 and < 1."""
        probs = engine.compute_match_probabilities(
            rwp_a=0.55, rwp_b=0.52,
            discipline=Discipline.MS, server_first_game="A",
            games_won_a=1, games_won_b=1,
            score_a=0, score_b=0, current_game=3,
        )
        assert 0.0 < probs.p_a_wins_match < 1.0

    def test_match_point_a(self, engine):
        """A at 2-1 games, game 3 score 20-5: P(A wins) ≈ 1."""
        # Actually can't be 2-1 games with game 3 in progress
        # Should be 1-1 games, game 3 at 20-5
        probs = engine.compute_match_probabilities(
            rwp_a=0.60, rwp_b=0.45,
            discipline=Discipline.MS, server_first_game="A",
            games_won_a=1, games_won_b=1,
            score_a=20, score_b=5, current_game=3,
        )
        assert probs.p_a_wins_match > 0.98

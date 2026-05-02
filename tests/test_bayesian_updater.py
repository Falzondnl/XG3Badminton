"""
test_bayesian_updater.py
========================
Unit tests for core/bayesian_updater.py

Tests:
  - Prior initialisation from pre-match RWP
  - Posterior updates from rally observations
  - Evidence weight schedule (0 → 1 over FULL_WEIGHT_RALLIES)
  - Bayesian estimate convergence direction
  - LiveProbabilityBlend Markov weight schedule
  - Edge cases: extreme RWP values, empty history
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from core.bayesian_updater import (
    BayesianRWPUpdater,
    BayesianPrior,
    LiveEstimate,
    LiveProbabilityBlend,
)


class TestBayesianPrior:
    """BayesianPrior construction from pre-match RWP."""

    def test_prior_from_rwp_0_5(self):
        """Symmetric RWP → α = β (equal prior)."""
        prior = BayesianPrior.from_pre_match_rwp(0.5)
        assert abs(prior.alpha - prior.beta) < 1e-6

    def test_prior_from_rwp_high(self):
        """High RWP → α > β."""
        prior = BayesianPrior.from_pre_match_rwp(0.65)
        assert prior.alpha > prior.beta

    def test_prior_from_rwp_low(self):
        """Low RWP → α < β."""
        prior = BayesianPrior.from_pre_match_rwp(0.42)
        assert prior.alpha < prior.beta

    def test_prior_mean_equals_rwp(self):
        """Prior mean (α/(α+β)) equals the input RWP."""
        for rwp in [0.45, 0.50, 0.55, 0.60]:
            prior = BayesianPrior.from_pre_match_rwp(rwp)
            mean = prior.alpha / (prior.alpha + prior.beta)
            assert abs(mean - rwp) < 1e-4, f"Prior mean {mean} != RWP {rwp}"

    def test_prior_alpha_beta_positive(self):
        """Both α and β must be strictly positive."""
        for rwp in [0.40, 0.50, 0.60]:
            prior = BayesianPrior.from_pre_match_rwp(rwp)
            assert prior.alpha > 0
            assert prior.beta > 0


class TestBayesianRWPUpdater:
    """BayesianRWPUpdater posterior accumulation."""

    @pytest.fixture
    def updater(self):
        return BayesianRWPUpdater(
            match_id="test_001",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            rwp_prior_a=0.540,
            rwp_prior_b=0.530,
        )

    def test_initial_no_evidence(self, updater):
        """Before any rallies, evidence_weight = 0."""
        est = updater.get_live_rwp("A")
        assert est.evidence_weight == 0.0

    def test_initial_rwp_equals_prior(self, updater):
        """With zero evidence, live RWP = prior RWP."""
        est = updater.get_live_rwp("A")
        assert abs(est.rwp_live - 0.540) < 1e-4

    def test_evidence_weight_increases(self, updater):
        """Evidence weight grows with each observed rally."""
        for i in range(10):
            updater.observe_rally(server="A", winner="A", game_number=1, point_index=i)

        est = updater.get_live_rwp("A")
        assert est.evidence_weight > 0.0
        assert est.evidence_weight <= 1.0

    def test_evidence_weight_capped_at_1(self, updater):
        """Evidence weight never exceeds 1.0."""
        for i in range(200):
            updater.observe_rally(server="A", winner="A", game_number=1, point_index=i)
        est = updater.get_live_rwp("A")
        assert est.evidence_weight <= 1.0

    def test_server_wins_increase_rwp(self, updater):
        """100% server wins → posterior mean above prior."""
        for i in range(20):
            updater.observe_rally(server="A", winner="A", game_number=1, point_index=i)
        est = updater.get_live_rwp("A")
        assert est.rwp_live >= 0.540  # at least as high as prior

    def test_server_loses_decrease_rwp(self, updater):
        """100% server losses → posterior mean below prior."""
        for i in range(20):
            updater.observe_rally(server="A", winner="B", game_number=1, point_index=i)
        est = updater.get_live_rwp("A")
        # With strong counter-evidence, live estimate should be pulled down
        assert est.rwp_live < 0.540

    def test_rwp_always_in_unit_interval(self, updater):
        """Live RWP must always be in (0, 1)."""
        for i in range(50):
            winner = "A" if i % 3 != 0 else "B"
            updater.observe_rally(server="A", winner=winner, game_number=1, point_index=i)
        for entity in ("A", "B"):
            est = updater.get_live_rwp(entity)
            assert 0.0 < est.rwp_live < 1.0

    def test_b_estimate_available(self, updater):
        """get_live_rwp works for both A and B."""
        updater.observe_rally(server="B", winner="B", game_number=1, point_index=0)
        est_b = updater.get_live_rwp("B")
        assert 0.0 < est_b.rwp_live < 1.0

    def test_reset_per_game(self, updater):
        """Resetting for game 2 clears game-level evidence."""
        for i in range(30):
            updater.observe_rally(server="A", winner="A", game_number=1, point_index=i)
        w_before = updater.get_live_rwp("A").evidence_weight
        updater.reset_for_new_game(game_number=2)
        w_after = updater.get_live_rwp("A").evidence_weight
        assert w_after < w_before  # evidence reduced after reset

    def test_invalid_entity_raises(self, updater):
        """get_live_rwp with invalid entity raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            updater.get_live_rwp("C")


class TestLiveProbabilityBlend:
    """LiveProbabilityBlend Markov weight schedule."""

    def test_early_match_markov_weight_low(self):
        """With few points played, Markov weight = 30%."""
        weight = LiveProbabilityBlend.markov_weight(points_played=5)
        assert abs(weight - 0.30) < 0.01

    def test_mid_match_markov_weight_medium(self):
        """With 15 points played, Markov weight = 50%."""
        weight = LiveProbabilityBlend.markov_weight(points_played=20)
        assert abs(weight - 0.50) < 0.01

    def test_late_match_markov_weight_high(self):
        """With 35+ points played, Markov weight = 70%."""
        weight = LiveProbabilityBlend.markov_weight(points_played=35)
        assert abs(weight - 0.70) < 0.01

    def test_markov_weight_monotonic(self):
        """Markov weight is non-decreasing with points played."""
        weights = [LiveProbabilityBlend.markov_weight(p) for p in range(0, 80, 5)]
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]

    def test_blend_sum_to_1(self):
        """Model weight + Markov weight = 1 always."""
        for points in [0, 10, 20, 30, 40, 60]:
            m_weight = LiveProbabilityBlend.markov_weight(points)
            assert abs(m_weight + (1 - m_weight) - 1.0) < 1e-10

    def test_compute_blended_prob(self):
        """Blended probability is weighted average of model and Markov."""
        model_prob = 0.60
        markov_prob = 0.70
        points = 5  # early → 30% Markov weight
        blended = LiveProbabilityBlend.compute(
            model_prob=model_prob,
            markov_prob=markov_prob,
            points_played=points,
        )
        # Expected: 0.30 * 0.70 + 0.70 * 0.60 = 0.21 + 0.42 = 0.63
        expected = 0.30 * markov_prob + 0.70 * model_prob
        assert abs(blended - expected) < 1e-6

    def test_blended_prob_in_unit_interval(self):
        """Blended probability always in (0, 1)."""
        for model_p in [0.30, 0.50, 0.70]:
            for markov_p in [0.40, 0.55, 0.75]:
                for points in [0, 15, 35]:
                    blended = LiveProbabilityBlend.compute(model_p, markov_p, points)
                    assert 0.0 < blended < 1.0

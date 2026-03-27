"""
model_core_agent.py
===================
ModelCoreAgent — Runs the live Markov DP from current match state.

Responsibilities:
  1. Extract current live state from MatchLiveState
  2. Get Bayesian live RWP from BayesianRWPUpdater
  3. Run BadmintonMarkovEngine.compute_match_probabilities() from current state
  4. Compute LiveProbabilityBlend (30/50/70% Markov schedule)
  5. Return LiveModelOutput for downstream agents

This is the most compute-intensive agent in the live chain (~20ms on p95).
All computation is deterministic — no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from config.badminton_config import Discipline
from core.markov_engine import BadmintonMarkovEngine, MatchProbabilities
from core.bayesian_updater import BayesianRWPUpdater, LiveRWPEstimate, LiveProbabilityBlend
from core.momentum_detector import MomentumDetector, MomentumSnapshot
from core.match_state import MatchLiveState, LiveStateSummary

logger = structlog.get_logger(__name__)


@dataclass
class LiveModelOutput:
    """Output from ModelCoreAgent — consumed by InferenceAgent and LivePricingEngine."""
    # Blended match-winner probabilities
    p_a_wins_markov: float
    p_a_wins_blend: float
    markov_weight: float

    # Live RWP
    rwp_a: float
    rwp_b: float

    # Full Markov probability matrix
    markov_probs: MatchProbabilities

    # Momentum
    momentum: MomentumSnapshot

    # Live RWP estimates (for uncertainty quantification)
    rwp_a_estimate: LiveRWPEstimate
    rwp_b_estimate: LiveRWPEstimate

    # State snapshot
    snap: LiveStateSummary


class ModelCoreAgent:
    """
    Runs Bayesian RWP update and Markov DP from current match state.

    Called once per score event in the live supervisor pipeline.
    Thread-safe: all state is in the passed-in objects, not in this class.
    """

    def __init__(self) -> None:
        self._markov = BadmintonMarkovEngine()

    def compute(
        self,
        live_state: MatchLiveState,
        bayesian_updater: BayesianRWPUpdater,
        momentum_detector: MomentumDetector,
        winner: str,
        pre_match_p_a: float,
    ) -> LiveModelOutput:
        """
        Run the full live model pipeline for one score event.

        Args:
            live_state:       Current MatchLiveState after the point was applied
            bayesian_updater: BayesianRWPUpdater for this match (accumulates evidence)
            momentum_detector: MomentumDetector for this match
            winner:            "A" or "B" — winner of the latest rally
            pre_match_p_a:     Pre-match ML model P(A wins) — Bayesian blend anchor

        Returns:
            LiveModelOutput with all computed probabilities.
        """
        # Update Bayesian RWP with this rally result
        bayesian_updater.observe_rally(
            server=live_state.server,
            rally_winner=winner,
        )

        # Get live RWP estimates
        rwp_a_est = bayesian_updater.get_live_rwp("A")
        rwp_b_est = bayesian_updater.get_live_rwp("B")
        rwp_a = rwp_a_est.rwp_live
        rwp_b = rwp_b_est.rwp_live

        # Get live state summary
        snap = LiveStateSummary(
            current_game=live_state.current_game,
            score_a=live_state.score_a,
            score_b=live_state.score_b,
            games_won_a=live_state.games_won_a,
            games_won_b=live_state.games_won_b,
            server=live_state.server,
            total_points_played=live_state.total_points_played,
        )

        # Run Markov from current state
        markov_probs = self._markov.compute_match_probabilities(
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            discipline=live_state.discipline,
            server_first_game=live_state.server,
            games_won_a=live_state.games_won_a,
            games_won_b=live_state.games_won_b,
            score_a=live_state.score_a,
            score_b=live_state.score_b,
            current_game=live_state.current_game,
        )

        # Blend Markov with pre-match model
        blend = LiveProbabilityBlend.compute(
            p_markov=markov_probs.p_a_wins_match,
            p_model=pre_match_p_a,
            total_points_played=live_state.total_points_played,
        )

        # Update momentum detector
        momentum = momentum_detector.add_point(
            winner=winner,
            server=live_state.server,
            game_number=live_state.current_game,
            score_a=live_state.score_a,
            score_b=live_state.score_b,
        )

        logger.debug(
            "model_core_computed",
            match_id=live_state.match_id,
            rwp_a=round(rwp_a, 4),
            rwp_b=round(rwp_b, 4),
            p_a_markov=round(markov_probs.p_a_wins_match, 4),
            p_a_blend=round(blend.p_a_wins_match_blend, 4),
            markov_weight=round(blend.markov_weight, 3),
            momentum_regime=momentum.regime.value,
        )

        return LiveModelOutput(
            p_a_wins_markov=markov_probs.p_a_wins_match,
            p_a_wins_blend=blend.p_a_wins_match_blend,
            markov_weight=blend.markov_weight,
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            markov_probs=markov_probs,
            momentum=momentum,
            rwp_a_estimate=rwp_a_est,
            rwp_b_estimate=rwp_b_est,
            snap=snap,
        )

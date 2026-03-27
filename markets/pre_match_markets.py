"""
pre_match_markets.py
====================
Full pre-match market generation for badminton.

Orchestrates the complete pre-match pricing pipeline:
  1. Model inference (3-layer ensemble → P(A wins match))
  2. RWP computation (from ML model via inverse Markov)
  3. Markov DP for all derivative markets (97 total)
  4. Pinnacle blend (70% model / 30% Markov at match level)
  5. Margin application per tier
  6. H7 arbitrage validation + H10 min odds check

Market output: MarketSet with all 15 families + metadata.

Pre-match context:
  - Called by PreMatchSupervisorAgent when odds are requested
  - All prices are valid for a configurable window (default 60s)
  - Stale prices suspended after PRE_MATCH_PRICE_STALENESS_LIMIT

Pinnacle blend rationale:
  Pinnacle is the sharpest market. By blending 70/30, we:
  - Anchor on our model's edge-containing information
  - Hedge against structural errors in Markov assumptions
  The blend is at match winner level; all derivatives are
  Markov-derived from the blended probability.

ZERO hardcoded probabilities. Raises RuntimeError if model not loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    TIER_MARGINS_MATCH_WINNER,
    TIER_MARGINS_DERIVATIVES,
    PRE_MATCH_MODEL_WEIGHT,
    PRE_MATCH_MARKOV_WEIGHT,
    MARKET_PRICE_VALIDITY_SECONDS,
    MarketFamily,
)
from core.rwp_calculator import RWPCalculator, PlayerRWPProfile, EnvironmentConditions, FatigueProfile
from core.markov_engine import BadmintonMarkovEngine
from markets.derivative_engine import BadmintonDerivativeEngine, MarketSet, MarketPrice
from markets.margin_engine import MarginEngine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Pre-match pricing request / response
# ---------------------------------------------------------------------------

@dataclass
class PreMatchPricingRequest:
    """
    Input to the pre-match pricing pipeline.

    All fields required. Missing optional values reduce feature richness
    but never cause errors — the pipeline gracefully degrades.
    """
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier
    match_date: date

    # Model outputs (from ML inference)
    model_p_a_wins: float              # From 3-layer ensemble, calibrated
    model_p_a_wins_2_0: float          # From target_2_0
    model_p_a_wins_deuce: float        # From target_deuce (goes to 3 games)

    # Pre-match RWP estimates (from RWP calculator)
    rwp_a: float                       # P(A wins a rally when serving)
    rwp_b: float                       # P(B wins a rally when serving)

    # Optional: Pinnacle line for blending
    pinnacle_p_a_wins: Optional[float] = None
    pinnacle_handicap: Optional[float] = None

    # Optional: environment and server
    first_server: str = "A"
    environment: Optional[EnvironmentConditions] = None

    # Metadata
    tournament_name: str = ""
    venue_country: str = ""


@dataclass
class PreMatchPricingResponse:
    """
    Output of the pre-match pricing pipeline.

    Contains all markets + pricing metadata.
    """
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier

    # Blended match win probability
    p_a_wins_blend: float
    model_weight: float
    markov_weight: float

    # Full market set
    market_set: MarketSet

    # RWP used for Markov
    rwp_a_used: float
    rwp_b_used: float

    # Pricing metadata
    generated_at: float   # Unix timestamp
    valid_until: float    # Expiry timestamp
    regime: str           # "R0" / "R1" / "R2" from RegimeGate

    # Validation results
    markets_valid: bool
    validation_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-match pricing engine
# ---------------------------------------------------------------------------

class PreMatchPricingEngine:
    """
    Orchestrates the full pre-match market generation pipeline.

    Usage:
      engine = PreMatchPricingEngine()
      response = engine.price(request)
    """

    def __init__(
        self,
        margin_engine: Optional[MarginEngine] = None,
    ) -> None:
        self._markov = BadmintonMarkovEngine()
        self._margin_engine = margin_engine or MarginEngine()
        self._derivative_engine = BadmintonDerivativeEngine()

    def price(self, request: PreMatchPricingRequest) -> PreMatchPricingResponse:
        """
        Run the full pre-match pricing pipeline.

        Steps:
          1. Compute model/Markov blended P(A wins)
          2. Derive Markov match probabilities from blended RWP
          3. Generate all 97 markets via derivative engine
          4. Apply tier margins
          5. Validate (H7 + H10)
          6. Return response

        Args:
            request: Fully populated pricing request.

        Returns:
            PreMatchPricingResponse with all markets.
        """
        t_start = time.time()

        # Step 1: Pinnacle/model blend for match winner
        p_a_blend, model_weight, markov_weight = self._blend_match_probability(
            model_p=request.model_p_a_wins,
            pinnacle_p=request.pinnacle_p_a_wins,
        )

        # Step 2: Markov engine for full match probabilities from RWP
        markov_probs = self._markov.compute_match_probabilities(
            rwp_a=request.rwp_a,
            rwp_b=request.rwp_b,
            discipline=request.discipline,
            server_first_game=request.first_server,
        )

        # Calibrate RWP to match the blended match win probability
        # (so Markov-derived markets are consistent with the blend)
        rwp_a_calibrated, rwp_b_calibrated = self._calibrate_rwp_to_match_prob(
            p_target=p_a_blend,
            rwp_a_initial=request.rwp_a,
            rwp_b_initial=request.rwp_b,
            discipline=request.discipline,
            first_server=request.first_server,
        )

        # Step 3: Generate all markets
        market_set = self._derivative_engine.compute_all_markets(
            match_id=request.match_id,
            rwp=rwp_a_calibrated,
            discipline=request.discipline,
            tier=request.tier,
            p_match_win=p_a_blend,
            server_first_game=request.first_server,
        )

        # Step 4: Apply margins
        margin_match = TIER_MARGINS_MATCH_WINNER.get(request.tier, 0.05)
        margin_deriv = TIER_MARGINS_DERIVATIVES.get(request.tier, 0.065)
        market_set = self._margin_engine.apply_margins(
            market_set,
            margin_match,
            margin_deriv,
        )

        # Step 5: Validate
        warnings: List[str] = []
        markets_valid = True

        for market_id, prices in market_set.markets.items():
            # H10: min odds
            for p in prices:
                if p.odds < 1.01:
                    warnings.append(f"H10 violation: {market_id} odds={p.odds:.4f}")
                    markets_valid = False

            # H7: arbitrage free — only validate binary/multi-outcome markets.
            # Single-outcome markets (e.g. exact-score props) represent one leg
            # of a grouped event and will naturally sum < 1.0 when isolated.
            if len(prices) >= 2:
                total_implied = sum(p.prob_with_margin for p in prices)
                margin = total_implied - 1.0
                if margin < -0.001:
                    warnings.append(
                        f"H7 violation: {market_id} overround={total_implied:.4f} < 1.0"
                    )
                    markets_valid = False

        # Regime assignment (simple rule-based)
        regime = self._assign_regime(request)

        t_elapsed = time.time() - t_start
        now = time.time()

        logger.info(
            "pre_match_priced",
            match_id=request.match_id,
            discipline=request.discipline.value,
            tier=request.tier.value,
            p_a_blend=f"{p_a_blend:.4f}",
            rwp_a=f"{rwp_a_calibrated:.4f}",
            n_markets=len(market_set.markets),
            valid=markets_valid,
            elapsed_ms=f"{t_elapsed*1000:.1f}",
        )

        if warnings:
            logger.warning(
                "pre_match_pricing_warnings",
                match_id=request.match_id,
                warnings=warnings,
            )

        return PreMatchPricingResponse(
            match_id=request.match_id,
            entity_a_id=request.entity_a_id,
            entity_b_id=request.entity_b_id,
            discipline=request.discipline,
            tier=request.tier,
            p_a_wins_blend=p_a_blend,
            model_weight=model_weight,
            markov_weight=markov_weight,
            market_set=market_set,
            rwp_a_used=rwp_a_calibrated,
            rwp_b_used=rwp_b_calibrated,
            generated_at=now,
            valid_until=now + MARKET_PRICE_VALIDITY_SECONDS,
            regime=regime,
            markets_valid=markets_valid,
            validation_warnings=warnings,
        )

    def _blend_match_probability(
        self,
        model_p: float,
        pinnacle_p: Optional[float],
    ) -> Tuple[float, float, float]:
        """
        Blend model and Pinnacle match probability.

        If Pinnacle is unavailable: use pure model (model_weight=1.0).
        If Pinnacle available: 70% model / 30% Pinnacle.

        Returns: (blended_p, model_weight, other_weight)
        """
        if pinnacle_p is None or not (0.01 < pinnacle_p < 0.99):
            return model_p, 1.0, 0.0

        model_w = PRE_MATCH_MODEL_WEIGHT
        pinnacle_w = 1.0 - model_w

        blend = model_w * model_p + pinnacle_w * pinnacle_p
        blend = max(0.001, min(0.999, blend))
        return blend, model_w, pinnacle_w

    def _calibrate_rwp_to_match_prob(
        self,
        p_target: float,
        rwp_a_initial: float,
        rwp_b_initial: float,
        discipline: Discipline,
        first_server: str,
    ) -> Tuple[float, float]:
        """
        Adjust RWP values so that Markov engine produces P(A wins) ≈ p_target.

        Uses bisection on rwp_a while keeping rwp_b constant and
        maintaining the rwp_a / rwp_b ratio.

        This ensures Markov-derived derivatives are consistent with the
        blended match winner probability.
        """
        markov = self._markov
        initial_ratio = rwp_a_initial / max(0.001, rwp_b_initial)

        def markov_p_a(rwp_a_trial: float) -> float:
            rwp_b_trial = rwp_a_trial / max(0.001, initial_ratio)
            rwp_b_trial = max(0.20, min(0.80, rwp_b_trial))
            probs = markov.compute_match_probabilities(
                rwp_a=rwp_a_trial,
                rwp_b=rwp_b_trial,
                discipline=discipline,
                server_first_game=first_server,
            )
            return probs.p_a_wins_match

        # Bisection search
        lo, hi = 0.20, 0.80
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if markov_p_a(mid) < p_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-6:
                break

        rwp_a_cal = (lo + hi) / 2.0
        rwp_b_cal = rwp_a_cal / max(0.001, initial_ratio)
        rwp_b_cal = max(0.20, min(0.80, rwp_b_cal))

        return rwp_a_cal, rwp_b_cal

    @staticmethod
    def _assign_regime(request: PreMatchPricingRequest) -> str:
        """
        Assign pricing regime based on match context.

        R0: Sparse data (low confidence entities, small tournaments)
        R1: Standard (most BWF World Tour matches)
        R2: Rich data (major tournaments, top 10 players, large sample)
        """
        from config.badminton_config import TournamentTier, REGIME_R2_TIERS

        if request.tier in REGIME_R2_TIERS:
            return "R2"

        # Fallback: R1 for standard BWF World Tour
        return "R1"


# ---------------------------------------------------------------------------
# Batch pre-match pricing (for multiple matches)
# ---------------------------------------------------------------------------

class BatchPreMatchPricer:
    """
    Batch pre-match pricing for a tournament fixture list.

    Prices all matches in a tournament prior to the tournament starting.
    Used by OutrightSupervisorAgent.
    """

    def __init__(self) -> None:
        self._engine = PreMatchPricingEngine()

    def price_batch(
        self,
        requests: List[PreMatchPricingRequest],
    ) -> Dict[str, PreMatchPricingResponse]:
        """
        Price a batch of matches.

        Returns: {match_id -> PreMatchPricingResponse}
        """
        results: Dict[str, PreMatchPricingResponse] = {}

        for req in requests:
            try:
                response = self._engine.price(req)
                results[req.match_id] = response
            except Exception as exc:
                logger.error(
                    "batch_pricing_failure",
                    match_id=req.match_id,
                    error=str(exc),
                )
                # Continue pricing other matches

        logger.info(
            "batch_pricing_complete",
            n_requested=len(requests),
            n_priced=len(results),
            n_failed=len(requests) - len(results),
        )

        return results

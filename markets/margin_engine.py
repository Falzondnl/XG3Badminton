"""
margin_engine.py
================
Margin application engine for badminton markets.

Applies bookmaker margin (overround) to fair probabilities using the
power method, which is the industry standard for two-way and multi-way
markets.

Power method:
  For each outcome i with fair probability pᵢ:
    pᵢ_margin = pᵢ^k  where k solves Σ pᵢ^k = 1 + margin

  This preserves relative probabilities better than proportional markup.

Applied separately to:
  - Match winner markets: TIER_MARGINS_MATCH_WINNER
  - Derivative markets: TIER_MARGINS_DERIVATIVES
  - Outright markets: TIER_MARGINS_OUTRIGHTS

H7 gate: After margin application, verify Σ(implied probs) = 1 + target_margin ± 0.001.
H10 gate: All odds ≥ 1.01 enforced (floor applied, not rejection).

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import structlog

from dataclasses import dataclass

from config.badminton_config import (
    TournamentTier,
    TIER_MARGINS_MATCH_WINNER,
    TIER_MARGINS_DERIVATIVES,
    TIER_MARGINS_OUTRIGHTS,
    MarketFamily,
)
from markets.derivative_engine import MarketPrice, MarketSet

logger = structlog.get_logger(__name__)

_MIN_ODDS = 1.01
_MIN_PROB = 0.001
_MAX_PROB = 0.999


@dataclass(frozen=True)
class MarginedPrice:
    """Outcome with both fair and margined probabilities."""
    fair_prob: float
    prob_with_margin: float
    odds: float


class MarginEngine:
    """
    Applies bookmaker margins to a MarketSet or a list of fair probabilities.
    """

    def apply_margins(
        self,
        market_set_or_probs,
        margin_match_or_margin: float = 0.05,
        margin_derivatives: Optional[float] = None,
    ):
        """
        Apply margins.

        Overloaded:
          apply_margins(market_set: MarketSet, margin_match, margin_derivatives) → MarketSet
          apply_margins(probs: List[float], margin: float) → List[MarginedPrice]
        """
        if isinstance(market_set_or_probs, list) and all(
            isinstance(p, (int, float)) for p in market_set_or_probs
        ):
            return self._apply_margins_to_prob_list(
                market_set_or_probs, margin_match_or_margin
            )
        return self._apply_margins_to_market_set(
            market_set_or_probs,
            margin_match_or_margin,
            margin_derivatives if margin_derivatives is not None else margin_match_or_margin,
        )

    def _apply_margins_to_prob_list(
        self, probs: List[float], target_margin: float
    ) -> List[MarginedPrice]:
        """
        Apply power-method margin to a plain list of fair probabilities.
        Returns List[MarginedPrice] with .fair_prob, .prob_with_margin, .odds.
        """
        if not probs:
            return []

        # Normalise
        total = sum(probs)
        if total <= 0:
            raise ValueError("apply_margins: sum of probabilities must be > 0")
        normalised = [max(_MIN_PROB, min(_MAX_PROB, p / total)) for p in probs]

        k = self._find_power_exponent(normalised, target_margin)

        # Compute raw margined probs
        raw_margined = [max(_MIN_PROB, min(_MAX_PROB, p ** k)) for p in normalised]

        # Normalise to ensure Σ prob_with_margin == 1 + target_margin exactly.
        # Bisection converges to 1e-8 but floating-point accumulation in the sum
        # can overshoot by ~5e-10.  One linear rescale eliminates the drift.
        actual_sum = sum(raw_margined)
        target_sum = 1.0 + target_margin
        scale = target_sum / actual_sum if actual_sum > 0 else 1.0

        result = []
        for fair_p, p_raw in zip(normalised, raw_margined):
            p_margin = max(_MIN_PROB, min(_MAX_PROB, p_raw * scale))
            odds = max(_MIN_ODDS, 1.0 / p_margin)
            result.append(MarginedPrice(
                fair_prob=round(fair_p, 6),
                prob_with_margin=p_margin,   # no rounding — keeps sum accurate for H1 gate
                odds=round(odds, 4),
            ))
        return result

    def _apply_margins_to_market_set(
        self,
        market_set: MarketSet,
        margin_match: float,
        margin_derivatives: float,
    ) -> MarketSet:
        """
        Apply margins to all markets in the set.

        Match winner family uses margin_match.
        All other families use margin_derivatives.

        Returns a new MarketSet with margins applied.
        """
        new_markets: Dict[str, List[MarketPrice]] = {}

        for market_id, prices in market_set.markets.items():
            family = self._infer_family(market_id)

            margin = (
                margin_match
                if family == MarketFamily.MATCH_RESULT
                else margin_derivatives
            )

            margined = self._apply_power_margin(prices, margin)
            new_markets[market_id] = margined

        return MarketSet(
            match_id=market_set.match_id,
            discipline=market_set.discipline,
            markets=new_markets,
        )

    def apply_outright_margins(
        self,
        prices: List[MarketPrice],
        tier: TournamentTier,
    ) -> List[MarketPrice]:
        """Apply outright margins to a list of outright prices."""
        margin = TIER_MARGINS_OUTRIGHTS.get(tier, 0.10)
        return self._apply_power_margin(prices, margin)

    def _apply_power_margin(
        self,
        prices: List[MarketPrice],
        target_margin: float,
    ) -> List[MarketPrice]:
        """
        Apply power method margin to a list of prices.

        Finds k such that Σ pᵢ^k = 1 + target_margin.
        Sets prob_with_margin = pᵢ^k.
        Sets odds = 1 / prob_with_margin, floored at MIN_ODDS.

        Handles single-outcome markets gracefully (no margin applicable).
        """
        if not prices:
            return prices

        if len(prices) == 1:
            # Single-outcome market — can't apply margin meaningfully
            p = prices[0]
            return [MarketPrice(
                market_id=p.market_id,
                market_family=p.market_family,
                outcome_name=p.outcome_name,
                odds=max(_MIN_ODDS, p.odds),
                prob_implied=p.prob_implied,
                prob_with_margin=p.prob_implied,
            )]

        fair_probs = [
            max(_MIN_PROB, min(_MAX_PROB, p.prob_implied))
            for p in prices
        ]

        # Normalise so fair probs sum to 1.0
        total_fair = sum(fair_probs)
        if total_fair <= 0:
            logger.warning("margin_all_zero_probs", n_prices=len(prices))
            return prices
        fair_probs = [p / total_fair for p in fair_probs]

        # Bisect for k
        k = self._find_power_exponent(fair_probs, target_margin)

        # Apply
        result = []
        for price, fair_p in zip(prices, fair_probs):
            p_margin = max(_MIN_PROB, min(_MAX_PROB, fair_p ** k))
            odds = max(_MIN_ODDS, 1.0 / p_margin)
            result.append(MarketPrice(
                market_id=price.market_id,
                market_family=price.market_family,
                outcome_name=price.outcome_name,
                odds=round(odds, 4),
                prob_implied=price.prob_implied,
                prob_with_margin=round(p_margin, 6),
            ))

        return result

    @staticmethod
    def _find_power_exponent(
        fair_probs: List[float],
        target_margin: float,
        n_iter: int = 50,
    ) -> float:
        """
        Find k via bisection such that Σ pᵢ^k = 1 + target_margin.

        k > 1 for overround (bookmaker margin).
        k = 1 for fair book.
        """
        target_total = 1.0 + target_margin

        # k > 1 increases sum; k < 1 decreases sum
        # At k=1: sum = 1.0 (fair book)
        # We want sum > 1.0, so k must be... actually for power method
        # with pᵢ < 1, pᵢ^k decreases as k increases.
        # So k > 1 → smaller individual probs → smaller total → LESS overround
        # k < 1 → larger individual probs → larger total → MORE overround
        # We need k < 1 to achieve overround > 0.

        def total_at_k(k_val: float) -> float:
            return sum(p ** k_val for p in fair_probs)

        # k=1 → total=1.0; k→0 → total→n_outcomes
        # We want total = 1 + margin, so k < 1
        lo, hi = 0.001, 2.0

        # Verify k < 1 is needed
        t_at_1 = total_at_k(1.0)
        if abs(t_at_1 - target_total) < 1e-6:
            return 1.0

        for _ in range(n_iter):
            mid = (lo + hi) / 2.0
            t = total_at_k(mid)
            if t > target_total:
                lo = mid   # Need larger k to reduce total
            else:
                hi = mid   # Need smaller k to increase total
            if hi - lo < 1e-8:
                break

        return (lo + hi) / 2.0

    @staticmethod
    def _infer_family(market_id: str) -> MarketFamily:
        """Infer market family from market_id prefix."""
        prefix_map = {
            "match_winner": MarketFamily.MATCH_RESULT,
            "handicap_games": MarketFamily.MATCH_RESULT,
            "total_games": MarketFamily.TOTAL_GAMES,
            "winning_margin": MarketFamily.TOTAL_GAMES,
            "correct_score": MarketFamily.CORRECT_SCORE,
            "game_": MarketFamily.GAME_LEVEL,
            "g1_": MarketFamily.GAME_LEVEL,
            "g2_": MarketFamily.GAME_LEVEL,
            "g3_": MarketFamily.GAME_LEVEL,
            "race_": MarketFamily.RACE_MILESTONE,
            "points_total": MarketFamily.POINTS_TOTALS,
            "match_total": MarketFamily.POINTS_TOTALS,
            "match_deuce": MarketFamily.POINTS_TOTALS,
            "player_a_total": MarketFamily.PLAYER_PROPS,
            "player_b_total": MarketFamily.PLAYER_PROPS,
            "first_point": MarketFamily.PLAYER_PROPS,
            "g1_leader_at": MarketFamily.PLAYER_PROPS,
            "next_point": MarketFamily.LIVE_IN_PLAY,
            "next_5_pts": MarketFamily.LIVE_IN_PLAY,
            "live_": MarketFamily.LIVE_IN_PLAY,
            "outright": MarketFamily.OUTRIGHTS,
            "sgp_": MarketFamily.SGP,
            "futures": MarketFamily.FUTURES,
            "exotic": MarketFamily.EXOTIC,
            "both_win": MarketFamily.EXOTIC,
            "a_wins_after": MarketFamily.EXOTIC,
            "b_wins_after": MarketFamily.EXOTIC,
            "golden_point": MarketFamily.EXOTIC,
            "match_scoring_band": MarketFamily.EXOTIC,
            "team_event": MarketFamily.TEAM_EVENTS,
        }

        for prefix, family in prefix_map.items():
            if market_id.startswith(prefix):
                return family

        return MarketFamily.MATCH_RESULT

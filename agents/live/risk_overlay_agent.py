"""
risk_overlay_agent.py
=====================
RiskOverlayAgent — Live risk checks on generated odds.

Responsibilities:
  1. H7 gate: verify no live market is arbitrage-open
  2. H10 gate: verify no odds below 1.01
  3. H1 gate: verify overround ≥ 4% on match winner
  4. Live continuity check (H7-live): max 40% jump per update
  5. Flag markets that moved > JUMP_THRESHOLD for manual review
  6. Reject (zero click scale) markets that fail hard gates

This runs AFTER LivePricingEngine generates the odds.
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from config.badminton_config import MIN_ODDS
from markets.derivative_engine import MarketPrice

logger = structlog.get_logger(__name__)

_MAX_ODDS_JUMP_PCT = 0.40    # H7-live: max 40% move per rally
_MIN_OVERROUND = 0.04        # H1: at least 4% margin
_ARBITRAGE_TOLERANCE = 0.001


class RiskOverlayAgent:
    """
    Post-pricing risk validation for live markets.

    Stateful: tracks previous odds to compute jumps.
    """

    def __init__(self, match_id: str) -> None:
        self._match_id = match_id
        self._prev_probs: Dict[str, Dict[str, float]] = {}  # market_id → {outcome → prob}

    def validate(
        self,
        markets: Dict[str, List[MarketPrice]],
        click_scales: Dict[str, float],
    ) -> tuple[Dict[str, List[MarketPrice]], Dict[str, float], List[str]]:
        """
        Validate live markets. Mutates click_scales for failing markets.

        Args:
            markets:      market_id → [MarketPrice]
            click_scales: market_id → scale (mutated in place)

        Returns:
            (validated_markets, updated_click_scales, violations)
        """
        violations: List[str] = []

        for market_id, prices in markets.items():
            if not prices:
                continue

            # H10: min odds
            for mp in prices:
                if mp.odds < MIN_ODDS:
                    violations.append(f"H10: {market_id}/{mp.outcome_name} odds={mp.odds:.4f}")
                    click_scales[market_id] = 0.0

            # H7: arbitrage
            implied_sum = sum(1.0 / mp.odds for mp in prices if mp.odds > 0)
            if implied_sum < 1.0 - _ARBITRAGE_TOLERANCE:
                violations.append(
                    f"H7: {market_id} arb-open implied_sum={implied_sum:.4f}"
                )
                click_scales[market_id] = 0.0

            # H1: overround ≥ 4%
            if "match_winner" in market_id:
                overround = implied_sum - 1.0
                if overround < _MIN_OVERROUND:
                    violations.append(
                        f"H1: {market_id} overround={overround:.4f} < 4%"
                    )

            # H7-live: max 40% jump per rally
            prev = self._prev_probs.get(market_id, {})
            for mp in prices:
                prev_p = prev.get(mp.outcome_name)
                if prev_p is not None and prev_p > 0:
                    jump = abs(mp.prob_implied - prev_p) / prev_p
                    if jump > _MAX_ODDS_JUMP_PCT:
                        violations.append(
                            f"H7-live: {market_id}/{mp.outcome_name} "
                            f"jump={jump:.2%} > 40% ({prev_p:.4f}→{mp.prob_implied:.4f})"
                        )
                        # Reduce scale but don't zero out — jump may be legitimate
                        if click_scales.get(market_id, 1.0) > 0.3:
                            click_scales[market_id] = 0.30

            # Update prev_probs
            self._prev_probs[market_id] = {mp.outcome_name: mp.prob_implied for mp in prices}

        if violations:
            logger.warning(
                "risk_overlay_violations",
                match_id=self._match_id,
                n_violations=len(violations),
                violations=violations[:5],
            )

        return markets, click_scales, violations

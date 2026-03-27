"""
automover_agent.py
==================
AutomoverAgent — Base price calculation and initial odds setting.

First agent in the iMOVE chain. Responsibilities:
  1. Copy raw_prices into adjusted_prices (starting point)
  2. Apply Pinnacle reference blend if available
  3. Apply tier-specific margin floors (H1 gate: overround ≥ 4%)
  4. Set initial click scales based on tier
"""

from __future__ import annotations

from typing import List

import structlog

from config.badminton_config import (
    TournamentTier,
    TIER_MARGINS_MATCH_WINNER,
    TIER_MARGINS_DERIVATIVES,
    MIN_ODDS,
)
from markets.derivative_engine import MarketPrice
from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

# Default click scales by tier (max liability per click per market)
_DEFAULT_CLICK_SCALES: dict[str, float] = {
    "SUPER_1000": 1.0,
    "SUPER_750": 1.0,
    "SUPER_500": 0.85,
    "SUPER_300": 0.70,
    "BWF_1000": 0.70,
    "INTERNATIONAL_SERIES": 0.50,
    "INTERNATIONAL_CHALLENGE": 0.40,
    "TEAM_EVENT": 0.80,
}


class AutomoverAgent(BaseTradingAgent):
    """
    Base price calculation — first agent in the trading chain.

    Copies raw_prices → adjusted_prices and applies Pinnacle blend
    if reference prices are available in context.
    """

    @property
    def agent_name(self) -> str:
        return "automover"

    def process(self, context: TradingContext) -> TradingAgentResult:
        if context.prices_locked:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="prices locked — skip")

        # Step 1: Copy raw prices to adjusted (base state)
        context.adjusted_prices = {
            market_id: list(prices)
            for market_id, prices in context.raw_prices.items()
        }

        # Step 2: Blend with Pinnacle reference if available
        n_blended = 0
        if context.reference_prices:
            for market_id, prices in context.adjusted_prices.items():
                ref_prob = context.reference_prices.get(market_id)
                if ref_prob is None:
                    continue

                # Blend: 40% raw model, 60% Pinnacle reference for match winner
                # 70% raw model, 30% Pinnacle for derivatives
                is_match_winner = "match_winner" in market_id
                blend_model = 0.40 if is_match_winner else 0.70
                blend_ref = 1.0 - blend_model

                new_prices: List[MarketPrice] = []
                for mp in prices:
                    blended_implied = blend_model * mp.prob_implied + blend_ref * ref_prob
                    blended_implied = max(0.005, min(0.995, blended_implied))
                    # Rebuild price with blended probability
                    new_prices.append(MarketPrice(
                        market_id=mp.market_id,
                        market_family=mp.market_family,
                        outcome_name=mp.outcome_name,
                        odds=mp.odds,
                        prob_implied=blended_implied,
                        prob_with_margin=mp.prob_with_margin,
                    ))
                context.adjusted_prices[market_id] = new_prices
                n_blended += 1

        # Step 3: Apply H1 margin floor — overround must be ≥ 4%
        n_floored = 0
        try:
            tier = TournamentTier(context.tier)
        except ValueError:
            tier = TournamentTier.INTERNATIONAL_SERIES

        margin_floor = TIER_MARGINS_MATCH_WINNER.get(tier, 0.05)
        margin_floor = max(0.04, margin_floor)  # H1: never below 4%

        for market_id, prices in context.adjusted_prices.items():
            if len(prices) < 2:
                continue
            total_implied = sum(mp.prob_implied for mp in prices)
            current_overround = total_implied - 1.0
            if current_overround < 0.04:
                # Scale up implied probs uniformly to reach minimum overround
                scale = (1.0 + 0.04) / total_implied
                context.adjusted_prices[market_id] = [
                    MarketPrice(
                        market_id=mp.market_id,
                        market_family=mp.market_family,
                        outcome_name=mp.outcome_name,
                        odds=max(MIN_ODDS, 1.0 / (mp.prob_implied * scale)),
                        prob_implied=mp.prob_implied,
                        prob_with_margin=mp.prob_implied * scale,
                    )
                    for mp in prices
                ]
                n_floored += 1

        # Step 4: Set initial click scales
        base_scale = _DEFAULT_CLICK_SCALES.get(context.tier, 0.60)
        for market_id in context.adjusted_prices:
            if market_id not in context.click_scales:
                context.click_scales[market_id] = base_scale

        msg = (
            f"base prices set — {len(context.adjusted_prices)} markets, "
            f"{n_blended} Pinnacle-blended, {n_floored} margin-floored"
        )
        self._log(context, msg)

        return TradingAgentResult(
            agent_name=self.agent_name,
            success=True,
            context_mutated=True,
            notes=msg,
        )

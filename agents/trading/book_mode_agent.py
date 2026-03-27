"""
book_mode_agent.py
==================
BookModeAgent — Position management and book balancing.

Manages the book mode based on current positions:
  - balanced:    Roughly equal exposure both sides
  - overbroke:   Over-round (profitable regardless of outcome)
  - underbroke:  Under-round (arb opportunity exists — dangerous)
  - flat:        Near-zero book (very low volume matched)

Adjustments:
  - Moves prices toward balanced book where possible
  - If underbroke: tighten margins urgently
  - If overbroke: can relax margins (better customer prices)
"""

from __future__ import annotations

import structlog

from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

_UNDERBROKE_THRESHOLD = 0.99   # implied sum < 0.99 = underbroke
_OVERBROKE_THRESHOLD = 1.20    # implied sum > 1.20 = very overbroke


class BookModeAgent(BaseTradingAgent):
    """
    Determines book mode and applies margin adjustments for balance.
    """

    @property
    def agent_name(self) -> str:
        return "book_mode"

    def process(self, context: TradingContext) -> TradingAgentResult:
        if context.prices_locked or context.suspend_all:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="locked/suspended — skip")

        # Classify book mode from match winner market
        mw = context.adjusted_prices.get("match_winner", [])
        if not mw:
            return TradingAgentResult(
                agent_name=self.agent_name, success=True,
                notes="no match_winner market — cannot determine book mode"
            )

        implied_sum = sum(1.0 / mp.odds for mp in mw if mp.odds > 0)

        if implied_sum < _UNDERBROKE_THRESHOLD:
            context.book_mode = "underbroke"
            # Emergency: force minimum 4% overround on all markets
            # by capping all odds to 1/(prob_with_margin)
            n_fixed = 0
            for market_id, prices in context.adjusted_prices.items():
                for i, mp in enumerate(prices):
                    if mp.odds > 1.0 / max(0.001, mp.prob_with_margin):
                        # Odds too generous — cap at margined level
                        from config.badminton_config import MIN_ODDS
                        capped_odds = max(MIN_ODDS, 1.0 / max(0.001, mp.prob_with_margin))
                        from markets.derivative_engine import MarketPrice
                        prices[i] = MarketPrice(
                            market_id=mp.market_id,
                            market_family=mp.market_family,
                            outcome_name=mp.outcome_name,
                            odds=capped_odds,
                            prob_implied=mp.prob_implied,
                            prob_with_margin=mp.prob_with_margin,
                        )
                        n_fixed += 1
            self._log(context, f"UNDERBROKE (sum={implied_sum:.4f}) — fixed {n_fixed} odds")
            logger.warning(
                "book_mode_underbroke",
                match_id=context.match_id,
                implied_sum=implied_sum,
                n_fixed=n_fixed,
            )

        elif implied_sum > _OVERBROKE_THRESHOLD:
            context.book_mode = "overbroke"
            self._log(context, f"overbroke (sum={implied_sum:.4f}) — no action needed")

        elif abs(implied_sum - 1.05) < 0.03:
            context.book_mode = "balanced"

        else:
            context.book_mode = "balanced"

        logger.debug(
            "book_mode_set",
            match_id=context.match_id,
            book_mode=context.book_mode,
            implied_sum=round(implied_sum, 4),
        )

        return TradingAgentResult(
            agent_name=self.agent_name,
            success=True,
            context_mutated=context.book_mode == "underbroke",
            notes=f"book_mode={context.book_mode}, implied_sum={implied_sum:.4f}",
        )

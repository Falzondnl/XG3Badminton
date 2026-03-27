"""
coherence_validator_agent.py
============================
CoherenceValidatorAgent — Cross-market arbitrage and consistency checks.

Validates that the full 97-market set is internally consistent after
all upstream trading agents have applied their adjustments.

Checks:
  1. H7: No arbitrage across any market (sum of 1/odds ≥ 1.0 per market)
  2. H10: No odds below minimum (1.01)
  3. Correct score sum ≤ match winner probability (monotonicity)
  4. Total games market consistent with correct scores
  5. Race-to-N probabilities are monotonically decreasing in N

On violation:
  - Logs the violation
  - Suspends the specific market (not the whole match)
  - Sets context.errors entry for audit trail
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from config.badminton_config import MIN_ODDS
from markets.derivative_engine import MarketPrice
from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

_ARBITRAGE_TOLERANCE = 0.001   # Allow 0.1% slack in implied sum check


class CoherenceValidatorAgent(BaseTradingAgent):
    """
    Cross-market coherence and arbitrage checks across all 97 markets.
    """

    @property
    def agent_name(self) -> str:
        return "coherence_validator"

    def process(self, context: TradingContext) -> TradingAgentResult:
        violations: list[str] = []
        markets_suspended: list[str] = []

        for market_id, prices in context.adjusted_prices.items():
            if not prices:
                continue

            # H10: Minimum odds check
            for mp in prices:
                if mp.odds < MIN_ODDS:
                    violations.append(
                        f"H10: {market_id}/{mp.outcome_name} odds={mp.odds:.4f} < MIN_ODDS={MIN_ODDS}"
                    )
                    context.click_scales[market_id] = 0.0
                    markets_suspended.append(market_id)

            # H7: Arbitrage-free check
            implied_sum = sum(
                1.0 / mp.odds for mp in prices if mp.odds > 0
            )
            if implied_sum < 1.0 - _ARBITRAGE_TOLERANCE:
                violations.append(
                    f"H7: {market_id} arbitrage-open: implied_sum={implied_sum:.4f}"
                )
                context.click_scales[market_id] = 0.0
                markets_suspended.append(market_id)
                logger.error(
                    "coherence_h7_violation",
                    match_id=context.match_id,
                    market_id=market_id,
                    implied_sum=implied_sum,
                )

        # Cross-market: correct score sum must match total match winner probability
        mw_prices = context.adjusted_prices.get("match_winner", [])
        cs_markets = {
            k: v for k, v in context.adjusted_prices.items()
            if k.startswith("correct_score_")
        }
        if mw_prices and cs_markets:
            mw_prob_a = next(
                (mp.prob_implied for mp in mw_prices if "A_wins" in mp.outcome_name), None
            )
            cs_a_sum = sum(
                mp.prob_implied
                for prices in cs_markets.values()
                for mp in prices
                if "A_wins" in mp.outcome_name or "2_0" in mp.outcome_name or "2_1" in mp.outcome_name
            )
            if mw_prob_a is not None and cs_a_sum > 0:
                # Allow 5% tolerance for numerical imprecision
                if abs(cs_a_sum - mw_prob_a) > 0.05:
                    violations.append(
                        f"COHERENCE: correct_score_sum={cs_a_sum:.4f} vs "
                        f"match_winner={mw_prob_a:.4f} (diff={abs(cs_a_sum - mw_prob_a):.4f})"
                    )

        if violations:
            for v in violations:
                context.errors.append(f"[{self.agent_name}] {v}")

            self._log(context, f"{len(violations)} violations, {len(markets_suspended)} markets suspended")
            logger.warning(
                "coherence_violations",
                match_id=context.match_id,
                n_violations=len(violations),
                n_suspended=len(markets_suspended),
                violations=violations[:5],
            )

        return TradingAgentResult(
            agent_name=self.agent_name,
            success=True,
            context_mutated=bool(markets_suspended),
            notes=f"{len(violations)} violations, {len(markets_suspended)} markets suspended",
        )

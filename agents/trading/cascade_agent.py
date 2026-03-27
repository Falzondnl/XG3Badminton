"""
cascade_agent.py
================
CascadeAgent — Propagates match-winner price movements across all 97 markets.

When the match-winner probability changes, ALL derivative markets must be
repriced to remain consistent. This agent implements the cascade:

  match_winner price change → recompute:
    1. Total games O/U (directly derived from p_a_wins via Markov)
    2. Correct score markets
    3. Game handicaps
    4. All game-level markets (game 1/2/3 winner)
    5. Points totals (dependent on game structure)
    6. Player props (dependent on point share)
    7. Exotic markets (comeback, golden point)

If the match-winner price has NOT changed significantly (< REPRICE_THRESHOLD),
most derivative markets can retain stale prices — this avoids unnecessary
Markov DP calls and keeps p50 latency < 50ms.

REPRICE_THRESHOLD: 0.005 (0.5% change in match-winner implied probability)
"""

from __future__ import annotations

from typing import Optional

import structlog

from config.badminton_config import Discipline, TournamentTier
from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

_REPRICE_THRESHOLD = 0.005   # 0.5% change triggers full cascade
_MAJOR_MOVE_THRESHOLD = 0.02  # 2% change = major move, log prominently


class CascadeAgent(BaseTradingAgent):
    """
    Detects match-winner price movements and triggers full market cascade.

    Maintains previous match-winner probability to compute delta.
    When delta exceeds threshold, calls derivative engine for full reprice.
    """

    def __init__(self) -> None:
        self._prev_p_a: dict[str, float] = {}  # match_id → previous p_a_wins

    @property
    def agent_name(self) -> str:
        return "cascade"

    def process(self, context: TradingContext) -> TradingAgentResult:
        if context.prices_locked or context.suspend_all:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="locked/suspended — skip")

        mw_prices = context.adjusted_prices.get("match_winner", [])
        if not mw_prices:
            return TradingAgentResult(
                agent_name=self.agent_name, success=True,
                notes="no match_winner — cascade skipped"
            )

        # Get current p_a from adjusted prices
        p_a_current: Optional[float] = None
        for mp in mw_prices:
            if "A_wins" in mp.outcome_name:
                p_a_current = mp.prob_implied
                break

        if p_a_current is None:
            return TradingAgentResult(
                agent_name=self.agent_name, success=True,
                notes="cannot determine p_a from match_winner prices"
            )

        prev_p_a = self._prev_p_a.get(context.match_id)
        delta = abs(p_a_current - prev_p_a) if prev_p_a is not None else 1.0

        if delta < _REPRICE_THRESHOLD and prev_p_a is not None:
            # No significant price movement — derivatives retain stale prices
            self._log(
                context,
                f"no cascade needed: delta={delta:.4f} < threshold={_REPRICE_THRESHOLD}"
            )
            return TradingAgentResult(
                agent_name=self.agent_name, success=True,
                context_mutated=False,
                notes=f"delta={delta:.4f} below threshold — no cascade",
            )

        # Price movement exceeds threshold — trigger full cascade
        if delta >= _MAJOR_MOVE_THRESHOLD:
            logger.warning(
                "cascade_major_price_move",
                match_id=context.match_id,
                delta=round(delta, 4),
                prev_p_a=round(prev_p_a, 4) if prev_p_a else None,
                p_a_current=round(p_a_current, 4),
            )

        try:
            self._cascade_derivatives(context, p_a_current)
            self._prev_p_a[context.match_id] = p_a_current

            msg = (
                f"cascade triggered: delta={delta:.4f}, "
                f"p_a={prev_p_a:.4f}→{p_a_current:.4f} "
                f"({len(context.adjusted_prices)} markets repriced)"
            )
            self._log(context, msg)

            return TradingAgentResult(
                agent_name=self.agent_name,
                success=True,
                context_mutated=True,
                notes=msg,
            )

        except Exception as exc:
            context.errors.append(f"[{self.agent_name}] cascade failed: {exc}")
            logger.error(
                "cascade_error",
                match_id=context.match_id,
                error=str(exc),
            )
            return TradingAgentResult(
                agent_name=self.agent_name,
                success=False,
                error=str(exc),
                notes="cascade failed — retaining stale prices",
            )

    def _cascade_derivatives(
        self,
        context: TradingContext,
        p_a_current: float,
    ) -> None:
        """
        Reprice all derivative markets from updated match-winner probability.

        Uses the derivative engine to regenerate markets from the new
        implied probability, then merges into context.adjusted_prices.
        """
        from core.rwp_calculator import RWPCalculator, RWPEstimate
        from core.markov_engine import BadmintonMarkovEngine
        from markets.derivative_engine import BadmintonDerivativeEngine

        try:
            discipline = Discipline(context.discipline)
            tier = TournamentTier(context.tier)
        except ValueError as e:
            raise ValueError(f"Invalid discipline/tier in context: {e}")

        markov = BadmintonMarkovEngine()

        # Back-solve RWP from p_a_current using bisection
        # P(A wins) = Markov(rwp_a, rwp_b=1-rwp_a) → solve for rwp_a
        lo, hi = 0.40, 0.75
        for _ in range(30):
            mid = (lo + hi) / 2.0
            p_test = markov.p_win_match_from_rwp(mid, discipline=discipline)
            if p_test < p_a_current:
                lo = mid
            else:
                hi = mid
        rwp_a = (lo + hi) / 2.0
        rwp_b = 1.0 - rwp_a

        rwp_estimate = RWPEstimate(
            rwp_a_as_server=rwp_a,
            rwp_b_as_server=rwp_b,
            discipline=discipline,
        )

        engine = BadmintonDerivativeEngine()
        new_market_set = engine.compute_all_markets(
            match_id=context.match_id,
            rwp=rwp_estimate,
            discipline=discipline,
            tier=tier,
            p_match_win=p_a_current,
        )

        # Merge derivative markets into adjusted_prices
        # Always preserve match_winner (set by automover)
        for market_id, prices in new_market_set.markets.items():
            if market_id == "match_winner":
                continue  # Don't overwrite match winner set upstream
            context.adjusted_prices[market_id] = prices

"""
max_loss_tracker_agent.py
=========================
MaxLossTrackerAgent — Real-time max loss / total liability enforcement.

Responsibilities:
  1. Compute current max loss across all open positions
  2. Compare against match-level and global liability caps
  3. Suspend markets or reduce scales if caps are breached
  4. Emit structured liability log for monitoring

Max loss = Σ over all outcomes of: P(outcome) × max_payout_if_outcome_wins
         = effectively the maximum possible payout across all scenarios.

For a 2-outcome market with bets on both sides:
  max_loss = max(net_payout_if_A_wins, net_payout_if_B_wins)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import structlog

from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

# Global limits (GBP)
_MATCH_MAX_LOSS_CAP_DEFAULT = 500_000.0   # Per match across all markets
_MARKET_MAX_LOSS_CAP_DEFAULT = 100_000.0  # Per single market
_GLOBAL_SOFT_WARNING_RATIO = 0.75         # Warn at 75% of cap
_GLOBAL_HARD_SUSPEND_RATIO = 0.95         # Suspend at 95% of cap


@dataclass
class LiabilitySnapshot:
    match_id: str
    total_max_loss: float
    per_market_max_loss: Dict[str, float]
    is_within_cap: bool
    is_warning: bool
    pct_of_cap: float


class MaxLossTrackerAgent(BaseTradingAgent):
    """
    Enforces match-level and per-market maximum loss caps.
    """

    def __init__(
        self,
        match_max_loss_cap: float = _MATCH_MAX_LOSS_CAP_DEFAULT,
        market_max_loss_cap: float = _MARKET_MAX_LOSS_CAP_DEFAULT,
    ) -> None:
        self._match_cap = match_max_loss_cap
        self._market_cap = market_max_loss_cap

    @property
    def agent_name(self) -> str:
        return "max_loss_tracker"

    def process(self, context: TradingContext) -> TradingAgentResult:
        if context.suspend_all:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="already suspended — skip")

        # Compute max loss from current exposure
        # context.current_exposure: {outcome_key: net_liability_gbp}
        per_market_max_loss: Dict[str, float] = {}
        total_max_loss = 0.0

        # Group exposures by market
        market_exposures: Dict[str, Dict[str, float]] = {}
        for outcome_key, exposure in context.current_exposure.items():
            parts = outcome_key.split(":")
            if len(parts) >= 2:
                market_id, outcome_name = parts[0], parts[1]
            else:
                market_id, outcome_name = outcome_key, "unknown"

            if market_id not in market_exposures:
                market_exposures[market_id] = {}
            market_exposures[market_id][outcome_name] = exposure

        for market_id, outcomes in market_exposures.items():
            # Max loss for this market = max single-outcome payout
            market_max = max(outcomes.values()) if outcomes else 0.0
            per_market_max_loss[market_id] = market_max

            # Per-market hard cap
            if market_max > self._market_cap:
                context.click_scales[market_id] = 0.0
                self._log(
                    context,
                    f"MARKET SUSPENDED: {market_id} max_loss={market_max:.0f} "
                    f"exceeds cap={self._market_cap:.0f}"
                )

        total_max_loss = sum(per_market_max_loss.values())

        # Match-level cap
        pct = total_max_loss / self._match_cap if self._match_cap > 0 else 0.0
        is_warning = pct >= _GLOBAL_SOFT_WARNING_RATIO
        is_over_cap = pct >= _GLOBAL_HARD_SUSPEND_RATIO

        if is_over_cap:
            context.suspend_all = True
            context.suspend_reason = (
                f"max_loss_tracker: total_max_loss={total_max_loss:.0f} GBP "
                f"({pct:.1%} of cap={self._match_cap:.0f}) — hard suspend"
            )
            logger.error(
                "max_loss_hard_suspend",
                match_id=context.match_id,
                total_max_loss=total_max_loss,
                cap=self._match_cap,
                pct=pct,
            )
        elif is_warning:
            # Soft: reduce all click scales by 30%
            for market_id in context.click_scales:
                context.click_scales[market_id] *= 0.70
            logger.warning(
                "max_loss_soft_warning",
                match_id=context.match_id,
                total_max_loss=total_max_loss,
                pct=pct,
            )

        snapshot = LiabilitySnapshot(
            match_id=context.match_id,
            total_max_loss=total_max_loss,
            per_market_max_loss=per_market_max_loss,
            is_within_cap=not is_over_cap,
            is_warning=is_warning,
            pct_of_cap=pct,
        )

        logger.info(
            "max_loss_snapshot",
            match_id=context.match_id,
            total_max_loss=round(total_max_loss, 2),
            pct_of_cap=round(pct, 4),
            n_markets=len(per_market_max_loss),
            suspended=context.suspend_all,
        )

        return TradingAgentResult(
            agent_name=self.agent_name,
            success=True,
            context_mutated=is_over_cap or is_warning,
            notes=f"total_max_loss={total_max_loss:.0f} GBP ({pct:.1%} of cap)",
        )

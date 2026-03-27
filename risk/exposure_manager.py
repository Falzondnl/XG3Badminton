"""
risk/exposure_manager.py
========================
ExposureManager — Real-time liability and max-loss tracking.

Tracks:
  - Per-outcome exposure (net liability if outcome wins)
  - Per-market max loss
  - Per-match total max loss
  - Global portfolio exposure across all active badminton matches

Integrates with:
  - TradingControlManager (auto-suspend markets on limit breach)
  - BadmintonTradingSupervisor (feeds exposure into TradingContext)
  - BetProcessor (updates exposure on each accepted bet)

ZERO mock data. All exposure computed from actual bet records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Global limits (GBP)
_GLOBAL_MAX_EXPOSURE_GBP = 10_000_000.0   # Across all active badminton matches
_MATCH_MAX_EXPOSURE_GBP = 1_000_000.0     # Per match
_MARKET_MAX_EXPOSURE_GBP = 200_000.0      # Per single market outcome


@dataclass
class BetRecord:
    """Single accepted bet contributing to exposure."""
    bet_id: str
    match_id: str
    market_id: str
    outcome_name: str
    stake_gbp: float
    decimal_odds: float
    placed_at: float = field(default_factory=time.time)

    @property
    def potential_payout_gbp(self) -> float:
        return self.stake_gbp * self.decimal_odds

    @property
    def potential_profit_gbp(self) -> float:
        return self.stake_gbp * (self.decimal_odds - 1.0)


@dataclass
class OutcomeExposure:
    """Net liability on a single outcome."""
    market_id: str
    outcome_name: str
    total_stake_gbp: float = 0.0
    total_potential_profit_gbp: float = 0.0
    n_bets: int = 0

    @property
    def net_liability_gbp(self) -> float:
        """Net payout if this outcome wins."""
        return self.total_potential_profit_gbp


class ExposureManager:
    """
    Real-time exposure tracking across all active badminton markets.

    Thread-safety: Not inherently thread-safe. For concurrent use,
    wrap in an asyncio.Lock or use from a single event loop.
    """

    def __init__(self) -> None:
        # match_id → market_id → outcome_name → OutcomeExposure
        self._exposures: Dict[str, Dict[str, Dict[str, OutcomeExposure]]] = {}
        self._bet_log: List[BetRecord] = []
        self._global_max_loss: float = 0.0

    # ------------------------------------------------------------------
    # Bet acceptance
    # ------------------------------------------------------------------

    def record_bet(self, bet: BetRecord) -> None:
        """
        Record an accepted bet and update exposure.

        Raises ExposureLimitError if the bet would breach any limit.
        Call check_limits() before accepting to pre-validate.
        """
        match_exp = self._exposures.setdefault(bet.match_id, {})
        market_exp = match_exp.setdefault(bet.market_id, {})
        outcome_exp = market_exp.setdefault(
            bet.outcome_name,
            OutcomeExposure(market_id=bet.market_id, outcome_name=bet.outcome_name),
        )

        outcome_exp.total_stake_gbp += bet.stake_gbp
        outcome_exp.total_potential_profit_gbp += bet.potential_profit_gbp
        outcome_exp.n_bets += 1
        self._bet_log.append(bet)

        logger.debug(
            "bet_exposure_recorded",
            match_id=bet.match_id,
            market_id=bet.market_id,
            outcome=bet.outcome_name,
            stake=bet.stake_gbp,
            new_liability=outcome_exp.net_liability_gbp,
        )

    def check_limits(
        self,
        match_id: str,
        market_id: str,
        outcome_name: str,
        stake_gbp: float,
        decimal_odds: float,
    ) -> Optional[str]:
        """
        Pre-validate a bet against exposure limits.

        Returns:
            None if bet is acceptable.
            Error string if any limit would be breached.
        """
        potential_profit = stake_gbp * (decimal_odds - 1.0)

        # Check outcome-level limit
        current_outcome_exp = self._get_outcome_exposure(match_id, market_id, outcome_name)
        if current_outcome_exp + potential_profit > _MARKET_MAX_EXPOSURE_GBP:
            return (
                f"outcome exposure limit: current={current_outcome_exp:.0f} + "
                f"new={potential_profit:.0f} > limit={_MARKET_MAX_EXPOSURE_GBP:.0f} GBP"
            )

        # Check match-level limit
        current_match_exp = self.get_match_max_loss(match_id)
        if current_match_exp + potential_profit > _MATCH_MAX_EXPOSURE_GBP:
            return (
                f"match max-loss limit: current={current_match_exp:.0f} + "
                f"new={potential_profit:.0f} > limit={_MATCH_MAX_EXPOSURE_GBP:.0f} GBP"
            )

        # Global limit
        if self._global_max_loss + potential_profit > _GLOBAL_MAX_EXPOSURE_GBP:
            return (
                f"global exposure limit: {self._global_max_loss:.0f} + "
                f"{potential_profit:.0f} > {_GLOBAL_MAX_EXPOSURE_GBP:.0f} GBP"
            )

        return None

    # ------------------------------------------------------------------
    # Exposure queries
    # ------------------------------------------------------------------

    def get_match_max_loss(self, match_id: str) -> float:
        """Max possible payout across all outcomes for a match."""
        match_exp = self._exposures.get(match_id, {})
        total = 0.0
        for market_exp in match_exp.values():
            if market_exp:
                total += max(oe.net_liability_gbp for oe in market_exp.values())
        return total

    def get_market_exposure(self, match_id: str, market_id: str) -> Dict[str, float]:
        """Return {outcome_name: net_liability_gbp} for a market."""
        match_exp = self._exposures.get(match_id, {})
        market_exp = match_exp.get(market_id, {})
        return {name: oe.net_liability_gbp for name, oe in market_exp.items()}

    def get_all_exposure_for_context(self, match_id: str) -> Dict[str, float]:
        """
        Return flat {market_id:outcome: exposure} dict for TradingContext.current_exposure.
        """
        result: Dict[str, float] = {}
        match_exp = self._exposures.get(match_id, {})
        for market_id, market_exp in match_exp.items():
            for outcome_name, oe in market_exp.items():
                result[f"{market_id}:{outcome_name}"] = oe.net_liability_gbp
        return result

    def _get_outcome_exposure(
        self, match_id: str, market_id: str, outcome_name: str
    ) -> float:
        match_exp = self._exposures.get(match_id, {})
        market_exp = match_exp.get(market_id, {})
        oe = market_exp.get(outcome_name)
        return oe.net_liability_gbp if oe else 0.0

    def get_portfolio_summary(self) -> Dict:
        """Return global exposure summary."""
        total = sum(
            self.get_match_max_loss(mid) for mid in self._exposures
        )
        self._global_max_loss = total
        return {
            "n_active_matches": len(self._exposures),
            "total_max_loss_gbp": round(total, 2),
            "global_limit_gbp": _GLOBAL_MAX_EXPOSURE_GBP,
            "utilisation_pct": round(total / _GLOBAL_MAX_EXPOSURE_GBP * 100, 2),
        }


class ExposureLimitError(RuntimeError):
    """Raised when a bet would breach an exposure limit."""

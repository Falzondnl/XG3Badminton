"""
market_trading_control.py
=========================
Market trading control for badminton — operator controls for live and
pre-match markets.

Provides:
  - Manual market lock/unlock (operator override)
  - Click scaling (max payout per tick per market)
  - Suspension triggers (automatic and manual)
  - Liability position tracking per market
  - Price ladder management (accept/reject incoming bets)

Architecture:
  This layer sits between LivePricingEngine / PreMatchPricingEngine and
  the outgoing API. All prices pass through TradingControl before being
  served to clients.

  Liability tracking is per-match, per-market, per-selection.
  When a market exceeds its liability threshold, it is automatically
  suspended until manually re-opened or after the defined cooldown.

Click scaling rules:
  - Normal: scale=1.0 (no restriction)
  - Momentum signal: scale=0.5 (half max payout)
  - Manual override: scale=custom
  - Suspension: scale=0.0 (no bets accepted)

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import structlog

from config.badminton_config import (
    TRADING_DEFAULT_CLICK_MAX_GBP,
    TRADING_LIABILITY_SUSPEND_THRESHOLD_GBP,
    TRADING_COOLDOWN_SECONDS,
)
from markets.derivative_engine import MarketPrice

logger = structlog.get_logger(__name__)

# Re-export for downstream consumers (e.g. tests, BetValidator)
# BetRecord lives in risk.exposure_manager but is logically associated with trading.
try:
    from risk.exposure_manager import BetRecord, LiabilityPosition as _LiabilityPositionRisk  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MarketState(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    LOCKED = "locked"        # Manual operator lock
    RESULTED = "resulted"    # Market has been settled


class MarketStatus(str, Enum):
    """Public-facing market status used by BetValidator and API layer."""
    OPEN = "open"            # Bets accepted
    SUSPENDED = "suspended"  # Temporarily halted
    GHOST = "ghost"          # Live ghost (feed gap > 30s, ADR-018)
    CLOSED = "closed"        # Market closed (not yet resulted)
    RESULTED = "resulted"    # Market settled


class SuspensionReason(str, Enum):
    MANUAL = "manual"
    FEED_GAP = "feed_gap"
    LIABILITY = "liability"
    SCORE_UPDATE = "score_update"
    MOMENTUM = "momentum"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LiabilityPosition:
    """Current liability position for a single selection."""
    market_id: str
    outcome_name: str
    total_stake_gbp: float = 0.0
    total_payout_gbp: float = 0.0   # Max payout if this selection wins
    exposure_gbp: float = 0.0       # Net exposure (payout - stake on other selections)
    n_bets: int = 0

    def add_bet(self, stake_gbp: float, odds: float) -> None:
        """Record a new bet on this selection."""
        self.total_stake_gbp += stake_gbp
        self.total_payout_gbp += stake_gbp * odds
        self.exposure_gbp = self.total_payout_gbp - self.total_stake_gbp
        self.n_bets += 1


@dataclass(frozen=True)
class BetResult:
    """Result of a bet recording attempt via TradingControlManager.record_bet()."""
    accepted: bool
    rejection_reason: Optional[str] = None


@dataclass
class MarketControl:
    """Control state for a single market."""
    market_id: str
    state: MarketState = MarketState.ACTIVE
    suspension_reason: Optional[SuspensionReason] = None
    suspended_at: Optional[float] = None
    click_scale: float = 1.0                  # Fraction of max liability allowed
    max_payout_gbp: float = TRADING_DEFAULT_CLICK_MAX_GBP
    max_stake_per_bet: float = 0.0            # 0 = derive from max_payout / odds
    liability: Dict[str, LiabilityPosition] = field(default_factory=dict)
    cooldown_until: Optional[float] = None
    outcomes: List[str] = field(default_factory=list)   # Registered outcome names
    winning_outcome: Optional[str] = None               # Set on RESULTED
    liability_threshold_gbp: float = TRADING_LIABILITY_SUSPEND_THRESHOLD_GBP
    _total_bets: int = field(default=0, compare=False, repr=False)

    def is_tradeable(self) -> bool:
        """True if bets can be accepted on this market."""
        return self.state == MarketState.ACTIVE

    def current_max_stake(self, odds: float) -> float:
        """
        Maximum stake accepted for a bet at given odds.

        If max_stake_per_bet is set explicitly, use that.
        Otherwise derive from max_payout_gbp * click_scale / odds.
        """
        if self.max_stake_per_bet > 0:
            return self.max_stake_per_bet * self.click_scale
        max_payout = self.max_payout_gbp * self.click_scale
        return max(0.0, max_payout / max(1.01, odds))

    def total_liability_gbp(self) -> float:
        """Sum of all selection exposures."""
        return sum(pos.exposure_gbp for pos in self.liability.values())

    @property
    def total_bets(self) -> int:
        return self._total_bets


# ---------------------------------------------------------------------------
# Trading control manager
# ---------------------------------------------------------------------------

class TradingControlManager:
    """
    Manages trading controls for all markets in a match.

    One instance per match, lives for the duration of the match.
    """

    def __init__(
        self,
        match_id: str,
        default_click_max_gbp: float = TRADING_DEFAULT_CLICK_MAX_GBP,
    ) -> None:
        self.match_id = match_id
        self._default_click_max = default_click_max_gbp
        self._controls: Dict[str, MarketControl] = {}
        self._global_suspended = False
        self._global_lock = False

    def add_market(self, market_id: str, outcomes: Optional[List[str]] = None) -> MarketControl:
        """
        Explicitly register a market with known outcomes.

        Idempotent — if market already exists, returns existing control.
        """
        if market_id not in self._controls:
            self._controls[market_id] = MarketControl(
                market_id=market_id,
                max_payout_gbp=self._default_click_max,
                outcomes=list(outcomes or []),
            )
        return self._controls[market_id]

    def get_market(self, market_id: str) -> MarketControl:
        """Return MarketControl for a registered market. Raises KeyError if not found."""
        ctrl = self._controls.get(market_id)
        if ctrl is None:
            raise KeyError(f"Market {market_id!r} not registered")
        return ctrl

    def get_or_create_control(self, market_id: str) -> MarketControl:
        """Get existing market control or create with defaults."""
        if market_id not in self._controls:
            self._controls[market_id] = MarketControl(
                market_id=market_id,
                max_payout_gbp=self._default_click_max,
            )
        return self._controls[market_id]

    def filter_tradeable_prices(
        self, prices: Dict[str, List[MarketPrice]]
    ) -> Dict[str, List[MarketPrice]]:
        """
        Filter market prices to only tradeable markets.

        Suspended/locked markets are excluded from output.
        """
        if self._global_suspended or self._global_lock:
            return {}

        result = {}
        for market_id, market_prices in prices.items():
            ctrl = self.get_or_create_control(market_id)
            if ctrl.is_tradeable():
                result[market_id] = market_prices

        return result

    def apply_click_scales(
        self, prices: Dict[str, List[MarketPrice]]
    ) -> Dict[str, float]:
        """
        Return click scale factors per market.

        Callers use this to set max stake limits on each market.
        """
        scales: Dict[str, float] = {}
        for market_id in prices:
            ctrl = self.get_or_create_control(market_id)
            scales[market_id] = ctrl.click_scale
        return scales

    def suspend_market(
        self,
        market_id: str,
        reason: SuspensionReason | str = SuspensionReason.MANUAL,
        cooldown_seconds: float = TRADING_COOLDOWN_SECONDS,
    ) -> None:
        """Suspend a specific market. Accepts SuspensionReason or string."""
        if isinstance(reason, str):
            try:
                reason = SuspensionReason(reason)
            except ValueError:
                reason = SuspensionReason.MANUAL
        ctrl = self.get_or_create_control(market_id)
        ctrl.state = MarketState.SUSPENDED
        ctrl.suspension_reason = reason
        ctrl.suspended_at = time.time()
        ctrl.cooldown_until = time.time() + cooldown_seconds

        logger.info(
            "market_suspended",
            match_id=self.match_id,
            market_id=market_id,
            reason=reason.value,
        )

    def resume_market(self, market_id: str) -> None:
        """Resume a suspended market (manual operator action). Raises if LOCKED."""
        ctrl = self.get_or_create_control(market_id)
        if ctrl.state == MarketState.LOCKED:
            raise RuntimeError(
                f"Cannot resume locked market {market_id!r} — call unlock_market() first"
            )
        ctrl.state = MarketState.ACTIVE
        ctrl.suspension_reason = None
        ctrl.suspended_at = None
        ctrl.cooldown_until = None
        logger.info("market_resumed", match_id=self.match_id, market_id=market_id)

    def lock_market(self, market_id: str) -> None:
        """Lock market for manual price review."""
        ctrl = self.get_or_create_control(market_id)
        ctrl.state = MarketState.LOCKED
        logger.info("market_locked", match_id=self.match_id, market_id=market_id)

    def unlock_market(self, market_id: str) -> None:
        """Unlock market after manual review."""
        ctrl = self.get_or_create_control(market_id)
        if ctrl.state == MarketState.LOCKED:
            ctrl.state = MarketState.ACTIVE
        logger.info("market_unlocked", match_id=self.match_id, market_id=market_id)

    def suspend_all(self, reason: SuspensionReason | str = SuspensionReason.SYSTEM) -> None:
        """Suspend all markets (e.g., feed loss). Accepts SuspensionReason or string."""
        self._global_suspended = True
        if isinstance(reason, str):
            try:
                reason = SuspensionReason(reason)
            except ValueError:
                reason = SuspensionReason.SYSTEM
        for market_id in list(self._controls.keys()):
            ctrl = self._controls[market_id]
            if ctrl.state == MarketState.ACTIVE:
                ctrl.state = MarketState.SUSPENDED
                ctrl.suspension_reason = reason
        logger.warning(
            "all_markets_suspended",
            match_id=self.match_id,
            reason=reason.value,
        )

    def resume_all(self) -> None:
        """Resume all non-locked markets."""
        self._global_suspended = False
        for ctrl in self._controls.values():
            if ctrl.state == MarketState.SUSPENDED:
                ctrl.state = MarketState.ACTIVE
                ctrl.suspension_reason = None
        logger.info("all_markets_resumed", match_id=self.match_id)

    def set_click_scale(
        self,
        market_id: str,
        scale: float = 1.0,
        max_stake: Optional[float] = None,
    ) -> None:
        """
        Set click scale for a market.

        Args:
            market_id:  Market to update.
            scale:      Fraction of max payout (0.0–1.0). Ignored if max_stake given.
            max_stake:  Absolute max stake per bet in GBP (overrides scale).
        """
        ctrl = self.get_or_create_control(market_id)
        if max_stake is not None:
            ctrl.max_stake_per_bet = max(0.0, max_stake)
        else:
            ctrl.click_scale = max(0.0, min(1.0, scale))

    def set_liability_threshold(self, market_id: str, max_liability: float) -> None:
        """Override the per-market auto-suspension liability threshold."""
        ctrl = self.get_or_create_control(market_id)
        ctrl.liability_threshold_gbp = max_liability

    def get_liability(self, market_id: str) -> Dict[str, float]:
        """Return {outcome_name: exposure_gbp} for a market."""
        ctrl = self.get_or_create_control(market_id)
        return {name: pos.exposure_gbp for name, pos in ctrl.liability.items()}

    def get_market_stats(self, market_id: str) -> Dict:
        """Return operational statistics for a market."""
        ctrl = self.get_or_create_control(market_id)
        return {
            "market_id": market_id,
            "state": ctrl.state.value,
            "total_bets": ctrl._total_bets,
            "total_liability_gbp": round(ctrl.total_liability_gbp(), 2),
            "click_scale": ctrl.click_scale,
            "max_stake_per_bet": ctrl.max_stake_per_bet,
        }

    def list_active_markets(self) -> List[str]:
        """Return list of market_ids currently in ACTIVE state."""
        return [mid for mid, ctrl in self._controls.items() if ctrl.state == MarketState.ACTIVE]

    def mark_resulted(self, market_id: str, winning_outcome: str) -> None:
        """Mark a market as resulted with the winning outcome."""
        ctrl = self.get_or_create_control(market_id)
        ctrl.state = MarketState.RESULTED
        ctrl.winning_outcome = winning_outcome
        logger.info(
            "market_resulted",
            match_id=self.match_id,
            market_id=market_id,
            winner=winning_outcome,
        )

    def get_operational_summary(self) -> Dict:
        """Return operational summary of all markets."""
        counts = {s.value: 0 for s in MarketState}
        for ctrl in self._controls.values():
            counts[ctrl.state.value] += 1
        return {
            "total_markets": len(self._controls),
            "active_markets": counts[MarketState.ACTIVE.value],
            "suspended_markets": counts[MarketState.SUSPENDED.value],
            "locked_markets": counts[MarketState.LOCKED.value],
            "resulted_markets": counts[MarketState.RESULTED.value],
            "global_suspended": self._global_suspended,
        }

    def record_bet(
        self,
        market_id: str,
        outcome_name: Optional[str] = None,
        stake_gbp: float = 0.0,
        odds: float = 1.0,
        *,
        outcome: Optional[str] = None,
        stake: Optional[float] = None,
    ) -> BetResult:
        """
        Record a bet on a market selection.

        Returns BetResult with .accepted and .rejection_reason.
        Supports both 'outcome_name'/'stake_gbp' and 'outcome'/'stake' kwargs.
        """
        _outcome = outcome_name or outcome
        _stake = stake_gbp if stake is None else stake

        ctrl = self.get_or_create_control(market_id)

        if not ctrl.is_tradeable():
            reason = f"market {market_id!r} is {ctrl.state.value} — bets not accepted (suspended)"
            logger.debug("bet_rejected_not_tradeable", market_id=market_id, outcome=_outcome)
            return BetResult(accepted=False, rejection_reason=reason)

        max_stake = ctrl.current_max_stake(odds)
        if max_stake > 0 and _stake > max_stake:
            reason = (
                f"stake {_stake:.2f} exceeds max {max_stake:.2f} GBP "
                f"(click_scale={ctrl.click_scale:.3f})"
            )
            logger.debug("bet_rejected_exceeds_click", market_id=market_id, stake=_stake, max_stake=max_stake)
            return BetResult(accepted=False, rejection_reason=reason)

        # Record liability
        if _outcome not in ctrl.liability:
            ctrl.liability[_outcome] = LiabilityPosition(
                market_id=market_id,
                outcome_name=_outcome,
            )
        ctrl.liability[_outcome].add_bet(_stake, odds)
        ctrl._total_bets += 1

        # Check auto-suspension threshold (per-market configurable)
        total_liability = ctrl.total_liability_gbp()
        if total_liability > ctrl.liability_threshold_gbp:
            self.suspend_market(market_id, reason=SuspensionReason.LIABILITY)

        return BetResult(accepted=True)

    def get_market_status(self, market_id: str) -> MarketStatus:
        """
        Return the public-facing MarketStatus for a market.

        Used by BetValidator to gate bet acceptance.
        """
        if self._global_suspended:
            return MarketStatus.SUSPENDED
        ctrl = self._controls.get(market_id)
        if ctrl is None:
            # Market not yet registered → treat as open
            return MarketStatus.OPEN
        if ctrl.state == MarketState.RESULTED:
            return MarketStatus.RESULTED
        if ctrl.state == MarketState.LOCKED:
            return MarketStatus.CLOSED
        if ctrl.state == MarketState.SUSPENDED:
            # Check if it's a ghost (feed_gap reason)
            if ctrl.suspension_reason == SuspensionReason.FEED_GAP:
                return MarketStatus.GHOST
            return MarketStatus.SUSPENDED
        return MarketStatus.OPEN

    def result_market(self, market_id: str, winning_outcome: str) -> None:
        """Mark market as resulted (alias for mark_resulted)."""
        self.mark_resulted(market_id, winning_outcome)

    def get_liability_report(self) -> Dict[str, float]:
        """Return total liability per market."""
        return {
            market_id: ctrl.total_liability_gbp()
            for market_id, ctrl in self._controls.items()
        }

    def get_open_markets(self) -> Dict[str, List[str]]:
        """
        Return all open (ACTIVE or SUSPENDED) markets with their outcome names.

        Used by GradingService.settle_match() to determine which markets need
        settlement. Returns {market_id: [outcome_name, ...]} for all non-RESULTED
        and non-CLOSED markets.
        """
        result: Dict[str, List[str]] = {}
        for market_id, ctrl in self._controls.items():
            if ctrl.state not in (MarketState.RESULTED, MarketState.LOCKED):
                outcome_names = list(ctrl.liability.keys())
                result[market_id] = outcome_names
        return result


# ---------------------------------------------------------------------------
# Alias — used by BetValidator, TraderControlAgent, API layer
# ---------------------------------------------------------------------------

#: Alias for TradingControlManager — the canonical type for dependency injection.
MarketTradingControl = TradingControlManager

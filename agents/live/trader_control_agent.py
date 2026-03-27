"""
agents/live/trader_control_agent.py
=====================================
TraderControlAgent — Manual operator override handling for live markets.

Responsibilities:
  1. Accept operator commands (suspend/resume/scale/lock/unlock market)
  2. Apply commands to MarketTradingControl
  3. Feed overrides into TradingContext so the trading cycle respects them
  4. Log all manual interventions with operator_id + reason + timestamp
  5. Enforce cooldown periods on repeated overrides (prevent flip-flopping)

Commands supported:
  - SUSPEND_MARKET     — suspend a market immediately
  - RESUME_MARKET      — unsuspend a market (clears manual suspend)
  - SET_CLICK_SCALE    — set click scale for a market (0.0 = no bets)
  - SUSPEND_ALL        — suspend all markets for a match
  - RESUME_ALL         — resume all previously-manual-suspended markets
  - LOCK_MARKET        — lock market from any auto-repricing
  - UNLOCK_MARKET      — allow auto-repricing again

Override audit log is append-only — never deleted, queryable by match_id.

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import structlog

from markets.market_trading_control import MarketTradingControl

logger = structlog.get_logger(__name__)

# Minimum seconds between repeated overrides on the same market
_OVERRIDE_COOLDOWN_SECONDS = 5.0


class OperatorCommandType(str, Enum):
    SUSPEND_MARKET = "SUSPEND_MARKET"
    RESUME_MARKET = "RESUME_MARKET"
    SET_CLICK_SCALE = "SET_CLICK_SCALE"
    SUSPEND_ALL = "SUSPEND_ALL"
    RESUME_ALL = "RESUME_ALL"
    LOCK_MARKET = "LOCK_MARKET"
    UNLOCK_MARKET = "UNLOCK_MARKET"


class TraderControlError(RuntimeError):
    """Raised on invalid or rejected operator command."""


@dataclass(frozen=True)
class OperatorCommand:
    """A single operator intervention command."""
    command_type: OperatorCommandType
    match_id: str
    market_id: Optional[str]        # None for SUSPEND_ALL / RESUME_ALL
    operator_id: str
    reason: str
    click_scale: Optional[float] = None  # For SET_CLICK_SCALE only
    issued_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class OperatorCommandResult:
    """Result of applying an operator command."""
    command_type: OperatorCommandType
    market_id: Optional[str]
    applied: bool
    detail: str


class TraderControlAgent:
    """
    Live operator override handler.

    One instance per match.
    """

    def __init__(
        self,
        match_id: str,
        trading_control: MarketTradingControl,
    ) -> None:
        self._match_id = match_id
        self._trading_control = trading_control
        self._override_log: List[OperatorCommand] = []
        # market_id → last override timestamp (for cooldown check)
        self._last_override_at: Dict[str, float] = {}
        # market_ids that have been manually suspended (for RESUME_ALL)
        self._manually_suspended: set[str] = set()
        # market_ids that have been manually locked
        self._manually_locked: set[str] = set()

    def apply_command(self, command: OperatorCommand) -> OperatorCommandResult:
        """
        Apply an operator command to the trading control layer.

        Args:
            command:  Validated operator command.

        Returns:
            OperatorCommandResult — always returns even on cooldown (applied=False).

        Raises:
            TraderControlError on schema violations.
        """
        if command.match_id != self._match_id:
            raise TraderControlError(
                f"Command match_id={command.match_id!r} does not match "
                f"agent match_id={self._match_id!r}"
            )

        # Cooldown check (per market_id)
        if command.market_id:
            last = self._last_override_at.get(command.market_id, 0.0)
            elapsed = time.time() - last
            if elapsed < _OVERRIDE_COOLDOWN_SECONDS:
                logger.warning(
                    "trader_control_cooldown_rejected",
                    match_id=self._match_id,
                    market_id=command.market_id,
                    command=command.command_type.value,
                    elapsed_s=round(elapsed, 2),
                    cooldown_s=_OVERRIDE_COOLDOWN_SECONDS,
                    operator_id=command.operator_id,
                )
                return OperatorCommandResult(
                    command_type=command.command_type,
                    market_id=command.market_id,
                    applied=False,
                    detail=(
                        f"cooldown: {elapsed:.1f}s elapsed < "
                        f"{_OVERRIDE_COOLDOWN_SECONDS}s required"
                    ),
                )

        # Log before applying (audit trail)
        self._override_log.append(command)
        if command.market_id:
            self._last_override_at[command.market_id] = time.time()

        result = self._dispatch(command)

        logger.info(
            "trader_control_command_applied",
            match_id=self._match_id,
            command=command.command_type.value,
            market_id=command.market_id,
            operator_id=command.operator_id,
            reason=command.reason,
            applied=result.applied,
            detail=result.detail,
        )

        return result

    def _dispatch(self, command: OperatorCommand) -> OperatorCommandResult:
        ct = command.command_type

        if ct == OperatorCommandType.SUSPEND_MARKET:
            if not command.market_id:
                raise TraderControlError("SUSPEND_MARKET requires market_id")
            self._trading_control.suspend_market(command.market_id)
            self._manually_suspended.add(command.market_id)
            return OperatorCommandResult(
                command_type=ct,
                market_id=command.market_id,
                applied=True,
                detail=f"market {command.market_id!r} suspended by operator",
            )

        if ct == OperatorCommandType.RESUME_MARKET:
            if not command.market_id:
                raise TraderControlError("RESUME_MARKET requires market_id")
            self._trading_control.resume_market(command.market_id)
            self._manually_suspended.discard(command.market_id)
            return OperatorCommandResult(
                command_type=ct,
                market_id=command.market_id,
                applied=True,
                detail=f"market {command.market_id!r} resumed by operator",
            )

        if ct == OperatorCommandType.SET_CLICK_SCALE:
            if not command.market_id:
                raise TraderControlError("SET_CLICK_SCALE requires market_id")
            if command.click_scale is None:
                raise TraderControlError("SET_CLICK_SCALE requires click_scale value")
            scale = max(0.0, min(1.0, command.click_scale))
            self._trading_control.set_click_scale(command.market_id, scale)
            return OperatorCommandResult(
                command_type=ct,
                market_id=command.market_id,
                applied=True,
                detail=f"click_scale={scale:.3f} on market {command.market_id!r}",
            )

        if ct == OperatorCommandType.SUSPEND_ALL:
            open_markets = self._trading_control.get_open_markets()
            for mid in open_markets:
                self._trading_control.suspend_market(mid)
                self._manually_suspended.add(mid)
            return OperatorCommandResult(
                command_type=ct,
                market_id=None,
                applied=True,
                detail=f"suspended {len(open_markets)} markets",
            )

        if ct == OperatorCommandType.RESUME_ALL:
            resumed = list(self._manually_suspended)
            for mid in resumed:
                self._trading_control.resume_market(mid)
            self._manually_suspended.clear()
            return OperatorCommandResult(
                command_type=ct,
                market_id=None,
                applied=True,
                detail=f"resumed {len(resumed)} manually-suspended markets",
            )

        if ct == OperatorCommandType.LOCK_MARKET:
            if not command.market_id:
                raise TraderControlError("LOCK_MARKET requires market_id")
            self._trading_control.lock_market(command.market_id)
            self._manually_locked.add(command.market_id)
            return OperatorCommandResult(
                command_type=ct,
                market_id=command.market_id,
                applied=True,
                detail=f"market {command.market_id!r} locked",
            )

        if ct == OperatorCommandType.UNLOCK_MARKET:
            if not command.market_id:
                raise TraderControlError("UNLOCK_MARKET requires market_id")
            self._trading_control.unlock_market(command.market_id)
            self._manually_locked.discard(command.market_id)
            return OperatorCommandResult(
                command_type=ct,
                market_id=command.market_id,
                applied=True,
                detail=f"market {command.market_id!r} unlocked",
            )

        raise TraderControlError(f"Unknown command type: {ct!r}")

    def get_override_log(self) -> List[OperatorCommand]:
        """Return full append-only audit log of all operator commands."""
        return list(self._override_log)

    def get_locked_markets(self) -> set[str]:
        """Return set of manually locked market_ids."""
        return set(self._manually_locked)

    def is_market_locked(self, market_id: str) -> bool:
        return market_id in self._manually_locked

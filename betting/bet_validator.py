"""
betting/bet_validator.py
========================
BetValidator — Pre-acceptance validation pipeline for incoming bets.

Validation layers (in order):
  1. Schema validation  — required fields, numeric ranges
  2. Market state       — market must be OPEN and not suspended
  3. Click scale        — stake × click_scale ≤ max allowed stake
  4. Odds check         — offered odds within tolerance of current published odds
  5. Exposure limits    — ExposureManager.check_limits() gate
  6. Max stake per bet  — configurable per bettor tier

Raises BetValidationError with a structured reason code on any failure.
Returns BetValidationResult(accepted=True) on success.

ZERO default probabilities or hardcoded stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog

from markets.market_trading_control import MarketTradingControl, MarketStatus
from risk.exposure_manager import ExposureManager

logger = structlog.get_logger(__name__)

# Odds tolerance: bet offered_odds must be within ±2% of current published odds
_ODDS_TOLERANCE_PCT = 0.02

# Max stake per single bet by tier (GBP)
_MAX_STAKE_STANDARD = 5_000.0
_MAX_STAKE_VIP = 25_000.0
_MIN_STAKE = 0.50


class BetRejectionCode(str, Enum):
    SCHEMA_ERROR = "SCHEMA_ERROR"           # Missing/invalid fields
    MARKET_CLOSED = "MARKET_CLOSED"         # Market not OPEN
    MARKET_SUSPENDED = "MARKET_SUSPENDED"   # Market suspended/ghost
    STAKE_BELOW_MIN = "STAKE_BELOW_MIN"     # Stake < £0.50
    STAKE_ABOVE_MAX = "STAKE_ABOVE_MAX"     # Stake exceeds tier maximum
    STAKE_EXCEEDS_SCALE = "STAKE_EXCEEDS_SCALE"  # Stake > max_stake × click_scale
    ODDS_STALE = "ODDS_STALE"               # Offered odds differ > 2% from current
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"       # Would breach outcome/match/global limit


class BetValidationError(RuntimeError):
    """Raised when a bet fails validation."""

    def __init__(self, code: BetRejectionCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True)
class BetValidationResult:
    """Successful validation result — all checks passed."""
    bet_id: str
    match_id: str
    market_id: str
    outcome_name: str
    stake_gbp: float
    decimal_odds: float
    validated_stake_gbp: float   # May be capped by click_scale


@dataclass
class IncomingBetRequest:
    """Raw incoming bet from the API layer."""
    bet_id: str
    match_id: str
    market_id: str
    outcome_name: str
    stake_gbp: float
    offered_odds: float      # Odds the bettor expects
    is_vip: bool = False


class BetValidator:
    """
    Stateless bet validation pipeline.

    Dependencies are injected — one instance can serve multiple matches
    as long as the correct trading_control and exposure_manager are passed
    per validation call.
    """

    def validate(
        self,
        request: IncomingBetRequest,
        trading_control: MarketTradingControl,
        exposure_manager: ExposureManager,
        current_odds: Optional[float] = None,
        click_scale: float = 1.0,
    ) -> BetValidationResult:
        """
        Run all validation layers. Raises BetValidationError on first failure.

        Args:
            request:          Incoming bet request from API.
            trading_control:  Market state manager for the match.
            exposure_manager: Exposure tracker for the match.
            current_odds:     Currently published odds for this outcome (None = skip odds check).
            click_scale:      Current click scale for the market (from TradingContext).

        Returns:
            BetValidationResult on success.

        Raises:
            BetValidationError with BetRejectionCode on any failure.
        """
        self._validate_schema(request)
        self._validate_market_state(request, trading_control)
        validated_stake = self._validate_stake(request, click_scale)
        if current_odds is not None:
            self._validate_odds(request, current_odds)
        self._validate_exposure(request, validated_stake, exposure_manager)

        logger.info(
            "bet_validated",
            bet_id=request.bet_id,
            match_id=request.match_id,
            market_id=request.market_id,
            outcome=request.outcome_name,
            stake=validated_stake,
            odds=request.offered_odds,
        )

        return BetValidationResult(
            bet_id=request.bet_id,
            match_id=request.match_id,
            market_id=request.market_id,
            outcome_name=request.outcome_name,
            stake_gbp=request.stake_gbp,
            decimal_odds=request.offered_odds,
            validated_stake_gbp=validated_stake,
        )

    # ------------------------------------------------------------------
    # Validation layers
    # ------------------------------------------------------------------

    def _validate_schema(self, req: IncomingBetRequest) -> None:
        if not req.bet_id or not req.bet_id.strip():
            raise BetValidationError(BetRejectionCode.SCHEMA_ERROR, "bet_id is required")
        if not req.match_id or not req.match_id.strip():
            raise BetValidationError(BetRejectionCode.SCHEMA_ERROR, "match_id is required")
        if not req.market_id or not req.market_id.strip():
            raise BetValidationError(BetRejectionCode.SCHEMA_ERROR, "market_id is required")
        if not req.outcome_name or not req.outcome_name.strip():
            raise BetValidationError(BetRejectionCode.SCHEMA_ERROR, "outcome_name is required")
        if req.stake_gbp <= 0:
            raise BetValidationError(
                BetRejectionCode.SCHEMA_ERROR,
                f"stake_gbp must be positive, got {req.stake_gbp}",
            )
        if req.offered_odds < 1.0:
            raise BetValidationError(
                BetRejectionCode.SCHEMA_ERROR,
                f"offered_odds must be >= 1.0, got {req.offered_odds}",
            )

    def _validate_market_state(
        self, req: IncomingBetRequest, trading_control: MarketTradingControl
    ) -> None:
        status = trading_control.get_market_status(req.market_id)
        if status == MarketStatus.RESULTED:
            raise BetValidationError(
                BetRejectionCode.MARKET_CLOSED,
                f"market {req.market_id!r} has already resulted",
            )
        if status in (MarketStatus.SUSPENDED, MarketStatus.GHOST):
            raise BetValidationError(
                BetRejectionCode.MARKET_SUSPENDED,
                f"market {req.market_id!r} is currently {status.value}",
            )
        if status != MarketStatus.OPEN:
            raise BetValidationError(
                BetRejectionCode.MARKET_CLOSED,
                f"market {req.market_id!r} is not open (status={status.value})",
            )

    def _validate_stake(self, req: IncomingBetRequest, click_scale: float) -> float:
        if req.stake_gbp < _MIN_STAKE:
            raise BetValidationError(
                BetRejectionCode.STAKE_BELOW_MIN,
                f"stake {req.stake_gbp:.2f} < minimum {_MIN_STAKE:.2f} GBP",
            )

        max_stake = _MAX_STAKE_VIP if req.is_vip else _MAX_STAKE_STANDARD
        if req.stake_gbp > max_stake:
            raise BetValidationError(
                BetRejectionCode.STAKE_ABOVE_MAX,
                f"stake {req.stake_gbp:.2f} > tier maximum {max_stake:.2f} GBP",
            )

        # Click scale cap: maximum accepted stake = max_stake × click_scale
        max_allowed = max_stake * max(0.0, click_scale)
        if req.stake_gbp > max_allowed:
            raise BetValidationError(
                BetRejectionCode.STAKE_EXCEEDS_SCALE,
                f"stake {req.stake_gbp:.2f} > allowed {max_allowed:.2f} GBP "
                f"(click_scale={click_scale:.3f})",
            )

        return req.stake_gbp

    def _validate_odds(self, req: IncomingBetRequest, current_odds: float) -> None:
        if current_odds <= 0:
            raise BetValidationError(
                BetRejectionCode.ODDS_STALE,
                f"current_odds={current_odds} is invalid for {req.market_id!r}",
            )
        drift = abs(req.offered_odds - current_odds) / current_odds
        if drift > _ODDS_TOLERANCE_PCT:
            raise BetValidationError(
                BetRejectionCode.ODDS_STALE,
                f"offered_odds={req.offered_odds:.4f} differs from current "
                f"{current_odds:.4f} by {drift:.2%} > tolerance {_ODDS_TOLERANCE_PCT:.2%}",
            )

    def _validate_exposure(
        self,
        req: IncomingBetRequest,
        stake_gbp: float,
        exposure_manager: ExposureManager,
    ) -> None:
        error = exposure_manager.check_limits(
            match_id=req.match_id,
            market_id=req.market_id,
            outcome_name=req.outcome_name,
            stake_gbp=stake_gbp,
            decimal_odds=req.offered_odds,
        )
        if error:
            raise BetValidationError(
                BetRejectionCode.EXPOSURE_LIMIT,
                error,
            )

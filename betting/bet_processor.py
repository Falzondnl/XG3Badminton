"""
betting/bet_processor.py
========================
BetProcessor — End-to-end bet acceptance pipeline.

Pipeline:
  1. BetValidator.validate()         — all schema/market/odds/exposure checks
  2. ExposureManager.record_bet()    — update liability tracking
  3. Return BetAcceptanceRecord      — persisted confirmation

The BetProcessor is the single entry point for bet acceptance. It owns
the atomicity guarantee: if exposure recording fails after validation passes,
the exception propagates and the bet is NOT accepted.

Also exposes:
  - get_bet(bet_id)          → BetAcceptanceRecord | None
  - list_bets_for_match()    → [BetAcceptanceRecord]
  - get_cashout_value()      → CashoutResult (delegates to CashoutCalculator)

ZERO hardcoded probabilities. ZERO mock data.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

from betting.bet_validator import (
    BetValidator,
    BetValidationResult,
    IncomingBetRequest,
    BetValidationError,
)
from core.match_state import MatchLiveState
from markets.market_trading_control import MarketTradingControl
from risk.exposure_manager import BetRecord, ExposureManager
from risk.cashout_calculator import CashoutCalculator, CashoutResult

logger = structlog.get_logger(__name__)


class BetProcessorError(RuntimeError):
    """Raised when bet processing fails after validation."""


@dataclass(frozen=True)
class BetAcceptanceRecord:
    """Persisted record of an accepted bet."""
    bet_id: str
    match_id: str
    market_id: str
    outcome_name: str
    stake_gbp: float
    decimal_odds: float
    potential_payout_gbp: float
    placed_at: float   # unix timestamp


class BetProcessor:
    """
    End-to-end bet acceptance pipeline.

    One instance per match. Maintains in-memory bet log.
    For production, persistence layer (DB write) would be injected.
    """

    def __init__(
        self,
        match_id: str,
        trading_control: MarketTradingControl,
        exposure_manager: ExposureManager,
        cashout_calculator: CashoutCalculator,
    ) -> None:
        self._match_id = match_id
        self._trading_control = trading_control
        self._exposure_manager = exposure_manager
        self._cashout_calculator = cashout_calculator
        self._validator = BetValidator()
        self._accepted: Dict[str, BetAcceptanceRecord] = {}

    def accept_bet(
        self,
        request: IncomingBetRequest,
        current_odds: Optional[float] = None,
        click_scale: float = 1.0,
    ) -> BetAcceptanceRecord:
        """
        Run full validation pipeline and record the bet.

        Args:
            request:       Incoming bet from API.
            current_odds:  Currently published odds (for stale-odds check).
            click_scale:   Market click scale (from TradingContext).

        Returns:
            BetAcceptanceRecord on success.

        Raises:
            BetValidationError  — bet rejected, not recorded.
            BetProcessorError   — validation passed but recording failed.
        """
        if request.match_id != self._match_id:
            raise BetProcessorError(
                f"BetProcessor for match {self._match_id!r} received bet "
                f"for match {request.match_id!r}"
            )

        # Layer 1: validate
        validation: BetValidationResult = self._validator.validate(
            request=request,
            trading_control=self._trading_control,
            exposure_manager=self._exposure_manager,
            current_odds=current_odds,
            click_scale=click_scale,
        )

        # Layer 2: record exposure (atomic — any failure here propagates)
        import time
        bet_record = BetRecord(
            bet_id=validation.bet_id,
            match_id=validation.match_id,
            market_id=validation.market_id,
            outcome_name=validation.outcome_name,
            stake_gbp=validation.validated_stake_gbp,
            decimal_odds=validation.decimal_odds,
            placed_at=time.time(),
        )

        try:
            self._exposure_manager.record_bet(bet_record)
        except Exception as exc:
            raise BetProcessorError(
                f"Failed to record exposure for bet {validation.bet_id!r}: {exc}"
            ) from exc

        # Layer 3: persist acceptance record
        acceptance = BetAcceptanceRecord(
            bet_id=validation.bet_id,
            match_id=validation.match_id,
            market_id=validation.market_id,
            outcome_name=validation.outcome_name,
            stake_gbp=validation.validated_stake_gbp,
            decimal_odds=validation.decimal_odds,
            potential_payout_gbp=round(
                validation.validated_stake_gbp * validation.decimal_odds, 2
            ),
            placed_at=bet_record.placed_at,
        )
        self._accepted[validation.bet_id] = acceptance

        logger.info(
            "bet_accepted",
            bet_id=acceptance.bet_id,
            match_id=acceptance.match_id,
            market_id=acceptance.market_id,
            outcome=acceptance.outcome_name,
            stake=acceptance.stake_gbp,
            odds=acceptance.decimal_odds,
            payout=acceptance.potential_payout_gbp,
        )

        return acceptance

    def get_bet(self, bet_id: str) -> Optional[BetAcceptanceRecord]:
        """Return an accepted bet record by ID."""
        return self._accepted.get(bet_id)

    def list_bets_for_match(self) -> List[BetAcceptanceRecord]:
        """Return all accepted bets for this match."""
        return list(self._accepted.values())

    def get_cashout_value(
        self,
        bet_id: str,
        live_state: MatchLiveState,
        outcome_is_player_a: bool,
        is_premium_bettor: bool = False,
    ) -> CashoutResult:
        """
        Compute live cashout value for an accepted bet.

        Raises:
            BetProcessorError if bet_id not found.
            CashoutError if Markov computation fails.
        """
        record = self._accepted.get(bet_id)
        if record is None:
            raise BetProcessorError(
                f"bet_id={bet_id!r} not found in accepted bets for match {self._match_id!r}"
            )

        bet_record = BetRecord(
            bet_id=record.bet_id,
            match_id=record.match_id,
            market_id=record.market_id,
            outcome_name=record.outcome_name,
            stake_gbp=record.stake_gbp,
            decimal_odds=record.decimal_odds,
            placed_at=record.placed_at,
        )

        return self._cashout_calculator.compute(
            bet=bet_record,
            live_state=live_state,
            outcome_is_player_a=outcome_is_player_a,
            is_premium_bettor=is_premium_bettor,
        )

    @property
    def n_accepted_bets(self) -> int:
        return len(self._accepted)

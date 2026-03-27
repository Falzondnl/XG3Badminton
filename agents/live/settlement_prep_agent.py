"""
agents/live/settlement_prep_agent.py
=====================================
SettlementPrepAgent — Auto-grading trigger on match completion.

Responsibilities:
  1. Detect match completion from MatchLiveState.status == COMPLETED
  2. Validate final score via GradingService prerequisites
  3. Build MatchResult from the final live state
  4. Invoke GradingService.settle_match() to settle all 97 markets
  5. Transition MarketTradingControl to RESULTED state
  6. Log settlement audit trail
  7. Prevent double-settlement (idempotent — subsequent calls are no-ops)

Called by:
  - LiveSupervisorAgent after each score update
  - (Can also be invoked manually by operator via TraderControlAgent)

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import structlog

from core.match_state import MatchLiveState, MatchStatus
from markets.market_trading_control import MarketTradingControl
from settlement.grading_service import GradingService, MatchResult, SettlementRecord

logger = structlog.get_logger(__name__)


class SettlementPrepError(RuntimeError):
    """Raised when settlement preparation fails."""


@dataclass(frozen=True)
class SettlementPrepResult:
    """Result of a settlement prep cycle."""
    match_id: str
    settled: bool                        # True if settlement was triggered
    already_settled: bool               # True if this was a duplicate call
    n_markets_settled: int
    settlement_records: List[SettlementRecord]
    error: Optional[str] = None


class SettlementPrepAgent:
    """
    Auto-grading trigger — fires when live state reaches COMPLETED.

    One instance per match.
    """

    def __init__(
        self,
        match_id: str,
        trading_control: MarketTradingControl,
        grading_service: GradingService,
    ) -> None:
        self._match_id = match_id
        self._trading_control = trading_control
        self._grading_service = grading_service
        self._settled = False
        self._settlement_records: List[SettlementRecord] = []

    def check_and_settle(self, live_state: MatchLiveState) -> SettlementPrepResult:
        """
        Check if the match is complete and settle if so.

        Args:
            live_state:  Current authoritative MatchLiveState.

        Returns:
            SettlementPrepResult with settlement status.
        """
        if live_state.match_id != self._match_id:
            raise SettlementPrepError(
                f"live_state.match_id={live_state.match_id!r} does not match "
                f"agent match_id={self._match_id!r}"
            )

        # Idempotency guard
        if self._settled:
            return SettlementPrepResult(
                match_id=self._match_id,
                settled=False,
                already_settled=True,
                n_markets_settled=len(self._settlement_records),
                settlement_records=self._settlement_records,
            )

        if live_state.status != MatchStatus.COMPLETED:
            return SettlementPrepResult(
                match_id=self._match_id,
                settled=False,
                already_settled=False,
                n_markets_settled=0,
                settlement_records=[],
            )

        # Match is COMPLETED — proceed with settlement
        return self._settle(live_state)

    def _settle(self, live_state: MatchLiveState) -> SettlementPrepResult:
        """Execute settlement pipeline."""
        logger.info(
            "settlement_prep_triggered",
            match_id=self._match_id,
            winner=live_state.match_winner,
            games_a=live_state.games_won_a,
            games_b=live_state.games_won_b,
        )

        try:
            match_result = MatchResult.from_live_state(live_state)
        except Exception as exc:
            error_msg = f"MatchResult.from_live_state failed: {exc}"
            logger.error(
                "settlement_prep_build_result_failed",
                match_id=self._match_id,
                error=error_msg,
            )
            return SettlementPrepResult(
                match_id=self._match_id,
                settled=False,
                already_settled=False,
                n_markets_settled=0,
                settlement_records=[],
                error=error_msg,
            )

        try:
            open_markets = self._trading_control.get_open_markets()
            records = self._grading_service.settle_match(
                match_result=match_result,
                open_markets=open_markets,
            )
        except Exception as exc:
            error_msg = f"GradingService.settle_match failed: {exc}"
            logger.error(
                "settlement_prep_grading_failed",
                match_id=self._match_id,
                error=error_msg,
            )
            return SettlementPrepResult(
                match_id=self._match_id,
                settled=False,
                already_settled=False,
                n_markets_settled=0,
                settlement_records=[],
                error=error_msg,
            )

        # Transition trading control to settled state
        try:
            self._trading_control.transition_to_settled(
                match_id=self._match_id,
                n_markets=len(records),
            )
        except Exception as exc:
            # Log but don't fail — settlement records are already written
            logger.warning(
                "settlement_prep_transition_failed",
                match_id=self._match_id,
                error=str(exc),
            )

        self._settled = True
        self._settlement_records = records

        logger.info(
            "settlement_prep_complete",
            match_id=self._match_id,
            n_markets=len(records),
            winner=match_result.winner,
        )

        return SettlementPrepResult(
            match_id=self._match_id,
            settled=True,
            already_settled=False,
            n_markets_settled=len(records),
            settlement_records=records,
        )

    @property
    def is_settled(self) -> bool:
        return self._settled

    def get_settlement_records(self) -> List[SettlementRecord]:
        return list(self._settlement_records)

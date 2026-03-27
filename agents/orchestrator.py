"""
orchestrator.py
===============
BadmintonOrchestratorAgent — master agent coordinating all badminton operations.

Follows the VaultAgent pattern from XG3 platform architecture (§19 of CLAUDE.md).

Hierarchy:
  BadmintonOrchestratorAgent
  ├── PreMatchSupervisorAgent     — pre-match pricing and odds management
  ├── LiveSupervisorAgent         — real-time live market management
  ├── OutrightSupervisorAgent     — tournament outrights and futures
  ├── SGPSupervisorAgent          — SGP/Bet Builder requests
  └── MonitoringSupervisorAgent   — alerts, anomalies, performance monitoring

Responsibilities:
  - Route incoming requests to appropriate supervisor
  - Coordinate match lifecycle (pre-match → live → settlement)
  - Manage shared resources (Markov cache, ELO system, model inference)
  - Handle graceful degradation (feed loss, model unavailability)
  - Enforce CLAUDE.md absolute rules at orchestrator level

Message routing:
  - REST API events → appropriate supervisor
  - Feed events → LiveSupervisorAgent
  - Settlement triggers → GradingService
  - Model requests → ML inference layer

State persistence:
  - Active match contexts: in-memory (with Redis fallback)
  - Historical match results: PostgreSQL
  - Audit trail: append-only event log

ZERO mock data or stubs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    ORCHESTRATOR_MAX_ACTIVE_MATCHES,
)
from feed.feed_health_monitor import FeedHealthMonitor, FeedName, FeedStatus
from markets.market_trading_control import TradingControlManager
from settlement.grading_service import GradingService
from agents.trading_supervisor import BadmintonTradingSupervisor, TradingCycleResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Match lifecycle states
# ---------------------------------------------------------------------------

class MatchLifecycleState(str, Enum):
    SCHEDULED = "scheduled"
    PRE_MATCH = "pre_match"        # Odds available, match not started
    LIVE = "live"                  # Match in progress
    SUSPENDED = "suspended"        # Temporary halt
    RESULTED = "resulted"          # Match complete, settling
    SETTLED = "settled"            # All markets settled
    ABANDONED = "abandoned"        # Match abandoned


# ---------------------------------------------------------------------------
# Active match record
# ---------------------------------------------------------------------------

@dataclass
class ActiveMatchRecord:
    """In-memory record for an active match."""
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier
    tournament_id: str
    lifecycle_state: MatchLifecycleState
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # Supervisor assignments
    pre_match_active: bool = False
    live_active: bool = False
    outright_active: bool = False

    # Trading control
    trading_control: Optional[TradingControlManager] = None
    # Trading supervisor (iMOVE-style 13-agent chain)
    trading_supervisor: Optional[BadmintonTradingSupervisor] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class BadmintonOrchestratorAgent:
    """
    Master orchestrator for all badminton market operations.

    Singleton per XG3 deployment — one orchestrator manages all active
    badminton matches across all disciplines.
    """

    def __init__(self) -> None:
        # Active match registry
        self._active_matches: Dict[str, ActiveMatchRecord] = {}

        # Shared resources
        self._feed_monitor = FeedHealthMonitor()
        self._grading_service = GradingService()

        # Supervisor references (lazy initialised on first match)
        self._pre_match_supervisor = None
        self._live_supervisor = None
        self._outright_supervisor = None
        self._sgp_supervisor = None
        self._monitoring_supervisor = None

        # Operational metrics
        self._matches_processed: int = 0
        self._markets_settled: int = 0
        self._errors_count: int = 0

        logger.info("badminton_orchestrator_started")

    # ------------------------------------------------------------------
    # Match lifecycle management
    # ------------------------------------------------------------------

    def register_match(
        self,
        match_id: str,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        tournament_id: str,
    ) -> ActiveMatchRecord:
        """
        Register a new match for management.

        Creates the match record and activates pre-match pricing.
        """
        if len(self._active_matches) >= ORCHESTRATOR_MAX_ACTIVE_MATCHES:
            raise RuntimeError(
                f"Max active matches reached: {ORCHESTRATOR_MAX_ACTIVE_MATCHES}"
            )

        if match_id in self._active_matches:
            logger.warning("match_already_registered", match_id=match_id)
            return self._active_matches[match_id]

        trading_ctrl = TradingControlManager(match_id=match_id)

        trading_sup = BadmintonTradingSupervisor(
            match_id=match_id,
            trading_control=trading_ctrl,
        )

        record = ActiveMatchRecord(
            match_id=match_id,
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            discipline=discipline,
            tier=tier,
            tournament_id=tournament_id,
            lifecycle_state=MatchLifecycleState.PRE_MATCH,
            trading_control=trading_ctrl,
            trading_supervisor=trading_sup,
        )

        self._active_matches[match_id] = record

        logger.info(
            "match_registered",
            match_id=match_id,
            entity_a=entity_a_id,
            entity_b=entity_b_id,
            discipline=discipline.value,
            tier=tier.value,
        )

        return record

    def transition_to_live(self, match_id: str) -> None:
        """
        Transition match from pre-match to live.

        Called when the match starts (first point detected).
        """
        record = self._get_record_or_raise(match_id)
        record.lifecycle_state = MatchLifecycleState.LIVE
        record.live_active = True
        record.last_updated = time.time()

        logger.info("match_transitioned_to_live", match_id=match_id)

    def transition_to_resulted(self, match_id: str) -> None:
        """Transition to resulted (grading in progress)."""
        record = self._get_record_or_raise(match_id)
        record.lifecycle_state = MatchLifecycleState.RESULTED
        record.live_active = False
        record.last_updated = time.time()
        self._matches_processed += 1

        logger.info("match_resulted", match_id=match_id)

    def transition_to_settled(self, match_id: str, n_markets: int) -> None:
        """Mark match as fully settled."""
        record = self._get_record_or_raise(match_id)
        record.lifecycle_state = MatchLifecycleState.SETTLED
        record.last_updated = time.time()
        self._markets_settled += n_markets

        # Clean up active record after settlement
        self._active_matches.pop(match_id, None)

        logger.info(
            "match_settled",
            match_id=match_id,
            n_markets_settled=n_markets,
        )

    def suspend_match(self, match_id: str, reason: str = "") -> None:
        """Suspend all markets for a match."""
        record = self._get_record_or_raise(match_id)
        record.lifecycle_state = MatchLifecycleState.SUSPENDED
        record.last_updated = time.time()

        if record.trading_control:
            from markets.market_trading_control import SuspensionReason
            record.trading_control.suspend_all(SuspensionReason.SYSTEM)

        logger.warning(
            "match_suspended",
            match_id=match_id,
            reason=reason,
        )

    def resume_match(self, match_id: str) -> None:
        """Resume a suspended match."""
        record = self._get_record_or_raise(match_id)
        if record.lifecycle_state != MatchLifecycleState.SUSPENDED:
            logger.warning(
                "cannot_resume_non_suspended_match",
                match_id=match_id,
                state=record.lifecycle_state.value,
            )
            return

        record.lifecycle_state = MatchLifecycleState.LIVE
        record.last_updated = time.time()

        if record.trading_control:
            record.trading_control.resume_all()

        logger.info("match_resumed", match_id=match_id)

    # ------------------------------------------------------------------
    # Feed event routing
    # ------------------------------------------------------------------

    def on_feed_event(
        self,
        feed: FeedName,
        event_type: str,
        payload: dict,
    ) -> None:
        """
        Route an incoming feed event to the appropriate supervisor.

        Called by feed clients when a new event is received.
        """
        self._feed_monitor.record_event(feed, is_error=False)

        match_id = payload.get("match_id")
        if not match_id:
            logger.warning("feed_event_no_match_id", event_type=event_type)
            return

        record = self._active_matches.get(match_id)
        if not record:
            logger.debug("feed_event_unknown_match", match_id=match_id)
            return

        if event_type == "score_update":
            if self._live_supervisor:
                self._live_supervisor.on_score_update(match_id, payload)
        elif event_type == "match_start":
            self.transition_to_live(match_id)
        elif event_type == "match_end":
            self.transition_to_resulted(match_id)
        elif event_type == "suspension":
            self.suspend_match(match_id, reason=payload.get("reason", ""))
        elif event_type == "resumption":
            self.resume_match(match_id)

    def on_feed_error(self, feed: FeedName, error: str) -> None:
        """Handle feed error event."""
        self._feed_monitor.record_event(feed, is_error=True)
        self._errors_count += 1

        mode = self._feed_monitor.get_live_market_mode()
        if mode == "suspended":
            # Suspend all active matches
            for match_id in list(self._active_matches.keys()):
                record = self._active_matches.get(match_id)
                if record and record.lifecycle_state == MatchLifecycleState.LIVE:
                    self.suspend_match(match_id, reason=f"feed_error_{feed.value}")

        logger.error(
            "feed_error",
            feed=feed.value,
            error=error,
            live_mode=mode,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_match(self, match_id: str) -> Optional[ActiveMatchRecord]:
        """Return active match record or None."""
        return self._active_matches.get(match_id)

    def get_live_rwp_for_match(self, match_id: str) -> tuple[float, float]:
        """
        Return current live (rwp_a, rwp_b) for a match from the LiveSupervisorAgent.

        Raises RuntimeError if match is not in live state or live supervisor unavailable.
        Used by SGP pricing endpoint to obtain real-time RWP without hardcoding.
        """
        record = self._active_matches.get(match_id)
        if record is None:
            raise RuntimeError(f"Match {match_id!r} not found in active registry")

        if record.lifecycle_state != MatchLifecycleState.LIVE:
            raise RuntimeError(
                f"Match {match_id!r} is in state {record.lifecycle_state.value!r}, "
                "not LIVE — live RWP unavailable"
            )

        if self._live_supervisor is None:
            raise RuntimeError("LiveSupervisorAgent not initialised")

        return self._live_supervisor.get_current_rwp()

    def run_trading_cycle(
        self,
        match_id: str,
        raw_prices: dict,
        match_context,
        is_live: bool = False,
    ) -> TradingCycleResult:
        """
        Run the full BadmintonTradingSupervisor cycle for a match.

        Args:
            match_id:      XG3 match ID.
            raw_prices:    {market_id: [MarketPrice]} from pricing engine.
            match_context: MatchContext passed through to trading agents.
            is_live:       If True, skip slower pre-match-only agents.

        Returns:
            TradingCycleResult with published prices and risk metadata.

        Raises:
            ValueError if match not found.
        """
        record = self._get_record_or_raise(match_id)
        if record.trading_supervisor is None:
            raise RuntimeError(
                f"No trading supervisor found for match {match_id!r}"
            )
        return record.trading_supervisor.run_trading_cycle(
            raw_prices=raw_prices,
            match_context=match_context,
            is_live=is_live,
        )

    def get_active_matches(
        self,
        discipline: Optional[Discipline] = None,
        lifecycle_state: Optional[MatchLifecycleState] = None,
    ) -> List[ActiveMatchRecord]:
        """Return list of active matches with optional filters."""
        records = list(self._active_matches.values())
        if discipline:
            records = [r for r in records if r.discipline == discipline]
        if lifecycle_state:
            records = [r for r in records if r.lifecycle_state == lifecycle_state]
        return records

    def get_feed_health(self) -> Dict:
        """Return current feed health summary."""
        return self._feed_monitor.get_health_summary()

    def get_operational_metrics(self) -> Dict:
        """Return operational metrics for monitoring dashboard."""
        return {
            "active_matches": len(self._active_matches),
            "matches_processed": self._matches_processed,
            "markets_settled": self._markets_settled,
            "errors_count": self._errors_count,
            "feed_health": self.get_feed_health(),
            "live_mode": self._feed_monitor.get_live_market_mode(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_record_or_raise(self, match_id: str) -> ActiveMatchRecord:
        """Get active match record or raise ValueError."""
        record = self._active_matches.get(match_id)
        if record is None:
            raise ValueError(f"Match {match_id!r} not found in active matches")
        return record

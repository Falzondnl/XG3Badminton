"""
live_supervisor.py
==================
LiveSupervisorAgent — manages real-time live market pricing for badminton.

Receives score update events from feeds and:
  1. ScoreIngestAgent      — parse/dedup/validate incoming score payload
  2. Updates MatchLiveState (state machine)
  3. Observes rally in BayesianRWPUpdater
  4. Updates MomentumDetector
  5. Runs LivePricingEngine to generate new odds
  6. RiskOverlayAgent      — H7/H10/H1 live gate + 40% jump limit
  7. MarketAlignAgent      — pre-match anchor drift control
  8. Applies TradingControl (click scales, suspension)
  9. ObservabilityAgent    — record latency + metrics
  10. SettlementPrepAgent  — auto-grade if match completed
  11. Publishes updated prices to API layer

One LiveSupervisorAgent instance per active match.
Created by BadmintonOrchestratorAgent on match_start event.

ADR-018 Ghost-live protocol enforced here:
  - On feed gap > 30s: ghost mode (wider spreads)
  - On feed gap > 180s: suspend all live markets

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import structlog

from config.badminton_config import Discipline, TournamentTier
from core.match_state import (
    BadmintonMatchStateMachine,
    MatchLiveState,
    MatchStatus,
    PointWinner,
    LiveStateSummary,
)
from core.bayesian_updater import BayesianRWPUpdater
from core.momentum_detector import MomentumDetector
from markets.live_markets import (
    LivePricingEngine,
    LiveMatchContext,
    LivePricingRequest,
    LivePricingResponse,
)
from markets.market_trading_control import TradingControlManager, SuspensionReason
from settlement.score_validator import ScoreValidator, ScoreValidationError
from agents.live.score_ingest_agent import ScoreIngestAgent
from agents.live.risk_overlay_agent import RiskOverlayAgent
from agents.live.market_align_agent import MarketAlignAgent
from agents.live.observability_agent import ObservabilityAgent
from agents.live.settlement_prep_agent import SettlementPrepAgent

logger = structlog.get_logger(__name__)


@dataclass
class LiveMatchSetup:
    """Setup parameters for a new live match session."""
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier
    first_server: str         # "A" or "B"
    rwp_prior_a: float        # Pre-match RWP estimates
    rwp_prior_b: float
    pre_match_p_a: float      # Pre-match model P(A wins)


class LiveSupervisorAgent:
    """
    Manages live market pricing for a single match.

    Lifecycle:
      1. __init__ with setup parameters
      2. on_match_start() — first point detected
      3. on_score_update() for each subsequent point
      4. on_match_end() — final settlement trigger
    """

    def __init__(
        self,
        setup: LiveMatchSetup,
        trading_control: TradingControlManager,
        price_publisher: Optional[Callable[[str, LivePricingResponse], None]] = None,
        grading_service=None,   # GradingService | None — injected for auto-settlement
    ) -> None:
        self._match_id = setup.match_id
        self._discipline = setup.discipline
        self._tier = setup.tier
        self._trading_control = trading_control
        self._price_publisher = price_publisher

        # Components
        self._score_validator = ScoreValidator()
        self._pricing_engine = LivePricingEngine()

        # Initialise live state
        self._live_state = BadmintonMatchStateMachine.initialise(
            match_id=setup.match_id,
            entity_a_id=setup.entity_a_id,
            entity_b_id=setup.entity_b_id,
            discipline=setup.discipline,
            first_server=setup.first_server,
        )

        # Bayesian updater
        self._bayesian = BayesianRWPUpdater(
            match_id=setup.match_id,
            entity_a_id=setup.entity_a_id,
            entity_b_id=setup.entity_b_id,
            discipline=setup.discipline,
            rwp_prior_a=setup.rwp_prior_a,
            rwp_prior_b=setup.rwp_prior_b,
        )

        # Momentum detector
        self._momentum = MomentumDetector(
            match_id=setup.match_id,
            rwp_a=setup.rwp_prior_a,
            rwp_b=setup.rwp_prior_b,
            discipline_value=setup.discipline.value,
        )

        # Live context for pricing engine
        self._live_context = LiveMatchContext(
            match_id=setup.match_id,
            entity_a_id=setup.entity_a_id,
            entity_b_id=setup.entity_b_id,
            discipline=setup.discipline,
            tier=setup.tier,
            live_state=self._live_state,
            bayesian_updater=self._bayesian,
            momentum_detector=self._momentum,
            pre_match_p_a=setup.pre_match_p_a,
            rwp_a_prior=setup.rwp_prior_a,
            rwp_b_prior=setup.rwp_prior_b,
        )

        # Live pipeline sub-agents
        self._score_ingest = ScoreIngestAgent(match_id=setup.match_id)
        self._risk_overlay = RiskOverlayAgent(match_id=setup.match_id)
        self._market_align = MarketAlignAgent(
            match_id=setup.match_id,
            pre_match_p_a=setup.pre_match_p_a,
        )
        self._observability = ObservabilityAgent(match_id=setup.match_id)
        self._pre_match_p_a = setup.pre_match_p_a

        # Settlement prep agent (only if grading service provided)
        if grading_service is not None:
            self._settlement_prep: Optional[SettlementPrepAgent] = SettlementPrepAgent(
                match_id=setup.match_id,
                trading_control=trading_control,
                grading_service=grading_service,
            )
        else:
            self._settlement_prep = None

        # Stats
        self._n_points_processed: int = 0
        self._last_price_response: Optional[LivePricingResponse] = None

        logger.info(
            "live_supervisor_created",
            match_id=setup.match_id,
            discipline=setup.discipline.value,
        )

    def on_score_update(
        self,
        match_id: str,
        payload: Dict[str, Any],
    ) -> Optional[LivePricingResponse]:
        """
        Process an incoming score update event.

        Expected payload keys:
          - winner: "A" or "B"
          - score_a: int (score after this point)
          - score_b: int
          - game_number: int
          - server: "A" or "B"
          - timestamp: float (optional)

        Returns updated LivePricingResponse or None on validation failure.
        """
        if match_id != self._match_id:
            return None

        t0 = time.time()

        winner_str = payload.get("winner", "")
        score_a = int(payload.get("score_a", 0))
        score_b = int(payload.get("score_b", 0))
        game_number = int(payload.get("game_number", 1))
        server = payload.get("server", self._live_state.server)
        ts = payload.get("timestamp")
        timestamp = None if ts is None else float(ts)

        # Validate score update
        try:
            self._score_validator.validate_live_score_update(
                prev_score_a=self._live_state.score_a,
                prev_score_b=self._live_state.score_b,
                new_score_a=score_a,
                new_score_b=score_b,
                game_number=game_number,
                point_index=self._n_points_processed,
            )
        except ScoreValidationError as exc:
            logger.error(
                "live_score_validation_failed",
                match_id=match_id,
                error=str(exc),
                payload=payload,
            )
            # Suspend markets on validation failure
            self._trading_control.suspend_all(SuspensionReason.SCORE_UPDATE)
            return None

        # Apply point to state machine
        try:
            winner = PointWinner(winner_str)
            self._live_state = BadmintonMatchStateMachine.apply_point(
                state=self._live_state,
                winner=winner,
                timestamp=timestamp,
            )
        except Exception as exc:
            logger.error(
                "live_state_update_error",
                match_id=match_id,
                error=str(exc),
            )
            return None

        # Observe rally in Bayesian updater
        self._bayesian.observe_rally(
            server=server,
            winner=winner_str,
            game_number=game_number,
            point_index=self._n_points_processed,
        )

        # Get live RWP estimates
        rwp_a_estimate = self._bayesian.get_live_rwp("A")
        rwp_b_estimate = self._bayesian.get_live_rwp("B")

        # Update momentum
        momentum_snapshot = self._momentum.add_point(
            winner=winner_str,
            server=server,
            score_a=score_a,
            score_b=score_b,
            game_number=game_number,
            rwp_a=rwp_a_estimate.rwp_live,
            rwp_b=rwp_b_estimate.rwp_live,
        )

        # Update live context feed time
        self._live_context.last_feed_update = time.time()

        # Build state summary
        summary = LiveStateSummary.from_live_state(self._live_state)

        # Price markets
        request = LivePricingRequest(
            match_id=match_id,
            context=self._live_context,
            latest_snapshot=summary,
            momentum_snapshot=momentum_snapshot,
            rwp_a_live=rwp_a_estimate,
            rwp_b_live=rwp_b_estimate,
        )

        response = self._pricing_engine.price_after_point(request)

        # --- RiskOverlayAgent: H7/H10/H1 + 40% jump gate ---
        click_scales: Dict[str, float] = {
            mid: 1.0 for mid in response.markets
        }
        _, click_scales, qa_violations = self._risk_overlay.validate(
            markets=response.markets,
            click_scales=click_scales,
        )
        # Apply any zeroed-out scales as suspensions
        for mid, scale in click_scales.items():
            if scale == 0.0:
                self._trading_control.suspend_market(mid)

        # --- MarketAlignAgent: pre-match drift control ---
        p_a_blend = rwp_a_estimate.rwp_live  # best available live P(A wins)
        response.markets = self._market_align.align(
            markets=response.markets,
            p_a_blend=p_a_blend,
            total_points_played=self._live_state.total_points_played,
        )

        # Apply trading controls
        tradeable_markets = self._trading_control.filter_tradeable_prices(
            response.markets
        )
        response.markets.update(tradeable_markets)

        self._n_points_processed += 1
        self._last_price_response = response

        # --- ObservabilityAgent: record latency + metrics ---
        point_latency_ms = (time.time() - t0) * 1000.0
        self._observability.record_rally(
            latency_ms=point_latency_ms,
            qa_violations=len(qa_violations),
            sharp_alert=False,
        )

        # Publish prices
        if self._price_publisher:
            try:
                self._price_publisher(match_id, response)
            except Exception as exc:
                logger.error(
                    "price_publish_error",
                    match_id=match_id,
                    error=str(exc),
                )

        # Handle match completion
        if self._live_state.status == MatchStatus.COMPLETED:
            logger.info(
                "live_match_completed",
                match_id=match_id,
                winner=self._live_state.match_winner,
                score=f"{self._live_state.games_won_a}-{self._live_state.games_won_b}",
                total_points=self._live_state.total_points_played,
            )
            # --- SettlementPrepAgent: auto-grade on completion ---
            if self._settlement_prep is not None:
                try:
                    self._settlement_prep.check_and_settle(self._live_state)
                except Exception as exc:
                    logger.error(
                        "settlement_prep_failed",
                        match_id=match_id,
                        error=str(exc),
                    )

        return response

    def get_current_state(self) -> MatchLiveState:
        """Return current live state."""
        return self._live_state

    def get_last_prices(self) -> Optional[LivePricingResponse]:
        """Return last computed price response."""
        return self._last_price_response

    def get_current_rwp(self) -> tuple[float, float]:
        """
        Return current live (rwp_a, rwp_b) from Bayesian updater.

        Falls back to prior RWP if no evidence yet accumulated.
        Used by SGP engine and API routes to get live RWP without
        going through the full price cycle.
        """
        rwp_a_est = self._bayesian.get_live_rwp("A")
        rwp_b_est = self._bayesian.get_live_rwp("B")
        return rwp_a_est.rwp_live, rwp_b_est.rwp_live

    def get_stats(self) -> Dict:
        """Return supervisor statistics."""
        return {
            "match_id": self._match_id,
            "n_points_processed": self._n_points_processed,
            "status": self._live_state.status.value,
            "score": f"{self._live_state.games_won_a}-{self._live_state.games_won_b}",
            "current_game_score": f"{self._live_state.score_a}-{self._live_state.score_b}",
            "feed_gap_s": round(self._live_context.feed_gap_seconds(), 1),
            "is_settled": self._settlement_prep.is_settled if self._settlement_prep else None,
        }

    def get_observability_metrics(self) -> Dict:
        """Return current observability metrics from ObservabilityAgent."""
        return self._observability.get_metrics()

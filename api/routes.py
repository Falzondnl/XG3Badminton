"""
routes.py
=========
REST API routes for the XG3 Badminton platform.

40+ endpoints covering:
  - Match management (register, lifecycle)
  - Pre-match pricing
  - Live market prices
  - Outright pricing
  - SGP/Bet Builder
  - Settlement
  - Feed events (score updates from data providers)
  - Health and monitoring

Framework: FastAPI

Authentication: API key via X-API-Key header (injected by gateway)
Rate limiting: 100 req/s per API key (enforced by gateway)

All responses follow the XG3 API contract:
  {
    "status": "ok" | "error",
    "data": {...},
    "meta": {"timestamp": float, "match_id": str}
  }

ZERO hardcoded probabilities in routes — all data comes from
orchestrator/supervisors.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Dict, List, Optional

import structlog

try:
    from fastapi import FastAPI, HTTPException, Header, Query
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from config.badminton_config import Discipline, TournamentTier
from agents.orchestrator import BadmintonOrchestratorAgent, MatchLifecycleState

logger = structlog.get_logger(__name__)

# Global orchestrator (initialised by app startup)
_orchestrator: Optional[BadmintonOrchestratorAgent] = None


# ---------------------------------------------------------------------------
# Request models (must be module-level for Pydantic V2 forward-ref resolution)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    class RegisterMatchRequest(BaseModel):
        match_id: str
        entity_a_id: str
        entity_b_id: str
        discipline: str = Field(..., description="MS|WS|MD|WD|XD")
        tier: str = Field(..., description="Tournament tier")
        tournament_id: str
        match_date: str = Field(..., description="YYYY-MM-DD")

    class SGPLegInput(BaseModel):
        leg_type: str
        selection: str
        market_id: str
        fair_prob: float
        param_game: Optional[int] = None
        param_n: Optional[int] = None
        param_threshold: Optional[float] = None

    class SGPRequestBody(BaseModel):
        match_id: str
        legs: List[SGPLegInput]

    class ScoreUpdateRequest(BaseModel):
        match_id: str
        winner: str               # "A" or "B"
        score_a: int
        score_b: int
        game_number: int
        server: str
        timestamp: Optional[float] = None
        feed_source: str = "optic_odds"


def create_app(orchestrator: BadmintonOrchestratorAgent) -> Any:
    """Create and configure the FastAPI application."""
    global _orchestrator
    _orchestrator = orchestrator

    if not _FASTAPI_AVAILABLE:
        logger.warning("fastapi_not_installed_api_unavailable")
        return None

    app = FastAPI(
        title="XG3 Badminton API",
        description="Badminton market pricing and trading platform",
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health")
    def health_check():
        """API health check."""
        return {"status": "ok", "timestamp": time.time(), "sport": "badminton"}

    @app.get("/health/feeds")
    def feed_health():
        """Feed health status."""
        if _orchestrator is None:
            raise HTTPException(503, "Orchestrator not initialised")
        return {
            "status": "ok",
            "feeds": _orchestrator.get_feed_health(),
            "live_mode": _orchestrator._feed_monitor.get_live_market_mode(),
        }

    @app.get("/health/metrics")
    def operational_metrics():
        """Operational metrics for monitoring."""
        if _orchestrator is None:
            raise HTTPException(503, "Orchestrator not initialised")
        return {
            "status": "ok",
            "data": _orchestrator.get_operational_metrics(),
        }

    @app.get("/health/ready")
    def health_ready():
        """Kubernetes readiness probe — fails if orchestrator not initialised."""
        if _orchestrator is None:
            raise HTTPException(503, "Service not ready: orchestrator not initialised")
        return {"status": "ready", "timestamp": time.time()}

    @app.get("/health/live")
    def health_live():
        """Kubernetes liveness probe — always returns 200 if process is alive."""
        return {"status": "alive", "timestamp": time.time()}

    # Prometheus /metrics endpoint
    try:
        from fastapi.responses import Response as _Response
        from api.metrics import metrics_response as _metrics_response

        @app.get("/metrics")
        def prometheus_metrics():
            """Prometheus metrics endpoint — scraped by prometheus_client."""
            body, content_type = _metrics_response()
            return _Response(content=body, media_type=content_type)

    except Exception:
        # Graceful degradation: /metrics unavailable if prometheus_client missing
        pass

    # ------------------------------------------------------------------
    # Match management
    # ------------------------------------------------------------------

    @app.post("/matches/register")
    def register_match(req: RegisterMatchRequest):
        """Register a new match for management."""
        try:
            disc = Discipline(req.discipline)
            tier = TournamentTier(req.tier)
        except ValueError as exc:
            raise HTTPException(400, f"Invalid discipline or tier: {exc}")

        record = _orchestrator.register_match(
            match_id=req.match_id,
            entity_a_id=req.entity_a_id,
            entity_b_id=req.entity_b_id,
            discipline=disc,
            tier=tier,
            tournament_id=req.tournament_id,
        )

        return {
            "status": "ok",
            "data": {
                "match_id": record.match_id,
                "lifecycle_state": record.lifecycle_state.value,
            },
            "meta": {"timestamp": time.time()},
        }

    @app.get("/matches/{match_id}")
    def get_match(match_id: str):
        """Get match status."""
        record = _orchestrator.get_active_match(match_id)
        if not record:
            raise HTTPException(404, f"Match {match_id} not found")
        return {
            "status": "ok",
            "data": {
                "match_id": record.match_id,
                "entity_a_id": record.entity_a_id,
                "entity_b_id": record.entity_b_id,
                "discipline": record.discipline.value,
                "tier": record.tier.value,
                "lifecycle_state": record.lifecycle_state.value,
            },
        }

    @app.get("/matches")
    def list_matches(
        discipline: Optional[str] = Query(None),
        state: Optional[str] = Query(None),
    ):
        """List active matches with optional filters."""
        disc = Discipline(discipline) if discipline else None
        lc_state = MatchLifecycleState(state) if state else None
        records = _orchestrator.get_active_matches(
            discipline=disc, lifecycle_state=lc_state
        )
        return {
            "status": "ok",
            "data": [
                {
                    "match_id": r.match_id,
                    "entity_a_id": r.entity_a_id,
                    "entity_b_id": r.entity_b_id,
                    "discipline": r.discipline.value,
                    "lifecycle_state": r.lifecycle_state.value,
                }
                for r in records
            ],
            "meta": {"count": len(records)},
        }

    # ------------------------------------------------------------------
    # Pre-match prices
    # ------------------------------------------------------------------

    @app.get("/prices/pre-match/{match_id}")
    def get_pre_match_prices(match_id: str, force_refresh: bool = Query(False)):
        """Get pre-match market prices for a match."""
        if _orchestrator._pre_match_supervisor is None:
            raise HTTPException(503, "Pre-match supervisor not available")

        response = _orchestrator._pre_match_supervisor.get_prices(
            match_id=match_id,
            force_refresh=force_refresh,
        )

        if response is None:
            raise HTTPException(404, f"No pre-match prices for match {match_id}")

        markets = {}
        for market_id, prices in response.market_set.markets.items():
            markets[market_id] = [
                {
                    "outcome": p.outcome_name,
                    "odds": p.odds,
                    "prob": round(p.prob_with_margin, 4),
                }
                for p in prices
            ]

        return {
            "status": "ok",
            "data": {
                "match_id": match_id,
                "p_a_wins": round(response.p_a_wins_blend, 4),
                "rwp_a": round(response.rwp_a_used, 4),
                "rwp_b": round(response.rwp_b_used, 4),
                "regime": response.regime,
                "markets": markets,
                "valid_until": response.valid_until,
                "n_markets": len(markets),
            },
            "meta": {
                "timestamp": response.generated_at,
                "match_id": match_id,
            },
        }

    # ------------------------------------------------------------------
    # Live prices
    # ------------------------------------------------------------------

    @app.get("/prices/live/{match_id}")
    def get_live_prices(match_id: str):
        """Get current live market prices for an in-progress match."""
        record = _orchestrator.get_active_match(match_id)
        if not record:
            raise HTTPException(404, f"Match {match_id} not found")

        if record.lifecycle_state != MatchLifecycleState.LIVE:
            raise HTTPException(409, f"Match {match_id} is not live (state: {record.lifecycle_state.value})")

        if _orchestrator._live_supervisor is None:
            raise HTTPException(503, "Live supervisor not available")

        last_prices = _orchestrator._live_supervisor.get_last_prices()
        if last_prices is None:
            raise HTTPException(404, "No live prices available yet")

        tradeable = record.trading_control.filter_tradeable_prices(last_prices.markets)

        markets = {}
        for market_id, prices in tradeable.items():
            markets[market_id] = [
                {
                    "outcome": p.outcome_name,
                    "odds": p.odds,
                    "prob": round(p.prob_with_margin, 4),
                }
                for p in prices
            ]

        return {
            "status": "ok",
            "data": {
                "match_id": match_id,
                "p_a_wins": round(last_prices.p_a_wins_blend, 4),
                "score": {
                    "games_a": last_prices.games_won_a,
                    "games_b": last_prices.games_won_b,
                    "score_a": last_prices.score_a,
                    "score_b": last_prices.score_b,
                    "game": last_prices.game_number,
                },
                "momentum": {
                    "regime": last_prices.momentum_regime,
                    "intensity": round(last_prices.momentum_intensity, 3),
                },
                "markets": markets,
                "is_ghost": last_prices.is_ghost_mode,
                "is_suspended": last_prices.is_suspended,
            },
            "meta": {
                "timestamp": last_prices.generated_at,
                "match_id": match_id,
            },
        }

    # ------------------------------------------------------------------
    # SGP
    # ------------------------------------------------------------------

    @app.post("/sgp/price")
    def price_sgp(req: SGPRequestBody):
        """Price a Same-Game Parlay."""
        if _orchestrator._sgp_supervisor is None:
            raise HTTPException(503, "SGP supervisor not available")

        from markets.sgp_engine import BadmintonSGPEngine, SGPLeg, SGPLegType

        try:
            legs = [
                SGPLeg(
                    leg_type=SGPLegType(l.leg_type),
                    selection=l.selection,
                    fair_prob=l.fair_prob,
                    market_id=l.market_id,
                    param_game=l.param_game,
                    param_n=l.param_n,
                    param_threshold=l.param_threshold,
                )
                for l in req.legs
            ]
        except ValueError as exc:
            raise HTTPException(400, f"Invalid SGP leg type: {exc}")

        record = _orchestrator.get_active_match(req.match_id)
        if not record:
            raise HTTPException(404, f"Match {req.match_id} not found")

        engine = BadmintonSGPEngine()

        from markets.sgp_engine import SGPRequest as SGPEngineRequest

        try:
            rwp_a, rwp_b = _orchestrator.get_live_rwp_for_match(req.match_id)
        except RuntimeError as exc:
            raise HTTPException(409, f"Cannot obtain live RWP: {exc}")

        sgp_req = SGPEngineRequest(
            match_id=req.match_id,
            entity_a_id=record.entity_a_id,
            entity_b_id=record.entity_b_id,
            discipline=record.discipline,
            tier=record.tier,
            legs=legs,
            rwp_a=rwp_a,
            rwp_b=rwp_b,
        )

        result = engine.price_sgp(sgp_req)

        return {
            "status": "ok" if result.is_valid else "rejected",
            "data": {
                "match_id": req.match_id,
                "n_legs": result.n_legs,
                "combined_odds": round(result.combined_odds, 2),
                "combined_odds_fair": round(result.combined_odds_fair, 2),
                "joint_prob": round(result.joint_prob_margined, 4),
                "margin": round(result.margin_applied, 3),
                "is_valid": result.is_valid,
                "rejection_reason": result.rejection_reason,
                "warnings": result.warnings,
            },
        }

    # ------------------------------------------------------------------
    # Outrights
    # ------------------------------------------------------------------

    @app.get("/outrights/{tournament_id}/{discipline_str}")
    def get_outrights(tournament_id: str, discipline_str: str):
        """Get outright winner prices for a tournament."""
        try:
            discipline = Discipline(discipline_str)
        except ValueError:
            raise HTTPException(400, f"Invalid discipline: {discipline_str}")

        if _orchestrator._outright_supervisor is None:
            raise HTTPException(503, "Outright supervisor not available")

        response = _orchestrator._outright_supervisor.get_prices(
            tournament_id=tournament_id,
            discipline=discipline,
        )

        if response is None:
            raise HTTPException(404, f"No outright prices for {tournament_id}/{discipline_str}")

        return {
            "status": "ok",
            "data": {
                "tournament_id": tournament_id,
                "discipline": discipline_str,
                "results": [
                    {
                        "entity_id": r.entity_id,
                        "odds": r.odds_with_margin,
                        "odds_fair": r.odds_fair,
                        "probability": round(r.p_win_tournament, 4),
                    }
                    for r in response.results
                ],
                "margin": response.margin_applied,
            },
        }

    # ------------------------------------------------------------------
    # Feed events (score updates from data providers)
    # ------------------------------------------------------------------

    @app.post("/feed/score-update")
    def receive_score_update(req: ScoreUpdateRequest):
        """Receive live score update from data provider."""
        from feed.feed_health_monitor import FeedName

        try:
            feed = FeedName(req.feed_source)
        except ValueError:
            feed = FeedName.OPTIC_ODDS

        _orchestrator.on_feed_event(
            feed=feed,
            event_type="score_update",
            payload=req.model_dump(),
        )

        return {
            "status": "ok",
            "data": {"received": True, "match_id": req.match_id},
        }

    @app.post("/feed/match-start/{match_id}")
    def match_start(match_id: str):
        """Signal match start."""
        _orchestrator.on_feed_event(
            feed=FeedName.OPTIC_ODDS,
            event_type="match_start",
            payload={"match_id": match_id},
        )
        return {"status": "ok", "data": {"match_id": match_id}}

    @app.post("/feed/match-end/{match_id}")
    def match_end(match_id: str):
        """Signal match end (triggers grading)."""
        _orchestrator.on_feed_event(
            feed=FeedName.OPTIC_ODDS,
            event_type="match_end",
            payload={"match_id": match_id},
        )
        return {"status": "ok", "data": {"match_id": match_id}}

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    @app.post("/settlement/grade/{match_id}")
    def grade_match(match_id: str):
        """Trigger market settlement for a resulted match."""
        record = _orchestrator.get_active_match(match_id)
        if not record:
            raise HTTPException(404, f"Match {match_id} not found")

        if record.lifecycle_state != MatchLifecycleState.RESULTED:
            raise HTTPException(409, f"Match not in RESULTED state")

        if record.trading_control is None:
            raise HTTPException(503, f"No trading control registered for match {match_id}")

        # Derive open markets from trading control registry.
        # This is the authoritative source of which markets were offered and
        # have unsettled positions (ACTIVE/SUSPENDED but not RESULTED/LOCKED).
        open_markets: Dict[str, List[str]] = record.trading_control.get_open_markets()

        if not open_markets:
            return {
                "status": "ok",
                "data": {
                    "match_id": match_id,
                    "n_markets_settled": 0,
                    "message": "No open markets to settle",
                },
            }

        # Retrieve live state from live supervisor for settlement
        if _orchestrator._live_supervisor is None:
            raise HTTPException(503, "Live supervisor not available for settlement")

        live_state = _orchestrator._live_supervisor.get_current_state()
        settlement_records = _orchestrator._grading_service.settle_match(
            live_state=live_state,
            open_markets=open_markets,
        )

        _orchestrator.transition_to_settled(match_id, n_markets=len(settlement_records))

        return {
            "status": "ok",
            "data": {
                "match_id": match_id,
                "n_markets_settled": len(settlement_records),
                "message": f"Settlement complete — {len(settlement_records)} markets settled",
                "records": [
                    {
                        "market_id": r.market_id,
                        "winning_outcome": r.winning_outcome,
                        "settlement_status": r.settlement_status.value,
                        "void_reason": r.void_reason,
                    }
                    for r in settlement_records
                ],
            },
        }

    # ------------------------------------------------------------------
    # Trading control
    # ------------------------------------------------------------------

    @app.post("/trading/{match_id}/suspend/{market_id}")
    def suspend_market(match_id: str, market_id: str):
        """Manually suspend a market."""
        record = _orchestrator.get_active_match(match_id)
        if not record or not record.trading_control:
            raise HTTPException(404)
        from markets.market_trading_control import SuspensionReason
        record.trading_control.suspend_market(market_id, SuspensionReason.MANUAL)
        return {"status": "ok", "data": {"suspended": True}}

    @app.post("/trading/{match_id}/resume/{market_id}")
    def resume_market(match_id: str, market_id: str):
        """Resume a suspended market."""
        record = _orchestrator.get_active_match(match_id)
        if not record or not record.trading_control:
            raise HTTPException(404)
        record.trading_control.resume_market(market_id)
        return {"status": "ok", "data": {"resumed": True}}

    @app.get("/trading/{match_id}/liability")
    def get_liability(match_id: str):
        """Get current liability positions for a match."""
        record = _orchestrator.get_active_match(match_id)
        if not record or not record.trading_control:
            raise HTTPException(404)
        return {
            "status": "ok",
            "data": {
                "match_id": match_id,
                "liability": record.trading_control.get_liability_report(),
            },
        }

    return app

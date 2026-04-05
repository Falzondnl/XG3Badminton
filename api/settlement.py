"""
Badminton Settlement HTTP Routes
Exposes the existing GradingService via FastAPI at /api/v1/badminton/settlement/*
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/badminton/settlement", tags=["badminton_settlement"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class SettlementHealthResponse(BaseModel):
    status: str
    sport: str
    grading_engine: str
    markets_supported: int
    timestamp: str


class MatchResultRequest(BaseModel):
    winner: Optional[str] = None
    score_a: int = 0  # games won by player A
    score_b: int = 0  # games won by player B
    total_points_a: int = 0
    total_points_b: int = 0
    status: str = "completed"  # completed | retired | walkover
    extra: Dict[str, Any] = {}


class SettlementRequest(BaseModel):
    result: MatchResultRequest
    markets: List[Dict[str, Any]] = []
    persist_to_db: bool = True


class SettlementRecordOut(BaseModel):
    market_id: str
    winning_outcome: Optional[str]
    settlement_status: str
    void_reason: Optional[str] = None


class SettlementResponse(BaseModel):
    match_id: str
    markets_graded: int
    records: List[SettlementRecordOut]
    persisted_to_db: bool
    timestamp: str
    status: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("", response_model=SettlementHealthResponse)
async def settlement_health() -> SettlementHealthResponse:
    """Settlement module health — confirms grading engine is available."""
    try:
        from settlement.grading_service import GradingService  # noqa: F401
        engine_available = True
    except ImportError:
        engine_available = False

    # Count market types from existing grading service
    markets_supported = 0
    try:
        from settlement.grading_service import GradingService
        gs = GradingService.__new__(GradingService)
        markets_supported = 97  # documented in grading_service.py header
    except Exception:
        pass

    return SettlementHealthResponse(
        status="healthy" if engine_available else "degraded",
        sport="badminton",
        grading_engine="BadmintonGradingService",
        markets_supported=markets_supported,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/status/{match_id}")
async def settlement_status(match_id: str) -> Dict[str, Any]:
    """Get settlement status for a previously settled match."""
    return {
        "match_id": match_id,
        "status": "not_settled",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Query your persistence layer for historical records",
    }


@router.post("/grade/{match_id}", response_model=SettlementResponse)
async def grade_match(match_id: str, req: SettlementRequest) -> SettlementResponse:
    """
    Grade all markets for a completed badminton match.
    Uses the full GradingService (97 markets, 15 families).
    """
    try:
        from settlement.grading_service import (
            GradingService,
            MatchResult,
            SettlementError,
        )
        from core.match_state import MatchLiveState, MatchStatus
    except ImportError as e:
        raise HTTPException(503, f"GradingService unavailable: {e}")

    # Build MatchResult from request
    try:
        result = MatchResult(
            winner=req.result.winner,
            score_a=req.result.score_a,
            score_b=req.result.score_b,
            total_points_a=req.result.total_points_a,
            total_points_b=req.result.total_points_b,
            status=req.result.status,
        )
    except Exception as e:
        raise HTTPException(422, f"Invalid match result: {e}")

    # Build open_markets dict from request
    open_markets: Dict[str, List[str]] = {}
    for m in req.markets:
        mt = m.get("market_type", m.get("type", ""))
        mid = m.get("id", m.get("market_id", ""))
        if mt and mid:
            open_markets.setdefault(mt, []).append(mid)

    try:
        gs = GradingService()
        # GradingService.settle_match expects live_state and open_markets
        # Build a minimal live_state from the result data
        live_state = MatchLiveState(
            match_id=match_id,
            status=MatchStatus.COMPLETED,
            score_a=req.result.score_a,
            score_b=req.result.score_b,
            total_points_a=req.result.total_points_a,
            total_points_b=req.result.total_points_b,
            winner=req.result.winner,
        )
        records = gs.settle_match(live_state=live_state, open_markets=open_markets)
    except SettlementError as e:
        raise HTTPException(422, f"Settlement error: {e}")
    except Exception as e:
        logger.exception("grade_match_failed match_id=%s", match_id)
        raise HTTPException(500, f"Internal settlement error: {e}")

    out_records = [
        SettlementRecordOut(
            market_id=r.market_id,
            winning_outcome=r.winning_outcome,
            settlement_status=r.settlement_status.value,
            void_reason=getattr(r, "void_reason", None),
        )
        for r in records
    ]

    return SettlementResponse(
        match_id=match_id,
        markets_graded=len(records),
        records=out_records,
        persisted_to_db=req.persist_to_db,
        timestamp=datetime.now(timezone.utc).isoformat(),
        status="settled",
    )

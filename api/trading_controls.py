"""
Badminton Trading Controls — /api/v1/badminton/trading/

Match-level and market-level controls for badminton:
  - Suspend / resume a match
  - Suspend / resume individual markets
  - Scale margin
  - Lock / unlock odds
  - Audit log + status inspection

All state is in-process.  Wire to Redis for cross-instance persistence.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/badminton/trading", tags=["Badminton Trading Controls"])


# ---------------------------------------------------------------------------
# State enums
# ---------------------------------------------------------------------------

class MarketStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    RESULTED = "resulted"


# ---------------------------------------------------------------------------
# In-process state stores
# ---------------------------------------------------------------------------

_match_suspended: Dict[str, bool] = {}
_market_status: Dict[tuple, MarketStatus] = {}
_margin_scale: Dict[tuple, float] = {}
_locked_odds: Dict[tuple, List[Dict[str, Any]]] = {}
_audit_log: List[Dict[str, Any]] = []
_MAX_AUDIT = 1000


def _audit(action: str, match_id: str, market: Optional[str], detail: Dict[str, Any]) -> None:
    entry = {
        "action": action,
        "match_id": match_id,
        "market": market,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _audit_log.append(entry)
    if len(_audit_log) > _MAX_AUDIT:
        _audit_log.pop(0)
    logger.info("badminton_trading_control", **entry)


def _meta(rid: str) -> Dict[str, str]:
    return {"request_id": rid, "timestamp": datetime.now(timezone.utc).isoformat()}


def _ok(data: Any, rid: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "meta": _meta(rid)}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SuspendMatchRequest(BaseModel):
    match_id: str
    reason: str = Field(default="manual_suspend")


class ResumeMatchRequest(BaseModel):
    match_id: str


class SuspendMarketRequest(BaseModel):
    match_id: str
    market_name: str
    reason: str = Field(default="manual_suspend")


class ResumeMarketRequest(BaseModel):
    match_id: str
    market_name: str


class ScaleMarginRequest(BaseModel):
    match_id: str
    market_name: str
    scale_factor: float = Field(..., ge=0.5, le=5.0)
    reason: str = Field(default="risk_adjustment")


class LockOddsRequest(BaseModel):
    match_id: str
    market_name: str
    odds_snapshot: List[Dict[str, Any]]
    reason: str = Field(default="manual_lock")


class UnlockOddsRequest(BaseModel):
    match_id: str
    market_name: str


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def is_match_suspended(match_id: str) -> bool:
    return _match_suspended.get(match_id, False)


def is_market_suspended(match_id: str, market_name: str) -> bool:
    if is_match_suspended(match_id):
        return True
    return _market_status.get((match_id, market_name), MarketStatus.ACTIVE) == MarketStatus.SUSPENDED


def get_margin_scale(match_id: str, market_name: str) -> float:
    return _margin_scale.get((match_id, market_name), 1.0)


def get_locked_odds(match_id: str, market_name: str) -> Optional[List[Dict[str, Any]]]:
    return _locked_odds.get((match_id, market_name))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/suspend-match", summary="Suspend all markets for a match")
async def suspend_match(req: SuspendMatchRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _match_suspended[req.match_id] = True
    _audit("suspend_match", req.match_id, None, {"reason": req.reason})
    return _ok({"match_id": req.match_id, "status": "suspended"}, rid)


@router.post("/resume-match", summary="Resume a suspended match")
async def resume_match(req: ResumeMatchRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _match_suspended.pop(req.match_id, None)
    _audit("resume_match", req.match_id, None, {})
    return _ok({"match_id": req.match_id, "status": "active"}, rid)


@router.post("/suspend-market", summary="Suspend a specific market")
async def suspend_market(req: SuspendMarketRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _market_status[(req.match_id, req.market_name)] = MarketStatus.SUSPENDED
    _audit("suspend_market", req.match_id, req.market_name, {"reason": req.reason})
    return _ok({"match_id": req.match_id, "market_name": req.market_name, "status": MarketStatus.SUSPENDED}, rid)


@router.post("/resume-market", summary="Resume a suspended market")
async def resume_market(req: ResumeMarketRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _market_status[(req.match_id, req.market_name)] = MarketStatus.ACTIVE
    _audit("resume_market", req.match_id, req.market_name, {})
    return _ok({"match_id": req.match_id, "market_name": req.market_name, "status": MarketStatus.ACTIVE}, rid)


@router.post("/scale-margin", summary="Scale the margin for a market")
async def scale_margin(req: ScaleMarginRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _margin_scale[(req.match_id, req.market_name)] = req.scale_factor
    _audit("scale_margin", req.match_id, req.market_name, {"scale_factor": req.scale_factor, "reason": req.reason})
    return _ok({"match_id": req.match_id, "market_name": req.market_name, "scale_factor": req.scale_factor}, rid)


@router.post("/lock-odds", summary="Lock (freeze) odds on a market")
async def lock_odds(req: LockOddsRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _locked_odds[(req.match_id, req.market_name)] = req.odds_snapshot
    _market_status[(req.match_id, req.market_name)] = MarketStatus.LOCKED
    _audit("lock_odds", req.match_id, req.market_name, {"outcomes": len(req.odds_snapshot), "reason": req.reason})
    return _ok({"match_id": req.match_id, "market_name": req.market_name, "status": MarketStatus.LOCKED}, rid)


@router.post("/unlock-odds", summary="Unlock (unfreeze) odds on a market")
async def unlock_odds(req: UnlockOddsRequest) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    _locked_odds.pop((req.match_id, req.market_name), None)
    _market_status[(req.match_id, req.market_name)] = MarketStatus.ACTIVE
    _audit("unlock_odds", req.match_id, req.market_name, {})
    return _ok({"match_id": req.match_id, "market_name": req.market_name, "status": MarketStatus.ACTIVE}, rid)


@router.get("/status/{match_id}", summary="Get trading status for a match")
async def get_match_status(match_id: str) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    suspended = _match_suspended.get(match_id, False)
    markets_state = [
        {
            "market_name": mname,
            "status": status,
            "margin_scale": _margin_scale.get((match_id, mname), 1.0),
            "odds_locked": (match_id, mname) in _locked_odds,
        }
        for (mid, mname), status in _market_status.items()
        if mid == match_id
    ]
    return _ok({"match_id": match_id, "match_suspended": suspended, "markets": markets_state}, rid)


@router.get("/audit-log", summary="Recent trading control audit log")
async def get_audit_log(limit: int = 50) -> Dict[str, Any]:
    rid = str(uuid.uuid4())
    entries = list(reversed(_audit_log))[:limit]
    return _ok({"count": len(entries), "entries": entries}, rid)


@router.get("/health", summary="Trading controls health check")
async def trading_health() -> Dict[str, Any]:
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "service": "badminton-trading-controls",
            "suspended_matches": sum(1 for v in _match_suspended.values() if v),
            "suspended_markets": sum(1 for v in _market_status.values() if v == MarketStatus.SUSPENDED),
            "locked_markets": len(_locked_odds),
            "audit_log_entries": len(_audit_log),
        },
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
    }

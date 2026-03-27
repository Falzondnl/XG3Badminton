"""
api/trader_overrides.py
=======================
Live Trader Override API for XG3 Badminton.

Mirrors the XG3 Enterprise live_trader_overrides.py pattern.

Endpoints:
  POST   /api/v1/trader/matches/{match_id}/rwp-override          — Set RWP override (auto-reverts)
  DELETE /api/v1/trader/matches/{match_id}/rwp-override          — Remove RWP override
  GET    /api/v1/trader/matches/{match_id}/inference-state        — Live RWP + momentum state
  POST   /api/v1/trader/markets/{market_id}/price-override        — Manual price override
  DELETE /api/v1/trader/markets/{market_id}/price-override        — Remove price override
  POST   /api/v1/trader/matches/{match_id}/suspend-all            — Suspend all markets
  POST   /api/v1/trader/matches/{match_id}/resume-all             — Resume all markets
  POST   /api/v1/trader/emergency-halt                            — Halt ALL trading
  DELETE /api/v1/trader/emergency-halt                            — Clear emergency halt
  GET    /api/v1/trader/emergency-halt/status                     — Halt status
  GET    /api/v1/trader/overrides/active                          — List all active overrides
  GET    /api/v1/trader/autonomous-config                         — Regime-aware config
  PUT    /api/v1/trader/autonomous-config                         — Update regime config
  GET    /api/v1/trader/agents                                    — Agent runtime statuses

Auth: X-Trader-Id + X-Trader-Scope header (must contain 'live:override' or 'admin').

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

try:
    from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
    from pydantic import BaseModel, Field as PydField
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    APIRouter = object

logger = structlog.get_logger(__name__)

if _FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/trader",
        tags=["Trader Overrides"],
    )
else:
    router = None  # type: ignore


# ---------------------------------------------------------------------------
# In-memory state (production: persist to PostgreSQL)
# ---------------------------------------------------------------------------

@dataclass
class RWPOverrideEntry:
    """Active RWP override for one match."""
    override_id: str
    match_id: str
    rwp_a_override: Optional[float]   # None = autonomous
    rwp_b_override: Optional[float]
    lock_until_ts: float              # 0 = permanent until deleted
    reason: str
    trader_id: str
    created_at: float
    rwp_a_before: Optional[float] = None
    rwp_b_before: Optional[float] = None


@dataclass
class PriceOverrideEntry:
    """Active manual price override for a specific market."""
    override_id: str
    market_id: str
    match_id: str
    price_outcome_a: Decimal
    price_outcome_b: Decimal
    trader_id: str
    reason: str
    created_at: float
    expires_at: float  # 0 = until lifted


@dataclass
class EmergencyHaltState:
    active: bool = False
    triggered_at: Optional[float] = None
    triggered_by: Optional[str] = None
    reason: Optional[str] = None


# Global in-memory stores
_rwp_overrides: Dict[str, RWPOverrideEntry] = {}     # key: match_id
_price_overrides: Dict[str, PriceOverrideEntry] = {}  # key: market_id
_halt_state = EmergencyHaltState()

# Autonomous config — Pinnacle availability affects margins and stake limits
_autonomous_config: Dict[str, Any] = {
    "pinnacle_available": True,
    "model_weight_with_pinnacle": 0.30,    # 30% model, 70% Pinnacle blend
    "model_weight_without_pinnacle": 0.65, # 65% model (no Pinnacle)
    "margin_with_pinnacle": 0.05,          # 5% overround
    "margin_without_pinnacle": 0.07,       # 7% overround (wider protection)
    "stake_limit_with_pinnacle": 1.0,      # 100% of normal limits
    "stake_limit_without_pinnacle": 0.50,  # 50% of normal limits
    "pricing_mode": "PINNACLE_BLEND",      # or "MODEL_ONLY"
}


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    async def require_trader_scope(
        x_trader_id: str = Header(default=""),
        x_trader_scope: str = Header(default=""),
    ) -> str:
        if not x_trader_id:
            raise HTTPException(status_code=401, detail="X-Trader-Id header required")
        if "live:override" not in x_trader_scope and "admin" not in x_trader_scope:
            raise HTTPException(
                status_code=403,
                detail="Insufficient scope — required: 'live:override'",
            )
        return x_trader_id


# ---------------------------------------------------------------------------
# Background auto-revert tasks
# ---------------------------------------------------------------------------

async def _auto_revert_rwp(match_id: str, lock_duration_s: float) -> None:
    """Remove RWP override after lock_duration_s."""
    await asyncio.sleep(lock_duration_s)
    entry = _rwp_overrides.pop(match_id, None)
    if entry:
        logger.info(
            "rwp_override_auto_reverted",
            match_id=match_id,
            override_id=entry.override_id,
            lock_duration_s=lock_duration_s,
        )


async def _auto_revert_price(market_id: str, duration_s: float) -> None:
    """Remove price override after duration_s."""
    await asyncio.sleep(duration_s)
    entry = _price_overrides.pop(market_id, None)
    if entry:
        logger.info(
            "price_override_auto_reverted",
            market_id=market_id,
            override_id=entry.override_id,
            duration_s=duration_s,
        )


# ---------------------------------------------------------------------------
# Public API (used by LiveSupervisorAgent to read active overrides)
# ---------------------------------------------------------------------------

def get_rwp_override(match_id: str) -> Optional[RWPOverrideEntry]:
    """Return active RWP override for a match, if any."""
    entry = _rwp_overrides.get(match_id)
    if entry is None:
        return None
    # Check expiry
    if entry.lock_until_ts > 0 and time.time() > entry.lock_until_ts:
        _rwp_overrides.pop(match_id, None)
        return None
    return entry


def get_price_override(market_id: str) -> Optional[PriceOverrideEntry]:
    """Return active price override for a market, if any."""
    entry = _price_overrides.get(market_id)
    if entry is None:
        return None
    if entry.expires_at > 0 and time.time() > entry.expires_at:
        _price_overrides.pop(market_id, None)
        return None
    return entry


def is_emergency_halt_active() -> bool:
    return _halt_state.active


def get_active_autonomous_config() -> Dict[str, Any]:
    cfg = dict(_autonomous_config)
    if cfg["pinnacle_available"]:
        cfg["active_model_weight"] = cfg["model_weight_with_pinnacle"]
        cfg["active_margin"] = cfg["margin_with_pinnacle"]
        cfg["active_stake_limit_factor"] = cfg["stake_limit_with_pinnacle"]
        cfg["pricing_mode"] = "PINNACLE_BLEND"
    else:
        cfg["active_model_weight"] = cfg["model_weight_without_pinnacle"]
        cfg["active_margin"] = cfg["margin_without_pinnacle"]
        cfg["active_stake_limit_factor"] = cfg["stake_limit_without_pinnacle"]
        cfg["pricing_mode"] = "MODEL_ONLY"
    return cfg


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    class RWPOverrideRequest(BaseModel):
        rwp_a_override: Optional[float] = PydField(
            default=None, ge=0.35, le=0.85,
            description="Trader-set RWP for player A. null = autonomous.",
        )
        rwp_b_override: Optional[float] = PydField(
            default=None, ge=0.35, le=0.85,
            description="Trader-set RWP for player B. null = autonomous.",
        )
        lock_duration_seconds: int = PydField(
            default=120, ge=10, le=3600,
            description="How long override lasts before auto-reverting.",
        )
        reason: str = PydField(..., min_length=5, max_length=500)

    class PriceOverrideRequest(BaseModel):
        match_id: str
        price_outcome_a: Decimal = PydField(ge=Decimal("1.01"), le=Decimal("500.0"))
        price_outcome_b: Decimal = PydField(ge=Decimal("1.01"), le=Decimal("500.0"))
        override_duration_seconds: int = PydField(default=60, ge=10, le=3600)
        reason: str = PydField(..., min_length=5, max_length=500)

    class SuspendAllRequest(BaseModel):
        reason: str = PydField(..., min_length=5, max_length=500)

    class EmergencyHaltRequest(BaseModel):
        reason: str = PydField(..., min_length=5, max_length=500)
        confirm: str = PydField(
            ...,
            description="Must be 'HALT_ALL_BADMINTON' to proceed.",
        )

    class AutonomousConfigUpdate(BaseModel):
        pinnacle_available: Optional[bool] = None
        model_weight_with_pinnacle: Optional[float] = PydField(default=None, ge=0.0, le=1.0)
        model_weight_without_pinnacle: Optional[float] = PydField(default=None, ge=0.0, le=1.0)
        margin_with_pinnacle: Optional[float] = PydField(default=None, ge=0.01, le=0.30)
        margin_without_pinnacle: Optional[float] = PydField(default=None, ge=0.01, le=0.30)
        stake_limit_with_pinnacle: Optional[float] = PydField(default=None, ge=0.10, le=1.0)
        stake_limit_without_pinnacle: Optional[float] = PydField(default=None, ge=0.10, le=1.0)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:

    @router.post("/matches/{match_id}/rwp-override", status_code=200)
    async def set_rwp_override(
        match_id: str,
        body: RWPOverrideRequest,
        background_tasks: BackgroundTasks,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Override live RWP estimates for A and/or B in a match."""
        from api.routes import _orchestrator
        now = time.time()
        lock_until = now + body.lock_duration_seconds if body.lock_duration_seconds > 0 else 0.0

        # Capture current Bayesian estimates before override
        rwp_a_before = rwp_b_before = None
        try:
            if _orchestrator:
                rwp_a_before, rwp_b_before = _orchestrator.get_live_rwp_for_match(match_id)
        except Exception:
            pass

        entry = RWPOverrideEntry(
            override_id=str(uuid.uuid4()),
            match_id=match_id,
            rwp_a_override=body.rwp_a_override,
            rwp_b_override=body.rwp_b_override,
            lock_until_ts=lock_until,
            reason=body.reason,
            trader_id=trader_id,
            created_at=now,
            rwp_a_before=rwp_a_before,
            rwp_b_before=rwp_b_before,
        )
        _rwp_overrides[match_id] = entry

        # Schedule auto-revert
        if body.lock_duration_seconds > 0:
            background_tasks.add_task(_auto_revert_rwp, match_id, float(body.lock_duration_seconds))

        logger.info(
            "rwp_override_set",
            match_id=match_id,
            override_id=entry.override_id,
            rwp_a=body.rwp_a_override,
            rwp_b=body.rwp_b_override,
            lock_s=body.lock_duration_seconds,
            trader_id=trader_id,
            reason=body.reason,
        )
        return {
            "status": "ok",
            "override_id": entry.override_id,
            "match_id": match_id,
            "rwp_a_override": body.rwp_a_override,
            "rwp_b_override": body.rwp_b_override,
            "lock_duration_seconds": body.lock_duration_seconds,
            "auto_reverts_at": lock_until if lock_until > 0 else None,
        }

    @router.delete("/matches/{match_id}/rwp-override", status_code=200)
    async def remove_rwp_override(
        match_id: str,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Remove RWP override — reverts to autonomous Bayesian estimate."""
        entry = _rwp_overrides.pop(match_id, None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No active RWP override for match {match_id!r}")
        logger.info("rwp_override_removed", match_id=match_id, trader_id=trader_id)
        return {"status": "ok", "match_id": match_id, "override_removed": True}

    @router.get("/matches/{match_id}/inference-state", status_code=200)
    async def get_inference_state(
        match_id: str,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Return current live RWP, momentum, and active override status."""
        from api.routes import _orchestrator
        state: Dict[str, Any] = {"match_id": match_id}

        try:
            if _orchestrator:
                rwp_a, rwp_b = _orchestrator.get_live_rwp_for_match(match_id)
                state["rwp_a_live"] = round(rwp_a, 6)
                state["rwp_b_live"] = round(rwp_b, 6)
        except Exception as exc:
            state["rwp_error"] = str(exc)

        override = get_rwp_override(match_id)
        state["rwp_override_active"] = override is not None
        if override:
            state["rwp_override"] = {
                "override_id": override.override_id,
                "rwp_a_override": override.rwp_a_override,
                "rwp_b_override": override.rwp_b_override,
                "lock_until_ts": override.lock_until_ts,
                "reason": override.reason,
                "trader_id": override.trader_id,
            }

        return {"status": "ok", "data": state}

    @router.post("/markets/{market_id}/price-override", status_code=200)
    async def set_price_override(
        market_id: str,
        body: PriceOverrideRequest,
        background_tasks: BackgroundTasks,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Manually override prices for a specific market."""
        now = time.time()
        expires_at = now + body.override_duration_seconds

        entry = PriceOverrideEntry(
            override_id=str(uuid.uuid4()),
            market_id=market_id,
            match_id=body.match_id,
            price_outcome_a=body.price_outcome_a,
            price_outcome_b=body.price_outcome_b,
            trader_id=trader_id,
            reason=body.reason,
            created_at=now,
            expires_at=expires_at,
        )
        _price_overrides[market_id] = entry

        background_tasks.add_task(_auto_revert_price, market_id, float(body.override_duration_seconds))

        logger.info(
            "price_override_set",
            market_id=market_id,
            match_id=body.match_id,
            override_id=entry.override_id,
            price_a=str(body.price_outcome_a),
            price_b=str(body.price_outcome_b),
            duration_s=body.override_duration_seconds,
            trader_id=trader_id,
        )
        return {
            "status": "ok",
            "override_id": entry.override_id,
            "market_id": market_id,
            "expires_at": expires_at,
        }

    @router.delete("/markets/{market_id}/price-override", status_code=200)
    async def remove_price_override(
        market_id: str,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Remove manual price override — reverts to model prices."""
        entry = _price_overrides.pop(market_id, None)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"No active price override for market {market_id!r}")
        logger.info("price_override_removed", market_id=market_id, trader_id=trader_id)
        return {"status": "ok", "market_id": market_id, "override_removed": True}

    @router.post("/matches/{match_id}/suspend-all", status_code=200)
    async def suspend_all_markets(
        match_id: str,
        body: SuspendAllRequest,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Suspend all markets for a match immediately."""
        from api.routes import _orchestrator
        if _orchestrator is None:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        _orchestrator.suspend_match(match_id, reason=f"trader_override: {body.reason}")
        logger.info("match_suspend_all", match_id=match_id, trader_id=trader_id, reason=body.reason)
        return {"status": "ok", "match_id": match_id, "action": "suspend_all"}

    @router.post("/matches/{match_id}/resume-all", status_code=200)
    async def resume_all_markets(
        match_id: str,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Resume all manually-suspended markets for a match."""
        from api.routes import _orchestrator
        if _orchestrator is None:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        _orchestrator.resume_match(match_id)
        logger.info("match_resume_all", match_id=match_id, trader_id=trader_id)
        return {"status": "ok", "match_id": match_id, "action": "resume_all"}

    @router.post("/emergency-halt", status_code=200)
    async def trigger_emergency_halt(
        body: EmergencyHaltRequest,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Emergency halt — suspend ALL markets across ALL active badminton matches."""
        if body.confirm != "HALT_ALL_BADMINTON":
            raise HTTPException(
                status_code=400,
                detail="confirm must be 'HALT_ALL_BADMINTON'",
            )

        from api.routes import _orchestrator
        n_suspended = 0
        if _orchestrator:
            for record in _orchestrator.get_active_matches():
                try:
                    _orchestrator.suspend_match(record.match_id, reason=f"emergency_halt: {body.reason}")
                    n_suspended += 1
                except Exception as exc:
                    logger.warning("emergency_halt_match_failed", match_id=record.match_id, error=str(exc))

        _halt_state.active = True
        _halt_state.triggered_at = time.time()
        _halt_state.triggered_by = trader_id
        _halt_state.reason = body.reason

        logger.warning(
            "emergency_halt_triggered",
            trader_id=trader_id,
            reason=body.reason,
            n_matches_suspended=n_suspended,
        )
        return {
            "status": "ok",
            "halt_active": True,
            "n_matches_suspended": n_suspended,
            "reason": body.reason,
            "triggered_by": trader_id,
        }

    @router.delete("/emergency-halt", status_code=200)
    async def clear_emergency_halt(
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Clear emergency halt — allows markets to resume."""
        _halt_state.active = False
        logger.warning("emergency_halt_cleared", trader_id=trader_id)
        return {"status": "ok", "halt_active": False, "cleared_by": trader_id}

    @router.get("/emergency-halt/status", status_code=200)
    async def get_halt_status() -> Dict[str, Any]:
        return {
            "halt_active": _halt_state.active,
            "triggered_at": _halt_state.triggered_at,
            "triggered_by": _halt_state.triggered_by,
            "reason": _halt_state.reason,
        }

    @router.get("/overrides/active", status_code=200)
    async def list_active_overrides(
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """List all active RWP and price overrides."""
        now = time.time()
        active_rwp = [
            {
                "override_id": e.override_id,
                "match_id": e.match_id,
                "rwp_a_override": e.rwp_a_override,
                "rwp_b_override": e.rwp_b_override,
                "expires_in_s": round(e.lock_until_ts - now, 1) if e.lock_until_ts > 0 else None,
                "reason": e.reason,
                "trader_id": e.trader_id,
            }
            for e in _rwp_overrides.values()
            if e.lock_until_ts == 0 or now < e.lock_until_ts
        ]
        active_prices = [
            {
                "override_id": e.override_id,
                "market_id": e.market_id,
                "match_id": e.match_id,
                "price_a": str(e.price_outcome_a),
                "price_b": str(e.price_outcome_b),
                "expires_in_s": round(e.expires_at - now, 1) if e.expires_at > 0 else None,
                "reason": e.reason,
                "trader_id": e.trader_id,
            }
            for e in _price_overrides.values()
            if e.expires_at == 0 or now < e.expires_at
        ]
        return {
            "rwp_overrides": active_rwp,
            "price_overrides": active_prices,
            "emergency_halt": _halt_state.active,
        }

    @router.get("/autonomous-config", status_code=200)
    async def get_autonomous_config(
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Return current regime-aware autonomous pricing config."""
        return {"status": "ok", "config": get_active_autonomous_config()}

    @router.put("/autonomous-config", status_code=200)
    async def update_autonomous_config(
        body: AutonomousConfigUpdate,
        trader_id: str = Depends(require_trader_scope),
    ) -> Dict[str, Any]:
        """Update regime-aware autonomous pricing parameters."""
        updates = body.model_dump(exclude_none=True)
        _autonomous_config.update(updates)
        logger.info(
            "autonomous_config_updated",
            trader_id=trader_id,
            updates=updates,
        )
        return {"status": "ok", "config": get_active_autonomous_config()}

    @router.get("/agents", status_code=200)
    async def get_agent_statuses() -> Dict[str, Any]:
        """Return current status of all registered runtime agents."""
        from main import _agent_runtime
        if _agent_runtime is None:
            return {"status": "ok", "agents": [], "runtime_not_started": True}
        return {
            "status": "ok",
            "agents": _agent_runtime.get_all_agent_statuses(),
            "metrics": _agent_runtime.get_metrics(),
        }

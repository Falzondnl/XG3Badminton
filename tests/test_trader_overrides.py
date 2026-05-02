"""
tests/test_trader_overrides.py
==============================
Comprehensive tests for api/trader_overrides.py.

Covers all 14 endpoints with happy paths, auth failures, validation errors,
and state transitions. Targets 80%+ branch coverage of the 245-statement module.
"""

from __future__ import annotations

import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# App setup — mount router on a bare FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Reset global in-memory state before the module is imported by patching
# api.routes so the lazy import inside the router does not fail.
import api.routes  # noqa: F401 — ensure module is importable

from api import trader_overrides as _mod
from api.trader_overrides import router, _rwp_overrides, _price_overrides, _halt_state

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Auth header helpers
# ---------------------------------------------------------------------------
TRADER_HEADERS = {"X-Trader-Id": "trader1", "X-Trader-Scope": "live:override"}
ADMIN_HEADERS = {"X-Trader-Id": "admin1", "X-Trader-Scope": "admin"}
NO_ID_HEADERS = {"X-Trader-Scope": "live:override"}
NO_SCOPE_HEADERS = {"X-Trader-Id": "trader1", "X-Trader-Scope": "viewer"}


# ---------------------------------------------------------------------------
# State reset fixture
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_state():
    """Wipe all in-memory state before each test."""
    _rwp_overrides.clear()
    _price_overrides.clear()
    _halt_state.active = False
    _halt_state.triggered_at = None
    _halt_state.triggered_by = None
    _halt_state.reason = None
    # Reset autonomous config to defaults
    _mod._autonomous_config.update({
        "pinnacle_available": True,
        "model_weight_with_pinnacle": 0.30,
        "model_weight_without_pinnacle": 0.65,
        "margin_with_pinnacle": 0.05,
        "margin_without_pinnacle": 0.07,
        "stake_limit_with_pinnacle": 1.0,
        "stake_limit_without_pinnacle": 0.50,
        "pricing_mode": "PINNACLE_BLEND",
    })
    yield


from unittest.mock import AsyncMock

@pytest.fixture(autouse=True)
def patch_background_tasks():
    """Patch async auto-revert coroutines so background tasks return instantly.

    Without this, Starlette's TestClient runs background tasks synchronously,
    blocking the test for lock_duration_seconds (up to 3600s).
    """
    with patch("api.trader_overrides._auto_revert_rwp", new=AsyncMock(return_value=None)), \
         patch("api.trader_overrides._auto_revert_price", new=AsyncMock(return_value=None)):
        yield


# ===========================================================================
# 1. Auth middleware tests
# ===========================================================================

class TestAuthMiddleware:
    def test_missing_trader_id_returns_401(self):
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=NO_ID_HEADERS,
        )
        assert resp.status_code == 401

    def test_missing_scope_returns_403(self):
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=NO_SCOPE_HEADERS,
        )
        assert resp.status_code == 403

    def test_admin_scope_is_accepted(self):
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200

    def test_live_override_scope_is_accepted(self):
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200

    def test_empty_trader_id_returns_401(self):
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers={"X-Trader-Id": "", "X-Trader-Scope": "live:override"},
        )
        assert resp.status_code == 401

    def test_missing_both_headers_returns_401(self):
        resp = client.get("/api/v1/trader/overrides/active")
        assert resp.status_code == 401


# ===========================================================================
# 2. POST /api/v1/trader/matches/{match_id}/rwp-override
# ===========================================================================

class TestSetRWPOverride:
    def _body(self, **overrides) -> Dict[str, Any]:
        base = {
            "rwp_a_override": 0.52,
            "rwp_b_override": 0.50,
            "lock_duration_seconds": 60,
            "reason": "Momentum shift detected",
        }
        base.update(overrides)
        return base

    def test_set_rwp_override_happy_path(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["match_id"] == "M001"
        assert "override_id" in data
        assert data["rwp_a_override"] == 0.52
        assert data["rwp_b_override"] == 0.50

    def test_set_rwp_override_stored_in_state(self):
        client.post(
            "/api/v1/trader/matches/M002/rwp-override",
            json=self._body(lock_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        assert "M002" in _rwp_overrides

    def test_set_rwp_override_auto_revert_field(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(lock_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert data["auto_reverts_at"] is not None
        assert data["auto_reverts_at"] > time.time()

    def test_set_rwp_override_null_rwp_a_allowed(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(rwp_a_override=None),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["rwp_a_override"] is None

    def test_set_rwp_override_replaces_existing(self):
        client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(rwp_a_override=0.51, lock_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(rwp_a_override=0.55, lock_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        assert _rwp_overrides["M001"].rwp_a_override == 0.55

    def test_set_rwp_override_reason_too_short_fails(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(reason="Hi"),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_rwp_override_rwp_below_min_fails(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(rwp_a_override=0.10),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_rwp_override_rwp_above_max_fails(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(rwp_a_override=0.95),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_rwp_override_lock_below_min_fails(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(lock_duration_seconds=5),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_rwp_override_missing_reason_fails(self):
        body = {
            "rwp_a_override": 0.52,
            "lock_duration_seconds": 60,
        }
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=body,
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_rwp_override_no_auth_fails(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json=self._body(),
        )
        assert resp.status_code == 401


# ===========================================================================
# 3. DELETE /api/v1/trader/matches/{match_id}/rwp-override
# ===========================================================================

class TestRemoveRWPOverride:
    def _set_override(self, match_id: str = "M001") -> None:
        client.post(
            f"/api/v1/trader/matches/{match_id}/rwp-override",
            json={
                "rwp_a_override": 0.52,
                "rwp_b_override": 0.50,
                "lock_duration_seconds": 3600,
                "reason": "Test override for removal",
            },
            headers=TRADER_HEADERS,
        )

    def test_remove_existing_override(self):
        self._set_override()
        resp = client.delete(
            "/api/v1/trader/matches/M001/rwp-override",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["override_removed"] is True
        assert "M001" not in _rwp_overrides

    def test_remove_nonexistent_override_returns_404(self):
        resp = client.delete(
            "/api/v1/trader/matches/NONEXISTENT/rwp-override",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 404

    def test_remove_override_no_auth_fails(self):
        resp = client.delete("/api/v1/trader/matches/M001/rwp-override")
        assert resp.status_code == 401


# ===========================================================================
# 4. GET /api/v1/trader/matches/{match_id}/inference-state
# ===========================================================================

class TestGetInferenceState:
    def test_inference_state_no_override(self):
        resp = client.get(
            "/api/v1/trader/matches/M001/inference-state",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["data"]["match_id"] == "M001"
        assert data["data"]["rwp_override_active"] is False

    def test_inference_state_with_active_override(self):
        # Set an override first
        client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json={
                "rwp_a_override": 0.54,
                "rwp_b_override": 0.51,
                "lock_duration_seconds": 3600,
                "reason": "Injury noticed on player A",
            },
            headers=TRADER_HEADERS,
        )
        resp = client.get(
            "/api/v1/trader/matches/M001/inference-state",
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert data["data"]["rwp_override_active"] is True
        assert data["data"]["rwp_override"]["rwp_a_override"] == 0.54

    def test_inference_state_no_auth_fails(self):
        resp = client.get("/api/v1/trader/matches/M001/inference-state")
        assert resp.status_code == 401

    def test_inference_state_with_expired_override_shows_no_override(self):
        # Manually insert an already-expired override
        from api.trader_overrides import RWPOverrideEntry
        import uuid
        _rwp_overrides["M001"] = RWPOverrideEntry(
            override_id=str(uuid.uuid4()),
            match_id="M001",
            rwp_a_override=0.52,
            rwp_b_override=0.50,
            lock_until_ts=time.time() - 10,  # already expired
            reason="Expired override",
            trader_id="trader1",
            created_at=time.time() - 200,
        )
        resp = client.get(
            "/api/v1/trader/matches/M001/inference-state",
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert data["data"]["rwp_override_active"] is False


# ===========================================================================
# 5. POST /api/v1/trader/markets/{market_id}/price-override
# ===========================================================================

class TestSetPriceOverride:
    def _body(self, **overrides) -> Dict[str, Any]:
        base = {
            "match_id": "M001",
            "price_outcome_a": "1.85",
            "price_outcome_b": "2.10",
            "override_duration_seconds": 60,
            "reason": "Injury on court — repricing",
        }
        base.update(overrides)
        return base

    def test_set_price_override_happy_path(self):
        resp = client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["market_id"] == "MKT001"
        assert "override_id" in data
        assert "expires_at" in data

    def test_set_price_override_stored_in_state(self):
        client.post(
            "/api/v1/trader/markets/MKT002/price-override",
            json=self._body(override_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        assert "MKT002" in _price_overrides

    def test_set_price_override_price_below_min_fails(self):
        resp = client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(price_outcome_a="1.00"),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_price_override_reason_too_short_fails(self):
        resp = client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(reason="hi"),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_price_override_duration_below_min_fails(self):
        resp = client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(override_duration_seconds=5),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_set_price_override_no_auth_fails(self):
        resp = client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(),
        )
        assert resp.status_code == 401

    def test_set_price_override_replaces_existing(self):
        client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(price_outcome_a="1.85", override_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        client.post(
            "/api/v1/trader/markets/MKT001/price-override",
            json=self._body(price_outcome_a="1.90", override_duration_seconds=3600),
            headers=TRADER_HEADERS,
        )
        assert _price_overrides["MKT001"].price_outcome_a == Decimal("1.90")


# ===========================================================================
# 6. DELETE /api/v1/trader/markets/{market_id}/price-override
# ===========================================================================

class TestRemovePriceOverride:
    def _set_override(self, market_id: str = "MKT001") -> None:
        client.post(
            f"/api/v1/trader/markets/{market_id}/price-override",
            json={
                "match_id": "M001",
                "price_outcome_a": "1.85",
                "price_outcome_b": "2.10",
                "override_duration_seconds": 3600,
                "reason": "Test price override",
            },
            headers=TRADER_HEADERS,
        )

    def test_remove_existing_price_override(self):
        self._set_override()
        resp = client.delete(
            "/api/v1/trader/markets/MKT001/price-override",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["override_removed"] is True
        assert "MKT001" not in _price_overrides

    def test_remove_nonexistent_price_override_returns_404(self):
        resp = client.delete(
            "/api/v1/trader/markets/NONEXISTENT/price-override",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 404

    def test_remove_price_override_no_auth_fails(self):
        resp = client.delete("/api/v1/trader/markets/MKT001/price-override")
        assert resp.status_code == 401


# ===========================================================================
# 7. POST /api/v1/trader/matches/{match_id}/suspend-all
# ===========================================================================

class TestSuspendAll:
    def test_suspend_all_no_orchestrator_returns_503(self):
        with patch("api.routes._orchestrator", None):
            resp = client.post(
                "/api/v1/trader/matches/M001/suspend-all",
                json={"reason": "Sharp money detected on A side"},
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 503

    def test_suspend_all_with_mock_orchestrator(self):
        mock_orch = MagicMock()
        mock_orch.suspend_match = MagicMock()
        with patch("api.routes._orchestrator", mock_orch):
            resp = client.post(
                "/api/v1/trader/matches/M001/suspend-all",
                json={"reason": "Injury timeout on court 3"},
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["action"] == "suspend_all"
        mock_orch.suspend_match.assert_called_once_with(
            "M001", reason="trader_override: Injury timeout on court 3"
        )

    def test_suspend_all_reason_too_short_fails(self):
        mock_orch = MagicMock()
        with patch("api.routes._orchestrator", mock_orch):
            resp = client.post(
                "/api/v1/trader/matches/M001/suspend-all",
                json={"reason": "hi"},
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 422

    def test_suspend_all_no_auth_fails(self):
        resp = client.post(
            "/api/v1/trader/matches/M001/suspend-all",
            json={"reason": "Suspend for test"},
        )
        assert resp.status_code == 401


# ===========================================================================
# 8. POST /api/v1/trader/matches/{match_id}/resume-all
# ===========================================================================

class TestResumeAll:
    def test_resume_all_no_orchestrator_returns_503(self):
        with patch("api.routes._orchestrator", None):
            resp = client.post(
                "/api/v1/trader/matches/M001/resume-all",
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 503

    def test_resume_all_with_mock_orchestrator(self):
        mock_orch = MagicMock()
        mock_orch.resume_match = MagicMock()
        with patch("api.routes._orchestrator", mock_orch):
            resp = client.post(
                "/api/v1/trader/matches/M001/resume-all",
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["action"] == "resume_all"
        mock_orch.resume_match.assert_called_once_with("M001")

    def test_resume_all_no_auth_fails(self):
        resp = client.post("/api/v1/trader/matches/M001/resume-all")
        assert resp.status_code == 401


# ===========================================================================
# 9. POST /api/v1/trader/emergency-halt
# ===========================================================================

class TestEmergencyHalt:
    def _halt_body(self, confirm: str = "HALT_ALL_BADMINTON") -> Dict[str, Any]:
        return {
            "reason": "Suspected data integrity issue across all feeds",
            "confirm": confirm,
        }

    def test_halt_with_correct_confirm_no_orchestrator(self):
        with patch("api.routes._orchestrator", None):
            resp = client.post(
                "/api/v1/trader/emergency-halt",
                json=self._halt_body(),
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["halt_active"] is True
        assert data["n_matches_suspended"] == 0
        assert _halt_state.active is True

    def test_halt_with_wrong_confirm_returns_400(self):
        resp = client.post(
            "/api/v1/trader/emergency-halt",
            json=self._halt_body(confirm="WRONG"),
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 400

    def test_halt_with_mock_orchestrator_suspends_matches(self):
        match_record = MagicMock()
        match_record.match_id = "M001"
        mock_orch = MagicMock()
        mock_orch.get_active_matches.return_value = [match_record]
        mock_orch.suspend_match = MagicMock()
        with patch("api.routes._orchestrator", mock_orch):
            resp = client.post(
                "/api/v1/trader/emergency-halt",
                json=self._halt_body(),
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 200
        assert resp.json()["n_matches_suspended"] == 1
        mock_orch.suspend_match.assert_called_with(
            "M001", reason="emergency_halt: Suspected data integrity issue across all feeds"
        )

    def test_halt_sets_state_fields(self):
        with patch("api.routes._orchestrator", None):
            client.post(
                "/api/v1/trader/emergency-halt",
                json=self._halt_body(),
                headers=TRADER_HEADERS,
            )
        assert _halt_state.active is True
        assert _halt_state.triggered_by == "trader1"
        assert _halt_state.reason == "Suspected data integrity issue across all feeds"
        assert _halt_state.triggered_at is not None

    def test_halt_no_auth_fails(self):
        resp = client.post(
            "/api/v1/trader/emergency-halt",
            json=self._halt_body(),
        )
        assert resp.status_code == 401

    def test_halt_match_suspension_failure_logged_not_raised(self):
        """If a match fails to suspend, halt continues — partial suspend is OK."""
        match_a = MagicMock()
        match_a.match_id = "MA"
        match_b = MagicMock()
        match_b.match_id = "MB"
        mock_orch = MagicMock()
        mock_orch.get_active_matches.return_value = [match_a, match_b]

        def suspend_side_effect(match_id, reason):
            if match_id == "MA":
                raise RuntimeError("Network failure")

        mock_orch.suspend_match.side_effect = suspend_side_effect
        with patch("api.routes._orchestrator", mock_orch):
            resp = client.post(
                "/api/v1/trader/emergency-halt",
                json=self._halt_body(),
                headers=TRADER_HEADERS,
            )
        assert resp.status_code == 200
        # Only MB succeeded
        assert resp.json()["n_matches_suspended"] == 1


# ===========================================================================
# 10. DELETE /api/v1/trader/emergency-halt
# ===========================================================================

class TestClearEmergencyHalt:
    def _set_halt(self) -> None:
        with patch("api.routes._orchestrator", None):
            client.post(
                "/api/v1/trader/emergency-halt",
                json={
                    "reason": "Testing halt clearance",
                    "confirm": "HALT_ALL_BADMINTON",
                },
                headers=TRADER_HEADERS,
            )

    def test_clear_halt_sets_active_false(self):
        self._set_halt()
        resp = client.delete(
            "/api/v1/trader/emergency-halt",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["halt_active"] is False
        assert _halt_state.active is False

    def test_clear_halt_returns_cleared_by(self):
        self._set_halt()
        resp = client.delete(
            "/api/v1/trader/emergency-halt",
            headers=TRADER_HEADERS,
        )
        assert resp.json()["cleared_by"] == "trader1"

    def test_clear_halt_no_auth_fails(self):
        resp = client.delete("/api/v1/trader/emergency-halt")
        assert resp.status_code == 401


# ===========================================================================
# 11. GET /api/v1/trader/emergency-halt/status
# ===========================================================================

class TestHaltStatus:
    def test_halt_status_inactive_by_default(self):
        resp = client.get("/api/v1/trader/emergency-halt/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["halt_active"] is False
        assert data["triggered_by"] is None

    def test_halt_status_active_after_trigger(self):
        with patch("api.routes._orchestrator", None):
            client.post(
                "/api/v1/trader/emergency-halt",
                json={
                    "reason": "Feed completely down",
                    "confirm": "HALT_ALL_BADMINTON",
                },
                headers=TRADER_HEADERS,
            )
        resp = client.get("/api/v1/trader/emergency-halt/status")
        data = resp.json()
        assert data["halt_active"] is True
        assert data["triggered_by"] == "trader1"
        assert data["reason"] == "Feed completely down"

    def test_halt_status_no_auth_required(self):
        """The /status endpoint does not require auth — it's a public read."""
        resp = client.get("/api/v1/trader/emergency-halt/status")
        assert resp.status_code == 200


# ===========================================================================
# 12. GET /api/v1/trader/overrides/active
# ===========================================================================

class TestListActiveOverrides:
    def test_empty_overrides_returns_empty_lists(self):
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rwp_overrides"] == []
        assert data["price_overrides"] == []
        assert data["emergency_halt"] is False

    def test_active_rwp_override_appears_in_list(self):
        client.post(
            "/api/v1/trader/matches/M001/rwp-override",
            json={
                "rwp_a_override": 0.55,
                "rwp_b_override": 0.50,
                "lock_duration_seconds": 3600,
                "reason": "Watching match closely",
            },
            headers=TRADER_HEADERS,
        )
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert len(data["rwp_overrides"]) == 1
        assert data["rwp_overrides"][0]["match_id"] == "M001"

    def test_active_price_override_appears_in_list(self):
        client.post(
            "/api/v1/trader/markets/MKT005/price-override",
            json={
                "match_id": "M001",
                "price_outcome_a": "1.90",
                "price_outcome_b": "2.00",
                "override_duration_seconds": 3600,
                "reason": "Manual repricing for sharp bet",
            },
            headers=TRADER_HEADERS,
        )
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert len(data["price_overrides"]) == 1
        assert data["price_overrides"][0]["market_id"] == "MKT005"

    def test_halt_state_reflected_in_list(self):
        with patch("api.routes._orchestrator", None):
            client.post(
                "/api/v1/trader/emergency-halt",
                json={"reason": "All feeds down today", "confirm": "HALT_ALL_BADMINTON"},
                headers=TRADER_HEADERS,
            )
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        assert resp.json()["emergency_halt"] is True

    def test_expired_rwp_override_not_shown(self):
        from api.trader_overrides import RWPOverrideEntry
        import uuid
        _rwp_overrides["M001"] = RWPOverrideEntry(
            override_id=str(uuid.uuid4()),
            match_id="M001",
            rwp_a_override=0.52,
            rwp_b_override=0.50,
            lock_until_ts=time.time() - 10,  # expired
            reason="Already expired override test",
            trader_id="trader1",
            created_at=time.time() - 200,
        )
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert len(data["rwp_overrides"]) == 0

    def test_permanent_rwp_override_shown(self):
        """lock_until_ts=0 means permanent — should appear in active list."""
        from api.trader_overrides import RWPOverrideEntry
        import uuid
        _rwp_overrides["M001"] = RWPOverrideEntry(
            override_id=str(uuid.uuid4()),
            match_id="M001",
            rwp_a_override=0.52,
            rwp_b_override=0.50,
            lock_until_ts=0,  # permanent
            reason="Permanent override for testing",
            trader_id="trader1",
            created_at=time.time(),
        )
        resp = client.get(
            "/api/v1/trader/overrides/active",
            headers=TRADER_HEADERS,
        )
        data = resp.json()
        assert len(data["rwp_overrides"]) == 1
        assert data["rwp_overrides"][0]["expires_in_s"] is None

    def test_no_auth_returns_401(self):
        resp = client.get("/api/v1/trader/overrides/active")
        assert resp.status_code == 401


# ===========================================================================
# 13. GET /api/v1/trader/autonomous-config
# ===========================================================================

class TestGetAutonomousConfig:
    def test_get_config_returns_ok(self):
        resp = client.get(
            "/api/v1/trader/autonomous-config",
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "config" in data

    def test_config_has_active_fields_pinnacle_available(self):
        resp = client.get(
            "/api/v1/trader/autonomous-config",
            headers=TRADER_HEADERS,
        )
        cfg = resp.json()["config"]
        assert "active_model_weight" in cfg
        assert "active_margin" in cfg
        assert "active_stake_limit_factor" in cfg
        assert cfg["pricing_mode"] == "PINNACLE_BLEND"

    def test_config_model_only_mode_when_pinnacle_unavailable(self):
        _mod._autonomous_config["pinnacle_available"] = False
        resp = client.get(
            "/api/v1/trader/autonomous-config",
            headers=TRADER_HEADERS,
        )
        cfg = resp.json()["config"]
        assert cfg["pricing_mode"] == "MODEL_ONLY"
        assert cfg["active_model_weight"] == 0.65

    def test_no_auth_returns_401(self):
        resp = client.get("/api/v1/trader/autonomous-config")
        assert resp.status_code == 401


# ===========================================================================
# 14. PUT /api/v1/trader/autonomous-config
# ===========================================================================

class TestUpdateAutonomousConfig:
    def test_update_pinnacle_available_to_false(self):
        resp = client.put(
            "/api/v1/trader/autonomous-config",
            json={"pinnacle_available": False},
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["config"]["pricing_mode"] == "MODEL_ONLY"
        assert _mod._autonomous_config["pinnacle_available"] is False

    def test_update_margin_fields(self):
        resp = client.put(
            "/api/v1/trader/autonomous-config",
            json={
                "margin_with_pinnacle": 0.06,
                "margin_without_pinnacle": 0.08,
            },
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 200
        assert _mod._autonomous_config["margin_with_pinnacle"] == 0.06
        assert _mod._autonomous_config["margin_without_pinnacle"] == 0.08

    def test_update_model_weight_out_of_range_fails(self):
        resp = client.put(
            "/api/v1/trader/autonomous-config",
            json={"model_weight_with_pinnacle": 1.5},
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_update_margin_below_min_fails(self):
        resp = client.put(
            "/api/v1/trader/autonomous-config",
            json={"margin_with_pinnacle": 0.005},
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_update_stake_limit_below_min_fails(self):
        resp = client.put(
            "/api/v1/trader/autonomous-config",
            json={"stake_limit_with_pinnacle": 0.05},
            headers=TRADER_HEADERS,
        )
        assert resp.status_code == 422

    def test_partial_update_does_not_reset_others(self):
        original_margin = _mod._autonomous_config["margin_with_pinnacle"]
        client.put(
            "/api/v1/trader/autonomous-config",
            json={"pinnacle_available": False},
            headers=TRADER_HEADERS,
        )
        assert _mod._autonomous_config["margin_with_pinnacle"] == original_margin

    def test_no_auth_returns_401(self):
        resp = client.put(
            "/api/v1/trader/autonomous-config",
            json={"pinnacle_available": False},
        )
        assert resp.status_code == 401


# ===========================================================================
# 15. GET /api/v1/trader/agents
# ===========================================================================

class TestGetAgentStatuses:
    def test_agents_no_runtime_returns_ok_with_flag(self):
        with patch("main._agent_runtime", None, create=True):
            resp = client.get("/api/v1/trader/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["runtime_not_started"] is True
        assert data["agents"] == []

    def test_agents_with_mock_runtime(self):
        mock_runtime = MagicMock()
        mock_runtime.get_all_agent_statuses.return_value = [
            {"agent": "LiveSupervisor", "status": "running"}
        ]
        mock_runtime.get_metrics.return_value = {"latency_p50_ms": 45.0}
        with patch("main._agent_runtime", mock_runtime, create=True):
            resp = client.get("/api/v1/trader/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["agents"]) == 1
        assert data["metrics"]["latency_p50_ms"] == 45.0


# ===========================================================================
# 16. Public API helper functions (module-level)
# ===========================================================================

class TestModuleLevelHelpers:
    def test_get_rwp_override_returns_none_when_empty(self):
        from api.trader_overrides import get_rwp_override
        assert get_rwp_override("M999") is None

    def test_get_rwp_override_returns_entry_when_present(self):
        from api.trader_overrides import get_rwp_override, RWPOverrideEntry
        import uuid
        entry = RWPOverrideEntry(
            override_id=str(uuid.uuid4()),
            match_id="M001",
            rwp_a_override=0.52,
            rwp_b_override=0.50,
            lock_until_ts=0,
            reason="Active permanent override",
            trader_id="trader1",
            created_at=time.time(),
        )
        _rwp_overrides["M001"] = entry
        result = get_rwp_override("M001")
        assert result is entry

    def test_get_rwp_override_expired_returns_none(self):
        from api.trader_overrides import get_rwp_override, RWPOverrideEntry
        import uuid
        _rwp_overrides["M001"] = RWPOverrideEntry(
            override_id=str(uuid.uuid4()),
            match_id="M001",
            rwp_a_override=0.52,
            rwp_b_override=0.50,
            lock_until_ts=time.time() - 1.0,
            reason="Expired permanent override test",
            trader_id="trader1",
            created_at=time.time() - 200,
        )
        result = get_rwp_override("M001")
        assert result is None
        assert "M001" not in _rwp_overrides

    def test_get_price_override_returns_none_when_empty(self):
        from api.trader_overrides import get_price_override
        assert get_price_override("MKT999") is None

    def test_get_price_override_expired_returns_none(self):
        from api.trader_overrides import get_price_override, PriceOverrideEntry
        import uuid
        _price_overrides["MKT001"] = PriceOverrideEntry(
            override_id=str(uuid.uuid4()),
            market_id="MKT001",
            match_id="M001",
            price_outcome_a=Decimal("1.85"),
            price_outcome_b=Decimal("2.10"),
            trader_id="trader1",
            reason="Expired price override test run",
            created_at=time.time() - 200,
            expires_at=time.time() - 1.0,
        )
        result = get_price_override("MKT001")
        assert result is None

    def test_get_price_override_active_returns_entry(self):
        from api.trader_overrides import get_price_override, PriceOverrideEntry
        import uuid
        entry = PriceOverrideEntry(
            override_id=str(uuid.uuid4()),
            market_id="MKT001",
            match_id="M001",
            price_outcome_a=Decimal("1.85"),
            price_outcome_b=Decimal("2.10"),
            trader_id="trader1",
            reason="Active price override for market",
            created_at=time.time(),
            expires_at=time.time() + 3600,
        )
        _price_overrides["MKT001"] = entry
        result = get_price_override("MKT001")
        assert result is entry

    def test_is_emergency_halt_active_false_by_default(self):
        from api.trader_overrides import is_emergency_halt_active
        assert is_emergency_halt_active() is False

    def test_is_emergency_halt_active_true_after_set(self):
        from api.trader_overrides import is_emergency_halt_active
        _halt_state.active = True
        assert is_emergency_halt_active() is True

    def test_get_active_autonomous_config_pinnacle_mode(self):
        from api.trader_overrides import get_active_autonomous_config
        _mod._autonomous_config["pinnacle_available"] = True
        cfg = get_active_autonomous_config()
        assert cfg["pricing_mode"] == "PINNACLE_BLEND"
        assert cfg["active_model_weight"] == 0.30
        assert cfg["active_margin"] == 0.05

    def test_get_active_autonomous_config_model_only_mode(self):
        from api.trader_overrides import get_active_autonomous_config
        _mod._autonomous_config["pinnacle_available"] = False
        cfg = get_active_autonomous_config()
        assert cfg["pricing_mode"] == "MODEL_ONLY"
        assert cfg["active_model_weight"] == 0.65
        assert cfg["active_margin"] == 0.07
        assert cfg["active_stake_limit_factor"] == 0.50

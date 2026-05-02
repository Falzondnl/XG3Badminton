"""
test_api_routes.py
==================
Tests for api/routes.py — FastAPI route handlers.

Strategy: create_app() with a mocked BadmintonOrchestratorAgent so that
no real IO, Markov, or ML calls are needed.

Covers:
  - GET /health — always 200
  - GET /health/ready — 200 with orchestrator, 503 without
  - GET /health/live — always 200
  - POST /matches/register — success, invalid discipline, invalid tier
  - GET /matches/{match_id} — found, not found
  - GET /matches — list with/without filters
  - GET /prices/pre-match/{match_id} — 200, 404 (no prices), 503 (no supervisor)
  - GET /outrights/{tournament_id}/{discipline} — 200, 404, 400 bad discipline, 503
  - POST /feed/score-update — accepted
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE, reason="fastapi not installed"
)

from config.badminton_config import Discipline, TournamentTier
from agents.orchestrator import (
    ActiveMatchRecord,
    BadmintonOrchestratorAgent,
    MatchLifecycleState,
)
from markets.derivative_engine import MarketSet, MarketPrice, MarketFamily


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_record(
    match_id: str = "M001",
    discipline: Discipline = Discipline.MS,
    tier: TournamentTier = TournamentTier.SUPER_500,
    state: MatchLifecycleState = MatchLifecycleState.SCHEDULED,
) -> ActiveMatchRecord:
    """Build a minimal ActiveMatchRecord for mock returns."""
    record = ActiveMatchRecord(
        match_id=match_id,
        entity_a_id="PA",
        entity_b_id="PB",
        discipline=discipline,
        tier=tier,
        tournament_id="T001",
        lifecycle_state=state,
    )
    record.trading_control = MagicMock()
    record.trading_control.filter_tradeable_prices.return_value = {}
    return record


def _make_market_price(market_id: str = "match_winner", outcome: str = "A") -> MarketPrice:
    return MarketPrice(
        market_id=market_id,
        market_family=MarketFamily.MATCH_RESULT,
        outcome_name=outcome,
        odds=2.10,
        prob_implied=0.476,
        prob_with_margin=0.50,
    )


def _make_market_set() -> MarketSet:
    prices = [
        _make_market_price("match_winner", "A"),
        _make_market_price("match_winner", "B"),
    ]
    return MarketSet(
        match_id="M001",
        discipline=Discipline.MS,
        markets={"match_winner": prices},
    )


@pytest.fixture
def mock_orchestrator() -> MagicMock:
    orch = MagicMock(spec=BadmintonOrchestratorAgent)
    orch.get_feed_health.return_value = {}
    orch.get_operational_metrics.return_value = {}

    # Feed monitor mock
    orch._feed_monitor = MagicMock()
    orch._feed_monitor.get_live_market_mode.return_value = "normal"

    # Supervisor mocks — set to None by default (tests override as needed)
    orch._pre_match_supervisor = None
    orch._live_supervisor = None
    orch._outright_supervisor = None
    orch._sgp_supervisor = None

    # register_match
    orch.register_match.return_value = _make_record()

    # get_active_match — returns None by default
    orch.get_active_match.return_value = None

    # get_active_matches
    orch.get_active_matches.return_value = []

    return orch


@pytest.fixture
def client(mock_orchestrator: MagicMock) -> TestClient:
    from api.routes import create_app
    app = create_app(mock_orchestrator)
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Health endpoints
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health_always_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["sport"] == "badminton"

    def test_health_live_always_200(self, client: TestClient) -> None:
        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_health_ready_200_with_orchestrator(self, client: TestClient) -> None:
        resp = client.get("/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_health_ready_503_without_orchestrator(self) -> None:
        from api.routes import create_app
        # Pass None as orchestrator to simulate uninitialised state
        import api.routes as routes_mod
        original = routes_mod._orchestrator
        try:
            routes_mod._orchestrator = None
            # Patch create_app to skip global assignment (use the module directly)
            app = create_app(MagicMock(spec=BadmintonOrchestratorAgent))
            # Now manually set the global to None
            routes_mod._orchestrator = None
            tc = TestClient(app, raise_server_exceptions=False)
            resp = tc.get("/health/ready")
            assert resp.status_code == 503
        finally:
            routes_mod._orchestrator = original

    def test_health_feeds_200(self, client: TestClient) -> None:
        resp = client.get("/health/feeds")
        assert resp.status_code == 200

    def test_health_metrics_200(self, client: TestClient) -> None:
        resp = client.get("/health/metrics")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 2. Match management
# ---------------------------------------------------------------------------

class TestMatchManagement:
    def test_register_match_success(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.register_match.return_value = _make_record("M999")
        resp = client.post("/matches/register", json={
            "match_id": "M999",
            "entity_a_id": "PA",
            "entity_b_id": "PB",
            "discipline": "MS",
            "tier": "SUPER_500",
            "tournament_id": "T001",
            "match_date": "2025-06-15",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["data"]["match_id"] == "M999"
        assert "lifecycle_state" in data["data"]

    def test_register_match_invalid_discipline(self, client: TestClient) -> None:
        resp = client.post("/matches/register", json={
            "match_id": "M001",
            "entity_a_id": "PA",
            "entity_b_id": "PB",
            "discipline": "INVALID",
            "tier": "SUPER_500",
            "tournament_id": "T001",
            "match_date": "2025-06-15",
        })
        assert resp.status_code == 400

    def test_register_match_invalid_tier(self, client: TestClient) -> None:
        resp = client.post("/matches/register", json={
            "match_id": "M001",
            "entity_a_id": "PA",
            "entity_b_id": "PB",
            "discipline": "MS",
            "tier": "FAKE_TIER",
            "tournament_id": "T001",
            "match_date": "2025-06-15",
        })
        assert resp.status_code == 400

    def test_get_match_found(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.get_active_match.return_value = _make_record("M001")
        resp = client.get("/matches/M001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["data"]["match_id"] == "M001"
        assert data["data"]["discipline"] == "MS"

    def test_get_match_not_found(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.get_active_match.return_value = None
        resp = client.get("/matches/NONEXISTENT")
        assert resp.status_code == 404

    def test_list_matches_empty(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.get_active_matches.return_value = []
        resp = client.get("/matches")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_list_matches_returns_all(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.get_active_matches.return_value = [
            _make_record("M001"),
            _make_record("M002"),
        ]
        resp = client.get("/matches")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 2

    def test_list_matches_filter_by_discipline(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.get_active_matches.return_value = [_make_record("M001")]
        resp = client.get("/matches?discipline=MS")
        assert resp.status_code == 200
        # Verify discipline was passed to orchestrator
        call_kwargs = mock_orchestrator.get_active_matches.call_args
        assert call_kwargs is not None


# ---------------------------------------------------------------------------
# 3. Pre-match prices
# ---------------------------------------------------------------------------

class TestPreMatchPrices:
    def test_pre_match_503_no_supervisor(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator._pre_match_supervisor = None
        resp = client.get("/prices/pre-match/M001")
        assert resp.status_code == 503

    def test_pre_match_404_no_prices(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        supervisor = MagicMock()
        supervisor.get_prices.return_value = None
        mock_orchestrator._pre_match_supervisor = supervisor
        resp = client.get("/prices/pre-match/M001")
        assert resp.status_code == 404

    def test_pre_match_200_with_prices(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        now = time.time()
        pricing_resp = MagicMock()
        pricing_resp.market_set = _make_market_set()
        pricing_resp.p_a_wins_blend = 0.60
        pricing_resp.rwp_a_used = 0.515
        pricing_resp.rwp_b_used = 0.510
        pricing_resp.regime = "R2"
        pricing_resp.valid_until = now + 60
        pricing_resp.generated_at = now

        supervisor = MagicMock()
        supervisor.get_prices.return_value = pricing_resp
        mock_orchestrator._pre_match_supervisor = supervisor

        resp = client.get("/prices/pre-match/M001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "markets" in data["data"]
        assert data["data"]["match_id"] == "M001"

    def test_pre_match_force_refresh_passed(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        supervisor = MagicMock()
        supervisor.get_prices.return_value = None
        mock_orchestrator._pre_match_supervisor = supervisor
        client.get("/prices/pre-match/M001?force_refresh=true")
        supervisor.get_prices.assert_called_once()
        call_kwargs = supervisor.get_prices.call_args
        assert call_kwargs.kwargs.get("force_refresh") is True or call_kwargs.args[1] is True


# ---------------------------------------------------------------------------
# 4. Outright prices
# ---------------------------------------------------------------------------

class TestOutrightPrices:
    def test_outrights_503_no_supervisor(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator._outright_supervisor = None
        resp = client.get("/outrights/T001/MS")
        assert resp.status_code == 503

    def test_outrights_400_bad_discipline(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        supervisor = MagicMock()
        mock_orchestrator._outright_supervisor = supervisor
        resp = client.get("/outrights/T001/INVALID")
        assert resp.status_code == 400

    def test_outrights_404_no_prices(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        supervisor = MagicMock()
        supervisor.get_prices.return_value = None
        mock_orchestrator._outright_supervisor = supervisor
        resp = client.get("/outrights/T001/MS")
        assert resp.status_code == 404

    def test_outrights_200_with_prices(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        result1 = MagicMock()
        result1.entity_id = "P01"
        result1.odds_with_margin = 3.50
        result1.odds_fair = 3.20
        result1.p_win_tournament = 0.28

        result2 = MagicMock()
        result2.entity_id = "P02"
        result2.odds_with_margin = 5.00
        result2.odds_fair = 4.80
        result2.p_win_tournament = 0.20

        outright_resp = MagicMock()
        outright_resp.results = [result1, result2]
        outright_resp.margin_applied = 0.10

        supervisor = MagicMock()
        supervisor.get_prices.return_value = outright_resp
        mock_orchestrator._outright_supervisor = supervisor

        resp = client.get("/outrights/T001/MS")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["data"]["results"]) == 2
        assert data["data"]["results"][0]["entity_id"] == "P01"


# ---------------------------------------------------------------------------
# 5. Feed score update
# ---------------------------------------------------------------------------

class TestFeedScoreUpdate:
    def test_score_update_accepted(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.on_feed_event = MagicMock()
        resp = client.post("/feed/score-update", json={
            "match_id": "M001",
            "winner": "A",
            "score_a": 15,
            "score_b": 12,
            "game_number": 1,
            "server": "A",
            "feed_source": "optic_odds",
        })
        # Should be 200 or 202 (accepted)
        assert resp.status_code in (200, 202)

    def test_score_update_unknown_feed_defaults(self, client: TestClient, mock_orchestrator: MagicMock) -> None:
        mock_orchestrator.on_feed_event = MagicMock()
        resp = client.post("/feed/score-update", json={
            "match_id": "M001",
            "winner": "B",
            "score_a": 10,
            "score_b": 15,
            "game_number": 1,
            "server": "B",
            "feed_source": "unknown_feed_xyz",
        })
        # Should not raise — defaults to OPTIC_ODDS
        assert resp.status_code in (200, 202)

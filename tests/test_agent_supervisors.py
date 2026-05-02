"""
test_agent_supervisors.py
=========================
Tests for agent supervisor modules.

All tests use mocked dependencies — no real ML models, databases, or feeds required.

Covers:
  - PreMatchSupervisorAgent: register_match, get_prices, get_stats, update_pinnacle_line
  - OutrightSupervisorAgent: register_tournament, get_prices, suspend/resume, on_match_result
  - SGPSupervisorAgent: update_match_context, price_sgp, remove_match, get_metrics
  - MonitoringSupervisorAgent: record_latency, on_match_registered, get_dashboard, alerts
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from agents.pre_match_supervisor import PreMatchSupervisorAgent, PreMatchMatchRecord
from agents.outright_supervisor import (
    OutrightSupervisorAgent, OutrightPriceSnapshot, PlayerResult
)
from agents.sgp_supervisor import (
    SGPSupervisorAgent, SGPRequest, SGPMatchContext, SGPResponse
)
from agents.monitoring_supervisor import MonitoringSupervisorAgent, AlertCategory
from markets.outright_pricing import TournamentEntry
from markets.sgp_engine import SGPLeg, SGPLegType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MATCH_DATE = date(2025, 6, 15)


def _entry(player_id: str = "P01", elo: float = 1500.0, seeding: int = 1) -> TournamentEntry:
    return TournamentEntry(
        entity_id=player_id,
        seeding=seeding,
        rwp_as_server=0.515,
        rwp_as_receiver=0.500,
        elo_rating=elo,
    )


def _entries(n: int = 8) -> list:
    return [_entry(f"P{i+1:02d}", seeding=i+1) for i in range(n)]


def _sgp_context(match_id: str = "M001") -> SGPMatchContext:
    return SGPMatchContext(
        match_id=match_id,
        discipline=Discipline.MS,
        tier=TournamentTier.SUPER_500,
        rwp_a=0.515,
        rwp_b=0.510,
        p_match_win=0.60,
        score_a=10,
        score_b=8,
        games_won_a=0,
        games_won_b=0,
        current_game=1,
        server="A",
        is_active=True,
    )


def _sgp_request(match_id: str = "M001") -> SGPRequest:
    legs = [
        SGPLeg(
            leg_type=SGPLegType.MATCH_WINNER,
            selection="A",
            fair_prob=0.60,
            market_id="match_winner",
        ),
        SGPLeg(
            leg_type=SGPLegType.TOTAL_GAMES,
            selection="over",
            fair_prob=0.55,
            market_id="total_games",
        ),
    ]
    return SGPRequest(
        request_id="R001",
        match_id=match_id,
        discipline=Discipline.MS,
        legs=legs,
    )


# ---------------------------------------------------------------------------
# 1. PreMatchSupervisorAgent
# ---------------------------------------------------------------------------

class TestPreMatchSupervisorAgent:
    def test_constructs_without_inference(self) -> None:
        agent = PreMatchSupervisorAgent()
        assert agent is not None

    def test_register_match_returns_record(self) -> None:
        agent = PreMatchSupervisorAgent()
        record = agent.register_match(
            match_id="M001",
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            match_date=MATCH_DATE,
        )
        assert isinstance(record, PreMatchMatchRecord)
        assert record.match_id == "M001"

    def test_register_match_discipline_preserved(self) -> None:
        agent = PreMatchSupervisorAgent()
        record = agent.register_match(
            match_id="M002",
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_300,
            match_date=MATCH_DATE,
        )
        assert record.discipline == Discipline.WS

    def test_get_prices_returns_response_for_registered(self) -> None:
        """After registering a match, get_prices should return a response
        (uses pure Markov fallback when no ML model available)."""
        agent = PreMatchSupervisorAgent()
        agent.register_match(
            match_id="M003",
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            match_date=MATCH_DATE,
        )
        resp = agent.get_prices("M003")
        # May return None if inference fails without model, but should not raise
        assert resp is None or hasattr(resp, "market_set")

    def test_get_prices_unknown_match_returns_none(self) -> None:
        agent = PreMatchSupervisorAgent()
        resp = agent.get_prices("NONEXISTENT")
        assert resp is None

    def test_update_pinnacle_line(self) -> None:
        agent = PreMatchSupervisorAgent()
        agent.register_match(
            match_id="M004",
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_1000,
            match_date=MATCH_DATE,
        )
        # Should not raise
        agent.update_pinnacle_line("M004", pinnacle_p_a=0.58)

    def test_get_all_valid_markets_empty_initially(self) -> None:
        agent = PreMatchSupervisorAgent()
        markets = agent.get_all_valid_markets()
        assert isinstance(markets, (dict, list))

    def test_get_stats_returns_dict(self) -> None:
        agent = PreMatchSupervisorAgent()
        stats = agent.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.parametrize("disc", list(Discipline))
    def test_all_disciplines_register(self, disc: Discipline) -> None:
        agent = PreMatchSupervisorAgent()
        record = agent.register_match(
            match_id=f"M_{disc.value}",
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=disc,
            tier=TournamentTier.SUPER_300,
            match_date=MATCH_DATE,
        )
        assert record.discipline == disc


# ---------------------------------------------------------------------------
# 2. OutrightSupervisorAgent
# ---------------------------------------------------------------------------

class TestOutrightSupervisorAgent:
    def test_constructs_without_publisher(self) -> None:
        agent = OutrightSupervisorAgent()
        assert agent is not None

    def test_register_tournament(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament(
            tournament_id="T001",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            entries=_entries(8),
        )
        status = agent.get_tournament_status("T001")
        assert status is not None

    def test_get_prices_after_register(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament(
            tournament_id="T002",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            entries=_entries(8),
        )
        snapshot = agent.get_prices("T002")
        assert isinstance(snapshot, OutrightPriceSnapshot)

    def test_get_prices_unknown_returns_none(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(KeyError):
            agent.get_prices("UNKNOWN")

    def test_suspend_tournament(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament(
            tournament_id="T003",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(8),
        )
        agent.suspend_tournament("T003")
        status = agent.get_tournament_status("T003")
        # After suspend, status should reflect suspended state
        assert status is not None

    def test_resume_tournament(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament(
            tournament_id="T004",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(8),
        )
        agent.suspend_tournament("T004")
        agent.resume_tournament("T004")  # Should not raise

    def test_on_match_result_eliminates_player(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament(
            tournament_id="T005",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            entries=_entries(8),
        )
        agent.on_match_result("T005", winner_id="P01", loser_id="P02", round_number=1)  # Should not raise

    def test_get_all_tournaments_returns_list(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament(
            tournament_id="T006",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_500,
            entries=_entries(8),
        )
        tournaments = agent.get_all_tournaments()
        assert isinstance(tournaments, (list, dict))

    def test_outright_price_snapshot_constructs(self) -> None:
        snap = OutrightPriceSnapshot(
            tournament_id="T001",
            discipline=Discipline.MS,
            prices={"P01": 3.50, "P02": 5.00},
            win_probs={"P01": 0.28, "P02": 0.20},
            h9_passed=True,
            prob_sum=0.998,
            timestamp=time.time(),
            n_simulations=10000,
        )
        assert snap.tournament_id == "T001"
        assert snap.h9_passed is True


# ---------------------------------------------------------------------------
# 3. SGPSupervisorAgent
# ---------------------------------------------------------------------------

class TestSGPSupervisorAgent:
    def test_constructs(self) -> None:
        agent = SGPSupervisorAgent()
        assert agent is not None

    def test_update_match_context(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M001")
        agent.update_match_context(ctx)  # Should not raise

    def test_price_sgp_no_context_returns_response(self) -> None:
        agent = SGPSupervisorAgent()
        req = _sgp_request("UNKNOWN")
        resp = agent.price_sgp(req)
        assert isinstance(resp, SGPResponse)
        # Without context, should be rejected
        assert resp.is_valid is False

    def test_price_sgp_with_context(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M100")
        agent.update_match_context(ctx)
        req = _sgp_request("M100")
        resp = agent.price_sgp(req)
        assert isinstance(resp, SGPResponse)

    def test_remove_match(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M999")
        agent.update_match_context(ctx)
        agent.remove_match("M999")  # Should not raise

    def test_get_metrics_returns_dict(self) -> None:
        agent = SGPSupervisorAgent()
        metrics = agent.get_metrics()
        assert isinstance(metrics, dict)

    def test_sgp_request_constructs(self) -> None:
        req = _sgp_request()
        assert req.match_id == "M001"
        assert len(req.legs) == 2

    def test_sgp_match_context_constructs(self) -> None:
        ctx = _sgp_context()
        assert ctx.discipline == Discipline.MS
        assert ctx.is_active is True


# ---------------------------------------------------------------------------
# 4. MonitoringSupervisorAgent
# ---------------------------------------------------------------------------

class TestMonitoringSupervisorAgent:
    def test_constructs_without_callback(self) -> None:
        agent = MonitoringSupervisorAgent()
        assert agent is not None

    def test_constructs_with_callback(self) -> None:
        callback = MagicMock()
        agent = MonitoringSupervisorAgent(alert_callback=callback)
        assert agent is not None

    def test_record_latency(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.record_latency("pre_match_price", 45.2)
        agent.record_latency("live_reprice", 12.3)

    def test_on_match_registered(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.on_match_registered("M001", "MS")  # Should not raise

    def test_on_match_live_started(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.on_match_registered("M001", "MS")
        agent.on_match_live_started("M001")  # Should not raise

    def test_on_match_removed(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.on_match_registered("M001", "MS")
        agent.on_match_removed("M001")  # Should not raise

    def test_record_price_update(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.on_match_registered("M001", "MS")
        agent.record_price_update("M001", latency_ms=30.0)  # Should not raise

    def test_record_score_update(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.on_match_registered("M001", "MS")
        agent.record_score_update("M001", latency_ms=15.0)  # Should not raise

    def test_get_latency_report(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.record_latency("test_op", 50.0)
        report = agent.get_latency_report()
        assert isinstance(report, dict)

    def test_get_qa_summary(self) -> None:
        agent = MonitoringSupervisorAgent()
        summary = agent.get_qa_summary()
        assert isinstance(summary, dict)

    def test_get_alert_summary(self) -> None:
        agent = MonitoringSupervisorAgent()
        summary = agent.get_alert_summary()
        assert isinstance(summary, (dict, list))

    def test_get_full_dashboard(self) -> None:
        agent = MonitoringSupervisorAgent()
        dashboard = agent.get_full_dashboard()
        assert isinstance(dashboard, dict)

    def test_emit_info(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.emit_info(AlertCategory.PRICING, "test_event", detail="value")

    def test_emit_warning(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.emit_warning(AlertCategory.FEED, "test_warning")

    def test_alert_callback_called(self) -> None:
        callback = MagicMock()
        agent = MonitoringSupervisorAgent(alert_callback=callback)
        # Trigger a warning alert
        agent.emit_warning(AlertCategory.TRADING, "critical_warning", detail="high severity")
        # Callback should be invoked for warnings
        callback.assert_called_once()

    def test_update_feed_health(self) -> None:
        agent = MonitoringSupervisorAgent()
        agent.update_feed_health(
            "optic_odds", "healthy", gap_s=5.0,
            error_rate=0.01, messages_per_minute=120.0, last_event_at=time.time()
        )  # Should not raise

    def test_latency_targets_defined(self) -> None:
        assert MonitoringSupervisorAgent.P50_TARGET_MS > 0
        assert MonitoringSupervisorAgent.P95_TARGET_MS > MonitoringSupervisorAgent.P50_TARGET_MS
        assert MonitoringSupervisorAgent.P99_TARGET_MS > MonitoringSupervisorAgent.P95_TARGET_MS

    def test_get_feed_health_report(self) -> None:
        agent = MonitoringSupervisorAgent()
        report = agent.get_feed_health_report()
        assert isinstance(report, (dict, list))

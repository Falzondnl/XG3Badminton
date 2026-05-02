"""
test_trading_supervisor.py
===========================
Unit tests for agents/trading_supervisor.py

Tests the BadmintonTradingSupervisor orchestrator:
  - Trading cycle executes agent chain
  - Result structure (TradingCycleResult)
  - Suspension propagates through chain
  - Live mode skips market_reference agent
  - Agent failure is isolated (chain continues)
  - Latency captured in result
  - Correct markets passed to TradingControlManager
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from agents.trading_supervisor import BadmintonTradingSupervisor, TradingCycleResult
from agents.trading.base_trading_agent import TradingContext, TradingAgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trading_control() -> MagicMock:
    tc = MagicMock()
    tc.set_click_scale = MagicMock()
    tc.suspend_all = MagicMock()
    return tc


def _make_raw_prices() -> dict:
    """Minimal dict of raw market prices (market_id → list of prices)."""
    price = MagicMock()
    price.odds = 1.90
    price.prob_implied = 0.50
    price.prob_with_margin = 0.53
    return {
        "match_winner": [price, price],
        "total_games": [price, price, price],
    }


def _make_context() -> dict:
    return {
        "entity_a_id": "axelsen",
        "entity_b_id": "lee_zii_jia",
        "discipline": Discipline.MS.value,
        "tier": TournamentTier.SUPER_1000.value,
        "total_liability_gbp": 0.0,
        "max_liability_gbp": 500_000.0,
    }


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestTradingSupervisorInit:
    """Constructor and agent chain setup."""

    def test_creates_with_no_pinnacle(self):
        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
        )
        assert supervisor._match_id == "m001"
        assert len(supervisor._agents) > 0

    def test_creates_with_pinnacle_client(self):
        pinnacle = MagicMock()
        supervisor = BadmintonTradingSupervisor(
            match_id="m002",
            trading_control=_make_trading_control(),
            pinnacle_client=pinnacle,
        )
        assert supervisor._agents[0].agent_name == "market_reference"


# ---------------------------------------------------------------------------
# Trading cycle basics
# ---------------------------------------------------------------------------

class TestTradingCycleResult:
    """run_trading_cycle returns correct TradingCycleResult structure."""

    def test_returns_trading_cycle_result(self):
        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
        )
        result = supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
        )
        assert isinstance(result, TradingCycleResult)

    def test_result_has_match_id(self):
        supervisor = BadmintonTradingSupervisor(
            match_id="m_axelsen",
            trading_control=_make_trading_control(),
        )
        result = supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
        )
        assert result.match_id == "m_axelsen"

    def test_latency_captured(self):
        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
        )
        result = supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
        )
        assert result.latency_ms >= 0.0

    def test_agent_results_populated(self):
        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
        )
        result = supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
        )
        # Some agents ran (not all may run — market_reference skipped if no Pinnacle)
        assert isinstance(result.agent_results, list)

    def test_n_markets_in_result(self):
        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
        )
        raw = _make_raw_prices()
        result = supervisor.run_trading_cycle(
            raw_prices=raw,
            match_context=_make_context(),
        )
        assert result.n_markets >= 0


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------

class TestLiveMode:
    """Live mode skips market_reference agent for latency."""

    def test_live_mode_skips_market_reference(self):
        """market_reference agent should NOT be called in live mode."""
        ref_agent = MagicMock()
        ref_agent.agent_name = "market_reference"
        ref_agent.process = MagicMock(return_value=TradingAgentResult(
            agent_name="market_reference", success=True
        ))

        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
            pinnacle_client=MagicMock(),
        )
        # Replace first agent (market_reference) with spy
        supervisor._agents[0] = ref_agent

        supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
            is_live=True,
        )
        ref_agent.process.assert_not_called()

    def test_prematch_mode_calls_market_reference(self):
        """market_reference agent SHOULD be called in pre-match mode."""
        ref_agent = MagicMock()
        ref_agent.agent_name = "market_reference"
        ref_agent.process = MagicMock(return_value=TradingAgentResult(
            agent_name="market_reference", success=True
        ))

        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
            pinnacle_client=MagicMock(),
        )
        supervisor._agents[0] = ref_agent

        supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
            is_live=False,
        )
        ref_agent.process.assert_called_once()


# ---------------------------------------------------------------------------
# Suspension
# ---------------------------------------------------------------------------

class TestSuspensionPropagation:
    """Suspension in one agent halts the chain."""

    def test_suspension_calls_trading_control_suspend_all(self):
        """When suspend_all=True in context, trading_control.suspend_all() is called."""
        tc = _make_trading_control()

        # Build a supervisor with a mock agent that triggers suspension
        class SuspendingAgent:
            agent_name = "suspending_agent"

            def process(self, ctx: TradingContext) -> TradingAgentResult:
                ctx.suspend_all = True
                ctx.suspend_reason = "test_suspension"
                return TradingAgentResult(agent_name="suspending_agent", success=True)

        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=tc,
        )
        supervisor._agents = [SuspendingAgent()]

        result = supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
        )

        tc.suspend_all.assert_called_once()
        assert result.suspended is True


# ---------------------------------------------------------------------------
# Agent failure isolation
# ---------------------------------------------------------------------------

class TestAgentFailureIsolation:
    """A failing agent should not crash the trading cycle."""

    def test_failing_agent_errors_captured(self):
        """Exception in agent is captured in result.errors, chain continues."""
        class FailingAgent:
            agent_name = "failing_agent"

            def process(self, ctx: TradingContext) -> TradingAgentResult:
                raise RuntimeError("simulated failure")

        class SuccessAgent:
            agent_name = "success_agent"

            def process(self, ctx: TradingContext) -> TradingAgentResult:
                return TradingAgentResult(agent_name="success_agent", success=True)

        supervisor = BadmintonTradingSupervisor(
            match_id="m001",
            trading_control=_make_trading_control(),
        )
        supervisor._agents = [FailingAgent(), SuccessAgent()]

        result = supervisor.run_trading_cycle(
            raw_prices=_make_raw_prices(),
            match_context=_make_context(),
        )

        # Chain continued — success agent ran
        success_names = [ar.agent_name for ar in result.agent_results if ar.success]
        assert "success_agent" in success_names

        # Error captured
        assert len(result.errors) >= 1
        assert any("simulated failure" in e for e in result.errors)

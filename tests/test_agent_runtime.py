"""
test_agent_runtime.py
=====================
Unit tests for agents/agent_runtime.py

Tests the BadmintonAgentRuntime lifecycle manager:
  - Agent registration / unregistration
  - Duplicate registration prevention
  - Agent discovery (get, list)
  - State management
  - Config defaults

All async methods are exercised via asyncio.run() so no pytest-asyncio
dependency is required.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_runtime import (
    BadmintonAgentRuntime,
    AgentRegistration,
    AgentState,
    RuntimeConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously in tests."""
    return asyncio.run(coro)


def _make_agent(name: str = "test_agent") -> MagicMock:
    """Create a minimal mock agent with a health_check() coroutine."""
    agent = MagicMock()
    agent.__class__.__name__ = name
    agent.health_check = AsyncMock(return_value=True)
    return agent


@pytest.fixture
def runtime():
    return BadmintonAgentRuntime()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestRuntimeInit:
    """Runtime initialisation state."""

    def test_starts_with_no_agents(self, runtime):
        assert runtime.list_agents() == []

    def test_not_running_at_init(self, runtime):
        assert runtime._running is False

    def test_default_config(self, runtime):
        assert runtime.config.health_check_interval_s == 30.0
        assert runtime.config.auto_restart is True
        assert runtime.config.max_restart_attempts == 3

    def test_custom_config(self):
        cfg = RuntimeConfig(health_check_interval_s=10.0, auto_restart=False)
        rt = BadmintonAgentRuntime(config=cfg)
        assert rt.config.health_check_interval_s == 10.0
        assert rt.config.auto_restart is False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestAgentRegistration:
    """Agent register / unregister / list behaviour."""

    def test_register_single_agent(self, runtime):
        _run(runtime.register_agent("pricer", _make_agent()))
        agents = runtime.list_agents()
        assert "pricer" in agents

    def test_register_multiple_agents(self, runtime):
        _run(runtime.register_agent("agent_a", _make_agent("A")))
        _run(runtime.register_agent("agent_b", _make_agent("B")))
        assert set(runtime.list_agents()) == {"agent_a", "agent_b"}

    def test_duplicate_registration_raises(self, runtime):
        _run(runtime.register_agent("dup", _make_agent()))
        with pytest.raises(ValueError, match="already registered"):
            _run(runtime.register_agent("dup", _make_agent()))

    def test_get_registered_agent(self, runtime):
        agent = _make_agent("my_agent")
        _run(runtime.register_agent("my_agent", agent))
        reg = runtime.get_agent("my_agent")
        assert isinstance(reg, AgentRegistration)
        assert reg.agent_id == "my_agent"
        assert reg.agent is agent

    def test_get_unknown_agent_returns_none(self, runtime):
        assert runtime.get_agent("nonexistent") is None

    def test_unregister_agent(self, runtime):
        _run(runtime.register_agent("to_remove", _make_agent()))
        assert "to_remove" in runtime.list_agents()
        _run(runtime.unregister_agent("to_remove"))
        assert "to_remove" not in runtime.list_agents()

    def test_unregister_unknown_agent_raises(self, runtime):
        with pytest.raises(ValueError, match="not found"):
            _run(runtime.unregister_agent("ghost"))


# ---------------------------------------------------------------------------
# Registration metadata
# ---------------------------------------------------------------------------

class TestAgentRegistrationMetadata:
    """AgentRegistration dataclass initial state."""

    def test_initial_state_is_registered(self, runtime):
        _run(runtime.register_agent("meta_agent", _make_agent()))
        reg = runtime.get_agent("meta_agent")
        assert reg.state == AgentState.REGISTERED

    def test_initial_restart_count_zero(self, runtime):
        _run(runtime.register_agent("restart_test", _make_agent()))
        reg = runtime.get_agent("restart_test")
        assert reg.restart_count == 0

    def test_can_restart_below_max(self, runtime):
        _run(runtime.register_agent("can_restart", _make_agent()))
        reg = runtime.get_agent("can_restart")
        assert reg.can_restart(max_attempts=3) is True

    def test_cannot_restart_at_max(self, runtime):
        _run(runtime.register_agent("no_restart", _make_agent()))
        reg = runtime.get_agent("no_restart")
        reg.restart_count = 3
        assert reg.can_restart(max_attempts=3) is False

    def test_metadata_stored_on_registration(self, runtime):
        meta = {"tier": "SUPER_1000", "discipline": "MS"}
        _run(runtime.register_agent("meta_test", _make_agent(), metadata=meta))
        reg = runtime.get_agent("meta_test")
        assert reg.metadata == meta


# ---------------------------------------------------------------------------
# AgentState enum coverage
# ---------------------------------------------------------------------------

class TestAgentStateEnum:
    """All expected states exist."""

    def test_all_expected_states(self):
        states = {s.value for s in AgentState}
        expected = {
            "registered", "starting", "running",
            "unhealthy", "restarting", "stopping", "stopped", "failed",
        }
        assert expected.issubset(states), f"Missing states: {expected - states}"


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------

class TestRuntimeConfig:
    """Config object defaults and overrides."""

    def test_all_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.health_check_interval_s > 0
        assert cfg.max_restart_attempts >= 1
        assert cfg.shutdown_timeout_s > 0
        assert cfg.restart_backoff_multiplier >= 1.0

    def test_shutdown_timeout_configurable(self):
        cfg = RuntimeConfig(shutdown_timeout_s=60.0)
        rt = BadmintonAgentRuntime(config=cfg)
        assert rt.config.shutdown_timeout_s == 60.0

"""
agents/agent_runtime.py
=======================
BadmintonAgentRuntime — lifecycle manager for all badminton autonomous agents.

Mirrors the XG3 Enterprise AgentRuntime pattern (apps/agents_runtime/engine.py).

Features:
  - Agent registration + discovery
  - Automatic health check every 30s (configurable)
  - Auto-restart on failure with exponential backoff (max 3 attempts)
  - Graceful shutdown (finish current task, 30s timeout)
  - Full observability via structlog metrics (every 60s)

Agent execution models:
  - run()     — long-running coroutine (e.g. AutoPricer.run_forever)
  - execute() — periodic task (called every second if agent has execute())
  - Neither   — event-driven: kept alive for event subscriptions

Usage:
    runtime = BadmintonAgentRuntime()
    await runtime.register_agent("orchestrator", orchestrator)
    await runtime.register_agent("auto_pricer", auto_pricer)
    await runtime.start_all()
    # ...
    await runtime.shutdown()

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums + config
# ---------------------------------------------------------------------------

class AgentState(str, Enum):
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class RuntimeConfig:
    health_check_interval_s: float = 30.0
    health_check_timeout_s: float = 5.0
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay_s: float = 5.0
    restart_backoff_multiplier: float = 2.0
    max_restart_delay_s: float = 60.0
    shutdown_timeout_s: float = 30.0
    metrics_interval_s: float = 60.0


# ---------------------------------------------------------------------------
# Agent registration record
# ---------------------------------------------------------------------------

@dataclass
class AgentRegistration:
    agent_id: str
    agent: Any
    state: AgentState = AgentState.REGISTERED
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    last_health_ok: Optional[bool] = None
    consecutive_failures: int = 0
    restart_count: int = 0
    last_restart_at: Optional[datetime] = None
    task: Optional[asyncio.Task] = None
    health_task: Optional[asyncio.Task] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_running(self) -> bool:
        return self.state == AgentState.RUNNING

    def can_restart(self, max_attempts: int) -> bool:
        return self.restart_count < max_attempts


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class BadmintonAgentRuntime:
    """
    Lifecycle manager for all badminton autonomous agents.

    One singleton per deployment.
    """

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig()
        self._agents: Dict[str, AgentRegistration] = {}
        self._running = False
        self._shutting_down = False
        self._metrics_task: Optional[asyncio.Task] = None
        self._stats = {
            "agents_started": 0,
            "agents_stopped": 0,
            "agents_restarted": 0,
            "health_checks_passed": 0,
            "health_checks_failed": 0,
        }
        logger.info("badminton_agent_runtime_initialized")

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        agent: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id!r} is already registered")
        self._agents[agent_id] = AgentRegistration(
            agent_id=agent_id,
            agent=agent,
            metadata=metadata or {},
        )
        logger.info(
            "agent_registered",
            agent_id=agent_id,
            agent_type=agent.__class__.__name__,
        )

    async def unregister_agent(self, agent_id: str) -> None:
        reg = self._agents.get(agent_id)
        if reg is None:
            raise ValueError(f"Agent {agent_id!r} not found")
        if reg.is_running():
            await self.stop_agent(agent_id)
        del self._agents[agent_id]
        logger.info("agent_unregistered", agent_id=agent_id)

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_agent(self, agent_id: str) -> None:
        reg = self._agents.get(agent_id)
        if reg is None:
            raise ValueError(f"Agent {agent_id!r} not found")
        if reg.is_running():
            logger.warning("agent_already_running", agent_id=agent_id)
            return

        reg.state = AgentState.STARTING
        reg.started_at = datetime.now(timezone.utc)

        try:
            reg.task = asyncio.create_task(
                self._run_agent(agent_id),
                name=f"agent_{agent_id}",
            )
            if self.config.health_check_interval_s > 0:
                reg.health_task = asyncio.create_task(
                    self._health_monitor(agent_id),
                    name=f"health_{agent_id}",
                )
            reg.state = AgentState.RUNNING
            self._stats["agents_started"] += 1
            logger.info("agent_started", agent_id=agent_id)
        except Exception as exc:
            reg.state = AgentState.FAILED
            logger.error("agent_start_failed", agent_id=agent_id, error=str(exc))
            raise

    async def stop_agent(self, agent_id: str, graceful: bool = True) -> None:
        reg = self._agents.get(agent_id)
        if reg is None:
            raise ValueError(f"Agent {agent_id!r} not found")
        if not reg.is_running():
            return

        reg.state = AgentState.STOPPING

        # Stop health monitor
        if reg.health_task and not reg.health_task.done():
            reg.health_task.cancel()
            try:
                await reg.health_task
            except asyncio.CancelledError:
                pass

        # Stop main task
        if reg.task and not reg.task.done():
            if graceful:
                try:
                    await asyncio.wait_for(reg.task, timeout=self.config.shutdown_timeout_s)
                except asyncio.TimeoutError:
                    logger.warning(
                        "agent_stop_timeout",
                        agent_id=agent_id,
                        timeout_s=self.config.shutdown_timeout_s,
                    )
                    reg.task.cancel()
            else:
                reg.task.cancel()
            try:
                await reg.task
            except asyncio.CancelledError:
                pass

        reg.state = AgentState.STOPPED
        reg.stopped_at = datetime.now(timezone.utc)
        self._stats["agents_stopped"] += 1
        logger.info("agent_stopped", agent_id=agent_id)

    async def restart_agent(self, agent_id: str) -> None:
        reg = self._agents.get(agent_id)
        if reg is None:
            raise ValueError(f"Agent {agent_id!r} not found")

        reg.state = AgentState.RESTARTING
        reg.restart_count += 1
        reg.last_restart_at = datetime.now(timezone.utc)

        await self.stop_agent(agent_id, graceful=True)

        delay = min(
            self.config.restart_delay_s
            * (self.config.restart_backoff_multiplier ** (reg.restart_count - 1)),
            self.config.max_restart_delay_s,
        )
        logger.info("agent_restart_wait", agent_id=agent_id, delay_s=round(delay, 1))
        await asyncio.sleep(delay)

        await self.start_agent(agent_id)
        self._stats["agents_restarted"] += 1
        logger.info("agent_restarted", agent_id=agent_id, restart_count=reg.restart_count)

    async def start_all(self) -> None:
        logger.info("starting_all_agents", n_agents=len(self._agents))
        for agent_id in list(self._agents.keys()):
            try:
                await self.start_agent(agent_id)
            except Exception as exc:
                logger.error("start_all_agent_failed", agent_id=agent_id, error=str(exc))

        self._running = True
        self._metrics_task = asyncio.create_task(
            self._collect_metrics(),
            name="runtime_metrics",
        )
        logger.info("all_agents_started")

    async def stop_all(self, graceful: bool = True) -> None:
        logger.info("stopping_all_agents")
        self._running = False

        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        for agent_id in list(self._agents.keys()):
            reg = self._agents.get(agent_id)
            if reg and reg.is_running():
                try:
                    await self.stop_agent(agent_id, graceful=graceful)
                except Exception as exc:
                    logger.error("stop_all_agent_failed", agent_id=agent_id, error=str(exc))

        logger.info("all_agents_stopped")

    async def shutdown(self, graceful: bool = True) -> None:
        self._shutting_down = True
        await self.stop_all(graceful=graceful)
        logger.info("badminton_agent_runtime_shutdown_complete")

    # ------------------------------------------------------------------
    # Internal: agent event loop
    # ------------------------------------------------------------------

    async def _run_agent(self, agent_id: str) -> None:
        reg = self._agents.get(agent_id)
        if reg is None:
            return

        agent = reg.agent
        logger.info("agent_event_loop_started", agent_id=agent_id)

        try:
            if hasattr(agent, "run_forever") and callable(agent.run_forever):
                await agent.run_forever()

            elif hasattr(agent, "run") and callable(agent.run):
                await agent.run()

            elif hasattr(agent, "execute") and callable(agent.execute):
                # Task-based: execute every second
                while not self._shutting_down:
                    try:
                        await agent.execute()
                    except Exception as exc:
                        logger.error(
                            "agent_execute_error",
                            agent_id=agent_id,
                            error=str(exc),
                        )
                    await asyncio.sleep(1.0)

            else:
                # Event-driven: keep alive
                logger.info(
                    "agent_event_driven",
                    agent_id=agent_id,
                    detail="no run/execute method — kept alive for event subscriptions",
                )
                while not self._shutting_down:
                    await asyncio.sleep(1.0)

        except asyncio.CancelledError:
            logger.info("agent_event_loop_cancelled", agent_id=agent_id)
            raise
        except Exception as exc:
            logger.error(
                "agent_event_loop_error",
                agent_id=agent_id,
                error=str(exc),
            )
            raise

    # ------------------------------------------------------------------
    # Internal: health monitoring
    # ------------------------------------------------------------------

    async def _health_monitor(self, agent_id: str) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval_s)

                reg = self._agents.get(agent_id)
                if reg is None or not reg.is_running():
                    break

                healthy = await self._check_agent_health(reg)
                reg.last_health_check = datetime.now(timezone.utc)
                reg.last_health_ok = healthy

                if healthy:
                    reg.consecutive_failures = 0
                    self._stats["health_checks_passed"] += 1
                    logger.debug("agent_health_ok", agent_id=agent_id)
                else:
                    reg.consecutive_failures += 1
                    reg.state = AgentState.UNHEALTHY
                    self._stats["health_checks_failed"] += 1
                    logger.warning(
                        "agent_health_failed",
                        agent_id=agent_id,
                        consecutive_failures=reg.consecutive_failures,
                    )

                    if self.config.auto_restart and reg.can_restart(self.config.max_restart_attempts):
                        logger.warning("agent_auto_restarting", agent_id=agent_id)
                        await self.restart_agent(agent_id)
                        break  # Health monitor recreated by restart_agent → start_agent
                    elif not reg.can_restart(self.config.max_restart_attempts):
                        reg.state = AgentState.FAILED
                        logger.error(
                            "agent_max_restarts_reached",
                            agent_id=agent_id,
                            restart_count=reg.restart_count,
                        )

            except asyncio.CancelledError:
                logger.info("agent_health_monitor_stopped", agent_id=agent_id)
                break
            except Exception as exc:
                logger.error(
                    "agent_health_monitor_error",
                    agent_id=agent_id,
                    error=str(exc),
                )

    async def _check_agent_health(self, reg: AgentRegistration) -> bool:
        """Call agent.health_check() if available, else check task is alive."""
        agent = reg.agent
        if hasattr(agent, "health_check") and callable(agent.health_check):
            try:
                result = await asyncio.wait_for(
                    agent.health_check(),
                    timeout=self.config.health_check_timeout_s,
                )
                return bool(result)
            except asyncio.TimeoutError:
                logger.warning(
                    "agent_health_check_timeout",
                    agent_id=reg.agent_id,
                    timeout_s=self.config.health_check_timeout_s,
                )
                return False
            except Exception as exc:
                logger.warning(
                    "agent_health_check_exception",
                    agent_id=reg.agent_id,
                    error=str(exc),
                )
                return False
        else:
            # No health_check() method → healthy if task is alive
            return reg.task is not None and not reg.task.done()

    # ------------------------------------------------------------------
    # Internal: metrics
    # ------------------------------------------------------------------

    async def _collect_metrics(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.config.metrics_interval_s)
                metrics = self.get_metrics()
                logger.info(
                    "agent_runtime_metrics",
                    total_agents=metrics["total_agents"],
                    running=metrics["running_agents"],
                    healthy=metrics["healthy_agents"],
                    failed=metrics["failed_agents"],
                    restarts=metrics["agents_restarted"],
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("metrics_collection_error", error=str(exc))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {s.value: 0 for s in AgentState}
        healthy = 0
        for reg in self._agents.values():
            counts[reg.state.value] += 1
            if reg.last_health_ok is True:
                healthy += 1
        return {
            "total_agents": len(self._agents),
            "running_agents": counts[AgentState.RUNNING.value],
            "healthy_agents": healthy,
            "unhealthy_agents": counts[AgentState.UNHEALTHY.value],
            "failed_agents": counts[AgentState.FAILED.value],
            "stopped_agents": counts[AgentState.STOPPED.value],
            **self._stats,
        }

    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        reg = self._agents.get(agent_id)
        if reg is None:
            raise ValueError(f"Agent {agent_id!r} not found")
        uptime_s = None
        if reg.started_at:
            uptime_s = (datetime.now(timezone.utc) - reg.started_at).total_seconds()
        return {
            "agent_id": agent_id,
            "agent_type": reg.agent.__class__.__name__,
            "state": reg.state.value,
            "is_running": reg.is_running(),
            "is_healthy": reg.last_health_ok,
            "registered_at": reg.registered_at.isoformat(),
            "started_at": reg.started_at.isoformat() if reg.started_at else None,
            "uptime_s": round(uptime_s, 1) if uptime_s else None,
            "restart_count": reg.restart_count,
            "consecutive_failures": reg.consecutive_failures,
            "last_health_check": reg.last_health_check.isoformat() if reg.last_health_check else None,
            "metadata": reg.metadata,
        }

    def get_all_agent_statuses(self) -> List[Dict[str, Any]]:
        return [self.get_agent_status(aid) for aid in self._agents]

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

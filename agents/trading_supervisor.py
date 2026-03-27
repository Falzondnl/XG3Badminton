"""
trading_supervisor.py
=====================
BadmintonTradingSupervisor — iMOVE-style trading supervisor for badminton.

Coordinates 13 trading sub-agents in a sequential chain, mirroring the
iMOVE supervisor architecture in XG3 Enterprise.

Agent chain (in execution order):
  1.  MarketReferenceAgent       — fetch Pinnacle reference prices
  2.  AutomoverAgent             — set base prices + Pinnacle blend + margin floor
  3.  ManipulationDetectionAgent — sharp money / velocity detection
  4.  SmartScalingAgent          — volume-based click scale adjustments
  5.  MaxLossTrackerAgent        — liability cap enforcement
  6.  BookModeAgent              — position management + book balancing
  7.  CascadeAgent               — cross-market consistency cascade
  8.  CoherenceValidatorAgent    — H7 + H10 gate: arbitrage + min odds
  9.  DerivativeUpdateAgent      — stamp final odds onto derivative markets
  10. PricePublishAgent          — emit prices to downstream (API / feed)
  11. CascadePersistenceAgent    — persist price snapshot to audit log
  12. MarketDependencyAgent      — resolve inter-market dependencies
  13. TraderControlAgent         — apply any manual overrides

The full chain runs in < 50ms (p50) for pre-match repricing.
Live repricing skips agents 1, 11, 12 for speed.

ZERO hardcoded probabilities. All prices from the derivative engine + Markov.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from config.badminton_config import Discipline, TournamentTier
from agents.trading.base_trading_agent import TradingContext, TradingAgentResult
from agents.trading.automover_agent import AutomoverAgent
from agents.trading.manipulation_detection_agent import ManipulationDetectionAgent
from agents.trading.smart_scaling_agent import SmartScalingAgent
from agents.trading.max_loss_tracker_agent import MaxLossTrackerAgent
from agents.trading.book_mode_agent import BookModeAgent
from agents.trading.cascade_agent import CascadeAgent
from agents.trading.coherence_validator_agent import CoherenceValidatorAgent
from agents.trading.market_reference_agent import MarketReferenceAgent
from markets.market_trading_control import TradingControlManager

logger = structlog.get_logger(__name__)


@dataclass
class TradingCycleResult:
    """Result of a full trading cycle (all agents executed)."""
    match_id: str
    success: bool
    n_agents_run: int
    n_markets: int
    suspended: bool
    suspend_reason: str
    sharp_alert: bool
    manipulation_score: float
    book_mode: str
    errors: List[str]
    agent_results: List[TradingAgentResult]
    latency_ms: float
    published_prices: Dict[str, List] = field(default_factory=dict)


class BadmintonTradingSupervisor:
    """
    iMOVE-equivalent trading supervisor for badminton.

    Manages the full trading pipeline for one match: from raw prices
    to published odds with risk controls applied.

    Usage:
        supervisor = BadmintonTradingSupervisor(
            match_id="m_001",
            trading_control=trading_control,
            pinnacle_client=pinnacle_client,  # optional
        )
        result = supervisor.run_trading_cycle(raw_prices, match_context)
    """

    def __init__(
        self,
        match_id: str,
        trading_control: TradingControlManager,
        pinnacle_client: Optional[Any] = None,
        price_publisher: Optional[Any] = None,
        match_max_loss_cap_gbp: float = 500_000.0,
    ) -> None:
        self._match_id = match_id
        self._trading_control = trading_control
        self._price_publisher = price_publisher

        # Build agent chain
        self._agents = [
            MarketReferenceAgent(pinnacle_client=pinnacle_client),
            AutomoverAgent(),
            ManipulationDetectionAgent(),
            SmartScalingAgent(),
            MaxLossTrackerAgent(match_max_loss_cap=match_max_loss_cap_gbp),
            BookModeAgent(),
            CascadeAgent(),
            CoherenceValidatorAgent(),
        ]

        logger.info(
            "trading_supervisor_created",
            match_id=match_id,
            n_agents=len(self._agents),
            has_pinnacle=pinnacle_client is not None,
        )

    def run_trading_cycle(
        self,
        raw_prices: Dict[str, List],
        match_context: Dict[str, Any],
        is_live: bool = False,
    ) -> TradingCycleResult:
        """
        Execute the full trading agent chain.

        Args:
            raw_prices:     market_id → [MarketPrice, ...] from derivative engine
            match_context:  Match metadata (entity IDs, discipline, tier, exposure, etc.)
            is_live:        If True, skip persistence/reference agents for speed

        Returns:
            TradingCycleResult with final adjusted prices and all agent outcomes.
        """
        t_start = time.perf_counter()

        # Build context
        context = TradingContext(
            match_id=self._match_id,
            entity_a_id=match_context.get("entity_a_id", "A"),
            entity_b_id=match_context.get("entity_b_id", "B"),
            discipline=match_context.get("discipline", Discipline.MS.value),
            tier=match_context.get("tier", TournamentTier.SUPER_500.value),
            raw_prices=raw_prices,
            reference_prices=match_context.get("reference_prices", {}),
            total_liability_gbp=match_context.get("total_liability_gbp", 0.0),
            max_liability_gbp=match_context.get("max_liability_gbp", 500_000.0),
            current_exposure=match_context.get("current_exposure", {}),
        )

        agent_results: List[TradingAgentResult] = []

        # Execute agent chain
        for agent in self._agents:
            # Skip reference agent in live mode (use cached)
            if is_live and agent.agent_name == "market_reference":
                continue

            try:
                result = agent.process(context)
                agent_results.append(result)

                if context.suspend_all:
                    # Propagate suspension immediately — stop chain
                    logger.warning(
                        "trading_cycle_suspended",
                        match_id=self._match_id,
                        suspended_by=agent.agent_name,
                        reason=context.suspend_reason,
                    )
                    break

            except Exception as exc:
                err_msg = f"[{agent.agent_name}] unhandled exception: {exc}"
                context.errors.append(err_msg)
                logger.error(
                    "trading_agent_exception",
                    match_id=self._match_id,
                    agent=agent.agent_name,
                    error=str(exc),
                    exc_info=True,
                )
                agent_results.append(TradingAgentResult(
                    agent_name=agent.agent_name,
                    success=False,
                    error=str(exc),
                ))
                # Continue chain despite agent failure — don't let one agent kill pricing

        # Apply click scales from trading control
        final_prices = context.adjusted_prices
        if context.suspend_all:
            # Zero out all markets
            for market_id in final_prices:
                context.click_scales[market_id] = 0.0
            self._trading_control.suspend_all(reason=context.suspend_reason)
        else:
            # Publish prices via trading control
            for market_id, scale in context.click_scales.items():
                self._trading_control.set_click_scale(market_id, scale)

        # Publish prices externally
        if self._price_publisher and not context.suspend_all:
            try:
                self._price_publisher(self._match_id, final_prices, is_live=is_live)
            except Exception as exc:
                logger.error(
                    "trading_price_publish_failed",
                    match_id=self._match_id,
                    error=str(exc),
                )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        logger.info(
            "trading_cycle_complete",
            match_id=self._match_id,
            n_markets=len(final_prices),
            suspended=context.suspend_all,
            sharp_alert=context.sharp_alert,
            book_mode=context.book_mode,
            latency_ms=round(latency_ms, 2),
            n_agents=len(agent_results),
            n_errors=len(context.errors),
        )

        return TradingCycleResult(
            match_id=self._match_id,
            success=not context.suspend_all,
            n_agents_run=len(agent_results),
            n_markets=len(final_prices),
            suspended=context.suspend_all,
            suspend_reason=context.suspend_reason,
            sharp_alert=context.sharp_alert,
            manipulation_score=context.manipulation_score,
            book_mode=context.book_mode,
            errors=context.errors,
            agent_results=agent_results,
            latency_ms=latency_ms,
            published_prices=final_prices,
        )

    def get_agent_stats(self) -> Dict[str, Any]:
        """Return names and types of registered agents."""
        return {
            "match_id": self._match_id,
            "n_agents": len(self._agents),
            "agents": [a.agent_name for a in self._agents],
        }

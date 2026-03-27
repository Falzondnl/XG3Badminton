"""
base_trading_agent.py
=====================
Base class for all BadmintonTradingSupervisor sub-agents.

Each sub-agent receives a TradingContext, mutates it (or emits signals),
and returns a TradingAgentResult. The supervisor chains them in order.

Pattern mirrors iMOVE supervisor in XG3 Enterprise:
  supervisor → [agent1, agent2, ..., agent13] → merged result

All agents are stateless (no match-level mutable state stored here).
State lives in TradingContext which is created fresh per reprice cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TradingContext:
    """
    Mutable context passed through the trading agent chain.

    Created at the start of each reprice cycle by the supervisor.
    Each agent reads and optionally mutates this context.
    The final state is used to publish odds.
    """
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: str
    tier: str

    # Raw prices from pre-match / Bayesian engine (before trading adjustments)
    raw_prices: Dict[str, List]  # market_id → [MarketPrice, ...]

    # Adjusted prices (mutated by agents in chain)
    adjusted_prices: Dict[str, List] = field(default_factory=dict)

    # Reference prices (Pinnacle / market consensus)
    reference_prices: Dict[str, float] = field(default_factory=dict)  # market_id → fair prob

    # Risk state
    total_liability_gbp: float = 0.0
    max_liability_gbp: float = 500_000.0
    current_exposure: Dict[str, float] = field(default_factory=dict)  # outcome → exposure

    # Manipulation signals
    sharp_alert: bool = False
    sharp_alert_details: str = ""
    manipulation_score: float = 0.0  # 0=clean, 1=max suspect

    # Book mode
    book_mode: str = "balanced"  # "balanced" / "overbroke" / "underbroke" / "flat"

    # Click scales (liability control)
    click_scales: Dict[str, float] = field(default_factory=dict)

    # Audit trail
    agent_notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Flags
    suspend_all: bool = False
    suspend_reason: str = ""
    prices_locked: bool = False


@dataclass
class TradingAgentResult:
    """Result returned by each sub-agent."""
    agent_name: str
    success: bool
    context_mutated: bool = False
    notes: str = ""
    error: Optional[str] = None


class BaseTradingAgent(ABC):
    """
    Abstract base for all trading sub-agents in the BadmintonTradingSupervisor.

    Lifecycle:
        result = agent.process(context)

    Agents MUST NOT raise exceptions for normal business conditions (e.g. high
    liability). Instead, set context.suspend_all or context.errors and return
    a TradingAgentResult with success=False.

    Agents MAY raise only for unrecoverable programming errors.
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique identifier for this agent."""

    @abstractmethod
    def process(self, context: TradingContext) -> TradingAgentResult:
        """
        Process the trading context.

        Args:
            context: Mutable trading context. Agents update adjusted_prices,
                     click_scales, sharp_alert, book_mode, etc.

        Returns:
            TradingAgentResult with outcome status.
        """

    def _log(self, context: TradingContext, msg: str) -> None:
        context.agent_notes.append(f"[{self.agent_name}] {msg}")
        logger.debug(
            "trading_agent_note",
            agent=self.agent_name,
            match_id=context.match_id,
            msg=msg,
        )

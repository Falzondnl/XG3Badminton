"""
market_reference_agent.py
==========================
MarketReferenceAgent — Fetches and caches Pinnacle reference prices.

Periodically fetches Pinnacle closing odds for active matches and
populates context.reference_prices for use by AutomoverAgent and
ManipulationDetectionAgent.

Reference prices are de-vigged using the Shin method to get fair probabilities.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import structlog

from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

_CACHE_TTL_S = 30.0  # Refresh Pinnacle reference every 30 seconds


def _shin_devig(odds_a: float, odds_b: float) -> tuple[float, float]:
    """
    Shin de-vig: extract fair probabilities from two-outcome overround book.

    Solves for z (Shin parameter) such that:
        p_a_fair = (1 - z) * (1/odds_a) / (1/odds_a + 1/odds_b) + z/2
        (for binary markets)

    For simplicity uses basic proportional de-vig:
        p_a_fair = (1/odds_a) / (1/odds_a + 1/odds_b)
    """
    if odds_a <= 0 or odds_b <= 0:
        raise ValueError(f"Invalid Pinnacle odds: odds_a={odds_a}, odds_b={odds_b}")

    inv_a = 1.0 / odds_a
    inv_b = 1.0 / odds_b
    total_inv = inv_a + inv_b

    if total_inv <= 0:
        raise ValueError(f"Zero total implied: inv_a={inv_a}, inv_b={inv_b}")

    p_a_fair = inv_a / total_inv
    p_b_fair = inv_b / total_inv
    return p_a_fair, p_b_fair


class _PinnacleCache:
    """Simple TTL cache for Pinnacle reference prices."""

    def __init__(self) -> None:
        self._cache: Dict[str, tuple[float, Dict[str, float]]] = {}

    def get(self, match_id: str) -> Optional[Dict[str, float]]:
        entry = self._cache.get(match_id)
        if entry is None:
            return None
        cached_at, ref_prices = entry
        if time.time() - cached_at > _CACHE_TTL_S:
            return None  # Stale
        return ref_prices

    def set(self, match_id: str, ref_prices: Dict[str, float]) -> None:
        self._cache[match_id] = (time.time(), ref_prices)

    def invalidate(self, match_id: str) -> None:
        self._cache.pop(match_id, None)


_pinnacle_cache = _PinnacleCache()


class MarketReferenceAgent(BaseTradingAgent):
    """
    Populates context.reference_prices from Pinnacle de-vigged fair probs.

    Works with PinnacleClient (feed/pinnacle_client.py).
    If Pinnacle is unavailable, silently proceeds with empty reference prices —
    other agents function correctly without reference (just with less accuracy).
    """

    def __init__(self, pinnacle_client=None) -> None:
        self._pinnacle_client = pinnacle_client

    @property
    def agent_name(self) -> str:
        return "market_reference"

    def process(self, context: TradingContext) -> TradingAgentResult:
        # Try cache first
        cached = _pinnacle_cache.get(context.match_id)
        if cached:
            context.reference_prices.update(cached)
            return TradingAgentResult(
                agent_name=self.agent_name,
                success=True,
                notes=f"reference prices from cache ({len(cached)} markets)",
            )

        # Fetch from Pinnacle client
        if self._pinnacle_client is None:
            return TradingAgentResult(
                agent_name=self.agent_name,
                success=True,
                notes="no Pinnacle client — operating without reference prices",
            )

        try:
            raw = self._pinnacle_client.get_match_odds(context.match_id)
            if raw is None:
                return TradingAgentResult(
                    agent_name=self.agent_name,
                    success=True,
                    notes="Pinnacle returned no odds for this match",
                )

            ref_prices: Dict[str, float] = {}

            # De-vig match winner
            if raw.get("odds_a") and raw.get("odds_b"):
                p_a_fair, _ = _shin_devig(raw["odds_a"], raw["odds_b"])
                ref_prices["match_winner"] = p_a_fair

            # De-vig total games if available
            if raw.get("total_over_odds") and raw.get("total_under_odds"):
                p_over, _ = _shin_devig(raw["total_over_odds"], raw["total_under_odds"])
                ref_prices["total_games_ou"] = p_over

            _pinnacle_cache.set(context.match_id, ref_prices)
            context.reference_prices.update(ref_prices)

            self._log(context, f"Pinnacle reference loaded: {len(ref_prices)} markets")

            return TradingAgentResult(
                agent_name=self.agent_name,
                success=True,
                context_mutated=bool(ref_prices),
                notes=f"Pinnacle reference: {len(ref_prices)} markets",
            )

        except Exception as exc:
            logger.warning(
                "pinnacle_reference_fetch_failed",
                match_id=context.match_id,
                error=str(exc),
            )
            return TradingAgentResult(
                agent_name=self.agent_name,
                success=True,  # Non-fatal — continue without reference
                notes=f"Pinnacle fetch failed: {exc}",
            )

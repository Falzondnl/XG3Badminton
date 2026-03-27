"""
platform/badminton_pricer.py
============================
BadmintonPricer — ISportPricer implementation for the XG3 platform.

Registers badminton as a first-class sport in the platform's sport pricer
registry, matching the tennis/football/basketball pattern.

Registration (called at app startup in main.py):
    from badminton.platform.badminton_pricer import BadmintonPricer
    from badminton_platform.interfaces.sport_pricer_registry import register
    register(BadmintonPricer(orchestrator=...))

Consumption (anywhere in the platform):
    pricer = get("badminton")
    prices = await pricer.price_match_winner(match_context)

match_context keys:
    match_id          str  — platform match ID
    entity_a_id       str  — player/pair ID
    entity_b_id       str  — player/pair ID
    discipline        str  — "MS"/"WS"/"MD"/"WD"/"XD"
    tournament_id     str  — platform tournament ID
    tier              str  — "SUPER_1000" / etc.
    first_server      str  — "A" or "B"
    pinnacle_odds_a   float|None — Pinnacle closing odds for A (de-vigged if available)
    pinnacle_odds_b   float|None

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

import structlog

from config.badminton_config import Discipline, TournamentTier
from agents.orchestrator import BadmintonOrchestratorAgent

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ISportPricer protocol (replicated here to avoid hard dependency on platform)
# ---------------------------------------------------------------------------

class ISportPricer:
    """
    Minimal copy of platform.interfaces.sport_pricer.ISportPricer.
    Used when badminton runs standalone (not embedded in XG3 Enterprise).
    When running inside XG3 Enterprise, import from badminton_platform.interfaces.
    """

    @property
    def sport_key(self) -> str:
        raise NotImplementedError

    async def price_match_winner(self, match_context: dict, margin_pct: float = 0.05) -> dict:
        raise NotImplementedError

    async def price_set_handicap(self, match_context: dict, line: float, margin_pct: float = 0.05) -> dict:
        raise NotImplementedError

    async def price_total_games(self, match_context: dict, line: float, margin_pct: float = 0.05) -> dict:
        raise NotImplementedError

    def validate_arbitrage_free(self, prices: dict, tolerance: float = 0.001) -> bool:
        if not prices:
            raise ValueError("validate_arbitrage_free: prices dict is empty")
        implied_sum = sum(Decimal("1") / v for v in prices.values() if v > 0)
        lower_bound = Decimal("1") - Decimal(str(tolerance))
        if implied_sum < lower_bound:
            raise ValueError(
                f"Arbitrage-OPEN: implied sum {float(implied_sum):.6f} < 1.0 "
                f"(tolerance {tolerance}). Prices: {prices}"
            )
        return True


# ---------------------------------------------------------------------------
# BadmintonPricer — full ISportPricer implementation
# ---------------------------------------------------------------------------

class BadmintonPricer(ISportPricer):
    """
    Platform-registered pricer for badminton.

    Delegates all pricing to BadmintonOrchestratorAgent's supervisor chain.
    Provides the ISportPricer interface expected by the XG3 platform registry.
    """

    def __init__(self, orchestrator: BadmintonOrchestratorAgent) -> None:
        self._orchestrator = orchestrator

    @property
    def sport_key(self) -> str:
        return "badminton"

    # ------------------------------------------------------------------
    # ISportPricer: match winner
    # ------------------------------------------------------------------

    async def price_match_winner(
        self,
        match_context: dict[str, Any],
        margin_pct: float = 0.05,
    ) -> dict[str, Decimal]:
        """
        Return {entity_a_id: odds, entity_b_id: odds} for match-winner market.

        Pulls from pre-match supervisor cache if available; falls back to
        fresh computation from ML model + Markov blend.
        """
        match_id = match_context.get("match_id")
        entity_a_id = match_context.get("entity_a_id")
        entity_b_id = match_context.get("entity_b_id")

        if not all([match_id, entity_a_id, entity_b_id]):
            raise ValueError(
                f"match_context missing required fields: match_id/entity_a_id/entity_b_id. "
                f"Got: {list(match_context.keys())}"
            )

        supervisor = self._orchestrator._pre_match_supervisor
        if supervisor is None:
            raise RuntimeError(
                "BadmintonPricer.price_match_winner: PreMatchSupervisorAgent not initialised. "
                "Call orchestrator.register_match() first."
            )

        prices_result = supervisor.get_prices(match_id)
        if prices_result is None:
            raise RuntimeError(
                f"PreMatchSupervisorAgent returned no prices for match {match_id!r}"
            )

        # Extract match-winner market from price set
        mw_prices = prices_result.markets.get("match_winner")
        if not mw_prices or len(mw_prices) < 2:
            raise RuntimeError(
                f"match_winner market missing or incomplete for {match_id!r}"
            )

        result: dict[str, Decimal] = {}
        for price in mw_prices:
            if "A_wins" in price.outcome_name or price.outcome_name == entity_a_id:
                result[entity_a_id] = Decimal(str(price.odds)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            elif "B_wins" in price.outcome_name or price.outcome_name == entity_b_id:
                result[entity_b_id] = Decimal(str(price.odds)).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

        if len(result) != 2:
            raise RuntimeError(
                f"Could not map match-winner prices to entity IDs for {match_id!r}: {result}"
            )

        self.validate_arbitrage_free(result)

        logger.info(
            "badminton_pricer_match_winner",
            match_id=match_id,
            prices={k: float(v) for k, v in result.items()},
        )
        return result

    # ------------------------------------------------------------------
    # ISportPricer: game handicap (equivalent to set handicap in tennis)
    # ------------------------------------------------------------------

    async def price_set_handicap(
        self,
        match_context: dict[str, Any],
        line: float,
        margin_pct: float = 0.05,
    ) -> dict[str, Decimal]:
        """
        Return {over: odds, under: odds} for game handicap market.

        line: +1.5 / -1.5 (A -1.5 means A must win 2-0)
        """
        match_id = match_context.get("match_id")
        supervisor = self._orchestrator._pre_match_supervisor
        if supervisor is None:
            raise RuntimeError("PreMatchSupervisorAgent not initialised")

        prices_result = supervisor.get_prices(match_id)

        # Map line to market_id
        if abs(line - (-1.5)) < 0.01:
            market_id = "handicap_games_a_minus_1_5"
        elif abs(line - 1.5) < 0.01:
            market_id = "handicap_games_b_minus_1_5"
        else:
            raise ValueError(
                f"Unsupported game handicap line {line} for badminton. "
                "Supported: -1.5, +1.5"
            )

        hcp_prices = prices_result.markets.get(market_id)
        if not hcp_prices or len(hcp_prices) < 2:
            raise RuntimeError(
                f"Handicap market {market_id!r} missing for match {match_id!r}"
            )

        result: dict[str, Decimal] = {}
        for price in hcp_prices:
            key = "over" if "wins_hcp" in price.outcome_name else "under"
            result[key] = Decimal(str(price.odds)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        self.validate_arbitrage_free(result)
        return result

    # ------------------------------------------------------------------
    # ISportPricer: total games
    # ------------------------------------------------------------------

    async def price_total_games(
        self,
        match_context: dict[str, Any],
        line: float,
        margin_pct: float = 0.05,
    ) -> dict[str, Decimal]:
        """
        Return {over: odds, under: odds} for total-games O/U market.

        line: 2.5 (standard badminton total games line)
        """
        match_id = match_context.get("match_id")
        supervisor = self._orchestrator._pre_match_supervisor
        if supervisor is None:
            raise RuntimeError("PreMatchSupervisorAgent not initialised")

        prices_result = supervisor.get_prices(match_id)

        market_id = "total_games_ou"
        tg_prices = prices_result.markets.get(market_id)
        if not tg_prices or len(tg_prices) < 2:
            raise RuntimeError(
                f"total_games market missing for match {match_id!r}"
            )

        result: dict[str, Decimal] = {}
        for price in tg_prices:
            key = "over" if "over" in price.outcome_name else "under"
            result[key] = Decimal(str(price.odds)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        self.validate_arbitrage_free(result)
        return result

    # ------------------------------------------------------------------
    # Extended: full market set
    # ------------------------------------------------------------------

    async def price_all_markets(
        self,
        match_context: dict[str, Any],
    ) -> dict[str, list]:
        """
        Return all pre-match markets.

        Returns raw MarketPrice list per market_id — richer than the
        ISportPricer interface which only covers 3 standard markets.
        """
        match_id = match_context.get("match_id")
        supervisor = self._orchestrator._pre_match_supervisor
        if supervisor is None:
            raise RuntimeError("PreMatchSupervisorAgent not initialised")

        prices_result = supervisor.get_prices(match_id)
        return prices_result.markets

    async def price_live_markets(
        self,
        match_id: str,
    ) -> dict[str, list]:
        """
        Return current live markets for a match.

        Pulls from LiveSupervisorAgent's last computed response.
        """
        supervisor = self._orchestrator._live_supervisor
        if supervisor is None:
            raise RuntimeError("LiveSupervisorAgent not initialised")

        last = supervisor.get_last_prices()
        if last is None:
            raise RuntimeError(
                f"No live prices available for match {match_id!r} yet"
            )
        return last.markets

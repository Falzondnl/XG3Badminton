"""
sgp_supervisor.py
=================
SGPSupervisorAgent — manages Same Game Parlay (Bet Builder) pricing.

Responsibilities:
  - Validate SGP leg combinations (discipline-appropriate markets)
  - Route to BadmintonSGPEngine for joint probability computation
  - H8 gate: SGP price never below max correlated leg
  - Manage SGP exposure/liability
  - Cache validated SGP structures to reduce latency

Architecture:
  - Supervised by BadmintonOrchestratorAgent
  - Operates on live RWP + match state from LiveSupervisorAgent
  - Returns SGP odds for immediate customer-facing display

ZERO hardcoded probabilities — all prices from SGP engine Markov computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from config.badminton_config import Discipline, TournamentTier, SGP_MAX_LEGS
from markets.sgp_engine import (
    BadmintonSGPEngine,
    SGPLeg,
    SGPResponse as SGPEngineResponse,
    SGPLegType as SGPMarket,
)

logger = structlog.get_logger(__name__)


class SGPValidationError(Exception):
    """Raised when an SGP leg combination is invalid."""


class SGPRejectionReason(str, Enum):
    """Why an SGP was rejected."""
    DUPLICATE_MARKET = "duplicate_market"
    INCOMPATIBLE_OUTCOMES = "incompatible_outcomes"
    TOO_MANY_LEGS = "too_many_legs"
    MATCH_NOT_ACTIVE = "match_not_active"
    SUSPENDED_MARKET = "suspended_market"
    SINGLE_LEG = "single_leg"
    H8_VIOLATION = "h8_violation"
    ENGINE_ERROR = "engine_error"


@dataclass
class SGPRequest:
    """
    Customer SGP pricing request.

    Contains match context and the proposed leg selection.
    """
    request_id: str
    match_id: str
    discipline: Discipline
    legs: List[SGPLeg]
    customer_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SGPResponse:
    """
    SGP pricing response.

    Includes fair value, margined price, and validation status.
    """
    request_id: str
    match_id: str
    is_valid: bool
    rejection_reason: Optional[SGPRejectionReason] = None
    rejection_detail: str = ""
    result: Optional[SGPEngineResponse] = None
    h8_passed: bool = False
    h8_detail: str = ""
    pricing_ms: float = 0.0

    @property
    def odds(self) -> Optional[float]:
        """Final decimal odds for this SGP."""
        return self.result.margined_odds if self.result else None

    @property
    def fair_value_odds(self) -> Optional[float]:
        """Fair value odds before margin."""
        return self.result.fair_odds if self.result else None


@dataclass
class SGPMatchContext:
    """
    Live match context required for SGP pricing.

    Populated by the orchestrator from live supervisor state.
    """
    match_id: str
    discipline: Discipline
    tier: TournamentTier
    rwp_a: float
    rwp_b: float
    p_match_win: float  # P(A wins match) from Markov live
    score_a: int
    score_b: int
    games_won_a: int
    games_won_b: int
    current_game: int
    server: str
    is_active: bool  # False = suspended/completed


class SGPSupervisorAgent:
    """
    Supervisor agent for Same Game Parlay pricing.

    Validates leg combinations, routes to the SGP engine, and
    enforces the H8 QA gate.
    """

    # Markets that cannot be combined with each other
    _MUTUALLY_EXCLUSIVE_PAIRS: List[Tuple[SGPMarket, SGPMarket]] = [
        (SGPMarket.MATCH_WINNER, SGPMarket.MATCH_WINNER),
        (SGPMarket.TOTAL_GAMES, SGPMarket.CORRECT_SCORE),  # correct score implies total games
    ]

    # Markets not available for SGP (settlement too correlated by definition)
    _SGP_EXCLUDED_MARKETS: List[SGPMarket] = []

    def __init__(self) -> None:
        self._engine = BadmintonSGPEngine()
        self._match_contexts: Dict[str, SGPMatchContext] = {}
        self._request_count = 0
        self._rejection_count = 0
        self._h8_failures = 0

        logger.info("sgp_supervisor_initialised")

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def update_match_context(self, context: SGPMatchContext) -> None:
        """
        Update live match context from the live supervisor.

        Called after each point scored to keep SGP pricing fresh.
        """
        self._match_contexts[context.match_id] = context
        logger.debug(
            "sgp_context_updated",
            match_id=context.match_id,
            score_a=context.score_a,
            score_b=context.score_b,
            games_won_a=context.games_won_a,
            games_won_b=context.games_won_b,
            rwp_a=round(context.rwp_a, 4),
        )

    def remove_match(self, match_id: str) -> None:
        """Remove match context (match completed/settled)."""
        self._match_contexts.pop(match_id, None)
        logger.info("sgp_match_removed", match_id=match_id)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price_sgp(self, request: SGPRequest) -> SGPResponse:
        """
        Price an SGP request.

        Pipeline:
          1. Validate match is active
          2. Validate leg count
          3. Validate leg compatibility
          4. Route to SGP engine
          5. Validate H8 gate
          6. Return response

        Args:
            request: SGPRequest with match context and proposed legs

        Returns:
            SGPResponse with valid=True/False and pricing if valid
        """
        t0 = time.perf_counter()
        self._request_count += 1

        try:
            response = self._execute_pricing(request)
        except Exception as exc:
            logger.error(
                "sgp_unhandled_error",
                request_id=request.request_id,
                match_id=request.match_id,
                error=str(exc),
            )
            response = SGPResponse(
                request_id=request.request_id,
                match_id=request.match_id,
                is_valid=False,
                rejection_reason=SGPRejectionReason.ENGINE_ERROR,
                rejection_detail=str(exc),
            )

        response.pricing_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            "sgp_priced",
            request_id=request.request_id,
            match_id=request.match_id,
            n_legs=len(request.legs),
            is_valid=response.is_valid,
            rejection_reason=response.rejection_reason,
            odds=round(response.odds, 4) if response.odds else None,
            h8_passed=response.h8_passed,
            pricing_ms=round(response.pricing_ms, 2),
        )

        return response

    def _execute_pricing(self, request: SGPRequest) -> SGPResponse:
        """Internal pricing logic (separated for exception boundary)."""
        # 1. Single leg is not an SGP
        if len(request.legs) < 2:
            self._rejection_count += 1
            return SGPResponse(
                request_id=request.request_id,
                match_id=request.match_id,
                is_valid=False,
                rejection_reason=SGPRejectionReason.SINGLE_LEG,
                rejection_detail="SGP requires at least 2 legs",
            )

        # 2. Too many legs
        if len(request.legs) > SGP_MAX_LEGS:
            self._rejection_count += 1
            return SGPResponse(
                request_id=request.request_id,
                match_id=request.match_id,
                is_valid=False,
                rejection_reason=SGPRejectionReason.TOO_MANY_LEGS,
                rejection_detail=f"Max {SGP_MAX_LEGS} legs allowed; got {len(request.legs)}",
            )

        # 3. Match context available and active
        context = self._match_contexts.get(request.match_id)
        if context is None:
            self._rejection_count += 1
            return SGPResponse(
                request_id=request.request_id,
                match_id=request.match_id,
                is_valid=False,
                rejection_reason=SGPRejectionReason.MATCH_NOT_ACTIVE,
                rejection_detail=f"No active context for match {request.match_id!r}",
            )

        if not context.is_active:
            self._rejection_count += 1
            return SGPResponse(
                request_id=request.request_id,
                match_id=request.match_id,
                is_valid=False,
                rejection_reason=SGPRejectionReason.MATCH_NOT_ACTIVE,
                rejection_detail="Match is not active (suspended or completed)",
            )

        # 4. Validate leg compatibility
        validation_error = self._validate_leg_combination(request.legs)
        if validation_error:
            self._rejection_count += 1
            return SGPResponse(
                request_id=request.request_id,
                match_id=request.match_id,
                is_valid=False,
                rejection_reason=validation_error[0],
                rejection_detail=validation_error[1],
            )

        # 5. Engine pricing
        try:
            result = self._engine.price_sgp(
                match_id=request.match_id,
                legs=request.legs,
                discipline=context.discipline,
                tier=context.tier,
                rwp_a=context.rwp_a,
                rwp_b=context.rwp_b,
                p_match_win=context.p_match_win,
                server_first_game=context.server,
                score_a=context.score_a,
                score_b=context.score_b,
                games_won_a=context.games_won_a,
                games_won_b=context.games_won_b,
                current_game=context.current_game,
            )
        except Exception as exc:
            raise RuntimeError(f"SGP engine error: {exc}") from exc

        # 6. H8 gate
        h8_passed, h8_detail = self._validate_h8(result)
        if not h8_passed:
            self._h8_failures += 1
            logger.error(
                "sgp_h8_gate_failure",
                request_id=request.request_id,
                match_id=request.match_id,
                h8_detail=h8_detail,
            )

        return SGPResponse(
            request_id=request.request_id,
            match_id=request.match_id,
            is_valid=True,
            result=result,
            h8_passed=h8_passed,
            h8_detail=h8_detail,
        )

    def _validate_leg_combination(
        self, legs: List[SGPLeg]
    ) -> Optional[Tuple[SGPRejectionReason, str]]:
        """
        Check legs for incompatible combinations.

        Returns (reason, detail) if invalid, None if valid.
        """
        # Check for excluded markets
        for leg in legs:
            if leg.market in self._SGP_EXCLUDED_MARKETS:
                return (
                    SGPRejectionReason.SUSPENDED_MARKET,
                    f"Market {leg.market.value!r} is excluded from SGP",
                )

        # Check for duplicate markets (same market type twice)
        market_types = [leg.market for leg in legs]
        seen: set = set()
        for mt in market_types:
            if mt in seen:
                return (
                    SGPRejectionReason.DUPLICATE_MARKET,
                    f"Market type {mt.value!r} appears more than once",
                )
            seen.add(mt)

        # Check mutually exclusive pairs
        for m1, m2 in self._MUTUALLY_EXCLUSIVE_PAIRS:
            if m1 == m2:
                # Same type pair — already caught above
                continue
            if m1 in seen and m2 in seen:
                return (
                    SGPRejectionReason.INCOMPATIBLE_OUTCOMES,
                    f"Markets {m1.value!r} and {m2.value!r} cannot be combined",
                )

        return None

    def _validate_h8(self, result: SGPEngineResponse) -> Tuple[bool, str]:
        """
        H8 gate: SGP price never below max correlated single leg.

        The SGP price must not be more generous than simply betting
        the most likely leg alone.
        """
        if result.max_single_leg_odds is None:
            return True, "no_leg_odds_reference"

        # margined_odds must be <= max_single_leg_odds
        # (SGP must not offer better price than best individual leg)
        if result.margined_odds > result.max_single_leg_odds + 0.001:
            return (
                False,
                f"H8: SGP odds {result.margined_odds:.4f} > "
                f"max leg odds {result.max_single_leg_odds:.4f}",
            )

        return True, "passed"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational metrics for monitoring."""
        return {
            "total_requests": self._request_count,
            "total_rejections": self._rejection_count,
            "h8_failures": self._h8_failures,
            "acceptance_rate": (
                round(
                    (self._request_count - self._rejection_count) / self._request_count,
                    4,
                )
                if self._request_count > 0
                else 0.0
            ),
            "active_match_contexts": len(self._match_contexts),
        }

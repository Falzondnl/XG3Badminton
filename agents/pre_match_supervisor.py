"""
pre_match_supervisor.py
=======================
PreMatchSupervisorAgent — manages pre-match pricing for badminton.

Responsibilities:
  - Trigger model inference for upcoming matches
  - Compute RWP estimates for both entities
  - Generate full pre-match market set
  - Schedule price refreshes
  - Respond to incoming odds requests from API layer
  - Handle Pinnacle blend when odds available
  - Escalate to OrchestratorAgent for lifecycle transitions

One PreMatchSupervisorAgent handles all pre-match matches.
Processes matches in priority order (tier → time to match start).

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    PRE_MATCH_REFRESH_INTERVAL_SECONDS,
    MARKET_PRICE_VALIDITY_SECONDS,
)
from markets.pre_match_markets import (
    PreMatchPricingEngine,
    PreMatchPricingRequest,
    PreMatchPricingResponse,
)
from ml.inference import BadmintonModelInference

logger = structlog.get_logger(__name__)


@dataclass
class PreMatchMatchRecord:
    """Record for a pre-match managed match."""
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier
    match_date: date

    # Last pricing
    last_response: Optional[PreMatchPricingResponse] = None
    last_priced_at: float = 0.0

    # Pinnacle line (updated when available)
    pinnacle_p_a: Optional[float] = None

    def needs_refresh(self, interval: float = PRE_MATCH_REFRESH_INTERVAL_SECONDS) -> bool:
        """True if prices are stale and need refreshing."""
        if self.last_response is None:
            return True
        return time.time() - self.last_priced_at >= interval

    def is_price_valid(self) -> bool:
        """True if current prices are within validity window."""
        if self.last_response is None:
            return False
        return time.time() < self.last_response.valid_until


class PreMatchSupervisorAgent:
    """
    Supervises pre-match pricing for all upcoming badminton matches.
    """

    def __init__(
        self,
        inference: Optional[BadmintonModelInference] = None,
    ) -> None:
        self._pricing_engine = PreMatchPricingEngine()
        self._inference = inference or BadmintonModelInference()
        self._matches: Dict[str, PreMatchMatchRecord] = {}

        logger.info("pre_match_supervisor_started")

    def register_match(
        self,
        match_id: str,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        match_date: date,
    ) -> PreMatchMatchRecord:
        """Register a match for pre-match management."""
        if match_id in self._matches:
            return self._matches[match_id]

        record = PreMatchMatchRecord(
            match_id=match_id,
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            discipline=discipline,
            tier=tier,
            match_date=match_date,
        )
        self._matches[match_id] = record
        logger.info(
            "pre_match_registered",
            match_id=match_id,
            discipline=discipline.value,
            tier=tier.value,
        )
        return record

    def get_prices(
        self,
        match_id: str,
        force_refresh: bool = False,
    ) -> Optional[PreMatchPricingResponse]:
        """
        Get current pre-match prices for a match.

        Returns cached response if still valid, otherwise refreshes.
        """
        record = self._matches.get(match_id)
        if not record:
            logger.warning("pre_match_prices_requested_unknown_match", match_id=match_id)
            return None

        if force_refresh or record.needs_refresh():
            self._refresh_prices(record)

        return record.last_response

    def update_pinnacle_line(
        self,
        match_id: str,
        pinnacle_p_a: float,
    ) -> None:
        """Update Pinnacle odds for a match (triggers price refresh)."""
        record = self._matches.get(match_id)
        if not record:
            return
        record.pinnacle_p_a = pinnacle_p_a
        self._refresh_prices(record)

    def _refresh_prices(self, record: PreMatchMatchRecord) -> None:
        """Fetch fresh model inference and reprice."""
        try:
            # Get model features and inference
            inference_result = self._inference.predict(
                entity_a_id=record.entity_a_id,
                entity_b_id=record.entity_b_id,
                discipline=record.discipline,
                tier=record.tier,
                match_date=record.match_date,
            )

            request = PreMatchPricingRequest(
                match_id=record.match_id,
                entity_a_id=record.entity_a_id,
                entity_b_id=record.entity_b_id,
                discipline=record.discipline,
                tier=record.tier,
                match_date=record.match_date,
                model_p_a_wins=inference_result.p_a_wins,
                model_p_a_wins_2_0=inference_result.p_a_wins_2_0,
                model_p_a_wins_deuce=inference_result.p_deuce,
                rwp_a=inference_result.rwp_a,
                rwp_b=inference_result.rwp_b,
                pinnacle_p_a_wins=record.pinnacle_p_a,
            )

            response = self._pricing_engine.price(request)
            record.last_response = response
            record.last_priced_at = time.time()

            logger.debug(
                "pre_match_prices_refreshed",
                match_id=record.match_id,
                p_a=f"{response.p_a_wins_blend:.4f}",
            )

        except Exception as exc:
            logger.error(
                "pre_match_pricing_error",
                match_id=record.match_id,
                error=str(exc),
            )

    def get_all_valid_markets(self) -> Dict[str, PreMatchPricingResponse]:
        """Return all valid pre-match markets."""
        return {
            match_id: record.last_response
            for match_id, record in self._matches.items()
            if record.is_price_valid()
        }

    def get_stats(self) -> Dict:
        """Return supervisor statistics."""
        n_valid = sum(1 for r in self._matches.values() if r.is_price_valid())
        return {
            "n_matches": len(self._matches),
            "n_valid_prices": n_valid,
        }

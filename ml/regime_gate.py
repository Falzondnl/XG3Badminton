"""
regime_gate.py
==============
Regime gate for badminton ML model — classifies matches into pricing regimes.

Regime classification:
  R0 — Sparse data: entity has < 10 matches or ELO is default (new player)
       Action: use wider margins, disable some derivative markets
  R1 — Standard: most BWF World Tour matches
       Action: standard margins and full market suite
  R2 — Rich data: top 10 players, major tournaments, large match sample
       Action: tighter margins, full derivative suite including exotic

Regime is used by:
  - PreMatchPricingEngine (margin selection)
  - DerivativeEngine (which families to offer)
  - LivePricingEngine (click scaling)

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog

from config.badminton_config import (
    TournamentTier,
    REGIME_R0_MIN_MATCHES,
    REGIME_R2_TIERS,
    REGIME_R2_MIN_MATCHES,
)

logger = structlog.get_logger(__name__)


class Regime(str, Enum):
    R0 = "R0"   # Sparse — wide margins
    R1 = "R1"   # Standard
    R2 = "R2"   # Rich — tight margins


@dataclass
class RegimeInput:
    """Input data for regime classification."""
    entity_a_match_count: Optional[int]
    entity_b_match_count: Optional[int]
    entity_a_elo_is_default: bool
    entity_b_elo_is_default: bool
    tier: TournamentTier
    discipline_value: str


class RegimeGate:
    """
    Classifies a match into a pricing regime.

    Called once per match pre-price, output cached for the match duration.
    """

    @staticmethod
    def classify(inputs: RegimeInput) -> Regime:
        """
        Classify match into R0/R1/R2.

        Priority (highest to lowest):
          1. If either entity has default ELO (< 5 matches): R0
          2. If tier is Super 1000/750/Olympics and both entities have >= R2 threshold: R2
          3. Otherwise: R1
        """
        count_a = inputs.entity_a_match_count or 0
        count_b = inputs.entity_b_match_count or 0

        # R0: sparse data
        if (
            inputs.entity_a_elo_is_default
            or inputs.entity_b_elo_is_default
            or count_a < REGIME_R0_MIN_MATCHES
            or count_b < REGIME_R0_MIN_MATCHES
        ):
            logger.debug(
                "regime_r0_assigned",
                count_a=count_a,
                count_b=count_b,
                a_default=inputs.entity_a_elo_is_default,
                b_default=inputs.entity_b_elo_is_default,
            )
            return Regime.R0

        # R2: rich data
        if (
            inputs.tier in REGIME_R2_TIERS
            and count_a >= REGIME_R2_MIN_MATCHES
            and count_b >= REGIME_R2_MIN_MATCHES
        ):
            return Regime.R2

        return Regime.R1

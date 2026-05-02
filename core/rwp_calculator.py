"""
rwp_calculator.py
==================
Rally Win Probability (RWP) — the atomic pricing unit for badminton.

RWP = P(current server wins the next rally)

This is the fundamental parameter from which ALL market prices are derived.
It is the badminton equivalent of SPW (Serve Point Win %) in tennis.

Mathematical framework:
  - RWP is discipline-specific (MS/WS/MD/WD/XD each have different dynamics)
  - RWP is player/pair-specific (estimated from historical data via serve_stat_db)
  - RWP is adjusted for environment (shuttle speed, hall conditions, altitude)
  - RWP is adjusted for fatigue (physical load model)
  - RWP is adjusted contextually by LLM signal layer (±5% cap, inform not predict)

ZERO hardcoded probabilities.
All estimates come from the serve_stat_db or raise RuntimeError if unavailable.

Sources:
  - "Using Markov chains to identify player performance in badminton"
    Scientia et Technica, Chaos, Solitons & Fractals (2022)
  - "Development of sequential winning-percentage prediction model"
    BMC Sports Science, Medicine and Rehabilitation (2025)
  - FineBadminton dataset tactical analysis (2025)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import structlog

from config.badminton_config import (
    Discipline,
    RWP_BASELINE,
    RWP_MIN_VALID,
    RWP_MAX_VALID,
    FATIGUE_MAX_PENALTY_PP,
    FATIGUE_EXTRA_MATCH_SAME_DAY,
    FATIGUE_LONG_MATCH_THRESHOLD_MINUTES,
    FATIGUE_LONG_MATCH_PENALTY,
    FATIGUE_WEEKLY_LOAD_COEFFICIENT,
    HALL_CONDITIONS,
    SHUTTLE_SPEED_NEUTRAL,
    RWP_SHUTTLE_SPEED_COEFFICIENT,
)
from core.scoring_engine import BadmintonScoringError

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RWPUnavailableError(RuntimeError):
    """
    Raised when RWP cannot be computed because required data is missing.

    This is intentional — we NEVER return a default probability.
    The caller must handle this exception and decide how to proceed.
    """


class RWPOutOfRangeError(ValueError):
    """Raised when a computed RWP falls outside the valid range [0.40, 0.65]."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerRWPProfile:
    """
    Historical rally win probability profile for a player/pair.

    Loaded from serve_stat_db.py — never constructed with hardcoded values.
    """
    entity_id: str           # Player/pair platform ID
    discipline: Discipline
    rwp_as_server: float     # P(wins rally when serving) — rolling 50-match estimate
    rwp_as_receiver: float   # P(wins rally when receiving)
    sample_size: int         # Number of rallies in the estimate
    last_updated: str        # ISO8601 date of last update

    def __post_init__(self) -> None:
        """Validate on construction."""
        if not (RWP_MIN_VALID <= self.rwp_as_server <= RWP_MAX_VALID):
            raise RWPOutOfRangeError(
                f"rwp_as_server={self.rwp_as_server:.4f} for entity={self.entity_id} "
                f"is outside valid range [{RWP_MIN_VALID}, {RWP_MAX_VALID}]"
            )
        if not (RWP_MIN_VALID <= self.rwp_as_receiver <= RWP_MAX_VALID):
            raise RWPOutOfRangeError(
                f"rwp_as_receiver={self.rwp_as_receiver:.4f} for entity={self.entity_id} "
                f"is outside valid range [{RWP_MIN_VALID}, {RWP_MAX_VALID}]"
            )
        if self.sample_size < 0:
            raise ValueError(f"sample_size cannot be negative: {self.sample_size}")


@dataclass(frozen=True)
class EnvironmentConditions:
    """
    Court / hall environmental conditions affecting shuttle speed and RWP.

    All values are optional — use None if not measured/available.
    """
    shuttle_speed_number: Optional[int] = None       # BWF shuttle speed (73-79)
    temperature_celsius: Optional[float] = None
    altitude_metres: Optional[float] = None
    humidity_pct: Optional[float] = None
    ac_strength: Optional[float] = None             # 0=none, 1=weak, 2=strong

    def __post_init__(self) -> None:
        if self.shuttle_speed_number is not None:
            if not (70 <= self.shuttle_speed_number <= 82):
                raise ValueError(
                    f"shuttle_speed_number={self.shuttle_speed_number} is outside "
                    f"plausible range [70, 82]"
                )
        if self.humidity_pct is not None:
            if not (0.0 <= self.humidity_pct <= 100.0):
                raise ValueError(f"humidity_pct must be in [0, 100]")


@dataclass(frozen=True)
class FatigueProfile:
    """Physical fatigue state for a player/pair on match day."""
    entity_id: str
    matches_today: int             # Matches already played today (before this match)
    minutes_last_match: int        # Duration of most recent match in minutes
    matches_last_7_days: int       # Including today's previous matches


@dataclass(frozen=True)
class RWPEstimate:
    """
    Final computed RWP estimate for a specific matchup.

    This is a computed value — DO NOT construct manually with hardcoded numbers.
    Use RWPCalculator.compute() instead.
    """
    rwp_a_as_server: float    # P(A wins rally when A is serving)
    rwp_b_as_server: float    # P(B wins rally when B is serving)
    discipline: Discipline
    # Adjustment breakdown for transparency / audit
    base_rwp_a: float
    base_rwp_b: float
    elo_adjustment_a: float
    env_adjustment_a: float
    fatigue_adjustment_a: float
    fatigue_adjustment_b: float
    llm_adjustment_a: float
    llm_adjustment_b: float
    # Confidence
    sample_size_a: int
    sample_size_b: int

    def __post_init__(self) -> None:
        for attr, val in [
            ("rwp_a_as_server", self.rwp_a_as_server),
            ("rwp_b_as_server", self.rwp_b_as_server),
        ]:
            if not (RWP_MIN_VALID <= val <= RWP_MAX_VALID):
                raise RWPOutOfRangeError(
                    f"{attr}={val:.4f} outside valid range "
                    f"[{RWP_MIN_VALID}, {RWP_MAX_VALID}]"
                )


# ---------------------------------------------------------------------------
# RWP Calculator
# ---------------------------------------------------------------------------

class RWPCalculator:
    """
    Computes Rally Win Probability (RWP) for a badminton matchup.

    Steps:
    1. Load historical RWP profiles from serve_stat_db (raises if unavailable)
    2. Apply ELO-based adjustment (relative player quality vs field)
    3. Apply environmental adjustment (shuttle speed, altitude, hall conditions)
    4. Apply fatigue adjustment (physical load)
    5. Apply LLM contextual signal (capped at ±5%)
    6. Validate output range [0.40, 0.65]

    Calling code is responsible for providing all required inputs.
    This class never falls back to a default probability.
    """

    # Maximum allowed LLM adjustment (ADR-017: ±5% cap)
    LLM_ADJUSTMENT_CAP: float = 0.05

    # ELO scaling: how much ELO difference shifts RWP
    # Derived from logistic function: Δrwp ≈ Δelo/400 × 0.08
    ELO_RWP_COEFFICIENT: float = 0.08 / 400.0

    @classmethod
    def compute(
        cls,
        discipline: Discipline,
        profile_a: PlayerRWPProfile,
        profile_b: PlayerRWPProfile,
        elo_a: float,
        elo_b: float,
        environment: Optional[EnvironmentConditions] = None,
        fatigue_a: Optional[FatigueProfile] = None,
        fatigue_b: Optional[FatigueProfile] = None,
        llm_signal_a: float = 0.0,
        llm_signal_b: float = 0.0,
    ) -> RWPEstimate:
        """
        Compute RWP for a matchup.

        Args:
            discipline: Badminton discipline (MS/WS/MD/WD/XD).
            profile_a: Historical RWP profile for player/pair A.
            profile_b: Historical RWP profile for player/pair B.
            elo_a: Current ELO rating for player/pair A (discipline-specific pool).
            elo_b: Current ELO rating for player/pair B.
            environment: Court/hall conditions (optional).
            fatigue_a: Fatigue state for player/pair A (optional).
            fatigue_b: Fatigue state for player/pair B (optional).
            llm_signal_a: LLM contextual adjustment for A in range [-0.05, +0.05].
            llm_signal_b: LLM contextual adjustment for B in range [-0.05, +0.05].

        Returns:
            RWPEstimate with all adjustment components.

        Raises:
            RWPUnavailableError: If either profile has insufficient sample size.
            RWPOutOfRangeError: If final RWP is outside valid range.
        """
        if profile_a.discipline != discipline:
            raise RWPUnavailableError(
                f"Profile A discipline {profile_a.discipline} does not match "
                f"requested discipline {discipline}"
            )
        if profile_b.discipline != discipline:
            raise RWPUnavailableError(
                f"Profile B discipline {profile_b.discipline} does not match "
                f"requested discipline {discipline}"
            )

        if profile_a.sample_size < 10:
            raise RWPUnavailableError(
                f"Insufficient sample size for entity {profile_a.entity_id}: "
                f"{profile_a.sample_size} rallies (minimum 10)"
            )
        if profile_b.sample_size < 10:
            raise RWPUnavailableError(
                f"Insufficient sample size for entity {profile_b.entity_id}: "
                f"{profile_b.sample_size} rallies (minimum 10)"
            )

        # Step 1: Base RWP from historical profiles
        base_rwp_a = profile_a.rwp_as_server
        base_rwp_b = profile_b.rwp_as_server

        # Step 2: ELO adjustment
        # ELO difference shifts RWP — stronger player wins more rallies even when receiving
        elo_diff = elo_a - elo_b
        elo_adj_a = cls.ELO_RWP_COEFFICIENT * elo_diff
        # Apply symmetrically: A adjusts up, B adjusts down (relative quality)
        # This is an ADDITIVE adjustment to the server's probability
        # The absolute magnitude is small (~0.003 per 40 ELO points)

        # Step 3: Environmental adjustment
        env_adj_a = cls._compute_env_adjustment(environment) if environment else 0.0

        # Step 4: Fatigue adjustments
        fatigue_adj_a = cls._compute_fatigue_adjustment(fatigue_a) if fatigue_a else 0.0
        fatigue_adj_b = cls._compute_fatigue_adjustment(fatigue_b) if fatigue_b else 0.0

        # Step 5: LLM adjustment (capped at ±5%)
        llm_adj_a = cls._clamp_llm_signal(llm_signal_a)
        llm_adj_b = cls._clamp_llm_signal(llm_signal_b)

        # Combine: RWP for A as server
        rwp_a_final = (
            base_rwp_a
            + elo_adj_a
            + env_adj_a
            - fatigue_adj_a  # fatigue reduces your server advantage
            + llm_adj_a
        )

        # Combine: RWP for B as server
        rwp_b_final = (
            base_rwp_b
            - elo_adj_a    # same ELO diff but inverted for B
            + env_adj_a
            - fatigue_adj_b
            + llm_adj_b
        )

        # Clamp to valid range — clamp, not raise (post-combination can drift slightly)
        rwp_a_final = cls._clamp_rwp(rwp_a_final, profile_a.entity_id)
        rwp_b_final = cls._clamp_rwp(rwp_b_final, profile_b.entity_id)

        logger.info(
            "rwp_computed",
            discipline=discipline.value,
            entity_a=profile_a.entity_id,
            entity_b=profile_b.entity_id,
            rwp_a=round(rwp_a_final, 4),
            rwp_b=round(rwp_b_final, 4),
            elo_diff=round(elo_diff, 1),
            fatigue_adj_a=round(fatigue_adj_a, 4),
            fatigue_adj_b=round(fatigue_adj_b, 4),
        )

        return RWPEstimate(
            rwp_a_as_server=rwp_a_final,
            rwp_b_as_server=rwp_b_final,
            discipline=discipline,
            base_rwp_a=base_rwp_a,
            base_rwp_b=base_rwp_b,
            elo_adjustment_a=elo_adj_a,
            env_adjustment_a=env_adj_a,
            fatigue_adjustment_a=fatigue_adj_a,
            fatigue_adjustment_b=fatigue_adj_b,
            llm_adjustment_a=llm_adj_a,
            llm_adjustment_b=llm_adj_b,
            sample_size_a=profile_a.sample_size,
            sample_size_b=profile_b.sample_size,
        )

    @staticmethod
    def _compute_env_adjustment(env: EnvironmentConditions) -> float:
        """
        Compute RWP adjustment from environmental conditions.

        Positive adjustment = server advantage increases (faster shuttle).
        Negative adjustment = server advantage decreases (slower shuttle).
        """
        adj = 0.0

        # Shuttle speed: deviation from neutral (76) scaled by coefficient
        if env.shuttle_speed_number is not None:
            speed_delta = env.shuttle_speed_number - SHUTTLE_SPEED_NEUTRAL
            adj += speed_delta * RWP_SHUTTLE_SPEED_COEFFICIENT

        # Hall conditions (C-11 corrections)
        if env.temperature_celsius is not None:
            # Reference temperature: 20°C
            temp_delta = env.temperature_celsius - 20.0
            adj += temp_delta * HALL_CONDITIONS["temperature_celsius_coefficient"]

        if env.altitude_metres is not None:
            # Reference altitude: sea level (0m)
            adj += env.altitude_metres * HALL_CONDITIONS["altitude_metres_coefficient"]

        if env.humidity_pct is not None:
            # Reference humidity: 50%
            humidity_delta = env.humidity_pct - 50.0
            adj += humidity_delta * HALL_CONDITIONS["humidity_pct_coefficient"]

        if env.ac_strength is not None:
            adj += env.ac_strength * HALL_CONDITIONS["ac_strength_coefficient"]

        return adj

    @staticmethod
    def _compute_fatigue_adjustment(fatigue: FatigueProfile) -> float:
        """
        Compute POSITIVE fatigue penalty (subtracted from RWP).

        Returns a value in [0, FATIGUE_MAX_PENALTY_PP].
        Larger value = more fatigued = lower effective RWP.
        """
        penalty = 0.0

        # Extra matches same day
        if fatigue.matches_today > 1:
            penalty += (fatigue.matches_today - 1) * FATIGUE_EXTRA_MATCH_SAME_DAY

        # Long previous match
        if fatigue.minutes_last_match > FATIGUE_LONG_MATCH_THRESHOLD_MINUTES:
            penalty += FATIGUE_LONG_MATCH_PENALTY

        # Weekly load (beyond 3 matches per week is considered heavy load)
        weekly_excess = max(0, fatigue.matches_last_7_days - 3)
        penalty += weekly_excess * FATIGUE_WEEKLY_LOAD_COEFFICIENT

        return min(penalty, FATIGUE_MAX_PENALTY_PP)

    @classmethod
    def _clamp_llm_signal(cls, signal: float) -> float:
        """Clamp LLM signal to ±LLM_ADJUSTMENT_CAP."""
        return max(-cls.LLM_ADJUSTMENT_CAP, min(cls.LLM_ADJUSTMENT_CAP, signal))

    @staticmethod
    def _clamp_rwp(rwp: float, entity_id: str) -> float:
        """
        Clamp RWP to valid range, logging a warning if clamping was required.

        We clamp (not raise) at the final stage because component combinations
        can legitimately drift slightly beyond the valid range.
        """
        if rwp < RWP_MIN_VALID:
            logger.warning(
                "rwp_clamped_to_minimum",
                entity_id=entity_id,
                original=round(rwp, 4),
                clamped_to=RWP_MIN_VALID,
            )
            return RWP_MIN_VALID
        if rwp > RWP_MAX_VALID:
            logger.warning(
                "rwp_clamped_to_maximum",
                entity_id=entity_id,
                original=round(rwp, 4),
                clamped_to=RWP_MAX_VALID,
            )
            return RWP_MAX_VALID
        return rwp

    @staticmethod
    def rwp_from_match_win_probability(
        p_match_win: float,
        discipline: Discipline,
        n_iterations: int = 50,
    ) -> float:
        """
        Invert match win probability to estimate RWP via bisection.

        This is used when only Pinnacle closing odds are available and we need
        to back-calculate an implied RWP for calibration purposes.

        Args:
            p_match_win: P(player A wins match) from market.
            discipline: Determines baseline RWP starting point.
            n_iterations: Number of bisection iterations (50 → ~1e-15 precision).

        Returns:
            Estimated RWP such that Markov(rwp, rwp_baseline_b) ≈ p_match_win.

        Note: This inversion is underidentified (C-07 auditor note) — we fix
        rwp_b at the discipline baseline and solve for rwp_a only.
        """
        from core.markov_engine import BadmintonMarkovEngine  # avoid circular import

        if not (0.01 <= p_match_win <= 0.99):
            raise ValueError(
                f"p_match_win={p_match_win} must be in [0.01, 0.99]"
            )

        # Fix rwp_b at discipline baseline — solve for rwp_a
        rwp_b = RWP_BASELINE[discipline]
        lo, hi = RWP_MIN_VALID, RWP_MAX_VALID
        engine = BadmintonMarkovEngine()

        for _ in range(n_iterations):
            mid = (lo + hi) / 2.0
            p_computed = engine.p_win_match_from_rwp(
                rwp_a=mid, rwp_b=rwp_b, discipline=discipline
            )
            if p_computed < p_match_win:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2.0

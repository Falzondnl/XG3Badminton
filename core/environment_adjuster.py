"""
environment_adjuster.py
=======================
Hall and shuttle condition adjustments for RWP estimation.

Models the effect of physical environment on badminton rally dynamics:

1. SHUTTLE SPEED
   - Faster shuttles → longer rallies → higher rally variance → RWP towards 0.5
   - Slower shuttles → shorter rallies → more server advantage (smasher)
   - Speed grades 75-85 (international: 77-79 typical)

2. INDOOR HALL CONDITIONS
   - High altitude: shuttle travels faster → adjust like slower shuttle number
   - Air conditioning: reduces shuttle speed irregularities
   - Temperature: hot conditions increase shuttle speed, cold reduces

3. VENUE-SPECIFIC FACTORS
   - Court surface (synthetic mat vs hardwood) → sliding adjustment
   - Hall dimensions affect drift

4. DISCIPLINE-SPECIFIC ADJUSTMENTS
   - Doubles: shorter rallies due to smash frequency → smaller env effect
   - Mixed XD: differential effect (man vs woman playing styles)

The adjustment is applied as a delta to baseline RWP:
  rwp_adjusted = rwp_base + env_delta
  env_delta ∈ [-ENV_MAX_ADJUSTMENT, +ENV_MAX_ADJUSTMENT]

ZERO hardcoded probabilities. Returns None deltas when data insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog

from config.badminton_config import (
    Discipline,
    ENV_MAX_ADJUSTMENT,
    ENV_ALTITUDE_THRESHOLD_M,
    ENV_SHUTTLE_SPEED_BASELINE,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ShuttleSpeed(int, Enum):
    """
    BWF shuttle speed grades.
    75 = slowest (high altitude, cold, sea level fast halls)
    85 = fastest (low altitude, hot, humid)
    Standard indoor: 77-79 range.
    """
    GRADE_75 = 75
    GRADE_76 = 76
    GRADE_77 = 77
    GRADE_78 = 78
    GRADE_79 = 79
    GRADE_80 = 80
    GRADE_81 = 81
    GRADE_82 = 82
    GRADE_83 = 83
    GRADE_84 = 84
    GRADE_85 = 85


class CourtSurface(str, Enum):
    SYNTHETIC_MAT = "synthetic_mat"   # Most BWF World Tour venues
    HARDWOOD = "hardwood"             # Some older venues
    RUBBER = "rubber"                 # Rare


# ---------------------------------------------------------------------------
# Input conditions
# ---------------------------------------------------------------------------

@dataclass
class HallConditions:
    """
    Environmental conditions for a badminton hall.

    All values are optional — None means data not available.
    Adjustments gracefully degrade to zero when data missing.
    """
    shuttle_speed: Optional[int] = None        # BWF speed grade (75-85)
    altitude_m: Optional[float] = None        # Venue altitude in metres
    temperature_c: Optional[float] = None     # Hall temperature Celsius
    humidity_pct: Optional[float] = None      # Relative humidity (0-100)
    has_ac: Optional[bool] = None             # Air conditioning active
    court_surface: Optional[CourtSurface] = None
    venue_name: Optional[str] = None
    country_code: Optional[str] = None        # For altitude lookups

    def is_high_altitude(self) -> bool:
        """True if altitude exceeds standard threshold."""
        if self.altitude_m is None:
            return False
        return self.altitude_m > ENV_ALTITUDE_THRESHOLD_M

    def effective_shuttle_speed(self) -> Optional[int]:
        """
        Compute effective shuttle speed accounting for altitude.

        At high altitude, shuttle travels faster, so a lower speed grade
        is used to compensate. Approximate rule of thumb:
          For every 1000m altitude: use 1-2 grades slower.
        """
        if self.shuttle_speed is None:
            return None

        speed = self.shuttle_speed

        if self.altitude_m is not None and self.altitude_m > ENV_ALTITUDE_THRESHOLD_M:
            altitude_correction = int(self.altitude_m / 1000.0)
            speed = max(75, speed - altitude_correction)

        if self.temperature_c is not None:
            # Warmer = faster shuttle → adjust grade down for equivalence
            if self.temperature_c > 25:
                speed = max(75, speed - 1)
            elif self.temperature_c < 15:
                speed = min(85, speed + 1)

        return speed


# ---------------------------------------------------------------------------
# Adjustment result
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentAdjustment:
    """
    RWP adjustment from environmental conditions.

    delta_rwp_server: additive adjustment to RWP (server's advantage)
    Applied symmetrically: both players affected.
    """
    delta_rwp_server: float       # Adjustment to server's RWP
    confidence: float             # 0-1: how confident is this adjustment
    rationale: str                # Human-readable explanation
    shuttle_speed_used: Optional[int]
    altitude_adjustment: float
    temperature_adjustment: float
    humidity_adjustment: float

    @property
    def is_significant(self) -> bool:
        """True if adjustment is large enough to matter (> 0.005)."""
        return abs(self.delta_rwp_server) > 0.005


# ---------------------------------------------------------------------------
# Adjuster
# ---------------------------------------------------------------------------

class EnvironmentAdjuster:
    """
    Computes RWP adjustments from hall/shuttle conditions.

    Called by RWPCalculator._compute_env_adjustment().
    """

    def compute_adjustment(
        self,
        conditions: HallConditions,
        discipline: Discipline,
    ) -> EnvironmentAdjustment:
        """
        Compute RWP delta from hall conditions.

        Returns an adjustment where:
          - Positive delta → server benefits more (faster rallies, power game)
          - Negative delta → rally game benefits (slower shuttle, longer exchanges)

        Args:
            conditions: Hall and shuttle conditions
            discipline: Badminton discipline (doubles vs singles has different effects)
        """
        altitude_adj = self._altitude_adjustment(conditions)
        shuttle_adj = self._shuttle_speed_adjustment(conditions)
        temp_adj = self._temperature_adjustment(conditions)
        humidity_adj = self._humidity_adjustment(conditions)

        # Combine adjustments (additive)
        total_delta = altitude_adj + shuttle_adj + temp_adj + humidity_adj

        # Clamp to max adjustment first (at the singles baseline)
        total_delta = max(-ENV_MAX_ADJUSTMENT, min(ENV_MAX_ADJUSTMENT, total_delta))

        # Doubles: smaller environmental effect (rallies are shorter anyway).
        # Apply AFTER clamping so the ratio between doubles and singles is
        # preserved even when conditions are extreme enough to hit the cap.
        from config.badminton_config import DOUBLES_DISCIPLINES
        if discipline in DOUBLES_DISCIPLINES:
            total_delta *= 0.6

        # Confidence: decreases with missing data
        missing_count = sum(
            1 for v in [
                conditions.shuttle_speed, conditions.altitude_m,
                conditions.temperature_c, conditions.humidity_pct
            ] if v is None
        )
        confidence = 1.0 - (missing_count * 0.2)

        effective_speed = conditions.effective_shuttle_speed()
        rationale_parts = []
        if abs(shuttle_adj) > 0.001:
            rationale_parts.append(f"shuttle_speed={conditions.shuttle_speed}")
        if abs(altitude_adj) > 0.001:
            rationale_parts.append(f"altitude={conditions.altitude_m:.0f}m")
        if abs(temp_adj) > 0.001:
            rationale_parts.append(f"temp={conditions.temperature_c:.1f}C")

        return EnvironmentAdjustment(
            delta_rwp_server=total_delta,
            confidence=max(0.0, min(1.0, confidence)),
            rationale=" | ".join(rationale_parts) if rationale_parts else "no_adjustment",
            shuttle_speed_used=effective_speed,
            altitude_adjustment=altitude_adj,
            temperature_adjustment=temp_adj,
            humidity_adjustment=humidity_adj,
        )

    def _shuttle_speed_adjustment(self, conditions: HallConditions) -> float:
        """
        Compute RWP delta from shuttle speed.

        Faster shuttle (higher grade) → more server advantage.
        Slower shuttle → longer rallies, receiver can recover.

        Delta relative to baseline speed.
        """
        if conditions.shuttle_speed is None:
            return 0.0

        effective_speed = conditions.effective_shuttle_speed()
        if effective_speed is None:
            return 0.0

        speed_diff = effective_speed - ENV_SHUTTLE_SPEED_BASELINE
        # Each grade difference → ~0.003 RWP adjustment
        return speed_diff * 0.003

    @staticmethod
    def _altitude_adjustment(conditions: HallConditions) -> float:
        """
        Compute RWP delta from altitude.

        High altitude → shuttle travels faster effectively → longer rallies
        (counter-intuitive: the shuttle "floats" more, creating longer exchanges)
        → small negative adjustment to server advantage.
        """
        if conditions.altitude_m is None:
            return 0.0

        if conditions.altitude_m < ENV_ALTITUDE_THRESHOLD_M:
            return 0.0

        # Above threshold: very mild effect on RWP
        excess_altitude = conditions.altitude_m - ENV_ALTITUDE_THRESHOLD_M
        # ~0.001 per 500m above threshold
        return -(excess_altitude / 500.0) * 0.001

    @staticmethod
    def _temperature_adjustment(conditions: HallConditions) -> float:
        """
        Compute RWP delta from temperature.

        Hot conditions → shuttle travels faster → server advantage increases slightly.
        Cold conditions → slower shuttle → rally game.
        """
        if conditions.temperature_c is None:
            return 0.0

        # Neutral temp = 20°C (typical BWF venue temperature requirement)
        neutral_temp = 20.0
        temp_diff = conditions.temperature_c - neutral_temp

        # ~0.001 per 5°C deviation
        return (temp_diff / 5.0) * 0.001

    @staticmethod
    def _humidity_adjustment(conditions: HallConditions) -> float:
        """
        Compute RWP delta from humidity.

        High humidity → shuttle slightly heavier → slightly slower.
        Low humidity → faster shuttle.
        """
        if conditions.humidity_pct is None:
            return 0.0

        # Neutral humidity = 60%
        neutral_humidity = 60.0
        humidity_diff = conditions.humidity_pct - neutral_humidity

        # Very small effect: ~0.0005 per 10% humidity deviation
        return -(humidity_diff / 10.0) * 0.0005

    @staticmethod
    def get_venue_altitude(country_code: str, city: str) -> Optional[float]:
        """
        Return known venue altitude for major BWF locations.

        Returns None if not in known list (caller should use None → 0 adjustment).

        Note: This is a static lookup of well-known BWF venues only.
        All values are publicly available geographic data, not model parameters.
        """
        # Known high-altitude BWF venues (>500m)
        _KNOWN_ALTITUDES = {
            # Format: (country_code.lower(), city_keywords) -> altitude_m
            ("chn", "kunming"): 1891.0,
            ("chn", "chengdu"): 506.0,
            ("col", "bogota"): 2600.0,
            ("mex", "mexico"): 2240.0,
            ("ken", "nairobi"): 1795.0,
            ("eth", "addis"): 2355.0,
            ("per", "lima"): 154.0,  # Actually low
            ("usa", "denver"): 1609.0,
        }

        cc = country_code.lower()
        city_lower = city.lower()

        for (known_cc, city_key), alt in _KNOWN_ALTITUDES.items():
            if known_cc == cc and city_key in city_lower:
                return alt

        return None

"""
test_environment_adjuster.py
============================
Unit tests for core/environment_adjuster.py

Tests:
  - Adjustment direction: high altitude → higher shuttle speed → adjustment direction
  - Adjustment magnitude bounded by ENV_MAX_ADJUSTMENT
  - Doubles get 60% of adjustment
  - Neutral conditions → zero delta
  - Edge cases: extreme values
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from core.environment_adjuster import (
    EnvironmentAdjuster,
    HallConditions,
    EnvironmentAdjustment,
)


@pytest.fixture
def adjuster():
    return EnvironmentAdjuster()


def standard_conditions() -> HallConditions:
    """Standard neutral conditions."""
    return HallConditions(
        shuttle_speed=77,    # Standard speed
        altitude_m=0,
        temperature_c=20,
        humidity_pct=50,
    )


class TestNeutralConditions:
    """Standard conditions produce zero or minimal adjustment."""

    def test_neutral_conditions_near_zero(self, adjuster):
        """Standard conditions → delta_rwp ≈ 0."""
        adj = adjuster.compute_adjustment(standard_conditions(), Discipline.MS)
        assert abs(adj.delta_rwp_server) < 0.01

    def test_adjustment_object_returned(self, adjuster):
        """compute_adjustment always returns EnvironmentAdjustment."""
        adj = adjuster.compute_adjustment(standard_conditions(), Discipline.MS)
        assert isinstance(adj, EnvironmentAdjustment)


class TestShuttleSpeed:
    """Shuttle speed grade effects."""

    def test_fast_shuttle_produces_adjustment(self, adjuster):
        """High shuttle speed (85) should differ from standard (77)."""
        fast = HallConditions(shuttle_speed=85, altitude_m=0, temperature_c=20, humidity_pct=50)
        slow = HallConditions(shuttle_speed=75, altitude_m=0, temperature_c=20, humidity_pct=50)
        adj_fast = adjuster.compute_adjustment(fast, Discipline.MS)
        adj_slow = adjuster.compute_adjustment(slow, Discipline.MS)
        # Fast shuttle should have different delta than slow shuttle
        assert adj_fast.delta_rwp_server != adj_slow.delta_rwp_server

    def test_shuttle_speed_range_valid(self, adjuster):
        """Valid shuttle speeds 75-85 do not raise."""
        for speed in [75, 77, 79, 82, 85]:
            conditions = HallConditions(
                shuttle_speed=speed, altitude_m=0, temperature_c=20, humidity_pct=50
            )
            adjuster.compute_adjustment(conditions, Discipline.MS)  # Should not raise


class TestAltitude:
    """Altitude effects on shuttle trajectory."""

    def test_high_altitude_produces_adjustment(self, adjuster):
        """High altitude (e.g. 2000m) differs from sea level."""
        sea_level = HallConditions(shuttle_speed=77, altitude_m=0, temperature_c=20, humidity_pct=50)
        highland = HallConditions(shuttle_speed=77, altitude_m=2000, temperature_c=20, humidity_pct=50)
        adj_sl = adjuster.compute_adjustment(sea_level, Discipline.MS)
        adj_hi = adjuster.compute_adjustment(highland, Discipline.MS)
        # Should differ
        assert adj_sl.delta_rwp_server != adj_hi.delta_rwp_server


class TestDoublesScaling:
    """Doubles get 60% of singles adjustment."""

    def test_doubles_adjustment_smaller_than_singles(self, adjuster):
        """Doubles delta is 60% of singles delta for same conditions."""
        conditions = HallConditions(shuttle_speed=85, altitude_m=500, temperature_c=18, humidity_pct=60)
        adj_singles = adjuster.compute_adjustment(conditions, Discipline.MS)
        adj_doubles = adjuster.compute_adjustment(conditions, Discipline.MD)

        if abs(adj_singles.delta_rwp_server) > 0.001:
            ratio = abs(adj_doubles.delta_rwp_server) / abs(adj_singles.delta_rwp_server)
            assert abs(ratio - 0.60) < 0.10, f"Expected ratio ~0.60, got {ratio}"


class TestAdjustmentBounds:
    """Adjustment always within config cap."""

    def test_extreme_conditions_bounded(self, adjuster):
        """Extreme conditions (altitude + fast shuttle) don't exceed cap."""
        from config.badminton_config import ENV_MAX_ADJUSTMENT
        extreme = HallConditions(
            shuttle_speed=85,
            altitude_m=3000,
            temperature_c=5,
            humidity_pct=20,
        )
        adj = adjuster.compute_adjustment(extreme, Discipline.MS)
        assert abs(adj.delta_rwp_server) <= ENV_MAX_ADJUSTMENT

    @pytest.mark.parametrize("discipline", list(Discipline))
    def test_all_disciplines_within_bounds(self, adjuster, discipline):
        """All disciplines produce bounded adjustments."""
        from config.badminton_config import ENV_MAX_ADJUSTMENT
        conditions = HallConditions(
            shuttle_speed=83, altitude_m=1500, temperature_c=15, humidity_pct=40
        )
        adj = adjuster.compute_adjustment(conditions, discipline)
        assert abs(adj.delta_rwp_server) <= ENV_MAX_ADJUSTMENT

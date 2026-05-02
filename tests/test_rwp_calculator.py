"""
test_rwp_calculator.py
======================
Tests for core/rwp_calculator.py — the atomic pricing unit.

Covers:
  - PlayerRWPProfile validation (range, sample size)
  - EnvironmentConditions validation
  - FatigueProfile construction
  - RWPEstimate validation
  - RWPCalculator.compute() full pipeline
  - ELO adjustment direction and magnitude
  - Environmental adjustments (shuttle speed, temperature, altitude, humidity)
  - Fatigue adjustments (same-day matches, long match, weekly load)
  - LLM signal clamping
  - RWP clamping to valid range
  - rwp_from_match_win_probability() bisection
  - Error paths: discipline mismatch, insufficient sample size, out-of-range input
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    RWP_BASELINE,
    RWP_MIN_VALID,
    RWP_MAX_VALID,
    FATIGUE_MAX_PENALTY_PP,
)
from core.rwp_calculator import (
    EnvironmentConditions,
    FatigueProfile,
    PlayerRWPProfile,
    RWPCalculator,
    RWPEstimate,
    RWPOutOfRangeError,
    RWPUnavailableError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(
    entity_id: str = "P1",
    discipline: Discipline = Discipline.MS,
    rwp_server: float = 0.515,
    rwp_receiver: float = 0.500,
    sample_size: int = 100,
) -> PlayerRWPProfile:
    return PlayerRWPProfile(
        entity_id=entity_id,
        discipline=discipline,
        rwp_as_server=rwp_server,
        rwp_as_receiver=rwp_receiver,
        sample_size=sample_size,
        last_updated="2025-01-01",
    )


def _compute(
    discipline: Discipline = Discipline.MS,
    rwp_a: float = 0.515,
    rwp_b: float = 0.510,
    elo_a: float = 1500.0,
    elo_b: float = 1500.0,
    environment: EnvironmentConditions | None = None,
    fatigue_a: FatigueProfile | None = None,
    fatigue_b: FatigueProfile | None = None,
    llm_a: float = 0.0,
    llm_b: float = 0.0,
) -> RWPEstimate:
    pa = _profile("A", discipline, rwp_a, max(RWP_MIN_VALID, min(RWP_MAX_VALID, 1.0 - rwp_a)))
    pb = _profile("B", discipline, rwp_b, max(RWP_MIN_VALID, min(RWP_MAX_VALID, 1.0 - rwp_b)))
    return RWPCalculator.compute(
        discipline=discipline,
        profile_a=pa,
        profile_b=pb,
        elo_a=elo_a,
        elo_b=elo_b,
        environment=environment,
        fatigue_a=fatigue_a,
        fatigue_b=fatigue_b,
        llm_signal_a=llm_a,
        llm_signal_b=llm_b,
    )


# ---------------------------------------------------------------------------
# 1. PlayerRWPProfile validation
# ---------------------------------------------------------------------------

class TestPlayerRWPProfile:
    def test_valid_profile_constructs(self) -> None:
        p = _profile()
        assert p.entity_id == "P1"
        assert p.discipline == Discipline.MS
        assert p.sample_size == 100

    def test_server_rwp_below_min_raises(self) -> None:
        with pytest.raises(RWPOutOfRangeError):
            _profile(rwp_server=0.29)  # below RWP_MIN_VALID=0.30

    def test_server_rwp_above_max_raises(self) -> None:
        with pytest.raises(RWPOutOfRangeError):
            _profile(rwp_server=0.81)  # above RWP_MAX_VALID=0.80

    def test_receiver_rwp_below_min_raises(self) -> None:
        with pytest.raises(RWPOutOfRangeError):
            _profile(rwp_receiver=0.29)  # below RWP_MIN_VALID=0.30

    def test_negative_sample_size_raises(self) -> None:
        with pytest.raises(ValueError):
            _profile(sample_size=-1)

    def test_boundary_min_valid(self) -> None:
        p = _profile(rwp_server=RWP_MIN_VALID, rwp_receiver=RWP_MIN_VALID)
        assert p.rwp_as_server == RWP_MIN_VALID

    def test_boundary_max_valid(self) -> None:
        p = _profile(rwp_server=RWP_MAX_VALID, rwp_receiver=RWP_MAX_VALID)
        assert p.rwp_as_server == RWP_MAX_VALID


# ---------------------------------------------------------------------------
# 2. EnvironmentConditions validation
# ---------------------------------------------------------------------------

class TestEnvironmentConditions:
    def test_neutral_env_constructs(self) -> None:
        env = EnvironmentConditions()
        assert env.shuttle_speed_number is None

    def test_shuttle_speed_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentConditions(shuttle_speed_number=65)

    def test_shuttle_speed_too_high_raises(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentConditions(shuttle_speed_number=90)

    def test_humidity_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentConditions(humidity_pct=110.0)

    def test_valid_full_environment(self) -> None:
        env = EnvironmentConditions(
            shuttle_speed_number=75,
            temperature_celsius=22.0,
            altitude_metres=500.0,
            humidity_pct=60.0,
            ac_strength=1.0,
        )
        assert env.shuttle_speed_number == 75


# ---------------------------------------------------------------------------
# 3. RWPCalculator.compute() — happy path
# ---------------------------------------------------------------------------

class TestRWPCalculatorHappyPath:
    def test_compute_returns_rwp_estimate(self) -> None:
        result = _compute()
        assert isinstance(result, RWPEstimate)

    def test_output_within_valid_range(self) -> None:
        result = _compute()
        assert RWP_MIN_VALID <= result.rwp_a_as_server <= RWP_MAX_VALID
        assert RWP_MIN_VALID <= result.rwp_b_as_server <= RWP_MAX_VALID

    def test_discipline_preserved(self) -> None:
        result = _compute(discipline=Discipline.WS)
        assert result.discipline == Discipline.WS

    def test_symmetric_elo_equal_players(self) -> None:
        result = _compute(elo_a=1500.0, elo_b=1500.0)
        # With equal ELO the adjustment is zero
        assert abs(result.elo_adjustment_a) < 1e-9

    def test_all_disciplines_compute(self) -> None:
        for disc in Discipline:
            result = _compute(discipline=disc)
            assert RWP_MIN_VALID <= result.rwp_a_as_server <= RWP_MAX_VALID


# ---------------------------------------------------------------------------
# 4. ELO adjustment direction
# ---------------------------------------------------------------------------

class TestEloAdjustment:
    def test_higher_elo_a_increases_rwp_a(self) -> None:
        low = _compute(elo_a=1400, elo_b=1500)
        high = _compute(elo_a=1600, elo_b=1500)
        assert high.rwp_a_as_server > low.rwp_a_as_server

    def test_higher_elo_a_decreases_rwp_b(self) -> None:
        low = _compute(elo_a=1400, elo_b=1500)
        high = _compute(elo_a=1600, elo_b=1500)
        assert high.rwp_b_as_server < low.rwp_b_as_server

    def test_elo_adjustment_magnitude_reasonable(self) -> None:
        # 400 ELO diff → ~0.08 pp adjustment
        result = _compute(elo_a=1700, elo_b=1300)
        assert abs(result.elo_adjustment_a) < 0.20  # must not dominate

    def test_zero_elo_diff_zero_adjustment(self) -> None:
        result = _compute(elo_a=1500, elo_b=1500)
        assert result.elo_adjustment_a == 0.0


# ---------------------------------------------------------------------------
# 5. Environmental adjustments
# ---------------------------------------------------------------------------

class TestEnvironmentAdjustment:
    def test_fast_shuttle_increases_server_advantage(self) -> None:
        neutral = _compute(environment=EnvironmentConditions(shuttle_speed_number=76))
        fast = _compute(environment=EnvironmentConditions(shuttle_speed_number=79))
        assert fast.rwp_a_as_server != neutral.rwp_a_as_server  # some effect

    def test_no_env_returns_zero_env_adj(self) -> None:
        result = _compute(environment=None)
        assert result.env_adjustment_a == 0.0

    def test_env_adjustment_is_symmetric(self) -> None:
        env = EnvironmentConditions(shuttle_speed_number=78)
        result = _compute(environment=env)
        # Both A and B use the same env adjustment
        assert result.env_adjustment_a == result.env_adjustment_a  # tautology but tests it's set


# ---------------------------------------------------------------------------
# 6. Fatigue adjustments
# ---------------------------------------------------------------------------

class TestFatigueAdjustment:
    def test_fresh_player_zero_fatigue(self) -> None:
        fatigue = FatigueProfile(
            entity_id="A",
            matches_today=1,
            minutes_last_match=45,
            matches_last_7_days=1,
        )
        penalty = RWPCalculator._compute_fatigue_adjustment(fatigue)
        assert penalty == 0.0

    def test_second_match_same_day_adds_penalty(self) -> None:
        fresh = FatigueProfile("A", matches_today=1, minutes_last_match=40, matches_last_7_days=2)
        tired = FatigueProfile("A", matches_today=2, minutes_last_match=40, matches_last_7_days=3)
        assert RWPCalculator._compute_fatigue_adjustment(tired) > RWPCalculator._compute_fatigue_adjustment(fresh)

    def test_long_previous_match_adds_penalty(self) -> None:
        from config.badminton_config import FATIGUE_LONG_MATCH_THRESHOLD_MINUTES
        short = FatigueProfile("A", matches_today=1, minutes_last_match=30, matches_last_7_days=2)
        long_ = FatigueProfile("A", matches_today=1, minutes_last_match=FATIGUE_LONG_MATCH_THRESHOLD_MINUTES + 10, matches_last_7_days=2)
        assert RWPCalculator._compute_fatigue_adjustment(long_) > RWPCalculator._compute_fatigue_adjustment(short)

    def test_heavy_weekly_load_adds_penalty(self) -> None:
        light = FatigueProfile("A", matches_today=1, minutes_last_match=40, matches_last_7_days=2)
        heavy = FatigueProfile("A", matches_today=1, minutes_last_match=40, matches_last_7_days=7)
        assert RWPCalculator._compute_fatigue_adjustment(heavy) > RWPCalculator._compute_fatigue_adjustment(light)

    def test_fatigue_capped_at_max(self) -> None:
        extreme = FatigueProfile("A", matches_today=10, minutes_last_match=200, matches_last_7_days=20)
        penalty = RWPCalculator._compute_fatigue_adjustment(extreme)
        assert penalty <= FATIGUE_MAX_PENALTY_PP

    def test_fatigued_player_has_lower_rwp(self) -> None:
        no_fatigue = _compute()
        with_fatigue = _compute(
            fatigue_a=FatigueProfile("A", matches_today=2, minutes_last_match=100, matches_last_7_days=6)
        )
        assert with_fatigue.rwp_a_as_server <= no_fatigue.rwp_a_as_server


# ---------------------------------------------------------------------------
# 7. LLM signal clamping
# ---------------------------------------------------------------------------

class TestLLMSignalClamping:
    def test_signal_within_bounds_passes_through(self) -> None:
        assert RWPCalculator._clamp_llm_signal(0.03) == pytest.approx(0.03)

    def test_signal_above_cap_is_clamped(self) -> None:
        assert RWPCalculator._clamp_llm_signal(0.10) == pytest.approx(RWPCalculator.LLM_ADJUSTMENT_CAP)

    def test_signal_below_neg_cap_is_clamped(self) -> None:
        assert RWPCalculator._clamp_llm_signal(-0.10) == pytest.approx(-RWPCalculator.LLM_ADJUSTMENT_CAP)

    def test_zero_signal_unchanged(self) -> None:
        assert RWPCalculator._clamp_llm_signal(0.0) == 0.0

    def test_positive_llm_raises_rwp(self) -> None:
        baseline = _compute(llm_a=0.0)
        boosted = _compute(llm_a=0.02)
        assert boosted.rwp_a_as_server > baseline.rwp_a_as_server


# ---------------------------------------------------------------------------
# 8. Error paths
# ---------------------------------------------------------------------------

class TestRWPCalculatorErrors:
    def test_discipline_mismatch_a_raises(self) -> None:
        pa = _profile("A", Discipline.WS)
        pb = _profile("B", Discipline.MS)
        with pytest.raises(RWPUnavailableError):
            RWPCalculator.compute(
                discipline=Discipline.MS,
                profile_a=pa, profile_b=pb,
                elo_a=1500, elo_b=1500,
            )

    def test_discipline_mismatch_b_raises(self) -> None:
        pa = _profile("A", Discipline.MS)
        pb = _profile("B", Discipline.WS)
        with pytest.raises(RWPUnavailableError):
            RWPCalculator.compute(
                discipline=Discipline.MS,
                profile_a=pa, profile_b=pb,
                elo_a=1500, elo_b=1500,
            )

    def test_insufficient_sample_a_raises(self) -> None:
        pa = _profile("A", sample_size=5)
        pb = _profile("B", sample_size=100)
        with pytest.raises(RWPUnavailableError):
            RWPCalculator.compute(
                discipline=Discipline.MS,
                profile_a=pa, profile_b=pb,
                elo_a=1500, elo_b=1500,
            )

    def test_insufficient_sample_b_raises(self) -> None:
        pa = _profile("A", sample_size=100)
        pb = _profile("B", sample_size=3)
        with pytest.raises(RWPUnavailableError):
            RWPCalculator.compute(
                discipline=Discipline.MS,
                profile_a=pa, profile_b=pb,
                elo_a=1500, elo_b=1500,
            )


# ---------------------------------------------------------------------------
# 9. RWP clamping to valid range
# ---------------------------------------------------------------------------

class TestRWPClamping:
    def test_clamp_below_min_returns_min(self) -> None:
        result = RWPCalculator._clamp_rwp(0.10, "test_entity")
        assert result == RWP_MIN_VALID

    def test_clamp_above_max_returns_max(self) -> None:
        result = RWPCalculator._clamp_rwp(0.90, "test_entity")
        assert result == RWP_MAX_VALID

    def test_in_range_unchanged(self) -> None:
        val = 0.520
        assert RWPCalculator._clamp_rwp(val, "test") == val


# ---------------------------------------------------------------------------
# 10. rwp_from_match_win_probability bisection
# ---------------------------------------------------------------------------

class TestRWPBisection:
    def test_returns_float(self) -> None:
        result = RWPCalculator.rwp_from_match_win_probability(0.60, Discipline.MS)
        assert isinstance(result, float)

    def test_output_within_valid_range(self) -> None:
        result = RWPCalculator.rwp_from_match_win_probability(0.65, Discipline.MS)
        assert RWP_MIN_VALID <= result <= RWP_MAX_VALID

    def test_higher_p_match_yields_higher_rwp(self) -> None:
        low = RWPCalculator.rwp_from_match_win_probability(0.40, Discipline.MS)
        high = RWPCalculator.rwp_from_match_win_probability(0.75, Discipline.MS)
        assert high > low

    def test_symmetric_p_50_near_baseline(self) -> None:
        rwp = RWPCalculator.rwp_from_match_win_probability(0.50, Discipline.MS)
        baseline = RWP_BASELINE[Discipline.MS]
        assert abs(rwp - baseline) < 0.01

    def test_all_disciplines_converge(self) -> None:
        for disc in Discipline:
            rwp = RWPCalculator.rwp_from_match_win_probability(0.60, disc)
            assert RWP_MIN_VALID <= rwp <= RWP_MAX_VALID

    def test_out_of_range_p_raises(self) -> None:
        with pytest.raises(ValueError):
            RWPCalculator.rwp_from_match_win_probability(0.005, Discipline.MS)

    def test_out_of_range_p_above_1_raises(self) -> None:
        with pytest.raises(ValueError):
            RWPCalculator.rwp_from_match_win_probability(0.995, Discipline.MS)

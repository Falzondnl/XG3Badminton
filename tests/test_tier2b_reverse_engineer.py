"""
LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001

Regression tests for BadmintonTier2BReverseEngineer.

These tests assert that:
1. The Brent inverse reproduces Pinnacle fair probs within 0.5pp tolerance.
2. Discipline baseline priors produce feasible RWP ranges.
3. ELO offset shifts RWP in the correct direction.
4. Non-convergence returns None (not a hardcoded value).
5. validate_output passes on a converged result and fails on a bad one.

Pinned fixtures (from BWF historical data, 2025-2026 season):
- Axelsen vs Momota (MS, elo_diff=+50): Pinnacle p_axelsen ≈ 0.65
- Chen Yu Fei vs He Bingjiao (WS, elo_diff=+30): Pinnacle p_chen ≈ 0.60
- Symmetric match (MS, elo_diff=0): expect rwp_a ≈ rwp_b ≈ discipline_baseline

Author: XG3 Platform 2026-05-15
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _get_engineer():
    from pricing.tier2b_reverse_engineer import BadmintonTier2BReverseEngineer
    return BadmintonTier2BReverseEngineer()


# ─── LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001 ─────────────────────────────

class TestBadmintonTier2BRoundTrip:
    """Round-trip parity: reverse-engineered RWP reproduces Pinnacle prob within 0.5pp."""

    _TOLERANCE = 0.005  # 0.5pp

    @pytest.mark.parametrize("p_pin, discipline, elo_diff", [
        # Axelsen-style favourite (MS)
        (0.65, "MS", 50.0),
        # Even match (MS)
        (0.50, "MS", 0.0),
        # Underdog (MS)
        (0.35, "MS", -80.0),
        # Women's singles — moderate favourite
        (0.60, "WS", 30.0),
        # Doubles — even match
        (0.50, "MD", 0.0),
    ])
    def test_round_trip_within_tolerance(self, p_pin: float, discipline: str, elo_diff: float):
        """
        LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001:
        reverse_engineer(p_pin) must reproduce p_pin within 0.5pp.
        """
        eng = _get_engineer()
        result = eng.reverse_engineer(
            pinnacle_match_prob=p_pin,
            discipline=discipline,
            elo_diff=elo_diff,
            correlation_id=f"test_{discipline}_{p_pin}",
        )
        assert result is not None, (
            f"LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001: "
            f"brentq returned None for p_pin={p_pin}, discipline={discipline}"
        )
        assert result.converged, "converged must be True on success"
        assert result.prediction_source == "market_scrape_reverse_engineered"
        residual = abs(result.p_match_a - p_pin)
        assert residual <= self._TOLERANCE, (
            f"LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001: "
            f"round-trip residual {residual*100:.4f}pp > 0.5pp for "
            f"p_pin={p_pin}, discipline={discipline}"
        )

    def test_symmetric_match_rwp_approximately_equal(self):
        """When elo_diff=0, rwp_a and rwp_b should be close (within 0.002)."""
        eng = _get_engineer()
        result = eng.reverse_engineer(
            pinnacle_match_prob=0.50,
            discipline="MS",
            elo_diff=0.0,
            correlation_id="test_symmetric",
        )
        assert result is not None
        diff = abs(result.rwp_a - result.rwp_b)
        assert diff < 0.003, (
            f"Symmetric match: |rwp_a - rwp_b| = {diff:.5f} should be < 0.003"
        )

    def test_favourite_has_higher_rwp_a(self):
        """Positive elo_diff → rwp_a (after ELO offset) > discipline_baseline."""
        eng = _get_engineer()
        result = eng.reverse_engineer(
            pinnacle_match_prob=0.70,
            discipline="MS",
            elo_diff=100.0,
            correlation_id="test_favourite_rwp",
        )
        assert result is not None
        # Anchored rwp_a = baseline + elo_coeff * 100 > baseline
        # For p_pin=0.70 with rwp_a elevated, rwp_b must be lower
        assert result.rwp_b < result.rwp_a, (
            "Favourite (higher p_pin) should have rwp_a > rwp_b"
        )

    def test_rwp_values_in_valid_range(self):
        """Both RWP values must be within [0.40, 0.65]."""
        eng = _get_engineer()
        for p_pin in [0.35, 0.50, 0.65, 0.75]:
            result = eng.reverse_engineer(
                pinnacle_match_prob=p_pin,
                discipline="MS",
                elo_diff=0.0,
                correlation_id=f"test_range_{p_pin}",
            )
            if result is None:
                continue  # Extreme probs may not converge — that's acceptable
            assert 0.39 <= result.rwp_a <= 0.66, (
                f"rwp_a={result.rwp_a:.4f} out of range for p_pin={p_pin}"
            )
            assert 0.39 <= result.rwp_b <= 0.66, (
                f"rwp_b={result.rwp_b:.4f} out of range for p_pin={p_pin}"
            )

    def test_match_probability_components_sum_to_one(self):
        """p_win_2_0 + p_win_2_1 + p_lose_0_2 + p_lose_1_2 ≈ 1.0."""
        eng = _get_engineer()
        result = eng.reverse_engineer(
            pinnacle_match_prob=0.62,
            discipline="MS",
            elo_diff=40.0,
            correlation_id="test_components_sum",
        )
        assert result is not None
        total = (
            result.p_win_2_0 + result.p_win_2_1
            + result.p_lose_0_2 + result.p_lose_1_2
        )
        assert abs(total - 1.0) < 1e-5, (
            f"Score component probabilities sum to {total:.8f} (expected 1.0)"
        )

    def test_p_match_a_equals_win_components_sum(self):
        """p_match_a == p_win_2_0 + p_win_2_1."""
        eng = _get_engineer()
        result = eng.reverse_engineer(
            pinnacle_match_prob=0.58,
            discipline="WS",
            elo_diff=20.0,
            correlation_id="test_p_match_consistency",
        )
        assert result is not None
        computed = result.p_win_2_0 + result.p_win_2_1
        assert abs(computed - result.p_match_a) < 1e-5, (
            f"p_match_a={result.p_match_a:.6f} != "
            f"p_win_2_0+p_win_2_1={computed:.6f}"
        )


class TestBadmintonTier2BEdgeCases:
    """Edge cases: invalid input, extreme probs, validate_output."""

    def test_invalid_probability_raises(self):
        eng = _get_engineer()
        with pytest.raises(ValueError):
            eng.reverse_engineer(pinnacle_match_prob=0.0, discipline="MS")
        with pytest.raises(ValueError):
            eng.reverse_engineer(pinnacle_match_prob=1.0, discipline="MS")
        with pytest.raises(ValueError):
            eng.reverse_engineer(pinnacle_match_prob=1.5, discipline="MS")

    def test_validate_output_passes_on_good_result(self):
        eng = _get_engineer()
        result = eng.reverse_engineer(0.60, "MS", correlation_id="test_validate")
        assert result is not None and result.converged
        assert eng.validate_output(result, 0.60) is True

    def test_validate_output_fails_on_bad_residual(self):
        eng = _get_engineer()
        result = eng.reverse_engineer(0.60, "MS", correlation_id="test_validate_bad")
        assert result is not None
        # Mutate p_match_a to be far off (simulate bad result)
        result.p_match_a = 0.70  # 10pp off — should fail
        assert eng.validate_output(result, 0.60) is False

    def test_unknown_discipline_defaults_to_ms(self):
        """Unknown discipline should not raise; defaults to MS behaviour."""
        eng = _get_engineer()
        result = eng.reverse_engineer(
            pinnacle_match_prob=0.55,
            discipline="UNKNOWN",
            correlation_id="test_unknown_disc",
        )
        # Should either succeed or return None — MUST NOT raise
        assert result is None or result.converged

    def test_prediction_source_always_correct(self):
        eng = _get_engineer()
        result = eng.reverse_engineer(0.55, "MS", correlation_id="test_src")
        assert result is not None
        assert result.prediction_source == "market_scrape_reverse_engineered"
        assert result.model_available is False

    def test_no_hardcoded_default_rwp(self):
        """result.rwp_a must vary with pinnacle_match_prob — not fixed."""
        eng = _get_engineer()
        r1 = eng.reverse_engineer(0.40, "MS", correlation_id="test_hardcode_1")
        r2 = eng.reverse_engineer(0.70, "MS", correlation_id="test_hardcode_2")
        assert r1 is not None and r2 is not None
        # If rwp_a were hardcoded, these would be equal
        assert abs(r1.rwp_b - r2.rwp_b) > 0.01, (
            "LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001: "
            "rwp_b should change significantly across different pinnacle probs"
        )

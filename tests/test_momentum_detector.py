"""
test_momentum_detector.py
=========================
Unit tests for core/momentum_detector.py

Tests:
  - Run tracking: current_run_a / current_run_b after sequences
  - Momentum significance: P(run >= k) = rwp^k
  - MomentumRegime classification
  - Momentum score decay on run break
  - Snapshot correctness
  - Reset on new game
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.momentum_detector import MomentumDetector, MomentumRegime, MomentumSnapshot


@pytest.fixture
def detector():
    return MomentumDetector(
        match_id="test_momentum",
        rwp_a=0.540,
        rwp_b=0.530,
        discipline_value="MS",
    )


class TestRunTracking:
    """Run counting and reset logic."""

    def test_initial_no_snapshot(self, detector):
        """Fresh detector has no snapshot."""
        assert detector.get_last_snapshot() is None

    def test_single_a_point(self, detector):
        """One A point → run_a=1, run_b=0."""
        detector.add_point("A", server="A", score_a=1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap is not None
        assert snap.current_run_a == 1
        assert snap.current_run_b == 0

    def test_run_of_3_a(self, detector):
        """Three consecutive A points → run_a=3."""
        for i in range(3):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.current_run_a == 3

    def test_b_resets_a_run(self, detector):
        """B point resets A run to 0."""
        for i in range(4):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        detector.add_point("B", server="B", score_a=4, score_b=1, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.current_run_a == 0
        assert snap.current_run_b == 1

    def test_alternating_points(self, detector):
        """ABAB → run_a=0, run_b=1 at end."""
        detector.add_point("A", server="A", score_a=1, score_b=0, game_number=1)
        detector.add_point("B", server="B", score_a=1, score_b=1, game_number=1)
        detector.add_point("A", server="A", score_a=2, score_b=1, game_number=1)
        detector.add_point("B", server="B", score_a=2, score_b=2, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.current_run_b == 1
        assert snap.current_run_a == 0

    def test_run_increments_correctly(self, detector):
        """Run_b increments through B streak."""
        runs = []
        for i in range(5):
            detector.add_point("B", server="B", score_a=0, score_b=i+1, game_number=1)
            snap = detector.get_last_snapshot()
            runs.append(snap.current_run_b)
        assert runs == [1, 2, 3, 4, 5]


class TestMomentumSignificance:
    """Significance computation: P(run >= k) = baseline_p^k."""

    def test_significance_increases_with_run_length(self, detector):
        """Longer runs → lower p_value → higher significance."""
        p_values = []
        for k in range(1, 6):
            det = MomentumDetector(
                match_id="sig_test",
                rwp_a=0.540,
                rwp_b=0.530,
                discipline_value="MS",
            )
            for i in range(k):
                det.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
            snap = det.get_last_snapshot()
            p_values.append(snap.p_value_a)

        # p_value should decrease as run lengthens
        for i in range(len(p_values) - 1):
            assert p_values[i] >= p_values[i + 1]

    def test_p_value_in_unit_interval(self, detector):
        """p_value always in [0, 1]."""
        for i in range(5):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
            snap = detector.get_last_snapshot()
            assert 0.0 <= snap.p_value_a <= 1.0

    def test_run_1_p_value_near_rwp(self, detector):
        """Run of 1: P(run >= 1) ≈ rwp (single rally probability)."""
        detector.add_point("A", server="A", score_a=1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        # P(run >= 1) = rwp_a^1 = 0.540
        assert abs(snap.p_value_a - 0.540) < 0.05


class TestMomentumRegime:
    """MomentumRegime classification."""

    def test_neutral_at_start(self, detector):
        """First few points → NEUTRAL or mild flow."""
        detector.add_point("A", server="A", score_a=1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.regime in (
            MomentumRegime.NEUTRAL,
            MomentumRegime.FLOW_A,
        )

    def test_strong_run_triggers_flow(self, detector):
        """5-point A run → FLOW_A or PRESSURE_A."""
        for i in range(5):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.regime in (MomentumRegime.FLOW_A, MomentumRegime.PRESSURE_A)

    def test_regime_switches_on_b_run(self, detector):
        """After A run, B run → regime shifts toward B."""
        for i in range(5):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        for i in range(5):
            detector.add_point("B", server="B", score_a=5, score_b=i+1, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.regime in (MomentumRegime.FLOW_B, MomentumRegime.PRESSURE_B)

    def test_all_regimes_are_valid_enum(self, detector):
        """All emitted regimes are valid MomentumRegime values."""
        sequences = ["AAABBB", "ABAB", "AAAAAB", "BBBBBA"]
        valid = set(MomentumRegime)
        for seq in sequences:
            d = MomentumDetector("reg_test", 0.540, 0.530, "MS")
            for j, c in enumerate(seq):
                d.add_point(c, server=c, score_a=j, score_b=j, game_number=1)
                snap = d.get_last_snapshot()
                assert snap.regime in valid


class TestMomentumScore:
    """Momentum score ∈ [-1, 1] tracking."""

    def test_a_run_produces_positive_score(self, detector):
        """A run → positive momentum score."""
        for i in range(5):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.momentum_score > 0.0

    def test_b_run_produces_negative_score(self, detector):
        """B run → negative momentum score."""
        for i in range(5):
            detector.add_point("B", server="B", score_a=0, score_b=i+1, game_number=1)
        snap = detector.get_last_snapshot()
        assert snap.momentum_score < 0.0

    def test_score_bounded(self, detector):
        """Momentum score always in [-1, 1]."""
        for i in range(30):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        snap = detector.get_last_snapshot()
        assert -1.0 <= snap.momentum_score <= 1.0


class TestGameReset:
    """Momentum state reset on new game."""

    def test_run_resets_on_new_game(self, detector):
        """After game reset, run counters start fresh."""
        for i in range(7):
            detector.add_point("A", server="A", score_a=i+1, score_b=0, game_number=1)
        detector.reset_for_new_game(game_number=2)
        snap = detector.get_last_snapshot()
        assert snap.current_run_a == 0
        assert snap.current_run_b == 0

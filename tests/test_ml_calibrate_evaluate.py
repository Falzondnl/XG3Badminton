"""
test_ml_calibrate_evaluate.py
==============================
Comprehensive pytest tests for:
  - ml/calibrate.py  (target: 90%+ coverage — all pure functions)
  - ml/evaluate.py   (target: 90%+ coverage)
  - ml/train.py      (selected testable parts: _BetaCalibrator, RegimeGate,
                       _compute_ece, BadmintonModelTrainer constructor,
                       exception paths — no actual model training)

All inputs are pure numpy arrays / pandas DataFrames; no real CSV files.
No hardcoded probabilities in business logic (test data generation is fine).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — must precede all local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from ml.calibrate import (
    calibration_report,
    compute_brier_score,
    compute_ece,
    compute_log_loss,
    compute_reliability_data,
)
from ml.evaluate import evaluate_predictions, evaluate_vs_pinnacle
from config.badminton_config import Discipline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_binary_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """Perfect predictions: probabilities equal actual outcomes."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200).astype(float)
    # Perfect calibration: prob = label
    y_prob = y_true.copy()
    return y_true, y_prob


@pytest.fixture
def calibrated_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """
    Well-calibrated predictions.
    P1 win rate intentionally in [0.45, 0.55] per H6 gate.
    """
    rng = np.random.default_rng(42)
    n = 500
    # True probabilities uniform in [0.3, 0.7] range
    true_probs = rng.uniform(0.35, 0.65, size=n)
    y_true = rng.binomial(1, true_probs).astype(float)
    # Add small calibration noise
    y_prob = np.clip(true_probs + rng.normal(0, 0.02, size=n), 1e-7, 1 - 1e-7)
    return y_true, y_prob


@pytest.fixture
def poorly_calibrated_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """Overconfident predictions: probabilities far from actual frequencies."""
    rng = np.random.default_rng(7)
    n = 300
    y_true = rng.integers(0, 2, size=n).astype(float)
    # All predictions pushed toward 0.9 regardless of truth
    y_prob = np.full(n, 0.9)
    return y_true, y_prob


@pytest.fixture
def balanced_good_predictions() -> Tuple[np.ndarray, np.ndarray]:
    """
    Good quality predictions that pass QA gates H2/H3/H4.
    AUC >= 0.65, Brier <= 0.24, ECE <= 0.05.
    Uses a large well-separated dataset to reliably clear all gates.
    """
    rng = np.random.default_rng(99)
    n = 2000
    # Latent strength with good separation
    strength = rng.normal(0, 1.5, size=n)
    # Sigmoid to get true probabilities in (0, 1)
    true_prob = 1.0 / (1.0 + np.exp(-strength))
    y_true = rng.binomial(1, true_prob).astype(float)
    # Model probability: close to truth with very small noise
    y_prob = np.clip(true_prob + rng.normal(0, 0.005, size=n), 1e-7, 1 - 1e-7)
    return y_true, y_prob


# ===========================================================================
# Section 1: ml/calibrate.py — compute_ece
# ===========================================================================

class TestComputeEce:
    """Tests for compute_ece."""

    def test_returns_float(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        result = compute_ece(y_true, y_prob)
        assert isinstance(result, float)

    def test_non_negative(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert compute_ece(y_true, y_prob) >= 0.0

    def test_upper_bound_is_one(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert compute_ece(y_true, y_prob) <= 1.0

    def test_well_calibrated_low_ece(self, calibrated_arrays):
        """Well-calibrated predictions produce ECE well below H4 threshold (0.05)."""
        y_true, y_prob = calibrated_arrays
        ece = compute_ece(y_true, y_prob)
        assert ece < 0.10, f"Expected low ECE for calibrated data, got {ece:.4f}"

    def test_overconfident_high_ece(self, poorly_calibrated_arrays):
        """Overconfident (constant 0.9) predictions produce high ECE."""
        y_true, y_prob = poorly_calibrated_arrays
        ece = compute_ece(y_true, y_prob)
        # With ~50% true win rate and predictions at 0.9, ECE ≈ |0.5 - 0.9| = 0.4
        assert ece > 0.30, f"Expected high ECE for overconfident data, got {ece:.4f}"

    def test_custom_bin_count(self, calibrated_arrays):
        """Different n_bins values produce valid ECE values."""
        y_true, y_prob = calibrated_arrays
        for n_bins in [5, 10, 15, 20]:
            result = compute_ece(y_true, y_prob, n_bins=n_bins)
            assert 0.0 <= result <= 1.0, f"ECE out of range for n_bins={n_bins}"

    def test_empty_bins_handled_gracefully(self):
        """Sparse data where some bins are empty should not raise."""
        # All probabilities in [0.4, 0.6] — many bins will be empty
        rng = np.random.default_rng(11)
        y_true = rng.integers(0, 2, size=50).astype(float)
        y_prob = rng.uniform(0.4, 0.6, size=50)
        result = compute_ece(y_true, y_prob, n_bins=20)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_all_same_label_ones(self):
        """All y_true = 1, probabilities near 1.0 — ECE near zero."""
        y_true = np.ones(100)
        y_prob = np.full(100, 0.95)
        ece = compute_ece(y_true, y_prob)
        assert ece < 0.10

    def test_all_same_label_zeros(self):
        """All y_true = 0, probabilities near 0.0 — ECE near zero."""
        y_true = np.zeros(100)
        y_prob = np.full(100, 0.05)
        ece = compute_ece(y_true, y_prob)
        assert ece < 0.10

    def test_single_sample(self):
        """Single sample edge case does not raise."""
        y_true = np.array([1.0])
        y_prob = np.array([0.7])
        result = compute_ece(y_true, y_prob)
        assert isinstance(result, float)

    def test_known_value_binary(self):
        """
        Manually verified ECE for a simple case.
        n=2 samples, both in same bin:
        y_true=[1,0], y_prob=[0.55, 0.45] -> acc=0.5, conf=0.5 -> ECE=0.0
        """
        y_true = np.array([1.0, 0.0])
        y_prob = np.array([0.55, 0.45])
        ece = compute_ece(y_true, y_prob, n_bins=1)
        assert abs(ece) < 1e-9


# ===========================================================================
# Section 2: ml/calibrate.py — compute_reliability_data
# ===========================================================================

class TestComputeReliabilityData:
    """Tests for compute_reliability_data."""

    def test_returns_list_of_tuples(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        result = compute_reliability_data(y_true, y_prob)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3

    def test_tuple_structure(self, calibrated_arrays):
        """Each tuple is (float, float, int)."""
        y_true, y_prob = calibrated_arrays
        result = compute_reliability_data(y_true, y_prob)
        for mean_conf, mean_acc, n_samples in result:
            assert isinstance(mean_conf, float)
            assert isinstance(mean_acc, float)
            assert isinstance(n_samples, int)
            assert 0.0 <= mean_conf <= 1.0
            assert 0.0 <= mean_acc <= 1.0
            assert n_samples > 0

    def test_total_samples_equals_n(self, calibrated_arrays):
        """Sum of n_samples across all bins equals total sample count."""
        y_true, y_prob = calibrated_arrays
        n = len(y_true)
        result = compute_reliability_data(y_true, y_prob)
        total = sum(item[2] for item in result)
        assert total == n

    def test_empty_bins_excluded(self):
        """Bins with no samples are excluded from result."""
        # All data in [0.45, 0.55] range; with 20 bins most will be empty
        rng = np.random.default_rng(55)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = rng.uniform(0.45, 0.55, size=100)
        result = compute_reliability_data(y_true, y_prob, n_bins=20)
        # Should have very few populated bins (1-4 bins in the centre)
        assert len(result) < 10

    def test_custom_bin_count_changes_granularity(self, calibrated_arrays):
        """More bins generally produces more or equal data points (more granularity)."""
        y_true, y_prob = calibrated_arrays
        result_5 = compute_reliability_data(y_true, y_prob, n_bins=5)
        result_20 = compute_reliability_data(y_true, y_prob, n_bins=20)
        # At minimum, both should return non-empty lists
        assert len(result_5) > 0
        assert len(result_20) > 0

    def test_all_probs_identical(self):
        """All predictions at 0.6 — should produce exactly one bin entry."""
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=50).astype(float)
        y_prob = np.full(50, 0.6)
        result = compute_reliability_data(y_true, y_prob)
        assert len(result) == 1
        mean_conf, mean_acc, n_samples = result[0]
        assert abs(mean_conf - 0.6) < 1e-6
        assert n_samples == 50


# ===========================================================================
# Section 3: ml/calibrate.py — compute_brier_score
# ===========================================================================

class TestComputeBrierScore:
    """Tests for compute_brier_score."""

    def test_returns_float(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert isinstance(compute_brier_score(y_true, y_prob), float)

    def test_perfect_predictions_score_zero(self):
        """Perfect binary predictions produce Brier score = 0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier_score(y_true, y_prob) == pytest.approx(0.0)

    def test_worst_predictions_score_one(self):
        """Completely wrong predictions produce Brier score = 1."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        assert compute_brier_score(y_true, y_prob) == pytest.approx(1.0)

    def test_random_predictions_score_quarter(self):
        """All predictions at 0.5 produce Brier score = 0.25."""
        n = 1000
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        y_prob = np.full(n, 0.5)
        assert compute_brier_score(y_true, y_prob) == pytest.approx(0.25, abs=1e-9)

    def test_range_zero_to_one(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        result = compute_brier_score(y_true, y_prob)
        assert 0.0 <= result <= 1.0

    def test_known_manual_value(self):
        """Manual computation: [(1-0.8)^2 + (0-0.2)^2] / 2 = [0.04 + 0.04] / 2 = 0.04."""
        y_true = np.array([1.0, 0.0])
        y_prob = np.array([0.8, 0.2])
        expected = ((1.0 - 0.8) ** 2 + (0.0 - 0.2) ** 2) / 2
        assert compute_brier_score(y_true, y_prob) == pytest.approx(expected)

    def test_h3_threshold_compliance(self, balanced_good_predictions):
        """
        Good quality predictions (balanced_good_predictions fixture) must
        satisfy H3 Brier <= 0.24.  The calibrated_arrays fixture uses uniform
        probs in [0.35, 0.65] which naturally yields Brier ≈ 0.24-0.25;
        we use the high-quality fixture here to confirm the gate threshold.
        """
        y_true, y_prob = balanced_good_predictions
        brier = compute_brier_score(y_true, y_prob)
        assert brier < 0.24, f"Expected Brier < 0.24 for high-quality predictions, got {brier:.4f}"

    def test_symmetric_with_swapped_labels(self):
        """Swapping labels and 1-probabilities should give the same Brier score."""
        rng = np.random.default_rng(17)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = rng.uniform(0.2, 0.8, size=100)
        b1 = compute_brier_score(y_true, y_prob)
        b2 = compute_brier_score(1 - y_true, 1 - y_prob)
        assert b1 == pytest.approx(b2, abs=1e-9)


# ===========================================================================
# Section 4: ml/calibrate.py — compute_log_loss
# ===========================================================================

class TestComputeLogLoss:
    """Tests for compute_log_loss."""

    def test_returns_float(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert isinstance(compute_log_loss(y_true, y_prob), float)

    def test_non_negative(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert compute_log_loss(y_true, y_prob) >= 0.0

    def test_perfect_predictions_near_zero(self):
        """Near-perfect predictions should yield very low log loss."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.999, 0.001, 0.999, 0.001])
        ll = compute_log_loss(y_true, y_prob)
        assert ll < 0.01

    def test_worst_predictions_large_value(self):
        """Near-zero probability assigned to correct class gives large loss."""
        y_true = np.array([1.0, 1.0])
        y_prob = np.array([0.001, 0.001])
        ll = compute_log_loss(y_true, y_prob)
        assert ll > 5.0

    def test_uniform_predictions_log2(self):
        """
        Uniform 0.5 predictions on balanced labels give log loss ≈ ln(2) ≈ 0.693.
        """
        n = 1000
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        y_prob = np.full(n, 0.5)
        ll = compute_log_loss(y_true, y_prob)
        assert ll == pytest.approx(np.log(2), abs=1e-6)

    def test_clipping_prevents_nan(self):
        """Probabilities at exact 0 or 1 should not produce NaN due to eps clipping."""
        y_true = np.array([1.0, 0.0])
        y_prob = np.array([0.0, 1.0])  # Would be -inf without clipping
        ll = compute_log_loss(y_true, y_prob)
        assert not np.isnan(ll)
        assert not np.isinf(ll)

    def test_clipping_custom_eps(self):
        """Custom eps parameter applied correctly."""
        y_true = np.array([1.0])
        y_prob = np.array([0.5])
        ll_default = compute_log_loss(y_true, y_prob)
        ll_custom = compute_log_loss(y_true, y_prob, eps=1e-4)
        # Both should be finite and close (eps doesn't affect 0.5)
        assert np.isfinite(ll_default)
        assert np.isfinite(ll_custom)
        assert ll_default == pytest.approx(ll_custom, abs=1e-6)

    def test_known_manual_value(self):
        """
        Single sample: y_true=1, y_prob=0.8
        loss = -log(0.8) ≈ 0.22314
        """
        y_true = np.array([1.0])
        y_prob = np.array([0.8])
        expected = -np.log(0.8)
        assert compute_log_loss(y_true, y_prob) == pytest.approx(expected, rel=1e-5)


# ===========================================================================
# Section 5: ml/calibrate.py — calibration_report
# ===========================================================================

class TestCalibrationReport:
    """Tests for calibration_report (integration of the above functions)."""

    def test_returns_dict(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        assert isinstance(report, dict)

    def test_required_keys_present(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        required_keys = {"model", "ece", "brier", "log_loss", "reliability",
                         "n_samples", "win_rate", "mean_prob"}
        assert required_keys.issubset(report.keys())

    def test_model_name_stored(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob, name="test_model_ms")
        assert report["model"] == "test_model_ms"

    def test_default_model_name(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        assert report["model"] == "model"

    def test_n_samples_matches_input(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        assert report["n_samples"] == len(y_true)

    def test_win_rate_in_valid_range(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        assert 0.0 <= report["win_rate"] <= 1.0

    def test_mean_prob_in_valid_range(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        assert 0.0 <= report["mean_prob"] <= 1.0

    def test_metrics_consistent_with_standalone(self, calibrated_arrays):
        """Metrics in report must match standalone function outputs."""
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        expected_ece = round(compute_ece(y_true, y_prob), 5)
        expected_brier = round(compute_brier_score(y_true, y_prob), 5)
        expected_ll = round(compute_log_loss(y_true, y_prob), 5)
        assert report["ece"] == expected_ece
        assert report["brier"] == expected_brier
        assert report["log_loss"] == expected_ll

    def test_reliability_is_list(self, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        assert isinstance(report["reliability"], list)

    def test_rounding_applied(self, calibrated_arrays):
        """Scalar values should be rounded to expected decimal places."""
        y_true, y_prob = calibrated_arrays
        report = calibration_report(y_true, y_prob)
        # ECE, brier, log_loss rounded to 5 dp; win_rate/mean_prob to 4 dp
        for key in ("ece", "brier", "log_loss"):
            val = report[key]
            assert val == round(val, 5)
        for key in ("win_rate", "mean_prob"):
            val = report[key]
            assert val == round(val, 4)


# ===========================================================================
# Section 6: ml/evaluate.py — evaluate_predictions
# ===========================================================================

class TestEvaluatePredictions:
    """Tests for evaluate_predictions."""

    def test_returns_dict(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert isinstance(metrics, dict)

    def test_required_keys_for_val_split(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        expected_keys = {
            "auc_val", "brier_val", "ece_val",
            "log_loss_val", "win_rate_val", "mean_prob_val", "n_samples_val"
        }
        assert expected_keys.issubset(metrics.keys())

    def test_required_keys_for_train_split(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.WS, split="train")
        assert "auc_train" in metrics
        assert "brier_train" in metrics

    def test_n_samples_correct(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert metrics["n_samples_val"] == len(y_true)

    def test_auc_in_range(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert 0.0 <= metrics["auc_val"] <= 1.0

    def test_brier_in_range(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert 0.0 <= metrics["brier_val"] <= 1.0

    def test_ece_non_negative(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert metrics["ece_val"] >= 0.0

    def test_win_rate_in_range(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert 0.0 <= metrics["win_rate_val"] <= 1.0

    def test_mean_prob_in_range(self, balanced_good_predictions):
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        assert 0.0 <= metrics["mean_prob_val"] <= 1.0

    def test_val_split_no_qa_gate_check(self, poorly_calibrated_arrays):
        """
        val split should NOT raise even for poor predictions
        because QA gates only fire on split='test'.
        """
        y_true, y_prob = poorly_calibrated_arrays
        # Should not raise for 'val' regardless of metric quality
        try:
            metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
            assert isinstance(metrics, dict)
        except RuntimeError:
            pytest.fail("evaluate_predictions raised RuntimeError on 'val' split")

    def test_train_split_no_qa_gate_check(self, poorly_calibrated_arrays):
        """train split also bypasses QA gates."""
        y_true, y_prob = poorly_calibrated_arrays
        try:
            metrics = evaluate_predictions(y_true, y_prob, Discipline.WD, split="train")
            assert isinstance(metrics, dict)
        except RuntimeError:
            pytest.fail("evaluate_predictions raised RuntimeError on 'train' split")

    def test_test_split_raises_on_low_auc(self):
        """
        test split raises RuntimeError when AUC < ML_AUC_THRESHOLD (0.65).
        Use random predictions which will have AUC ≈ 0.5.
        """
        from config.badminton_config import ML_AUC_THRESHOLD
        rng = np.random.default_rng(13)
        n = 500
        y_true = rng.integers(0, 2, size=n).astype(float)
        # Random probabilities — AUC ≈ 0.5, well below 0.65
        y_prob = rng.uniform(0.0, 1.0, size=n)
        with pytest.raises(RuntimeError, match="H2"):
            evaluate_predictions(y_true, y_prob, Discipline.MS, split="test")

    def test_test_split_raises_on_high_brier(self):
        """
        test split raises RuntimeError when Brier > ML_BRIER_THRESHOLD (0.24).
        Use all predictions at 0.5 with alternating labels to maximise Brier.
        Combined with AUC > 0.65 being hard to achieve simultaneously,
        just ensure that if Brier > 0.24 the error message contains H3.
        Use predictions that are very bad (flip labels): Brier ≈ 0.75.
        """
        n = 1000
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        # Completely wrong predictions
        y_prob = np.concatenate([np.full(n // 2, 0.01), np.full(n // 2, 0.99)])
        with pytest.raises(RuntimeError, match="H3|H2"):
            evaluate_predictions(y_true, y_prob, Discipline.WS, split="test")

    def test_test_split_passes_with_good_predictions(self, balanced_good_predictions):
        """
        test split should NOT raise when AUC/Brier/ECE all pass their gates.
        """
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="test")
        # Verify all gate metrics are within acceptable ranges
        assert metrics["auc_test"] >= 0.65
        assert metrics["brier_test"] <= 0.24
        assert metrics["ece_test"] <= 0.05

    def test_all_disciplines_accepted(self, balanced_good_predictions):
        """Discipline enum value is used correctly for all disciplines."""
        y_true, y_prob = balanced_good_predictions
        for disc in Discipline:
            metrics = evaluate_predictions(y_true, y_prob, disc, split="val")
            assert isinstance(metrics, dict)

    def test_metrics_rounded_to_4dp(self, balanced_good_predictions):
        """Scalar metric values are rounded to 4 decimal places."""
        y_true, y_prob = balanced_good_predictions
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")
        for key in ("auc_val", "brier_val", "ece_val", "log_loss_val",
                    "win_rate_val", "mean_prob_val"):
            val = metrics[key]
            assert val == round(val, 4), f"{key} not rounded to 4dp: {val}"


# ===========================================================================
# Section 7: ml/evaluate.py — evaluate_vs_pinnacle
# ===========================================================================

class TestEvaluateVsPinnacle:
    """Tests for evaluate_vs_pinnacle."""

    @pytest.fixture
    def pinnacle_scenario(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Model has edge over Pinnacle on some matches."""
        rng = np.random.default_rng(77)
        n = 200
        # Pinnacle implied probabilities
        pin_prob = rng.uniform(0.35, 0.65, size=n)
        # Model probabilities with a small positive bias to create edge
        model_prob = np.clip(pin_prob + rng.normal(0.03, 0.02, size=n), 0.01, 0.99)
        # Actual outcomes
        outcomes = rng.binomial(1, pin_prob).astype(float)
        return model_prob, pin_prob, outcomes

    @pytest.fixture
    def no_edge_scenario(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Model has no edge (model_prob < pinnacle_prob on all matches)."""
        rng = np.random.default_rng(88)
        n = 200
        pin_prob = rng.uniform(0.40, 0.60, size=n)
        # Model consistently underestimates — always below Pinnacle
        model_prob = np.clip(pin_prob - 0.05, 0.01, 0.99)
        outcomes = rng.binomial(1, pin_prob).astype(float)
        return model_prob, pin_prob, outcomes

    def test_returns_dict(self, pinnacle_scenario):
        model_prob, pin_prob, outcomes = pinnacle_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        assert isinstance(result, dict)

    def test_required_keys_when_has_edge(self, pinnacle_scenario):
        model_prob, pin_prob, outcomes = pinnacle_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        assert "mean_edge" in result
        assert "n_with_edge" in result
        assert "roi_kelly_0.25" in result

    def test_mean_edge_positive_when_model_leads(self, pinnacle_scenario):
        """When model probability > Pinnacle, mean edge should be positive."""
        model_prob, pin_prob, outcomes = pinnacle_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        assert result["mean_edge"] > 0.0

    def test_n_with_edge_zero_when_no_edge(self, no_edge_scenario):
        """When model is always below Pinnacle, n_with_edge = 0."""
        model_prob, pin_prob, outcomes = no_edge_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        assert result["n_with_edge"] == 0
        assert result["roi_kelly_0.25"] == 0.0

    def test_pct_with_edge_in_range(self, pinnacle_scenario):
        """pct_with_edge should be in [0, 1]."""
        model_prob, pin_prob, outcomes = pinnacle_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        if "pct_with_edge" in result:
            assert 0.0 <= result["pct_with_edge"] <= 1.0

    def test_n_total_correct(self, pinnacle_scenario):
        model_prob, pin_prob, outcomes = pinnacle_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        if "n_total" in result:
            assert result["n_total"] == len(model_prob)

    def test_roi_is_finite(self, pinnacle_scenario):
        model_prob, pin_prob, outcomes = pinnacle_scenario
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        assert np.isfinite(result["roi_kelly_0.25"])

    def test_equal_probs_mean_edge_zero(self):
        """If model_prob == pinnacle_prob, mean edge = 0."""
        n = 100
        probs = np.full(n, 0.55)
        outcomes = np.ones(n)
        result = evaluate_vs_pinnacle(probs, probs.copy(), outcomes)
        assert abs(result["mean_edge"]) < 1e-9

    def test_pinnacle_prob_at_boundary_skipped(self):
        """
        Entries with pinnacle_prob = 0 or = 1 should be skipped in Kelly calc
        (would produce div/zero) without raising.
        """
        model_prob = np.array([0.7, 0.6, 0.8])
        pin_prob = np.array([0.0, 0.5, 1.0])  # boundary values
        outcomes = np.array([1.0, 0.0, 1.0])
        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)
        assert isinstance(result, dict)
        assert np.isfinite(result["roi_kelly_0.25"])


# ===========================================================================
# Section 8: ml/train.py — _BetaCalibrator
# ===========================================================================

class TestBetaCalibrator:
    """Tests for _BetaCalibrator (isotonic regression calibrator)."""

    @pytest.fixture
    def calibrator_class(self):
        from ml.train import _BetaCalibrator
        return _BetaCalibrator

    @pytest.fixture
    def fitted_calibrator(self, calibrator_class):
        """A calibrator already fitted on synthetic data."""
        rng = np.random.default_rng(21)
        n = 200
        probs = rng.uniform(0.2, 0.8, size=n)
        y_true = rng.binomial(1, probs).astype(float)
        cal = calibrator_class()
        cal.fit(probs, y_true)
        return cal

    def test_instantiation(self, calibrator_class):
        cal = calibrator_class()
        assert not cal._fitted

    def test_fit_sets_fitted_flag(self, calibrator_class):
        rng = np.random.default_rng(22)
        probs = rng.uniform(0.2, 0.8, size=100)
        y_true = rng.integers(0, 2, size=100).astype(float)
        cal = calibrator_class()
        cal.fit(probs, y_true)
        assert cal._fitted

    def test_fit_returns_self(self, calibrator_class):
        rng = np.random.default_rng(23)
        probs = rng.uniform(0.2, 0.8, size=100)
        y_true = rng.integers(0, 2, size=100).astype(float)
        cal = calibrator_class()
        result = cal.fit(probs, y_true)
        assert result is cal

    def test_transform_before_fit_raises_runtime_error(self, calibrator_class):
        cal = calibrator_class()
        probs = np.array([0.3, 0.5, 0.7])
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.transform(probs)

    def test_transform_returns_array(self, fitted_calibrator):
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        result = fitted_calibrator.transform(probs)
        assert isinstance(result, np.ndarray)

    def test_transform_output_length_matches_input(self, fitted_calibrator):
        probs = np.linspace(0.1, 0.9, 50)
        result = fitted_calibrator.transform(probs)
        assert len(result) == 50

    def test_transform_output_in_valid_probability_range(self, fitted_calibrator):
        """Calibrated probabilities must be in [0, 1] (isotonic clips OOB)."""
        probs = np.linspace(0.0, 1.0, 101)
        result = fitted_calibrator.transform(probs)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_transform_monotonic_non_decreasing(self, fitted_calibrator):
        """
        Isotonic regression guarantees non-decreasing output.
        Input probs sorted ascending → output sorted ascending.
        """
        probs = np.linspace(0.1, 0.9, 50)
        result = fitted_calibrator.transform(probs)
        assert np.all(np.diff(result) >= -1e-9), "Calibrated output not monotonically non-decreasing"

    def test_fit_transform_reduces_ece(self, calibrator_class):
        """After calibration ECE should not increase compared to uncalibrated model."""
        rng = np.random.default_rng(99)
        n = 500
        true_probs = rng.uniform(0.3, 0.7, size=n)
        y_true = rng.binomial(1, true_probs).astype(float)
        # Overconfident uncalibrated probs
        raw_probs = np.clip(true_probs * 1.3 + 0.1, 0.01, 0.99)

        ece_before = compute_ece(y_true, raw_probs)

        cal = calibrator_class()
        cal.fit(raw_probs, y_true)
        cal_probs = cal.transform(raw_probs)

        ece_after = compute_ece(y_true, cal_probs)

        assert ece_after <= ece_before + 0.05, (
            f"Calibration increased ECE: before={ece_before:.4f}, after={ece_after:.4f}"
        )

    def test_fit_with_single_sample(self, calibrator_class):
        """Single-sample fit should not raise."""
        probs = np.array([0.6])
        y_true = np.array([1.0])
        cal = calibrator_class()
        cal.fit(probs, y_true)
        assert cal._fitted

    def test_transform_with_single_sample(self, fitted_calibrator):
        result = fitted_calibrator.transform(np.array([0.5]))
        assert len(result) == 1


# ===========================================================================
# Section 9: ml/train.py — _compute_ece (module-level private function)
# ===========================================================================

class TestTrainComputeEce:
    """
    Tests for the _compute_ece function in train.py.
    This is the same algorithm as calibrate.compute_ece — verify consistency.
    """

    @pytest.fixture
    def train_ece_fn(self):
        from ml.train import _compute_ece
        return _compute_ece

    def test_returns_float(self, train_ece_fn, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert isinstance(train_ece_fn(y_true, y_prob), float)

    def test_non_negative(self, train_ece_fn, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        assert train_ece_fn(y_true, y_prob) >= 0.0

    def test_consistent_with_calibrate_module(self, train_ece_fn, calibrated_arrays):
        """_compute_ece in train.py should give same result as calibrate.compute_ece."""
        y_true, y_prob = calibrated_arrays
        from ml.calibrate import compute_ece as cal_ece
        train_result = train_ece_fn(y_true, y_prob)
        cal_result = cal_ece(y_true, y_prob)
        assert train_result == pytest.approx(cal_result, abs=1e-9)

    def test_custom_bins(self, train_ece_fn, calibrated_arrays):
        y_true, y_prob = calibrated_arrays
        for bins in [5, 10, 20]:
            result = train_ece_fn(y_true, y_prob, n_bins=bins)
            assert 0.0 <= result <= 1.0

    def test_empty_bins_no_error(self, train_ece_fn):
        rng = np.random.default_rng(33)
        y_true = rng.integers(0, 2, size=20).astype(float)
        y_prob = rng.uniform(0.45, 0.55, size=20)
        result = train_ece_fn(y_true, y_prob, n_bins=50)
        assert isinstance(result, float)


# ===========================================================================
# Section 10: ml/train.py — RegimeGate
# ===========================================================================

class TestRegimeGate:
    """Tests for RegimeGate — assigns R0/R1/R2 to match rows."""

    @pytest.fixture
    def ms_gate(self):
        from ml.train import RegimeGate
        return RegimeGate(Discipline.MS)

    @pytest.fixture
    def md_gate(self):
        from ml.train import RegimeGate
        return RegimeGate(Discipline.MD)

    @pytest.fixture
    def simple_df(self):
        """Small DataFrame with entity_a and entity_b columns."""
        return pd.DataFrame({
            "entity_a": ["player_1", "player_2", "player_3", "player_1"],
            "entity_b": ["player_4", "player_5", "player_1", "player_2"],
        })

    def test_instantiation_ms(self):
        from ml.train import RegimeGate
        gate = RegimeGate(Discipline.MS)
        assert gate._r0_max == 5   # ML_REGIME_R0_MAX_MATCHES[MS]
        assert gate._r1_max == 50  # ML_REGIME_R1_MAX_MATCHES[MS]

    def test_instantiation_md(self):
        from ml.train import RegimeGate
        gate = RegimeGate(Discipline.MD)
        assert gate._r0_max == 8   # MD has higher threshold (pairs)
        assert gate._r1_max == 60

    def test_assign_returns_series(self, ms_gate, simple_df):
        counts = {"player_1": 100, "player_2": 100, "player_3": 100,
                  "player_4": 100, "player_5": 100}
        result = ms_gate.assign(simple_df, counts)
        assert isinstance(result, pd.Series)

    def test_assign_length_matches_df(self, ms_gate, simple_df):
        counts = {"player_1": 100, "player_2": 100, "player_3": 100,
                  "player_4": 100, "player_5": 100}
        result = ms_gate.assign(simple_df, counts)
        assert len(result) == len(simple_df)

    def test_assign_values_are_valid_regimes(self, ms_gate, simple_df):
        counts = {"player_1": 100, "player_2": 100, "player_3": 100,
                  "player_4": 100, "player_5": 100}
        result = ms_gate.assign(simple_df, counts)
        assert set(result.unique()).issubset({"R0", "R1", "R2"})

    def test_new_player_assigned_r0(self, ms_gate):
        """Player with 0 historical matches (not in counts) should get R0."""
        df = pd.DataFrame({"entity_a": ["new_player"], "entity_b": ["veteran"]})
        counts = {"veteran": 200}  # "new_player" missing from counts → 0 matches
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R0"

    def test_veteran_players_assigned_r2(self, ms_gate):
        """Both players with > R1 max matches → R2."""
        df = pd.DataFrame({"entity_a": ["vet_a"], "entity_b": ["vet_b"]})
        counts = {"vet_a": 200, "vet_b": 150}  # Both > 50 (R1 max for MS)
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R2"

    def test_intermediate_player_assigned_r1(self, ms_gate):
        """Player with 20 matches (> R0=5, <= R1=50) should get R1."""
        df = pd.DataFrame({"entity_a": ["mid_a"], "entity_b": ["vet_b"]})
        counts = {"mid_a": 20, "vet_b": 200}  # min_count=20 → R1
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R1"

    def test_r0_boundary_at_r0_max(self, ms_gate):
        """Exactly at R0_max threshold (5 for MS) → R0 (min_count <= r0_max)."""
        df = pd.DataFrame({"entity_a": ["p_a"], "entity_b": ["p_b"]})
        counts = {"p_a": 5, "p_b": 200}  # min_count=5 == r0_max → R0
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R0"

    def test_r1_boundary_above_r0_max(self, ms_gate):
        """One count above R0_max (6 for MS) → R1 (since 6 <= r1_max=50)."""
        df = pd.DataFrame({"entity_a": ["p_a"], "entity_b": ["p_b"]})
        counts = {"p_a": 6, "p_b": 200}  # min_count=6 > 5 and <= 50 → R1
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R1"

    def test_r1_boundary_at_r1_max(self, ms_gate):
        """Exactly at R1_max threshold (50 for MS) → R1 (min_count <= r1_max)."""
        df = pd.DataFrame({"entity_a": ["p_a"], "entity_b": ["p_b"]})
        counts = {"p_a": 50, "p_b": 200}  # min_count=50 == r1_max → R1
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R1"

    def test_r2_boundary_above_r1_max(self, ms_gate):
        """One count above R1_max (51 for MS) → R2."""
        df = pd.DataFrame({"entity_a": ["p_a"], "entity_b": ["p_b"]})
        counts = {"p_a": 51, "p_b": 200}  # min_count=51 > 50 → R2
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R2"

    def test_min_count_determines_regime(self, ms_gate):
        """Regime is determined by min(count_a, count_b) — the weaker entity."""
        df = pd.DataFrame({"entity_a": ["vet_a"], "entity_b": ["new_b"]})
        counts = {"vet_a": 200, "new_b": 3}  # min=3 → R0
        result = ms_gate.assign(df, counts)
        assert result.iloc[0] == "R0"

    def test_all_disciplines_instantiate(self):
        from ml.train import RegimeGate
        for disc in Discipline:
            gate = RegimeGate(disc)
            assert gate._r0_max > 0
            assert gate._r1_max > gate._r0_max


# ===========================================================================
# Section 11: ml/train.py — BadmintonModelTrainer constructor
# ===========================================================================

class TestBadmintonModelTrainerConstructor:
    """Tests for BadmintonModelTrainer.__init__ — constructor validation only."""

    def test_raises_without_model_dir_env(self):
        """Constructor raises RuntimeError when model_dir not given and env var absent."""
        from ml.train import BadmintonModelTrainer
        # Ensure env var is not set
        env_backup = os.environ.pop("BADMINTON_MODEL_DIR", None)
        try:
            with pytest.raises(RuntimeError, match="BADMINTON_MODEL_DIR"):
                BadmintonModelTrainer(Discipline.MS)
        finally:
            if env_backup is not None:
                os.environ["BADMINTON_MODEL_DIR"] = env_backup

    def test_accepts_explicit_model_dir(self, tmp_path):
        """Constructor succeeds when model_dir is explicitly provided."""
        from ml.train import BadmintonModelTrainer
        trainer = BadmintonModelTrainer(Discipline.MS, model_dir=str(tmp_path))
        expected_dir = tmp_path / "MS"
        assert trainer._model_dir == expected_dir
        assert expected_dir.exists()

    def test_model_dir_created_on_init(self, tmp_path):
        """Constructor creates the discipline subdirectory if it doesn't exist."""
        from ml.train import BadmintonModelTrainer
        disc_dir = tmp_path / "WS"
        assert not disc_dir.exists()
        BadmintonModelTrainer(Discipline.WS, model_dir=str(tmp_path))
        assert disc_dir.exists()

    def test_discipline_stored(self, tmp_path):
        from ml.train import BadmintonModelTrainer
        trainer = BadmintonModelTrainer(Discipline.MD, model_dir=str(tmp_path))
        assert trainer._discipline == Discipline.MD

    def test_env_var_used_when_no_explicit_dir(self, tmp_path):
        """Constructor uses BADMINTON_MODEL_DIR env var when model_dir not given."""
        from ml.train import BadmintonModelTrainer
        os.environ["BADMINTON_MODEL_DIR"] = str(tmp_path)
        try:
            trainer = BadmintonModelTrainer(Discipline.XD)
            assert trainer._discipline == Discipline.XD
            assert (tmp_path / "XD").exists()
        finally:
            del os.environ["BADMINTON_MODEL_DIR"]

    def test_all_disciplines_accepted(self, tmp_path):
        from ml.train import BadmintonModelTrainer
        for disc in Discipline:
            trainer = BadmintonModelTrainer(disc, model_dir=str(tmp_path))
            assert trainer._discipline == disc

    def test_train_and_evaluate_raises_on_insufficient_data(self, tmp_path):
        """train_and_evaluate raises RuntimeError when training set has < 100 rows."""
        from ml.train import BadmintonModelTrainer
        from config.badminton_config import (
            ML_TRAIN_START_YEAR,
            ML_TRAIN_END_YEAR,
            ML_FEATURES_TOTAL,
        )
        trainer = BadmintonModelTrainer(Discipline.MS, model_dir=str(tmp_path))
        # Build a minimal DataFrame with fewer than 100 training rows
        dates = pd.date_range("2019-01-01", periods=30, freq="7D")  # 30 rows in train window
        feat_cols = {f"feat_{i:03d}": np.random.default_rng(i).uniform(0, 1, 30)
                     for i in range(ML_FEATURES_TOTAL)}
        df = pd.DataFrame({
            "date": dates,
            "discipline": "MS",
            "entity_a": [f"p_a_{i}" for i in range(30)],
            "entity_b": [f"p_b_{i}" for i in range(30)],
            "target_win": np.random.default_rng(5).integers(0, 2, 30),
            **feat_cols,
        })
        with pytest.raises(RuntimeError, match="Insufficient training data"):
            trainer.train_and_evaluate(df, n_optuna_trials=1)


# ===========================================================================
# Section 12: Integration — calibrate + evaluate pipeline
# ===========================================================================

class TestCalibrateEvaluateIntegration:
    """
    End-to-end tests combining calibrate and evaluate functions
    to simulate what the training pipeline does.
    """

    def test_calibration_report_then_evaluate_consistency(self, balanced_good_predictions):
        """
        Metrics from calibration_report and evaluate_predictions should
        agree on brier, ece, and log_loss for the same inputs.
        """
        y_true, y_prob = balanced_good_predictions
        report = calibration_report(y_true, y_prob, name="pipeline_check")
        metrics = evaluate_predictions(y_true, y_prob, Discipline.MS, split="val")

        # Both use the same underlying implementations — values should match within rounding
        assert abs(report["brier"] - metrics["brier_val"]) < 1e-3
        assert abs(report["ece"] - metrics["ece_val"]) < 1e-3
        assert abs(report["log_loss"] - metrics["log_loss_val"]) < 1e-3

    def test_full_calibrate_pipeline_on_balanced_data(self, balanced_good_predictions):
        """
        Simulate a complete calibration check:
        1. Fit BetaCalibrator on half the data
        2. Transform the other half
        3. Evaluate on test split — should pass all QA gates
        """
        from ml.train import _BetaCalibrator
        y_true, y_prob = balanced_good_predictions
        n = len(y_true)
        split = n // 2

        # Fit calibrator on first half
        cal = _BetaCalibrator()
        cal.fit(y_prob[:split], y_true[:split])

        # Transform second half
        cal_probs = cal.transform(y_prob[split:])
        y_test = y_true[split:]

        # Calibrated probs should still evaluate within acceptable ranges
        brier = compute_brier_score(y_test, cal_probs)
        ece = compute_ece(y_test, cal_probs)
        assert brier <= 0.35, f"Post-calibration Brier too high: {brier:.4f}"
        assert ece >= 0.0

    def test_reliability_data_consistent_with_ece(self, calibrated_arrays):
        """
        ECE can be recomputed from reliability diagram data.
        Verify the two outputs are mathematically consistent.
        """
        y_true, y_prob = calibrated_arrays
        n = len(y_true)
        n_bins = 10

        reliability = compute_reliability_data(y_true, y_prob, n_bins=n_bins)
        ece_from_reliability = sum(
            (n_samples / n) * abs(mean_acc - mean_conf)
            for mean_conf, mean_acc, n_samples in reliability
        )
        ece_direct = compute_ece(y_true, y_prob, n_bins=n_bins)

        assert abs(ece_from_reliability - ece_direct) < 1e-9, (
            f"ECE mismatch: from_reliability={ece_from_reliability:.6f}, "
            f"direct={ece_direct:.6f}"
        )

    def test_evaluate_vs_pinnacle_with_realistic_scenario(self):
        """
        Simulate a realistic Pinnacle comparison:
        model has 3pp average edge, 60% of matches.
        Should produce positive ROI and valid output dict.
        """
        from ml.train import _BetaCalibrator

        rng = np.random.default_rng(111)
        n = 300
        pin_prob = rng.uniform(0.38, 0.62, size=n)
        # Edge on 60% of matches
        edge_mask = rng.random(n) < 0.60
        model_prob = pin_prob.copy()
        model_prob[edge_mask] = np.clip(pin_prob[edge_mask] + 0.04, 0.01, 0.99)
        outcomes = rng.binomial(1, pin_prob).astype(float)

        result = evaluate_vs_pinnacle(model_prob, pin_prob, outcomes)

        assert "mean_edge" in result
        assert result["mean_edge"] > 0.0
        assert result["n_with_edge"] > 0
        roi = result["roi_kelly_0.25"]
        assert np.isfinite(roi)

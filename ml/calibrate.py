"""
calibrate.py
============
Post-training calibration utilities for badminton models.

Provides:
  - Calibration quality evaluation (ECE, reliability diagram data)
  - Beta calibrator fitting (from train.py _BetaCalibrator)
  - Pinnacle-implied probability calibration
  - Calibration comparison: pre vs post

Calibration is applied as the final step in the training pipeline.
Models are re-calibrated on a hold-out validation set.

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (|B| / n) * |acc(B) - conf(B)|

    Args:
        y_true: Binary labels (0/1)
        y_prob: Predicted probabilities [0, 1]
        n_bins: Number of calibration bins

    Returns:
        ECE score (lower is better, 0 = perfect calibration)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[Tuple[float, float, int]]:
    """
    Compute reliability diagram data.

    Returns list of (mean_confidence, mean_accuracy, n_samples) per bin.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        mean_conf = float(y_prob[mask].mean())
        mean_acc = float(y_true[mask].mean())
        n_samples = int(mask.sum())
        reliability.append((mean_conf, mean_acc, n_samples))

    return reliability


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score = mean squared error of probabilities."""
    return float(np.mean((y_true - y_prob) ** 2))


def compute_log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Binary cross-entropy log loss."""
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    ))


def calibration_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "model",
) -> dict:
    """
    Full calibration quality report.

    Returns dict with ECE, Brier, log loss, and reliability data.
    """
    ece = compute_ece(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    ll = compute_log_loss(y_true, y_prob)
    reliability = compute_reliability_data(y_true, y_prob)

    report = {
        "model": name,
        "ece": round(ece, 5),
        "brier": round(brier, 5),
        "log_loss": round(ll, 5),
        "reliability": reliability,
        "n_samples": len(y_true),
        "win_rate": round(float(y_true.mean()), 4),
        "mean_prob": round(float(y_prob.mean()), 4),
    }

    logger.info(
        "calibration_report",
        model=name,
        ece=report["ece"],
        brier=report["brier"],
        log_loss=report["log_loss"],
    )

    return report

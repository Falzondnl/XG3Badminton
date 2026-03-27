"""
evaluate.py
===========
Model evaluation and Pinnacle comparison for badminton models.

Evaluates trained models against:
  1. Hold-out test set (standard ML metrics)
  2. Pinnacle historical odds (edge measurement)
  3. Kelly criterion simulation (theoretical expected value)

Metrics computed:
  - AUC-ROC (H2 gate threshold: >= 0.65)
  - Brier score (H3 gate threshold: <= 0.24)
  - ECE (H4 gate threshold: <= 0.05)
  - Log loss
  - Accuracy at various probability thresholds

Pinnacle comparison:
  - Load Pinnacle historical odds from Optic Odds snapshots
  - Compute implied probability from Pinnacle odds (remove margin)
  - Compare model probability vs Pinnacle implied probability
  - Edge = model_prob - pinnacle_implied_prob

ZERO hardcoded expected values — all thresholds from config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import structlog

from config.badminton_config import (
    Discipline,
    ML_AUC_THRESHOLD,
    ML_BRIER_THRESHOLD,
    ML_ECE_THRESHOLD,
)
from ml.calibrate import compute_ece, compute_brier_score, compute_log_loss

logger = structlog.get_logger(__name__)

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    discipline: Discipline,
    split: str = "test",
) -> Dict[str, float]:
    """
    Evaluate model predictions against known outcomes.

    Runs all QA gates and returns metrics dict.
    Raises RuntimeError if H2/H3/H4 gates fail.

    Args:
        y_true: Binary labels (1 = P1 won)
        y_prob: Calibrated probability P1 wins
        discipline: Discipline for logging context
        split: "train", "val", or "test"
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("sklearn_not_available_skipping_auc")
        auc = 0.0
    else:
        auc = float(roc_auc_score(y_true, y_prob))

    brier = compute_brier_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    ll = compute_log_loss(y_true, y_prob)

    # Win rate check (P1 win rate should be in [0.45, 0.55])
    win_rate = float(y_true.mean())
    mean_prob = float(y_prob.mean())

    metrics = {
        f"auc_{split}": round(auc, 4),
        f"brier_{split}": round(brier, 4),
        f"ece_{split}": round(ece, 4),
        f"log_loss_{split}": round(ll, 4),
        f"win_rate_{split}": round(win_rate, 4),
        f"mean_prob_{split}": round(mean_prob, 4),
        f"n_samples_{split}": len(y_true),
    }

    logger.info(
        "model_evaluation",
        discipline=discipline.value,
        split=split,
        **{k: v for k, v in metrics.items() if split in k},
    )

    # QA gate validation (only for test split)
    if split == "test":
        gate_failures = []

        if auc < ML_AUC_THRESHOLD:
            gate_failures.append(
                f"H2: AUC {auc:.4f} < {ML_AUC_THRESHOLD} "
                f"(discipline={discipline.value})"
            )
        if brier > ML_BRIER_THRESHOLD:
            gate_failures.append(
                f"H3: Brier {brier:.4f} > {ML_BRIER_THRESHOLD}"
            )
        if ece > ML_ECE_THRESHOLD:
            gate_failures.append(
                f"H4: ECE {ece:.4f} > {ML_ECE_THRESHOLD}"
            )

        if gate_failures:
            for failure in gate_failures:
                logger.error("qa_gate_failure", failure=failure)
            raise RuntimeError(
                f"QA gate failures for {discipline.value}: {gate_failures}"
            )

    return metrics


def evaluate_vs_pinnacle(
    model_probs: np.ndarray,
    pinnacle_probs: np.ndarray,
    outcomes: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate model edge against Pinnacle market.

    Computes:
    - Mean edge per bet (positive = model has edge)
    - Kelly recommended bet size
    - Simulated ROI at Kelly fractions

    Args:
        model_probs: Model P(A wins) per match
        pinnacle_probs: Pinnacle implied P(A wins) (margin-removed)
        outcomes: Actual outcomes (1 = A won)
    """
    edge = model_probs - pinnacle_probs
    mean_edge = float(np.mean(edge))

    # Only evaluate matches where we have edge
    edge_mask = edge > 0.01
    n_with_edge = int(edge_mask.sum())

    if n_with_edge == 0:
        return {
            "mean_edge": round(mean_edge, 4),
            "n_with_edge": 0,
            "roi_kelly_0.25": 0.0,
        }

    # ROI simulation at quarter Kelly
    kelly_fraction = 0.25
    returns = []
    for prob, pin_prob, outcome in zip(
        model_probs[edge_mask],
        pinnacle_probs[edge_mask],
        outcomes[edge_mask],
    ):
        if pin_prob <= 0 or pin_prob >= 1:
            continue
        implied_odds = 1.0 / pin_prob
        kelly_stake = kelly_fraction * (prob - (1 - prob) / (implied_odds - 1))
        kelly_stake = max(0.0, min(0.25, kelly_stake))
        profit = kelly_stake * (implied_odds - 1) if outcome == 1 else -kelly_stake
        returns.append(profit)

    roi = float(np.mean(returns)) if returns else 0.0

    return {
        "mean_edge": round(mean_edge, 4),
        "n_with_edge": n_with_edge,
        "n_total": len(model_probs),
        "roi_kelly_0.25": round(roi, 4),
        "pct_with_edge": round(n_with_edge / len(model_probs), 3),
    }

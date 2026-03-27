"""
BadmintonTrainer — trains per-discipline models.

Regime layout:
  R0 → Men's Singles  (MS)
  R1 → Women's Singles (WS)
  R2 → Doubles combined (MD + XD + WD) with discipline_enc feature

Temporal split: 70% train / 15% val / 15% test (by date).
Prints: discipline, AUC, Brier, class balance, n_samples.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from config import MD_CSV, MS_CSV, R0_DIR, R1_DIR, R2_DIR, WD_CSV, WS_CSV, XD_CSV
from ml.calibrator import BetaCalibrator
from ml.ensemble import StackingEnsemble
from ml.features import FEATURE_NAMES, BadmintonFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _temporal_split(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15
) -> tuple[np.ndarray, ...]:
    n = len(y)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Not enough data for split: n={n}")
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train : n_train + n_val]
    y_val = y[n_train : n_train + n_val]
    X_test = X[n_train + n_val :]
    y_test = y[n_train + n_val :]
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


class BadmintonTrainer:
    """Orchestrates training for all three regimes."""

    def __init__(self) -> None:
        self._extractors: dict[str, BadmintonFeatureExtractor] = {}

    # ---------------------------------------------------------------------- #
    # Public
    # ---------------------------------------------------------------------- #

    def train_all(self) -> None:
        """Train R0 (MS), R1 (WS), R2 (doubles) and save all artefacts."""
        print("\n" + "=" * 60)
        print("BADMINTON MODEL TRAINING - TIER-1")
        print("=" * 60)

        self._train_singles(discipline="MS", csv_path=MS_CSV, out_dir=R0_DIR)
        self._train_singles(discipline="WS", csv_path=WS_CSV, out_dir=R1_DIR)
        self._train_doubles(out_dir=R2_DIR)

        print("\n" + "=" * 60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("=" * 60)

    # ---------------------------------------------------------------------- #
    # Singles training
    # ---------------------------------------------------------------------- #

    def _train_singles(self, discipline: str, csv_path: str, out_dir: str) -> None:
        print(f"\n--- {discipline} (Singles) ---")
        df = _load_csv(csv_path)
        extractor = BadmintonFeatureExtractor()
        X, y = extractor.extract_training_dataset(df, discipline=discipline, apply_swap=True)
        self._extractors[discipline] = extractor

        print(f"  n={len(y)}, class_balance={y.mean():.3f}")

        X_train, y_train, X_val, y_val, X_test, y_test = _temporal_split(X, y)
        print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        ensemble = StackingEnsemble()
        metrics = ensemble.fit(X_train, y_train, X_val, y_val, feature_names=FEATURE_NAMES)

        calibrator = BetaCalibrator()
        raw_val = ensemble.predict_proba(X_val)
        calibrator.fit(raw_val, y_val)

        # Test metrics
        raw_test = ensemble.predict_proba(X_test)
        cal_test = calibrator.predict(raw_test)
        auc = roc_auc_score(y_test, cal_test)
        brier = brier_score_loss(y_test, cal_test)
        print(f"  TEST  AUC={auc:.4f}  Brier={brier:.4f}")
        print(f"  Base  metrics: {metrics}")

        # Save
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(out / "ensemble.pkl"))
        calibrator.save(str(out / "calibrator.pkl"))
        with open(out / "extractor.pkl", "wb") as f:
            pickle.dump(extractor, f)
        print(f"  Saved -> {out_dir}/")

    # ---------------------------------------------------------------------- #
    # Doubles training
    # ---------------------------------------------------------------------- #

    def _train_doubles(self, out_dir: str) -> None:
        print("\n--- Doubles (MD + XD + WD combined) ---")

        extractor = BadmintonFeatureExtractor()
        all_Xs: list[np.ndarray] = []
        all_ys: list[np.ndarray] = []

        disc_csv_map = {
            "MD": MD_CSV,
            "XD": XD_CSV,
            "WD": WD_CSV,
        }

        for discipline, csv_path in disc_csv_map.items():
            df = _load_csv(csv_path)
            # Use fresh extractor per discipline to avoid cross-contamination,
            # but share ELO pools within the combined model run.
            X_d, y_d = extractor.extract_training_dataset(
                df, discipline=discipline, apply_swap=True
            )
            all_Xs.append(X_d)
            all_ys.append(y_d)

        X = np.vstack(all_Xs)
        y = np.concatenate(all_ys)

        # Shuffle together (they are appended by discipline, not temporally)
        rng = np.random.default_rng(seed=0)
        idx = rng.permutation(len(y))
        X = X[idx]
        y = y[idx]

        self._extractors["doubles"] = extractor

        print(f"  n={len(y)}, class_balance={y.mean():.3f}")
        print(f"  disciplines: MD + XD + WD")

        X_train, y_train, X_val, y_val, X_test, y_test = _temporal_split(X, y)
        print(f"  train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        ensemble = StackingEnsemble()
        metrics = ensemble.fit(X_train, y_train, X_val, y_val, feature_names=FEATURE_NAMES)

        calibrator = BetaCalibrator()
        raw_val = ensemble.predict_proba(X_val)
        calibrator.fit(raw_val, y_val)

        raw_test = ensemble.predict_proba(X_test)
        cal_test = calibrator.predict(raw_test)
        auc = roc_auc_score(y_test, cal_test)
        brier = brier_score_loss(y_test, cal_test)
        print(f"  TEST  AUC={auc:.4f}  Brier={brier:.4f}")
        print(f"  Base  metrics: {metrics}")

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        ensemble.save(str(out / "ensemble.pkl"))
        calibrator.save(str(out / "calibrator.pkl"))
        with open(out / "extractor.pkl", "wb") as f:
            pickle.dump(extractor, f)
        print(f"  Saved -> {out_dir}/")

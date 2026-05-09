"""
fit_calibrators_p0.py
=====================

P0 CLAUDE.md FIX — fit BetaCalibrator for badminton r0/r1/r2 with REAL
predictions on a temporal holdout (last 20% of matches per regime).

The existing calibrator.pkl files have is_fitted=False (stub state) — the
calibrator raises RuntimeError on every prediction. This script:

  1. Replays training data chronologically through a fresh BadmintonFeatureExtractor
     (state correctly updates AFTER each match → pre-match features, no leakage)
  2. Splits into TRAIN-state / CALIBRATION-holdout = first 80% / last 20% by date
  3. Runs the loaded production ensemble.predict_proba() on the holdout
     (the same feature vectors the live system would compute)
  4. Fits BetaCalibrator on (raw_probs, y_true)
  5. Computes Expected Calibration Error (ECE) on holdout — must be < 0.05
  6. Saves fitted calibrator back to models/{r0,r1,r2}/calibrator.pkl

Regime mapping (from config.py + ml/predictor.py):
  r0 → MS  (Men's Singles)
  r1 → WS  (Women's Singles)
  r2 → doubles combined (MD + XD + WD)

This mirrors how the production predictor loads them in BadmintonPredictor.load().
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress sklearn version warning — we re-fit with the current version, so this is benign.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import MS_CSV, WS_CSV, MD_CSV, XD_CSV, WD_CSV, R0_DIR, R1_DIR, R2_DIR
from ml.calibrator import BetaCalibrator
from ml.ensemble import StackingEnsemble
from ml.features import BadmintonFeatureExtractor


# ---------------------------------------------------------------------- #
# sklearn 1.8 -> 1.7.1 compat shim for deserialized LogisticRegression.
# The production .pkl files were trained with sklearn 1.8 which removed
# the legacy `multi_class` attribute; sklearn 1.7.1 (current container
# requirements.txt) still references it inside predict_proba(). The
# patch sets multi_class='auto' on every LR object hanging off the loaded
# ensemble (matches the default that 1.8 effectively used: binary -> ovr).
# This does not modify the trained coefficients in any way.
# ---------------------------------------------------------------------- #

def patch_lr_multi_class(estimator) -> None:
    """Recursively set multi_class='auto' on any LogisticRegression we find."""
    from sklearn.linear_model import LogisticRegression

    def _walk(obj, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(obj, LogisticRegression):
            if not hasattr(obj, "multi_class"):
                obj.multi_class = "auto"
            return
        # walk simple containers
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                _walk(v, depth + 1)

    _walk(estimator)


def safe_load_ensemble(path: str) -> StackingEnsemble:
    e = StackingEnsemble.load(path)
    patch_lr_multi_class(e)
    return e


# ---------------------------------------------------------------------- #
# ECE
# ---------------------------------------------------------------------- #

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------- #
# Per-regime data builder
# ---------------------------------------------------------------------- #

def build_features_chronological(df: pd.DataFrame, discipline: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build feature matrix WITHOUT random P1/P2 swap so features stay deterministic
    and represent the same pre-match state the live system would produce.

    Returns (X, y, df_sorted). df_sorted is the chronologically-ordered df after
    void-match drops, so we can split temporally.
    """
    extractor = BadmintonFeatureExtractor()

    # Match data_loader's expectation: parse date, drop nulls, chronologically sort.
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)

    # Drop void matches (winner=0)
    df = df[df["winner"].isin([1, 2])].reset_index(drop=True)

    # extract_training_dataset re-sorts internally + applies the void filter again,
    # but resets the discipline state at the start. We want to drive it directly with
    # apply_swap=False so calibration-holdout predictions match live behaviour.
    X, y = extractor.extract_training_dataset(df, discipline, apply_swap=False)

    # Should match shape — extract_training_dataset filters winner==0 internally too.
    assert len(X) == len(y), f"X/y mismatch for {discipline}"
    # df was already void-filtered, so lengths align (extract_training_dataset will
    # produce same N because it re-applies the same filter).
    assert len(X) == len(df), f"length mismatch X={len(X)} df={len(df)} for {discipline}"

    return X, y, df


def fit_one_regime(
    regime_name: str,
    model_dir: str,
    csv_paths: list[tuple[str, str]],   # list of (csv_path, discipline_str)
) -> dict:
    """
    Load production ensemble, build features for the regime's discipline(s),
    take the last 20% of matches as the calibration holdout, fit BetaCalibrator,
    save it.
    """
    print(f"\n{'='*72}")
    print(f"REGIME {regime_name} -> {model_dir}")
    print(f"  disciplines: {[d for _, d in csv_paths]}")
    print(f"{'='*72}")

    # ---------- Load ensemble ----------
    ensemble_path = REPO_ROOT / model_dir / "ensemble.pkl"
    ensemble = safe_load_ensemble(str(ensemble_path))
    if not ensemble.is_fitted:
        raise RuntimeError(f"Ensemble at {ensemble_path} reports is_fitted=False — cannot fit calibrator")
    print(f"  ensemble loaded: is_fitted={ensemble.is_fitted}, n_features={len(ensemble.feature_names)}")

    # ---------- Build feature dataset across all disciplines for this regime ----------
    # For r2 (doubles), combine MD + XD + WD using a shared extractor so partnership
    # state is properly maintained per-discipline. Each discipline maintains its own
    # ELO pool inside the extractor (per ml/features.py:109).
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_dates: list[pd.Series] = []

    for csv_path, discipline in csv_paths:
        full_csv = csv_path  # already absolute from config.py
        df = pd.read_csv(full_csv, low_memory=False)
        print(f"  loading {discipline}: {full_csv} -> {len(df):,} raw rows")

        X, y, df_sorted = build_features_chronological(df, discipline)
        print(f"    after parse + void-drop: {len(X):,} usable matches, "
              f"date range {df_sorted['_date'].min().date()} -> {df_sorted['_date'].max().date()}, "
              f"target balance={y.mean():.3f}")

        all_X.append(X)
        all_y.append(y)
        all_dates.append(df_sorted["_date"].reset_index(drop=True))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    dates = pd.concat(all_dates, ignore_index=True)

    # Sort the combined matrix chronologically (across disciplines for r2)
    order = np.argsort(dates.values)
    X = X[order]
    y = y[order]
    dates = dates.iloc[order].reset_index(drop=True)

    n_total = len(X)
    n_holdout = int(0.20 * n_total)
    n_train = n_total - n_holdout

    X_holdout = X[n_train:]
    y_holdout = y[n_train:]
    holdout_start = dates.iloc[n_train].date()
    holdout_end = dates.iloc[-1].date()

    print(f"  combined dataset: n={n_total:,} matches, target_balance={y.mean():.3f}")
    print(f"  temporal split: TRAIN n={n_train:,} | HOLDOUT n={n_holdout:,} ({holdout_start} -> {holdout_end})")
    print(f"  holdout target_balance={y_holdout.mean():.3f}")

    # ---------- Predict on holdout ----------
    raw_probs = ensemble.predict_proba(X_holdout)
    raw_probs = np.asarray(raw_probs).astype(np.float64)
    print(f"  ensemble predictions: min={raw_probs.min():.4f} max={raw_probs.max():.4f} "
          f"mean={raw_probs.mean():.4f}")

    # ---------- Pre-calibration metrics ----------
    pre_brier = compute_brier(y_holdout, raw_probs)
    pre_ece = compute_ece(y_holdout, raw_probs, n_bins=10)
    print(f"  pre-calibration: Brier={pre_brier:.4f} ECE={pre_ece:.4f}")

    # ---------- Fit BetaCalibrator ----------
    calibrator = BetaCalibrator()
    calibrator.fit(raw_probs, y_holdout)
    assert calibrator.is_fitted, "BetaCalibrator.fit() did not set is_fitted=True"

    cal_probs = calibrator.predict(raw_probs)
    post_brier = compute_brier(y_holdout, cal_probs)
    post_ece = compute_ece(y_holdout, cal_probs, n_bins=10)
    print(f"  post-calibration: Brier={post_brier:.4f} ECE={post_ece:.4f}")
    print(f"  Brier improvement: {pre_brier - post_brier:+.4f}")
    print(f"  ECE improvement:   {pre_ece - post_ece:+.4f}")

    # ---------- H4 gate ----------
    h4_threshold = 0.05
    h4_passed = post_ece <= h4_threshold
    print(f"  H4 gate (ECE <= {h4_threshold}): {'PASS' if h4_passed else 'FAIL'} (got {post_ece:.4f})")

    # ---------- Save ----------
    out_path = REPO_ROOT / model_dir / "calibrator.pkl"
    calibrator.save(str(out_path))
    print(f"  saved: {out_path}")

    # Verify on disk
    reloaded = BetaCalibrator.load(str(out_path))
    assert reloaded.is_fitted is True, f"PERSISTENCE FAIL: reloaded calibrator is_fitted={reloaded.is_fitted}"
    print(f"  reload-verify: is_fitted=True OK")

    return {
        "regime": regime_name,
        "model_dir": model_dir,
        "n_train": n_train,
        "n_holdout": n_holdout,
        "holdout_date_start": str(holdout_start),
        "holdout_date_end": str(holdout_end),
        "target_balance_overall": float(y.mean()),
        "target_balance_holdout": float(y_holdout.mean()),
        "pre_brier": pre_brier,
        "pre_ece": pre_ece,
        "post_brier": post_brier,
        "post_ece": post_ece,
        "h4_passed": h4_passed,
    }


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #

def main() -> int:
    print("=" * 72)
    print("BADMINTON CALIBRATOR P0 FIT - r0 / r1 / r2")
    print("=" * 72)
    print(f"REPO_ROOT: {REPO_ROOT}")

    regimes: list[tuple[str, str, list[tuple[str, str]]]] = [
        ("r0", R0_DIR, [(MS_CSV, "MS")]),
        ("r1", R1_DIR, [(WS_CSV, "WS")]),
        ("r2", R2_DIR, [(MD_CSV, "MD"), (XD_CSV, "XD"), (WD_CSV, "WD")]),
    ]

    results: list[dict] = []
    for regime_name, model_dir, csvs in regimes:
        try:
            r = fit_one_regime(regime_name, model_dir, csvs)
            results.append(r)
        except Exception as exc:
            print(f"\nFAILED regime={regime_name}: {exc}")
            import traceback
            traceback.print_exc()
            return 1

    # ------------------ Summary ------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'REGIME':<8}{'N_HOLDOUT':<12}{'TGT_BAL':<10}{'PRE_ECE':<10}{'POST_ECE':<10}{'PRE_BRIER':<12}{'POST_BRIER':<12}{'H4':<6}")
    for r in results:
        print(f"{r['regime']:<8}{r['n_holdout']:<12,}{r['target_balance_holdout']:<10.4f}"
              f"{r['pre_ece']:<10.4f}{r['post_ece']:<10.4f}"
              f"{r['pre_brier']:<12.4f}{r['post_brier']:<12.4f}"
              f"{'PASS' if r['h4_passed'] else 'FAIL':<6}")

    all_passed = all(r["h4_passed"] for r in results)
    print(f"\nOVERALL: {'ALL H4 PASS' if all_passed else 'H4 FAILURES — see above'}")

    # Final assertion check
    print("\n" + "=" * 72)
    print("PERSISTENCE VERIFICATION (is_fitted=True on disk)")
    print("=" * 72)
    import pickle
    for regime_name, model_dir, _ in regimes:
        path = REPO_ROOT / model_dir / "calibrator.pkl"
        with open(path, "rb") as f:
            c = pickle.load(f)
        flag = c.is_fitted
        size = path.stat().st_size
        status = "OK" if flag is True else "FAIL"
        print(f"  {regime_name}: is_fitted={flag} size={size:,} bytes -> {status}")
        assert flag is True, f"REGRESSION: calibrator at {path} is_fitted={flag}"

    return 0 if all_passed else 2


if __name__ == "__main__":
    sys.exit(main())

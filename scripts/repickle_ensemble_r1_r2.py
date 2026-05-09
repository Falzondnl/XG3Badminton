"""
repickle_ensemble_r1_r2.py
==========================

Bet365-grade companion to scripts/fit_calibrators_p0.py.

Commit dd1f8dc fixed r0/ensemble.pkl which had been pickled under sklearn 1.8.0
(missing the 'multi_class' attribute that sklearn 1.7.1 still references in
LogisticRegression.predict_proba). However, r1 and r2 ensembles were left
on the 1.8.0 pickle format, meaning the SAME bug affects WS and doubles
predictions in production.

Per CLAUDE.md "BET365-LEVEL ONLY" rule:
  "Fixing the symptom and ignoring the related bugs — when a bug pattern
   affects multiple sports/services, all instances must be fixed in the
   same change, not piecemeal."

This script does the same fix dd1f8dc applied to r0:
  1. Loads r1/r2 ensembles
  2. Rebuilds the LogisticRegression meta natively in sklearn 1.7.1 by copying
     coef_, intercept_, classes_, n_features_in_, n_iter_ (NO refit — the
     learned coefficients are byte-for-byte preserved)
  3. Saves the ensemble back

After this runs, the calibrator fitting in fit_calibrators_p0.py will work
end-to-end through the predictor (no patch_lr_multi_class shim needed).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sklearn.linear_model import LogisticRegression

from config import R1_DIR, R2_DIR
from ml.ensemble import StackingEnsemble


def rebuild_lr_native(old_lr: LogisticRegression) -> LogisticRegression:
    """
    Build a fresh LogisticRegression in the current sklearn version that
    reproduces the OLD model's predictions exactly by copying fitted state.
    """
    new = LogisticRegression()
    # Constructor params kept at sklearn-1.7.1 defaults (binary problem, multi_class='auto')
    # Copy fitted attributes byte-for-byte
    new.coef_ = np.array(old_lr.coef_, copy=True)
    new.intercept_ = np.array(old_lr.intercept_, copy=True)
    new.classes_ = np.array(old_lr.classes_, copy=True)
    new.n_features_in_ = int(old_lr.n_features_in_)
    new.n_iter_ = np.array(old_lr.n_iter_, copy=True) if hasattr(old_lr, "n_iter_") else np.array([0])
    # 1.7.1 LR.predict_proba reads multi_class — set explicitly
    new.multi_class = "auto"
    return new


def repickle_one(model_dir: str, regime_name: str) -> None:
    path = REPO_ROOT / model_dir / "ensemble.pkl"
    print(f"\nREGIME {regime_name} -> {path}")

    e = StackingEnsemble.load(str(path))
    if e.meta is None:
        print(f"  meta is None — nothing to fix")
        return

    has_mc = hasattr(e.meta, "multi_class")
    print(f"  loaded: meta type={type(e.meta).__name__}, has multi_class={has_mc}")

    if has_mc:
        print(f"  already native — skipping")
        return

    # Smoke: capture current predictions on a neutral 3-input baseline using a
    # one-shot patched copy, so we can verify the rebuild reproduces them.
    e.meta.multi_class = "auto"  # temporary
    rng = np.random.default_rng(42)
    baseline_X = rng.uniform(0.0, 1.0, size=(20, e.meta.coef_.shape[1])).astype(np.float64)
    old_preds = e.meta.predict_proba(baseline_X)[:, 1]
    del e.meta.multi_class  # remove the temporary attr so the rebuild is clean

    new_meta = rebuild_lr_native(e.meta)
    new_preds = new_meta.predict_proba(baseline_X)[:, 1]
    diff = float(np.max(np.abs(old_preds - new_preds)))
    print(f"  prediction diff (old patched vs new native): max={diff:.3e}")
    if diff > 1e-9:
        raise RuntimeError(f"Rebuild did not preserve predictions; diff={diff}")

    e.meta = new_meta
    e.save(str(path))
    print(f"  saved: {path}")

    # Verify
    e2 = StackingEnsemble.load(str(path))
    assert hasattr(e2.meta, "multi_class"), "REGRESSION: rebuilt meta missing multi_class on reload"
    print(f"  reload-verify: meta has multi_class=True OK")


def main() -> int:
    print("=" * 72)
    print("RE-PICKLE BADMINTON r1 + r2 ENSEMBLES (sklearn 1.8.0 -> 1.7.1)")
    print("=" * 72)
    for regime, mdir in [("r1", R1_DIR), ("r2", R2_DIR)]:
        repickle_one(mdir, regime)
    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

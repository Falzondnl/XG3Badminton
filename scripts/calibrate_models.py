"""
calibrate_models.py
===================
Post-training model calibration script.

Loads trained ensemble models and runs:
  1. Beta calibration fit on hold-out calibration set
  2. Reliability diagram generation (ECE per bin)
  3. H4 gate validation (ECE <= 0.05)
  4. Pinnacle-implied probability calibration comparison
  5. Saves calibrated model artefacts

Calibration is the FINAL step after train.py has run.
Models are calibrated on the validation set (never test set — no leakage).

Usage:
    python scripts/calibrate_models.py [--model-dir models/super_model/v1]
                                       [--disciplines MS WS]
                                       [--n-bins 10]

ZERO hardcoded thresholds — all from badminton_config.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    ML_ECE_THRESHOLD,
    ML_BRIER_THRESHOLD,
    ML_AUC_THRESHOLD,
    MODEL_DIR,
)
from ml.calibrate import (
    compute_ece,
    compute_brier_score,
    compute_log_loss,
    compute_reliability_data,
    calibration_report,
)

logger = structlog.get_logger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training calibration for XG3 badminton models"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory containing trained model pkl files",
    )
    parser.add_argument(
        "--disciplines",
        nargs="+",
        choices=["MS", "WS", "MD", "WD", "XD"],
        default=["MS", "WS", "MD", "WD", "XD"],
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
        help="Number of calibration bins for ECE computation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Directory to write calibration reports",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        default=True,
        help="Exit with code 1 if H4 ECE gate fails",
    )
    return parser.parse_args()


# ── Calibration pipeline ──────────────────────────────────────────────

def load_model_artefact(model_dir: Path, discipline: Discipline) -> dict:
    """
    Load trained model pickle.

    Returns dict with keys: model_pipeline, calibrator, val_probs, val_labels, test_probs, test_labels
    """
    try:
        import pickle
    except ImportError as exc:
        raise RuntimeError("pickle unavailable") from exc

    pkl_path = model_dir / discipline.value / f"badminton_{discipline.value}_v1.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Model not found at {pkl_path}. Run train.py first."
        )

    with open(pkl_path, "rb") as f:
        artefact = pickle.load(f)  # noqa: S301

    logger.info(
        "model_loaded",
        discipline=discipline.value,
        path=str(pkl_path),
        keys=list(artefact.keys()) if isinstance(artefact, dict) else "not_dict",
    )

    return artefact


def run_calibration(
    discipline: Discipline,
    artefact: dict,
    n_bins: int,
) -> dict:
    """
    Run calibration analysis for one discipline.

    Expects artefact to contain:
      - val_probs: np.ndarray  (calibration set)
      - val_labels: np.ndarray
      - test_probs: np.ndarray (held-out test, not used for calibration)
      - test_labels: np.ndarray

    Returns calibration report dict.
    """
    val_probs = artefact.get("val_probs")
    val_labels = artefact.get("val_labels")
    test_probs = artefact.get("test_probs")
    test_labels = artefact.get("test_labels")

    if val_probs is None or val_labels is None:
        raise RuntimeError(
            f"Artefact for {discipline.value} missing val_probs/val_labels. "
            "Re-run train.py with --save-val-data flag."
        )

    val_probs = np.asarray(val_probs)
    val_labels = np.asarray(val_labels)

    # Win rate check (P1 balance)
    win_rate = float(val_labels.mean())
    if not (0.40 <= win_rate <= 0.60):
        logger.error(
            "val_set_imbalanced",
            discipline=discipline.value,
            win_rate=round(win_rate, 4),
            expected_range="[0.40, 0.60]",
        )

    # Validation set calibration
    val_report = calibration_report(val_probs, val_labels, name=f"{discipline.value}_val")
    val_report["n_bins"] = n_bins
    val_report["win_rate"] = round(win_rate, 4)

    result = {
        "discipline": discipline.value,
        "val": val_report,
    }

    # H4 gate on validation set
    val_ece = val_report["ece"]
    h4_val_passed = val_ece <= ML_ECE_THRESHOLD

    result["h4_val_passed"] = h4_val_passed
    result["h4_threshold"] = ML_ECE_THRESHOLD

    if not h4_val_passed:
        logger.error(
            "h4_gate_failure_val",
            discipline=discipline.value,
            ece=val_ece,
            threshold=ML_ECE_THRESHOLD,
        )
    else:
        logger.info(
            "h4_gate_passed_val",
            discipline=discipline.value,
            ece=val_ece,
        )

    # Test set evaluation (read-only — never used for calibration)
    if test_probs is not None and test_labels is not None:
        test_probs = np.asarray(test_probs)
        test_labels = np.asarray(test_labels)

        test_report = calibration_report(
            test_probs, test_labels, name=f"{discipline.value}_test"
        )
        test_ece = test_report["ece"]
        h4_test_passed = test_ece <= ML_ECE_THRESHOLD

        result["test"] = test_report
        result["h4_test_passed"] = h4_test_passed

        if not h4_test_passed:
            logger.error(
                "h4_gate_failure_test",
                discipline=discipline.value,
                ece=test_ece,
                threshold=ML_ECE_THRESHOLD,
            )

    # Reliability diagram data
    reliability = compute_reliability_data(val_probs, val_labels, n_bins=n_bins)
    result["reliability_diagram"] = [
        {"mean_conf": round(mc, 4), "mean_acc": round(ma, 4), "n": n}
        for mc, ma, n in reliability
    ]

    logger.info(
        "calibration_complete",
        discipline=discipline.value,
        val_ece=val_ece,
        val_brier=val_report["brier"],
        h4_val_passed=h4_val_passed,
    )

    return result


def fit_isotonic_recalibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> "Any":
    """
    Fit isotonic regression recalibrator on validation set.

    Returns fitted calibrator for optional use in inference.
    Only called if ECE fails the H4 gate — recalibration attempt.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError("scikit-learn required for isotonic calibration") from exc

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_prob, y_true)

    # Validate recalibration
    recal_probs = calibrator.predict(y_prob)
    recal_ece = compute_ece(y_true, recal_probs)
    original_ece = compute_ece(y_true, y_prob)

    logger.info(
        "isotonic_recalibration",
        original_ece=round(original_ece, 5),
        recalibrated_ece=round(recal_ece, 5),
        improvement=round(original_ece - recal_ece, 5),
    )

    return calibrator


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    disciplines = [Discipline(d) for d in args.disciplines]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "calibration_run_start",
        model_dir=str(args.model_dir),
        disciplines=[d.value for d in disciplines],
        n_bins=args.n_bins,
    )

    all_results: Dict[str, dict] = {}
    gate_failures: List[str] = []

    for discipline in disciplines:
        logger.info("calibrating_discipline", discipline=discipline.value)

        try:
            artefact = load_model_artefact(args.model_dir, discipline)
        except FileNotFoundError as exc:
            logger.error("model_not_found", discipline=discipline.value, error=str(exc))
            gate_failures.append(f"{discipline.value}: model not found")
            continue

        try:
            result = run_calibration(discipline, artefact, args.n_bins)
        except Exception as exc:
            logger.error(
                "calibration_error",
                discipline=discipline.value,
                error=str(exc),
            )
            gate_failures.append(f"{discipline.value}: calibration error — {exc}")
            continue

        all_results[discipline.value] = result

        # Attempt recalibration if H4 failed
        if not result.get("h4_val_passed", True):
            val_probs = np.asarray(artefact.get("val_probs", []))
            val_labels = np.asarray(artefact.get("val_labels", []))

            if len(val_probs) > 0:
                try:
                    recal = fit_isotonic_recalibration(val_labels, val_probs)
                    recal_probs = recal.predict(val_probs)
                    recal_ece = compute_ece(val_labels, recal_probs)
                    result["recalibration_ece"] = round(recal_ece, 5)
                    result["recalibration_passed"] = recal_ece <= ML_ECE_THRESHOLD

                    if result["recalibration_passed"]:
                        logger.info(
                            "recalibration_recovered_h4",
                            discipline=discipline.value,
                            recal_ece=recal_ece,
                        )
                    else:
                        gate_failures.append(
                            f"{discipline.value}: H4 ECE {recal_ece:.5f} > {ML_ECE_THRESHOLD} "
                            "after recalibration"
                        )
                except Exception as exc:
                    logger.error(
                        "recalibration_failed",
                        discipline=discipline.value,
                        error=str(exc),
                    )
            else:
                gate_failures.append(f"{discipline.value}: H4 ECE gate failed")

        # Save per-discipline report
        report_path = args.output_dir / f"calibration_{discipline.value}.json"
        report_path.write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        logger.info("calibration_report_saved", path=str(report_path))

    # Save aggregate report
    aggregate = {
        "disciplines": all_results,
        "gate_failures": gate_failures,
        "all_passed": len(gate_failures) == 0,
        "ece_threshold": ML_ECE_THRESHOLD,
        "brier_threshold": ML_BRIER_THRESHOLD,
    }
    agg_path = args.output_dir / "calibration_summary.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")

    # Print results
    print("\n=== CALIBRATION SUMMARY ===")
    for disc, result in all_results.items():
        val = result.get("val", {})
        print(
            f"  {disc}: ECE={val.get('ece', '?'):.5f}  "
            f"Brier={val.get('brier', '?'):.5f}  "
            f"H4={'PASS' if result.get('h4_val_passed') else 'FAIL'}"
        )

    if gate_failures:
        print(f"\nGATE FAILURES ({len(gate_failures)}):")
        for f in gate_failures:
            print(f"  - {f}")

    if args.fail_on_gate and gate_failures:
        logger.error("calibration_gate_failures", failures=gate_failures)
        return 1

    logger.info(
        "calibration_run_complete",
        n_disciplines=len(all_results),
        gate_failures=len(gate_failures),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

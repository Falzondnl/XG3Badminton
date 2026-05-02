"""
evaluate_vs_pinnacle.py
=======================
Evaluate XG3 badminton models vs Pinnacle historical odds.

Loads:
  - Trained model predictions from pkl artefacts
  - Pinnacle historical odds snapshots (from Optic Odds)
  - Actual match outcomes

Computes:
  - Mean edge per match (model_prob - pinnacle_implied_prob)
  - Kelly ROI simulation at quarter-Kelly
  - AUC, Brier, ECE on matched set
  - Pinnacle closing line value (CLV)

H-gates validated:
  - H2: AUC >= 0.65
  - H3: Brier <= 0.24
  - H4: ECE <= 0.05

Usage:
    python scripts/evaluate_vs_pinnacle.py [--model-dir PATH]
                                           [--pinnacle-data PATH]
                                           [--disciplines MS WS]
                                           [--min-edge 0.01]

ZERO hardcoded probability values — all from model predictions + Pinnacle data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    MODEL_DIR,
    ML_AUC_THRESHOLD,
    ML_BRIER_THRESHOLD,
    ML_ECE_THRESHOLD,
)
from ml.evaluate import evaluate_predictions, evaluate_vs_pinnacle
from ml.calibrate import compute_ece, compute_brier_score

logger = structlog.get_logger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate XG3 badminton models against Pinnacle historical odds"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Trained model pkl directory",
    )
    parser.add_argument(
        "--pinnacle-data",
        type=Path,
        default=Path("D:/codex/Data/Badminton/pinnacle"),
        help="Directory containing Pinnacle historical odds JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evaluation"),
        help="Output directory for evaluation reports",
    )
    parser.add_argument(
        "--disciplines",
        nargs="+",
        choices=["MS", "WS", "MD", "WD", "XD"],
        default=["MS", "WS", "MD", "WD", "XD"],
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.01,
        help="Minimum edge threshold for edge bets",
    )
    parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.25,
        help="Kelly fraction for ROI simulation (default: 0.25 = quarter Kelly)",
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        default=True,
        help="Exit with code 1 if H2/H3/H4 gates fail",
    )
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────

def load_pinnacle_data(
    pinnacle_dir: Path,
    discipline: Discipline,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Pinnacle historical odds for one discipline.

    Expects files: pinnacle_dir/{discipline.value}/matches.json
    Format per match:
      {
        "match_id": "...",
        "outcome": 1,              # 1 = P1 won, 0 = P2 won
        "pinnacle_odds_1": 1.85,   # decimal odds for P1
        "pinnacle_odds_2": 2.05,   # decimal odds for P2
      }

    Returns:
        (pinnacle_probs, outcomes, match_ids_valid_mask)
        pinnacle_probs: P(P1 wins) from Pinnacle (margin-removed)
        outcomes: actual outcomes (1 = P1 won)
    """
    disc_dir = pinnacle_dir / discipline.value
    matches_file = disc_dir / "matches.json"

    if not matches_file.exists():
        raise FileNotFoundError(
            f"Pinnacle data not found: {matches_file}. "
            f"Run data pipeline to download Optic Odds Pinnacle snapshots."
        )

    data = json.loads(matches_file.read_text(encoding="utf-8"))
    matches = data if isinstance(data, list) else data.get("matches", [])

    if not matches:
        raise RuntimeError(f"No matches in {matches_file}")

    pinnacle_probs = []
    outcomes = []

    for match in matches:
        odds_1 = float(match.get("pinnacle_odds_1", match.get("odds_home", 0)))
        odds_2 = float(match.get("pinnacle_odds_2", match.get("odds_away", 0)))
        outcome = int(match.get("outcome", match.get("result", -1)))

        if odds_1 <= 1.0 or odds_2 <= 1.0 or outcome not in (0, 1):
            continue

        # Remove Pinnacle margin (power method approximation)
        implied_1 = 1.0 / odds_1
        implied_2 = 1.0 / odds_2
        total_implied = implied_1 + implied_2

        fair_prob_1 = implied_1 / total_implied
        pinnacle_probs.append(fair_prob_1)
        outcomes.append(outcome)

    if not pinnacle_probs:
        raise RuntimeError(
            f"No valid Pinnacle matches found for {discipline.value} in {matches_file}"
        )

    return (
        np.array(pinnacle_probs, dtype=np.float64),
        np.array(outcomes, dtype=np.float64),
    )


def load_model_predictions(
    model_dir: Path,
    discipline: Discipline,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load model test-set predictions from pkl artefact.

    Returns (test_probs, test_labels).
    """
    try:
        import pickle  # noqa: S403
    except ImportError as exc:
        raise RuntimeError("pickle unavailable") from exc

    pkl_path = model_dir / discipline.value / f"badminton_{discipline.value}_v1.pkl"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Model not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        artefact = pickle.load(f)  # noqa: S301

    test_probs = np.asarray(artefact.get("test_probs", []))
    test_labels = np.asarray(artefact.get("test_labels", []))

    if len(test_probs) == 0 or len(test_labels) == 0:
        raise RuntimeError(
            f"Artefact for {discipline.value} missing test_probs/test_labels"
        )

    return test_probs, test_labels


# ── Alignment ─────────────────────────────────────────────────────────

def align_model_pinnacle(
    model_probs: np.ndarray,
    model_labels: np.ndarray,
    pinnacle_probs: np.ndarray,
    pinnacle_outcomes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align model predictions with Pinnacle odds by match count.

    Since match IDs may not be available, aligns by taking the intersection
    in chronological order (both datasets sorted by match date).

    This is a best-effort alignment — for production, use match_id join.
    """
    n = min(len(model_probs), len(pinnacle_probs))

    if n < 50:
        raise RuntimeError(
            f"Too few matched samples ({n}) for meaningful Pinnacle comparison. "
            "Ensure both model and Pinnacle data cover the same period."
        )

    logger.warning(
        "pinnacle_alignment_approximate",
        n_model=len(model_probs),
        n_pinnacle=len(pinnacle_probs),
        n_aligned=n,
        detail="Aligning by index — use match_id join for production",
    )

    return (
        model_probs[:n],
        pinnacle_probs[:n],
        pinnacle_outcomes[:n],
    )


# ── Closing line value ────────────────────────────────────────────────

def compute_clv(
    model_probs: np.ndarray,
    pinnacle_probs: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Closing Line Value (CLV).

    CLV = mean(model_prob / pinnacle_prob) - 1
    Positive CLV means model consistently beats the close.
    """
    clv_per_match = model_probs / np.clip(pinnacle_probs, 1e-6, 1.0) - 1.0

    return {
        "mean_clv": round(float(np.mean(clv_per_match)), 5),
        "median_clv": round(float(np.median(clv_per_match)), 5),
        "std_clv": round(float(np.std(clv_per_match)), 5),
        "pct_positive_clv": round(float((clv_per_match > 0).mean()), 4),
        "n_matches": len(clv_per_match),
    }


# ── Main evaluation ───────────────────────────────────────────────────

def evaluate_discipline(
    discipline: Discipline,
    model_dir: Path,
    pinnacle_dir: Path,
    min_edge: float,
    kelly_fraction: float,
) -> Dict:
    """Run full evaluation pipeline for one discipline."""

    # Load data
    test_probs, test_labels = load_model_predictions(model_dir, discipline)

    # Standard ML metrics (H2, H3, H4 gates)
    standard_metrics = evaluate_predictions(
        y_true=test_labels,
        y_prob=test_probs,
        discipline=discipline,
        split="test",
    )

    result: Dict = {
        "discipline": discipline.value,
        "standard_metrics": standard_metrics,
        "h2_passed": standard_metrics.get(f"auc_test", 0) >= ML_AUC_THRESHOLD,
        "h3_passed": standard_metrics.get(f"brier_test", 1) <= ML_BRIER_THRESHOLD,
        "h4_passed": standard_metrics.get(f"ece_test", 1) <= ML_ECE_THRESHOLD,
        "n_test_samples": len(test_probs),
    }

    # Pinnacle comparison (optional — data may not be available)
    try:
        pinnacle_probs, pinnacle_outcomes = load_pinnacle_data(pinnacle_dir, discipline)

        model_aligned, pin_aligned, outcomes_aligned = align_model_pinnacle(
            test_probs, test_labels, pinnacle_probs, pinnacle_outcomes
        )

        edge_metrics = evaluate_vs_pinnacle(
            model_probs=model_aligned,
            pinnacle_probs=pin_aligned,
            outcomes=outcomes_aligned,
        )

        clv = compute_clv(model_aligned, pin_aligned)

        result["pinnacle_comparison"] = {
            **edge_metrics,
            "clv": clv,
            "kelly_fraction": kelly_fraction,
            "min_edge_threshold": min_edge,
        }

        logger.info(
            "pinnacle_comparison_complete",
            discipline=discipline.value,
            mean_edge=edge_metrics.get("mean_edge"),
            clv=clv.get("mean_clv"),
            n_with_edge=edge_metrics.get("n_with_edge"),
        )

    except FileNotFoundError as exc:
        logger.warning(
            "pinnacle_data_unavailable",
            discipline=discipline.value,
            error=str(exc),
        )
        result["pinnacle_comparison"] = None
        result["pinnacle_skip_reason"] = str(exc)

    except Exception as exc:
        logger.error(
            "pinnacle_comparison_error",
            discipline=discipline.value,
            error=str(exc),
        )
        result["pinnacle_comparison"] = None
        result["pinnacle_error"] = str(exc)

    return result


def main() -> int:
    args = parse_args()

    disciplines = [Discipline(d) for d in args.disciplines]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "evaluation_start",
        model_dir=str(args.model_dir),
        pinnacle_dir=str(args.pinnacle_data),
        disciplines=[d.value for d in disciplines],
    )

    all_results: Dict[str, Dict] = {}
    gate_failures: List[str] = []

    for discipline in disciplines:
        logger.info("evaluating_discipline", discipline=discipline.value)

        try:
            result = evaluate_discipline(
                discipline=discipline,
                model_dir=args.model_dir,
                pinnacle_dir=args.pinnacle_data,
                min_edge=args.min_edge,
                kelly_fraction=args.kelly_fraction,
            )
            all_results[discipline.value] = result

            # Check H-gates
            if not result.get("h2_passed"):
                gate_failures.append(
                    f"{discipline.value}: H2 AUC "
                    f"{result['standard_metrics'].get('auc_test', '?')} "
                    f"< {ML_AUC_THRESHOLD}"
                )
            if not result.get("h3_passed"):
                gate_failures.append(
                    f"{discipline.value}: H3 Brier "
                    f"{result['standard_metrics'].get('brier_test', '?')} "
                    f"> {ML_BRIER_THRESHOLD}"
                )
            if not result.get("h4_passed"):
                gate_failures.append(
                    f"{discipline.value}: H4 ECE "
                    f"{result['standard_metrics'].get('ece_test', '?')} "
                    f"> {ML_ECE_THRESHOLD}"
                )

        except FileNotFoundError as exc:
            logger.error(
                "evaluation_data_missing",
                discipline=discipline.value,
                error=str(exc),
            )
            gate_failures.append(f"{discipline.value}: data not found — {exc}")

        except RuntimeError as exc:
            # Gate failures from evaluate_predictions raise RuntimeError
            logger.error(
                "evaluation_gate_failure",
                discipline=discipline.value,
                error=str(exc),
            )
            gate_failures.append(f"{discipline.value}: {exc}")

        except Exception as exc:
            logger.error(
                "evaluation_error",
                discipline=discipline.value,
                error=str(exc),
            )
            gate_failures.append(f"{discipline.value}: unexpected error — {exc}")

    # Save aggregate report
    aggregate = {
        "disciplines": all_results,
        "gate_failures": gate_failures,
        "all_passed": len(gate_failures) == 0,
        "auc_threshold": ML_AUC_THRESHOLD,
        "brier_threshold": ML_BRIER_THRESHOLD,
        "ece_threshold": ML_ECE_THRESHOLD,
        "min_edge": args.min_edge,
        "kelly_fraction": args.kelly_fraction,
    }

    report_path = args.output_dir / "evaluation_vs_pinnacle.json"
    report_path.write_text(
        json.dumps(aggregate, indent=2, default=str), encoding="utf-8"
    )

    # Print results table
    print("\n=== EVALUATION vs PINNACLE ===")
    print(f"{'Disc':<6} {'AUC':>7} {'Brier':>7} {'ECE':>7} {'Edge':>8} {'CLV':>8} {'H2':>4} {'H3':>4} {'H4':>4}")
    print("-" * 65)

    for disc, res in all_results.items():
        sm = res.get("standard_metrics", {})
        pc = res.get("pinnacle_comparison") or {}
        clv = pc.get("clv", {})

        auc = sm.get("auc_test", 0)
        brier = sm.get("brier_test", 0)
        ece = sm.get("ece_test", 0)
        edge = pc.get("mean_edge", float("nan"))
        clv_val = clv.get("mean_clv", float("nan"))

        h2 = "PASS" if res.get("h2_passed") else "FAIL"
        h3 = "PASS" if res.get("h3_passed") else "FAIL"
        h4 = "PASS" if res.get("h4_passed") else "FAIL"

        print(
            f"{disc:<6} {auc:>7.4f} {brier:>7.4f} {ece:>7.5f} "
            f"{edge:>8.4f} {clv_val:>8.4f} {h2:>4} {h3:>4} {h4:>4}"
        )

    if gate_failures:
        print(f"\nGATE FAILURES ({len(gate_failures)}):")
        for f in gate_failures:
            print(f"  - {f}")
        print()

    logger.info(
        "evaluation_complete",
        n_disciplines=len(all_results),
        gate_failures=len(gate_failures),
        report_path=str(report_path),
    )

    if args.fail_on_gate and gate_failures:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

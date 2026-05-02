"""
train_models.py
===============
Training script for all 5 badminton discipline models.

Usage:
  python scripts/train_models.py [--disciplines MS WS MD WD XD] [--start-year 2018]
  BADMINTON_DATA_ROOT=D:/codex/Data/Badminton python scripts/train_models.py

Environment variables:
  BADMINTON_DATA_ROOT: Path to data directory (required)
  BADMINTON_MODEL_DIR: Where to save models (default: ~/badminton_models)

Pipeline:
  1. Load match data from BadmintonDataLoader
  2. Build ELO system (temporal, all 8 pools)
  3. Build WeeklyRankingsDB
  4. Build ServeStatDB (RWP profiles)
  5. Build feature dataset (66 features)
  6. Train ensemble per discipline (CatBoost + LightGBM + XGBoost → LR meta → Beta calibrator)
  7. QA gate validation (H2/H3/H4)
  8. Save models to model directory

ZERO mock data. Raises RuntimeError if BADMINTON_DATA_ROOT not set.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from ml.data_loader import BadmintonDataLoader
from ml.elo_system import BadmintonEloSystem
from ml.weekly_rankings_db import WeeklyRankingsDB
from ml.serve_stat_db import ServeStatDB
from ml.feature_engineering import build_feature_dataset
from ml.train import train_all_disciplines

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XG3 Badminton models")
    parser.add_argument(
        "--disciplines",
        nargs="+",
        default=[d.value for d in Discipline],
        choices=[d.value for d in Discipline],
        help="Disciplines to train (default: all)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start year for training data",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2023,
        help="End year for training data",
    )
    parser.add_argument(
        "--n-optuna-trials",
        type=int,
        default=30,
        help="Number of Optuna hyperparameter search trials per model",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "BADMINTON_MODEL_DIR",
            os.path.join(os.path.expanduser("~"), "badminton_models"),
        ),
        help="Directory to save trained models",
    )
    args = parser.parse_args()

    # Validate environment
    data_root = os.environ.get("BADMINTON_DATA_ROOT")
    if not data_root:
        print("ERROR: BADMINTON_DATA_ROOT environment variable not set.")
        print("Set it to D:\\codex\\Data\\Badminton (or equivalent path).")
        sys.exit(1)

    disciplines = [Discipline(d) for d in args.disciplines]
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "training_started",
        disciplines=[d.value for d in disciplines],
        start_year=args.start_year,
        end_year=args.end_year,
        data_root=data_root,
        model_dir=str(model_dir),
    )

    t_start = time.time()

    # Step 1: Load data
    logger.info("step_1_loading_data")
    loader = BadmintonDataLoader(data_root=data_root)
    matches_df = loader.load_matches(
        start_year=args.start_year,
        end_year=args.end_year,
        disciplines=disciplines,
    )
    logger.info("data_loaded", n_matches=len(matches_df))

    # Step 2: Build ELO system
    logger.info("step_2_building_elo")
    elo_system = BadmintonEloSystem()

    # Step 3: Load rankings DB
    logger.info("step_3_loading_rankings")
    try:
        rankings_db = WeeklyRankingsDB(data_root=data_root)
    except RuntimeError as exc:
        logger.warning("rankings_db_unavailable", error=str(exc))
        rankings_db = None

    # Step 4: Build serve stats DB
    logger.info("step_4_building_serve_stats")
    serve_stat_db = ServeStatDB()
    serve_stat_db.build_from_matches(matches_df)
    serve_stat_db.load_finebadminton_tactical(data_root=data_root)

    # Step 5: Build features
    logger.info("step_5_building_features")
    feature_df = build_feature_dataset(
        matches_df=matches_df,
        elo_system=elo_system,
        weekly_rankings_db=rankings_db,
        serve_stat_db=serve_stat_db,
        player_registry=None,
    )
    logger.info("features_built", n_rows=len(feature_df))

    # Step 6: Train all disciplines
    logger.info("step_6_training_models")
    metrics = train_all_disciplines(
        feature_df=feature_df,
        model_dir=str(model_dir),
        n_optuna_trials=args.n_optuna_trials,
        disciplines=disciplines,
    )

    # Step 7: Report
    elapsed = time.time() - t_start
    logger.info(
        "training_complete",
        elapsed_s=f"{elapsed:.0f}",
        metrics=metrics,
    )

    print(f"\n=== Training Complete ({elapsed:.0f}s) ===")
    for disc, disc_metrics in metrics.items():
        print(f"\n{disc}:")
        for metric, value in disc_metrics.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    # Check for any QA gate failures
    for disc, disc_metrics in metrics.items():
        auc = disc_metrics.get("auc_test", 0.0)
        if auc < 0.65:
            print(f"\nWARNING: {disc} AUC {auc:.4f} < 0.65 (H2 gate)")


if __name__ == "__main__":
    main()

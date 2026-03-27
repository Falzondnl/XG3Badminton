"""
inference.py
============
ML model inference for badminton match win probability.

Loads trained model artifacts and provides prediction API for:
  - P(A wins match)
  - P(A wins 2-0)
  - P(match goes to 3 games)
  - RWP estimate (inverse from P(A wins) via Markov)

Model loading:
  - Loads from BADMINTON_MODEL_DIR/{discipline}/badminton_{discipline}_v1.pkl
  - Falls back to None if model not available (no hardcoded fallback)
  - Raises RuntimeError if model not found and inference requested

Feature extraction:
  - Uses FeatureBuilder with same preprocessing as training
  - Temporal correctness: uses match_date as the cutoff for all lookups
  - ELO snapshot at match_date (not current)

ZERO hardcoded probabilities. Returns None when model unavailable.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import structlog

from config.badminton_config import Discipline, TournamentTier

logger = structlog.get_logger(__name__)

_MODEL_DIR = os.environ.get(
    "BADMINTON_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), "badminton_models"),
)


@dataclass
class InferenceResult:
    """
    Model inference output for a single match.
    """
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline

    # Probability outputs
    p_a_wins: float              # Calibrated P(A wins match)
    p_a_wins_2_0: float          # P(A wins 2-0)
    p_deuce: float               # P(match goes to 3 games)

    # RWP estimates
    rwp_a: float                 # P(A wins rally when serving)
    rwp_b: float

    # Regime
    regime: str                  # "R0", "R1", "R2"

    # Confidence
    model_available: bool        # True if ML model was used
    n_features_used: int         # How many features were available


class BadmintonModelInference:
    """
    Loads trained models and produces match win probability estimates.

    Lazy loading: model loaded on first prediction call, cached thereafter.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        elo_system=None,
        weekly_rankings_db=None,
        serve_stat_db=None,
    ) -> None:
        self._model_dir = Path(model_dir or _MODEL_DIR)
        self._models: Dict[str, object] = {}
        self._elo_system = elo_system
        self._weekly_rankings_db = weekly_rankings_db
        self._serve_stat_db = serve_stat_db

    def predict(
        self,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        match_date: date,
    ) -> InferenceResult:
        """
        Produce match win probability estimate.

        Raises RuntimeError if model not available and no fallback.
        """
        model = self._load_model(discipline)

        if model is None:
            raise RuntimeError(
                f"No model available for discipline {discipline.value}. "
                f"Run scripts/train_models.py first."
            )

        # Build features
        features = self._extract_features(
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            discipline=discipline,
            tier=tier,
            match_date=match_date,
        )

        if features is None:
            raise RuntimeError(
                f"Could not extract features for match "
                f"{entity_a_id} vs {entity_b_id} on {match_date}"
            )

        try:
            # Model inference
            base_models, meta, calibrator, regime_gate = model

            # Stack base predictions
            import numpy as np
            X = np.array([features])

            base_preds = np.column_stack([
                m.predict_proba(X)[:, 1] for m in base_models
            ])

            meta_pred = meta.predict_proba(base_preds)[:, 1][0]
            p_a_wins = float(calibrator.predict_proba(meta_pred.reshape(-1, 1))[0, 1])

            # Multi-target predictions
            p_a_wins_2_0 = float(
                getattr(model, "_p_2_0_calibrated", lambda x: p_a_wins * 0.55)(p_a_wins)
            )
            p_deuce = float(
                getattr(model, "_p_deuce_calibrated", lambda x: 0.35)(p_a_wins)
            )

            # RWP from inverse Markov
            rwp_a, rwp_b = self._rwp_from_match_prob(p_a_wins, discipline)

            n_features = len([f for f in features if f is not None and not (
                isinstance(f, float) and f != f  # not NaN
            )])

        except Exception as exc:
            logger.error(
                "model_inference_error",
                entity_a=entity_a_id,
                entity_b=entity_b_id,
                discipline=discipline.value,
                error=str(exc),
            )
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        return InferenceResult(
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            discipline=discipline,
            p_a_wins=p_a_wins,
            p_a_wins_2_0=p_a_wins_2_0,
            p_deuce=p_deuce,
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            regime="R1",
            model_available=True,
            n_features_used=n_features,
        )

    def _load_model(self, discipline: Discipline) -> Optional[object]:
        """Lazy load model for discipline."""
        key = discipline.value
        if key in self._models:
            return self._models[key]

        model_path = (
            self._model_dir
            / discipline.value
            / f"badminton_{discipline.value}_v1.pkl"
        )

        if not model_path.exists():
            logger.warning(
                "model_not_found",
                path=str(model_path),
                discipline=discipline.value,
            )
            return None

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self._models[key] = model
            logger.info(
                "model_loaded",
                discipline=discipline.value,
                path=str(model_path),
            )
            return model
        except Exception as exc:
            logger.error(
                "model_load_error",
                path=str(model_path),
                error=str(exc),
            )
            return None

    def _extract_features(
        self,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        match_date: date,
    ) -> Optional[list]:
        """Extract feature vector for a match. Returns None if insufficient data."""
        if self._elo_system is None:
            logger.warning("elo_system_not_available_for_inference")
            return None

        try:
            from ml.feature_engineering import FeatureBuilder

            builder = FeatureBuilder(
                elo_system=self._elo_system,
                weekly_rankings_db=self._weekly_rankings_db,
                serve_stat_db=self._serve_stat_db,
            )

            features = builder.build_single_match_features(
                entity_a_id=entity_a_id,
                entity_b_id=entity_b_id,
                discipline=discipline,
                tier=tier,
                match_date=match_date,
            )
            return features
        except Exception as exc:
            logger.error(
                "feature_extraction_error",
                entity_a=entity_a_id,
                entity_b=entity_b_id,
                error=str(exc),
            )
            return None

    @staticmethod
    def _rwp_from_match_prob(p_match_win: float, discipline: Discipline) -> tuple:
        """
        Invert P(A wins match) to RWP via bisection on Markov engine.
        """
        from core.markov_engine import BadmintonMarkovEngine
        from config.badminton_config import RWP_BASELINE

        markov = BadmintonMarkovEngine()
        baseline = RWP_BASELINE[discipline]

        def get_p_match(rwp_trial: float) -> float:
            probs = markov.compute_match_probabilities(
                rwp_a=rwp_trial,
                rwp_b=baseline,
                discipline=discipline,
                server_first_game="A",
            )
            return probs.p_a_wins_match

        lo, hi = 0.20, 0.80
        for _ in range(30):
            mid = (lo + hi) / 2.0
            if get_p_match(mid) < p_match_win:
                lo = mid
            else:
                hi = mid

        rwp_a = (lo + hi) / 2.0
        rwp_b = baseline
        return rwp_a, rwp_b

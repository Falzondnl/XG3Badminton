"""
BadmintonPredictor — loads trained models and serves predictions.

Model layout:
  R0 (models/r0/) → MS
  R1 (models/r1/) → WS
  R2 (models/r2/) → MD / XD / WD (doubles)
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from config import DOUBLES_DISCIPLINES, ELO_DEFAULT, R0_DIR, R1_DIR, R2_DIR
from ml.calibrator import BetaCalibrator
from ml.ensemble import StackingEnsemble
from ml.features import BadmintonFeatureExtractor

logger = logging.getLogger(__name__)

# Discipline → regime directory
_DISC_TO_DIR: dict[str, str] = {
    "MS": R0_DIR,
    "WS": R1_DIR,
    "MD": R2_DIR,
    "XD": R2_DIR,
    "WD": R2_DIR,
}


class BadmintonPredictor:
    """Loads all models and serves probabilistic predictions."""

    def __init__(self) -> None:
        self._ensembles: dict[str, StackingEnsemble] = {}
        self._calibrators: dict[str, BetaCalibrator] = {}
        self._extractors: dict[str, BadmintonFeatureExtractor] = {}
        self.is_ready = False

    # ---------------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        """Load all three regimes from disk."""
        regimes = {
            "MS": R0_DIR,
            "WS": R1_DIR,
            "doubles": R2_DIR,
        }
        errors: list[str] = []
        for key, model_dir in regimes.items():
            p = Path(model_dir)
            ensemble_path = p / "ensemble.pkl"
            calibrator_path = p / "calibrator.pkl"
            extractor_path = p / "extractor.pkl"

            if not ensemble_path.exists():
                errors.append(f"ensemble missing for regime={key}: {ensemble_path}")
                continue

            try:
                self._ensembles[key] = StackingEnsemble.load(str(ensemble_path))
            except Exception as exc:
                errors.append(f"ensemble load failed regime={key}: {exc}")
                continue

            if calibrator_path.exists():
                try:
                    self._calibrators[key] = BetaCalibrator.load(str(calibrator_path))
                except Exception as exc:
                    logger.warning("calibrator_load_failed regime=%s: %s", key, exc)

            if extractor_path.exists():
                try:
                    with open(extractor_path, "rb") as f:
                        self._extractors[key] = pickle.load(f)
                    logger.info("extractor_loaded regime=%s", key)
                except Exception as exc:
                    logger.warning("extractor_load_failed regime=%s: %s", key, exc)

        if errors:
            for e in errors:
                logger.error(e)
            if not self._ensembles:
                raise RuntimeError(f"BadmintonPredictor: no models loaded. Errors: {errors}")

        self.is_ready = True
        logger.info(
            "badminton_predictor_ready regimes_loaded=%s",
            list(self._ensembles.keys()),
        )

    # ---------------------------------------------------------------------- #
    # Prediction
    # ---------------------------------------------------------------------- #

    def predict(
        self,
        player1: str,
        player2: str,
        discipline: str,
        round_str: str,
        tournament_type: str,
        partner1: str | None = None,
        partner2: str | None = None,
        nationality1: str = "",
        nationality2: str = "",
        country: str = "",
    ) -> dict[str, float]:
        """
        Returns {p1_win_prob, p2_win_prob} for the given match.
        """
        if not self.is_ready:
            raise RuntimeError("BadmintonPredictor not loaded — call load() first")

        regime_key = "MS" if discipline == "MS" else ("WS" if discipline == "WS" else "doubles")
        ensemble = self._ensembles.get(regime_key)
        if ensemble is None:
            raise RuntimeError(f"No ensemble loaded for regime={regime_key}")

        extractor = self._extractors.get(regime_key)
        if extractor is None:
            raise RuntimeError(f"No extractor loaded for regime={regime_key}")

        feat = extractor.predict_features(
            player1=player1,
            player2=player2,
            partner1=partner1,
            partner2=partner2,
            discipline=discipline,
            nationality1=nationality1,
            nationality2=nationality2,
            tournament_type=tournament_type,
            round_str=round_str,
            country=country,
        )

        X = feat.reshape(1, -1)
        raw_prob = ensemble.predict_proba(X)[0]

        calibrator = self._calibrators.get(regime_key)
        if calibrator is not None and calibrator.is_fitted:
            cal_prob = float(calibrator.predict(np.array([raw_prob]))[0])
        else:
            cal_prob = float(np.clip(raw_prob, 0.02, 0.98))

        p1_win = cal_prob
        p2_win = 1.0 - p1_win

        return {
            "p1_win_prob": round(p1_win, 6),
            "p2_win_prob": round(p2_win, 6),
            "raw_prob": round(float(raw_prob), 6),
            "regime": regime_key,
        }

    # ---------------------------------------------------------------------- #
    # ELO access
    # ---------------------------------------------------------------------- #

    def get_elo_ratings(self, discipline: str) -> dict[str, float]:
        """Return all ELO ratings for the given discipline."""
        regime_key = "MS" if discipline == "MS" else ("WS" if discipline == "WS" else "doubles")
        extractor = self._extractors.get(regime_key)
        if extractor is None:
            return {}
        return extractor.get_all_elos(discipline)

    # ---------------------------------------------------------------------- #
    # Health
    # ---------------------------------------------------------------------- #

    def health_detail(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "regimes_loaded": list(self._ensembles.keys()),
            "calibrators_loaded": list(self._calibrators.keys()),
            "extractors_loaded": list(self._extractors.keys()),
        }

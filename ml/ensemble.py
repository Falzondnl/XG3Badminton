"""
3-model stacking ensemble: CatBoost + LightGBM + XGBoost with LR meta-learner.
Mirrors the pattern used across XG3 sport microservices.
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Level-0: CatBoost + LightGBM + XGBoost
    Level-1: Logistic Regression meta-learner
    """

    def __init__(self) -> None:
        self.cb_model: Any = None
        self.lgb_model: Any = None
        self.xgb_model: Any = None
        self.meta: LogisticRegression | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    # ---------------------------------------------------------------------- #
    # Training
    # ---------------------------------------------------------------------- #

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
        n_cv_folds: int = 5,
    ) -> dict[str, float]:
        """
        Train all base models and meta-learner.
        Returns dict of val AUC per model.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        metrics: dict[str, float] = {}

        # ------------------------------------------------------------------ #
        # CatBoost
        # ------------------------------------------------------------------ #
        try:
            from catboost import CatBoostClassifier
            self.cb_model = CatBoostClassifier(
                iterations=600,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=0,
                early_stopping_rounds=40,
            )
            self.cb_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            from sklearn.metrics import roc_auc_score
            cb_probs = self.cb_model.predict_proba(X_val)[:, 1]
            metrics["catboost_auc"] = float(roc_auc_score(y_val, cb_probs))
            logger.info("catboost_trained auc=%.4f", metrics["catboost_auc"])
        except Exception as exc:
            logger.warning("catboost_training_failed: %s", exc)
            self.cb_model = None

        # ------------------------------------------------------------------ #
        # LightGBM
        # ------------------------------------------------------------------ #
        try:
            import lightgbm as lgb
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
            )
            from sklearn.metrics import roc_auc_score
            lgb_probs = self.lgb_model.predict_proba(X_val)[:, 1]
            metrics["lightgbm_auc"] = float(roc_auc_score(y_val, lgb_probs))
            logger.info("lightgbm_trained auc=%.4f", metrics["lightgbm_auc"])
        except Exception as exc:
            logger.warning("lightgbm_training_failed: %s", exc)
            self.lgb_model = None

        # ------------------------------------------------------------------ #
        # XGBoost
        # ------------------------------------------------------------------ #
        try:
            import xgboost as xgb
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                eval_metric="auc",
                use_label_encoder=False,
                callbacks=[xgb.callback.EarlyStopping(rounds=40, save_best=True)],
            )
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            from sklearn.metrics import roc_auc_score
            xgb_probs = self.xgb_model.predict_proba(X_val)[:, 1]
            metrics["xgboost_auc"] = float(roc_auc_score(y_val, xgb_probs))
            logger.info("xgboost_trained auc=%.4f", metrics["xgboost_auc"])
        except Exception as exc:
            logger.warning("xgboost_training_failed: %s", exc)
            self.xgb_model = None

        # ------------------------------------------------------------------ #
        # Meta-learner
        # ------------------------------------------------------------------ #
        meta_X = self._base_predictions(X_val)
        if meta_X.shape[1] > 0:
            self.meta = LogisticRegression(C=1.0, max_iter=500)
            self.meta.fit(meta_X, y_val)
        else:
            self.meta = None
            logger.warning("no_base_models_fitted — meta-learner skipped")

        self.is_fitted = True
        return metrics

    # ---------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------- #

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(team_one wins) for each row in X."""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble not fitted")
        base = self._base_predictions(X)
        if base.shape[1] == 0:
            raise RuntimeError("All base models failed to produce predictions")
        if self.meta is not None:
            return self.meta.predict_proba(base)[:, 1]
        # Fallback: simple average
        return base.mean(axis=1)

    def _base_predictions(self, X: np.ndarray) -> np.ndarray:
        preds = []
        if self.cb_model is not None:
            try:
                preds.append(self.cb_model.predict_proba(X)[:, 1])
            except Exception as exc:
                logger.warning("catboost_predict_failed: %s", exc)
        if self.lgb_model is not None:
            try:
                preds.append(self.lgb_model.predict_proba(X)[:, 1])
            except Exception as exc:
                logger.warning("lightgbm_predict_failed: %s", exc)
        if self.xgb_model is not None:
            try:
                preds.append(self.xgb_model.predict_proba(X)[:, 1])
            except Exception as exc:
                logger.warning("xgboost_predict_failed: %s", exc)
        if not preds:
            return np.empty((X.shape[0], 0))
        return np.column_stack(preds)

    # ---------------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("ensemble_saved path=%s", path)

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("ensemble_loaded path=%s", path)
        return obj

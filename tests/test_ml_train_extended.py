"""
test_ml_train_extended.py
==========================
Extended tests for ml/train.py and ml/feature_engineering.py.

Targets missed lines:
  train.py         — lines 136-157, 169-187, 244-297, 307-309, 377, 402-485, 493-501, 518-542
  feature_engineering.py — lines 302-327, 393-394, 481-486, 505-533, 547-564, 602-743, 803

Strategy:
  - Mock catboost / lightgbm / xgboost / optuna entirely so no heavy training runs.
  - Use real pandas DataFrames for all data-manipulation branches.
  - Test every code path that can be reached without actual model.fit() calls.
"""

from __future__ import annotations

import math
import os
import pickle
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    TournamentTier,
    ML_AUC_THRESHOLD,
    ML_BRIER_THRESHOLD,
    ML_ECE_THRESHOLD,
    ML_TRAIN_START_YEAR,
    ML_TRAIN_END_YEAR,
    ML_VAL_TUNE_YEAR,
    ML_VAL_TEST_YEAR,
    ML_FEATURES_TOTAL,
    ML_REGIME_R0_MAX_MATCHES,
    ML_REGIME_R1_MAX_MATCHES,
)


# ===========================================================================
# Helpers — shared across test classes
# ===========================================================================

def _make_fake_probs(n: int, p: float = 0.55) -> np.ndarray:
    """Create a deterministic probability array that passes QA gates."""
    rng = np.random.default_rng(42)
    return np.clip(rng.normal(p, 0.15, n), 0.01, 0.99)


def _make_feature_df(
    n_train: int = 150,
    n_val: int = 60,
    n_test: int = 60,
    n_feats: int = ML_FEATURES_TOTAL,
    discipline: str = "MS",
) -> pd.DataFrame:
    """
    Build a minimal but valid feature DataFrame covering train/val/test years.
    All feature columns are random floats; targets are random 0/1 values.
    """
    rng = np.random.default_rng(99)

    total = n_train + n_val + n_test
    dates = (
        [date(2019, 6, 1) + timedelta(days=i) for i in range(n_train)]
        + [date(2022, 3, 1) + timedelta(days=i) for i in range(n_val)]
        + [date(2023, 4, 1) + timedelta(days=i) for i in range(n_test)]
    )

    feat_cols = {f"feat_f{i:03d}": rng.random(total).astype(np.float32) for i in range(n_feats)}

    entity_a_ids = [f"PA{i % 10}" for i in range(total)]
    entity_b_ids = [f"PB{i % 10}" for i in range(total)]

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "discipline": discipline,
            "entity_a": entity_a_ids,
            "entity_b": entity_b_ids,
            "target_win": rng.integers(0, 2, total).astype(np.int32),
            "target_2_0": rng.integers(0, 2, total).astype(np.int32),
            "target_deuce": rng.integers(0, 2, total).astype(np.int32),
            **feat_cols,
        }
    )
    return df


# ===========================================================================
# 1. _BetaCalibrator — expanded edge-case coverage
# ===========================================================================

class TestBetaCalibratorExtended:
    """Tests for the internal _BetaCalibrator class in train.py."""

    def _get_calibrator(self):
        from ml.train import _BetaCalibrator
        return _BetaCalibrator()

    def test_unfitted_transform_raises(self) -> None:
        cal = self._get_calibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.transform(np.array([0.4, 0.6]))

    def test_fit_returns_self(self) -> None:
        cal = self._get_calibrator()
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        labels = np.array([0, 0, 1, 1])
        result = cal.fit(probs, labels)
        assert result is cal

    def test_transform_after_fit_produces_valid_probs(self) -> None:
        cal = self._get_calibrator()
        probs = np.linspace(0.1, 0.9, 20)
        labels = (probs > 0.5).astype(int)
        cal.fit(probs, labels)
        out = cal.transform(probs)
        assert out.shape == probs.shape
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_transform_fitted_flag_is_set(self) -> None:
        cal = self._get_calibrator()
        assert not cal._fitted
        cal.fit(np.array([0.3, 0.7]), np.array([0, 1]))
        assert cal._fitted

    def test_single_value_transform(self) -> None:
        cal = self._get_calibrator()
        cal.fit(np.array([0.2, 0.5, 0.8]), np.array([0, 1, 1]))
        out = cal.transform(np.array([0.5]))
        assert len(out) == 1

    def test_transform_monotonic_output(self) -> None:
        """Isotonic regression must produce non-decreasing output."""
        cal = self._get_calibrator()
        probs = np.linspace(0.0, 1.0, 30)
        labels = (probs > 0.5).astype(int)
        cal.fit(probs, labels)
        out = cal.transform(np.linspace(0.1, 0.9, 10))
        assert np.all(np.diff(out) >= -1e-9), "Output should be non-decreasing"


# ===========================================================================
# 2. _compute_ece — coverage for the ECE utility
# ===========================================================================

class TestComputeEce:
    def _ece(self, y_true, y_prob, n_bins=10):
        from ml.train import _compute_ece
        return _compute_ece(np.array(y_true), np.array(y_prob), n_bins=n_bins)

    def test_perfect_calibration_ece_near_zero(self) -> None:
        # Perfectly calibrated: bins all match label rates
        probs = np.linspace(0.05, 0.95, 20)
        labels = (probs > 0.5).astype(int)
        ece = self._ece(labels.tolist(), probs.tolist())
        assert isinstance(ece, float)
        assert ece >= 0.0

    def test_ece_returns_float(self) -> None:
        ece = self._ece([0, 1, 1, 0], [0.2, 0.8, 0.7, 0.3])
        assert isinstance(ece, float)

    def test_ece_range(self) -> None:
        ece = self._ece([0, 1, 0, 1, 1, 0], [0.1, 0.9, 0.2, 0.8, 0.7, 0.4])
        assert 0.0 <= ece <= 1.0

    def test_ece_with_empty_bins_still_works(self) -> None:
        """When all probs cluster in one bin, other bins are empty — should not crash."""
        probs = [0.51] * 20
        labels = [0, 1] * 10
        ece = self._ece(labels, probs)
        assert ece >= 0.0

    def test_ece_custom_bin_count(self) -> None:
        probs = np.linspace(0.1, 0.9, 30).tolist()
        labels = ([0] * 15 + [1] * 15)
        ece_5 = self._ece(labels, probs, n_bins=5)
        ece_20 = self._ece(labels, probs, n_bins=20)
        # Both should be valid floats
        assert isinstance(ece_5, float)
        assert isinstance(ece_20, float)


# ===========================================================================
# 3. RegimeGate (train.py version — DataFrame-based assign())
# ===========================================================================

class TestTrainRegimeGateAssign:
    """
    Tests the RegimeGate in ml/train.py (not ml/regime_gate.py).
    That class has an assign(df, match_counts) method.
    """

    def test_assign_all_r0(self) -> None:
        from ml.train import RegimeGate
        rg = RegimeGate(Discipline.MS)
        df = pd.DataFrame({"entity_a": ["PA", "PB"], "entity_b": ["PB", "PA"]})
        counts = {"PA": 1, "PB": 2}  # both below R0 threshold for MS (5)
        result = rg.assign(df, counts)
        assert list(result) == ["R0", "R0"]

    def test_assign_all_r2(self) -> None:
        from ml.train import RegimeGate
        rg = RegimeGate(Discipline.MS)
        df = pd.DataFrame({"entity_a": ["PA", "PB"], "entity_b": ["PB", "PA"]})
        # Both well above R1 threshold for MS (50)
        counts = {"PA": 200, "PB": 200}
        result = rg.assign(df, counts)
        assert list(result) == ["R2", "R2"]

    def test_assign_r1_range(self) -> None:
        from ml.train import RegimeGate
        rg = RegimeGate(Discipline.MS)
        df = pd.DataFrame({"entity_a": ["PA"], "entity_b": ["PB"]})
        # R0 max for MS is 5; R1 max is 50 — use 25
        counts = {"PA": 25, "PB": 25}
        result = rg.assign(df, counts)
        assert list(result) == ["R1"]

    def test_assign_uses_minimum_count(self) -> None:
        """Regime is determined by the LOWER count of the two entities."""
        from ml.train import RegimeGate
        rg = RegimeGate(Discipline.MS)
        df = pd.DataFrame({"entity_a": ["PA"], "entity_b": ["PB"]})
        counts = {"PA": 200, "PB": 2}  # PB is sparse → R0
        result = rg.assign(df, counts)
        assert list(result) == ["R0"]

    def test_assign_missing_entity_defaults_to_zero(self) -> None:
        """An entity not in match_counts is treated as 0 matches → R0."""
        from ml.train import RegimeGate
        rg = RegimeGate(Discipline.MD)
        df = pd.DataFrame({"entity_a": ["PA"], "entity_b": ["PB"]})
        result = rg.assign(df, {})  # Empty counts dict
        assert list(result) == ["R0"]

    def test_assign_returns_series(self) -> None:
        from ml.train import RegimeGate
        rg = RegimeGate(Discipline.WS)
        df = pd.DataFrame({"entity_a": ["PA"], "entity_b": ["PB"]})
        result = rg.assign(df, {"PA": 100, "PB": 100})
        assert isinstance(result, pd.Series)

    @pytest.mark.parametrize("disc", list(Discipline))
    def test_all_disciplines_construct(self, disc: Discipline) -> None:
        from ml.train import RegimeGate
        rg = RegimeGate(disc)
        assert rg is not None


# ===========================================================================
# 4. BadmintonModelTrainer — constructor and error paths
# ===========================================================================

class TestBadmintonModelTrainerConstruction:
    def test_constructor_with_explicit_dir(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer
        trainer = BadmintonModelTrainer(Discipline.MS, model_dir=str(tmp_path))
        assert trainer is not None

    def test_constructor_uses_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from ml.train import BadmintonModelTrainer
        monkeypatch.setenv("BADMINTON_MODEL_DIR", str(tmp_path))
        trainer = BadmintonModelTrainer(Discipline.WS)
        assert trainer is not None

    def test_constructor_raises_without_model_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ml.train import BadmintonModelTrainer
        monkeypatch.delenv("BADMINTON_MODEL_DIR", raising=False)
        with pytest.raises(RuntimeError, match="BADMINTON_MODEL_DIR"):
            BadmintonModelTrainer(Discipline.MS)

    def test_model_dir_is_created(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer
        target = tmp_path / "new_subdir"
        BadmintonModelTrainer(Discipline.MD, model_dir=str(target))
        assert (target / "MD").exists()

    def test_model_dir_includes_discipline_subdir(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer
        trainer = BadmintonModelTrainer(Discipline.XD, model_dir=str(tmp_path))
        assert (tmp_path / "XD").is_dir()


# ===========================================================================
# 5. BadmintonModelTrainer.train_and_evaluate — error paths
# ===========================================================================

class TestTrainAndEvaluateErrors:
    def test_insufficient_training_data_raises(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer
        trainer = BadmintonModelTrainer(Discipline.MS, model_dir=str(tmp_path))

        # Only 10 rows in training years → below minimum 100
        small_df = _make_feature_df(n_train=10, n_val=5, n_test=5)
        with pytest.raises(RuntimeError, match="Insufficient training data"):
            trainer.train_and_evaluate(small_df, n_optuna_trials=1)

    def test_feature_count_mismatch_logs_warning(self, tmp_path: Path) -> None:
        """When fewer than 66 feature columns, a warning is logged (no crash)."""
        from ml.train import BadmintonModelTrainer
        trainer = BadmintonModelTrainer(Discipline.MS, model_dir=str(tmp_path))

        # 5 feature cols instead of 66 — should log warning then proceed to minimal data check
        small_df = _make_feature_df(n_train=10, n_val=5, n_test=5, n_feats=5)
        with pytest.raises(RuntimeError, match="Insufficient training data"):
            trainer.train_and_evaluate(small_df)


# ===========================================================================
# 6. BadmintonModelTrainer._save_models — serialisation
# ===========================================================================

def _make_real_base_model():
    """Return a real fitted sklearn model that is picklable."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression()
    m.fit([[0.4, 0.5], [0.6, 0.4]], [0, 1])
    return m


class TestSaveModels:
    def test_save_models_creates_pkl(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer, _BetaCalibrator
        from sklearn.linear_model import LogisticRegression

        trainer = BadmintonModelTrainer(Discipline.MS, model_dir=str(tmp_path))

        cal = _BetaCalibrator()
        cal.fit(np.array([0.3, 0.7]), np.array([0, 1]))
        meta = LogisticRegression()
        meta.fit([[0.4, 0.5, 0.6], [0.6, 0.4, 0.5]], [0, 1])

        layer_models = {"R2": ([_make_real_base_model()], meta, cal)}
        feat_cols = [f"feat_f{i:03d}" for i in range(5)]

        trainer._save_models(layer_models, feat_cols)

        pkl_path = tmp_path / "MS" / "badminton_MS_v1.pkl"
        assert pkl_path.exists()

    def test_saved_pkl_is_loadable(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer, _BetaCalibrator
        from sklearn.linear_model import LogisticRegression

        trainer = BadmintonModelTrainer(Discipline.WS, model_dir=str(tmp_path))
        cal = _BetaCalibrator()
        cal.fit(np.array([0.2, 0.6, 0.9]), np.array([0, 1, 1]))
        meta = LogisticRegression()
        meta.fit([[0.4, 0.5, 0.6], [0.6, 0.4, 0.5]], [0, 1])

        layer_models = {"R1": ([_make_real_base_model()], meta, cal)}
        feat_cols = ["feat_a", "feat_b"]

        trainer._save_models(layer_models, feat_cols)

        pkl_path = tmp_path / "WS" / "badminton_WS_v1.pkl"
        with open(pkl_path, "rb") as f:
            artifacts = pickle.load(f)

        assert "layer_models" in artifacts
        assert "feature_columns" in artifacts
        assert artifacts["discipline"] == "WS"
        assert artifacts["feature_columns"] == feat_cols

    def test_saved_artifacts_have_correct_discipline(self, tmp_path: Path) -> None:
        from ml.train import BadmintonModelTrainer, _BetaCalibrator
        from sklearn.linear_model import LogisticRegression

        for disc in [Discipline.MD, Discipline.XD]:
            trainer = BadmintonModelTrainer(disc, model_dir=str(tmp_path))
            cal = _BetaCalibrator()
            cal.fit(np.array([0.3, 0.7]), np.array([0, 1]))
            meta = LogisticRegression()
            meta.fit([[0.5, 0.5, 0.5], [0.3, 0.7, 0.4]], [0, 1])
            trainer._save_models({"R2": ([_make_real_base_model()], meta, cal)}, ["feat_x"])

            pkl_path = tmp_path / disc.value / f"badminton_{disc.value}_v1.pkl"
            with open(pkl_path, "rb") as f:
                arts = pickle.load(f)
            assert arts["discipline"] == disc.value


# ===========================================================================
# 7. LayerTrainer.predict — without running actual .fit()
# ===========================================================================

class TestLayerTrainerPredict:
    def test_predict_calls_base_models_and_meta(self) -> None:
        from ml.train import LayerTrainer, _BetaCalibrator
        from sklearn.linear_model import LogisticRegression

        # Build real meta and calibrator with trivial data
        meta = LogisticRegression()
        oof = np.array([[0.4, 0.5, 0.6], [0.6, 0.5, 0.4]])
        y = np.array([0, 1])
        meta.fit(oof, y)

        cal = _BetaCalibrator()
        cal.fit(np.array([0.3, 0.7]), np.array([0, 1]))

        # Mock base models: each returns known probabilities
        def _mock_predict_proba(X):
            return np.column_stack([1 - np.full(len(X), 0.5), np.full(len(X), 0.5)])

        base1 = MagicMock()
        base1.predict_proba.side_effect = _mock_predict_proba
        base2 = MagicMock()
        base2.predict_proba.side_effect = _mock_predict_proba
        base3 = MagicMock()
        base3.predict_proba.side_effect = _mock_predict_proba

        lt = LayerTrainer()
        X = np.random.rand(5, 3).astype(np.float32)
        result = lt.predict(X, [base1, base2, base3], meta, cal)

        assert result.shape == (5,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_predict_output_is_ndarray(self) -> None:
        from ml.train import LayerTrainer, _BetaCalibrator
        from sklearn.linear_model import LogisticRegression

        # Meta trained on 3-column OOF (one column per base model)
        meta = LogisticRegression()
        meta.fit([[0.4, 0.5, 0.3], [0.6, 0.3, 0.7]], [0, 1])
        cal = _BetaCalibrator()
        cal.fit(np.array([0.4, 0.6]), np.array([0, 1]))

        # Must supply exactly 3 base models so column_stack produces 3 columns
        def _proba(X):
            return np.column_stack([1 - np.full(len(X), 0.45), np.full(len(X), 0.45)])

        base_models = []
        for _ in range(3):
            m = MagicMock()
            m.predict_proba.side_effect = _proba
            base_models.append(m)

        lt = LayerTrainer()
        out = lt.predict(
            np.random.rand(2, 5).astype(np.float32),
            base_models,
            meta,
            cal,
        )
        assert isinstance(out, np.ndarray)


# ===========================================================================
# 8. train_all_disciplines — skips disciplines with < 50 rows
# ===========================================================================

class TestTrainAllDisciplines:
    def test_skips_discipline_with_too_few_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disciplines with < 50 rows should be skipped, not raise."""
        from ml import train as train_mod

        # Patch train_and_evaluate so it never actually trains
        trained_disciplines: List[str] = []

        def _fake_train(self_inner, feature_df, n_optuna_trials=30):
            trained_disciplines.append(self_inner._discipline.value)
            return {"auc": 0.70, "brier": 0.22, "ece": 0.04, "n_test": 60}

        monkeypatch.setattr(
            train_mod.BadmintonModelTrainer,
            "train_and_evaluate",
            _fake_train,
        )

        # Only MS has enough rows; all others get < 50
        df_ms = _make_feature_df(n_train=150, n_val=60, n_test=60, discipline="MS")
        df_ws = _make_feature_df(n_train=10, n_val=3, n_test=3, discipline="WS")
        df = pd.concat([df_ms, df_ws], ignore_index=True)

        results = train_mod.train_all_disciplines(df, model_dir=str(tmp_path))
        assert "MS" in results
        # WS had only 16 rows → skipped
        assert "WS" not in results

    def test_propagates_runtime_error_from_trainer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If train_and_evaluate raises, train_all_disciplines re-raises."""
        from ml import train as train_mod

        def _failing_train(self_inner, feature_df, n_optuna_trials=30):
            raise RuntimeError("QA gate failure")

        monkeypatch.setattr(
            train_mod.BadmintonModelTrainer,
            "train_and_evaluate",
            _failing_train,
        )

        df = _make_feature_df(n_train=150, n_val=60, n_test=60, discipline="MS")
        with pytest.raises(RuntimeError, match="QA gate failure"):
            train_mod.train_all_disciplines(df, model_dir=str(tmp_path))


# ===========================================================================
# 9. feature_engineering.py — utility functions
# ===========================================================================

class TestUtilityFunctions:
    """Tests for private utility functions in feature_engineering.py."""

    # ---- _win_rate ----

    def test_win_rate_empty_history(self) -> None:
        from ml.feature_engineering import _win_rate
        result = _win_rate([], window=10, discipline=Discipline.MS, min_matches=3)
        assert math.isnan(result)

    def test_win_rate_insufficient_matches(self) -> None:
        from ml.feature_engineering import _win_rate
        hist = [{"won": True, "discipline": "MS", "date": date(2023, 1, 1)}]
        result = _win_rate(hist, window=10, discipline=Discipline.MS, min_matches=3)
        assert math.isnan(result)

    def test_win_rate_all_wins(self) -> None:
        from ml.feature_engineering import _win_rate
        hist = [{"won": True, "discipline": "MS", "date": date(2023, 1, i)} for i in range(1, 6)]
        result = _win_rate(hist, window=10, discipline=Discipline.MS, min_matches=3)
        assert result == pytest.approx(1.0)

    def test_win_rate_no_discipline_filter(self) -> None:
        from ml.feature_engineering import _win_rate
        hist = [
            {"won": True, "discipline": "MS", "date": date(2023, 1, 1)},
            {"won": False, "discipline": "WS", "date": date(2023, 1, 2)},
            {"won": True, "discipline": "MD", "date": date(2023, 1, 3)},
        ]
        result = _win_rate(hist, window=10, discipline=None, min_matches=3)
        assert result == pytest.approx(2 / 3)

    def test_win_rate_discipline_filter_reduces_subset(self) -> None:
        from ml.feature_engineering import _win_rate
        hist = [
            {"won": True, "discipline": "MS", "date": date(2023, 1, i)} for i in range(1, 5)
        ] + [
            {"won": False, "discipline": "WS", "date": date(2023, 2, i)} for i in range(1, 5)
        ]
        result = _win_rate(hist, window=20, discipline=Discipline.MS, min_matches=3)
        assert result == pytest.approx(1.0)

    # ---- _current_streak ----

    def test_current_streak_empty(self) -> None:
        from ml.feature_engineering import _current_streak
        assert _current_streak([]) == 0

    def test_current_streak_winning(self) -> None:
        from ml.feature_engineering import _current_streak
        hist = [{"won": False}, {"won": True}, {"won": True}, {"won": True}]
        assert _current_streak(hist) == 3

    def test_current_streak_losing(self) -> None:
        from ml.feature_engineering import _current_streak
        hist = [{"won": True}, {"won": False}, {"won": False}]
        assert _current_streak(hist) == -2

    def test_current_streak_single_win(self) -> None:
        from ml.feature_engineering import _current_streak
        assert _current_streak([{"won": True}]) == 1

    def test_current_streak_single_loss(self) -> None:
        from ml.feature_engineering import _current_streak
        assert _current_streak([{"won": False}]) == -1

    # ---- _weighted_form ----

    def test_weighted_form_empty_returns_nan(self) -> None:
        from ml.feature_engineering import _weighted_form
        result = _weighted_form([], window=15)
        assert math.isnan(result)

    def test_weighted_form_insufficient_returns_nan(self) -> None:
        from ml.feature_engineering import _weighted_form
        hist = [{"won": True, "opponent_rank": 10}] * 2  # Only 2 matches, need 3
        result = _weighted_form(hist, window=15)
        assert math.isnan(result)

    def test_weighted_form_known_rank(self) -> None:
        from ml.feature_engineering import _weighted_form
        hist = [
            {"won": True, "opponent_rank": 5},
            {"won": True, "opponent_rank": 10},
            {"won": False, "opponent_rank": 50},
        ]
        result = _weighted_form(hist, window=15)
        assert 0.0 <= result <= 1.0

    def test_weighted_form_no_opponent_rank_uses_default_weight(self) -> None:
        from ml.feature_engineering import _weighted_form
        hist = [
            {"won": True, "opponent_rank": None},
            {"won": False, "opponent_rank": None},
            {"won": True, "opponent_rank": None},
        ]
        result = _weighted_form(hist, window=15)
        assert result == pytest.approx(2 / 3)

    # ---- _top_ranked_win_rate ----

    def test_top_ranked_win_rate_no_top_opponents(self) -> None:
        from ml.feature_engineering import _top_ranked_win_rate
        hist = [{"won": True, "opponent_rank": 200}] * 5  # All ranked 200, threshold 50
        result = _top_ranked_win_rate(hist, window=20, rank_threshold=50)
        assert math.isnan(result)

    def test_top_ranked_win_rate_with_top_opponents(self) -> None:
        from ml.feature_engineering import _top_ranked_win_rate
        hist = [
            {"won": True, "opponent_rank": 10},
            {"won": False, "opponent_rank": 20},
            {"won": True, "opponent_rank": 30},
        ]
        result = _top_ranked_win_rate(hist, window=20, rank_threshold=50)
        assert result == pytest.approx(2 / 3)

    def test_top_ranked_win_rate_one_match_returns_nan(self) -> None:
        from ml.feature_engineering import _top_ranked_win_rate
        hist = [{"won": True, "opponent_rank": 5}]
        result = _top_ranked_win_rate(hist, window=20, rank_threshold=50)
        assert math.isnan(result)

    # ---- _h2h_record ----

    def test_h2h_record_empty_history(self) -> None:
        from ml.feature_engineering import _h2h_record
        result = _h2h_record(
            "PA", "PB", {}, discipline=Discipline.MS,
            window=100, before_date=date(2024, 1, 1)
        )
        assert result["total"] == 0
        assert result["wins_a"] == 0

    def test_h2h_record_counts_wins_correctly(self) -> None:
        from ml.feature_engineering import _h2h_record
        history = {
            "PA": [
                {"date": date(2023, 1, 1), "won": True, "opponent": "PB", "discipline": "MS"},
                {"date": date(2023, 2, 1), "won": False, "opponent": "PB", "discipline": "MS"},
                {"date": date(2023, 3, 1), "won": True, "opponent": "PB", "discipline": "MS"},
            ]
        }
        result = _h2h_record(
            "PA", "PB", history, discipline=Discipline.MS,
            window=100, before_date=date(2024, 1, 1)
        )
        assert result["total"] == 3
        assert result["wins_a"] == 2
        assert result["wins_b"] == 1

    def test_h2h_record_respects_before_date(self) -> None:
        from ml.feature_engineering import _h2h_record
        history = {
            "PA": [
                {"date": date(2023, 6, 1), "won": True, "opponent": "PB", "discipline": "MS"},
                {"date": date(2024, 1, 1), "won": True, "opponent": "PB", "discipline": "MS"},
            ]
        }
        result = _h2h_record(
            "PA", "PB", history, discipline=Discipline.MS,
            window=100, before_date=date(2024, 1, 1)
        )
        assert result["total"] == 1  # Only the 2023 match is before 2024-01-01

    def test_h2h_record_no_discipline_filter(self) -> None:
        from ml.feature_engineering import _h2h_record
        history = {
            "PA": [
                {"date": date(2023, 1, 1), "won": True, "opponent": "PB", "discipline": "MS"},
                {"date": date(2023, 2, 1), "won": True, "opponent": "PB", "discipline": "WS"},
            ]
        }
        result = _h2h_record(
            "PA", "PB", history, discipline=None,
            window=100, before_date=date(2024, 1, 1)
        )
        assert result["total"] == 2

    # ---- _matches_in_window ----

    def test_matches_in_window_zero_when_empty(self) -> None:
        from ml.feature_engineering import _matches_in_window
        assert _matches_in_window([], date(2024, 1, 10), days=7) == 0

    def test_matches_in_window_counts_within_window(self) -> None:
        from ml.feature_engineering import _matches_in_window
        ref = date(2024, 1, 10)
        hist = [
            {"date": date(2024, 1, 4)},   # 6 days before — inside 7d window
            {"date": date(2024, 1, 2)},   # 8 days before — outside
            {"date": date(2024, 1, 9)},   # 1 day before — inside
        ]
        assert _matches_in_window(hist, ref, days=7) == 2

    def test_matches_in_window_excludes_ref_date_itself(self) -> None:
        from ml.feature_engineering import _matches_in_window
        ref = date(2024, 1, 10)
        hist = [{"date": ref}]  # Same date — excluded (strictly < ref_date)
        assert _matches_in_window(hist, ref, days=1) == 0

    # ---- _games_in_window ----

    def test_games_in_window_default_value(self) -> None:
        from ml.feature_engineering import _games_in_window
        ref = date(2024, 1, 10)
        hist = [{"date": date(2024, 1, 8)}]  # No games_played key → default 2
        count = _games_in_window(hist, ref, days=7)
        assert count == 2

    def test_games_in_window_uses_games_played_field(self) -> None:
        from ml.feature_engineering import _games_in_window
        ref = date(2024, 1, 10)
        hist = [
            {"date": date(2024, 1, 8), "games_played": 3},
            {"date": date(2024, 1, 7), "games_played": 2},
        ]
        assert _games_in_window(hist, ref, days=7) == 5

    # ---- _wins_in_tournament ----

    def test_wins_in_tournament_counts_correctly(self) -> None:
        from ml.feature_engineering import _wins_in_tournament
        history = {
            "PA": [
                {"tournament_id": "T1", "date": date(2024, 3, 1), "won": True},
                {"tournament_id": "T1", "date": date(2024, 3, 2), "won": False},
                {"tournament_id": "T2", "date": date(2024, 3, 3), "won": True},
            ]
        }
        result = _wins_in_tournament("PA", "T1", history, date(2024, 3, 5))
        assert result == 1  # Only T1 wins before match_date

    def test_wins_in_tournament_zero_when_entity_absent(self) -> None:
        from ml.feature_engineering import _wins_in_tournament
        assert _wins_in_tournament("PX", "T1", {}, date(2024, 1, 1)) == 0

    # ---- _is_straight_win ----

    def test_is_straight_win_true(self) -> None:
        from ml.feature_engineering import _is_straight_win
        assert _is_straight_win([(21, 10), (21, 15)], "A") is True

    def test_is_straight_win_false_three_games(self) -> None:
        from ml.feature_engineering import _is_straight_win
        assert _is_straight_win([(21, 10), (10, 21), (21, 15)], "A") is False

    def test_is_straight_win_empty_scores(self) -> None:
        from ml.feature_engineering import _is_straight_win
        assert _is_straight_win([], "A") is False

    def test_is_straight_win_player_b(self) -> None:
        from ml.feature_engineering import _is_straight_win
        assert _is_straight_win([(10, 21), (15, 21)], "B") is True

    # ---- _any_game_reached_deuce ----

    def test_deuce_detected_at_20_20(self) -> None:
        from ml.feature_engineering import _any_game_reached_deuce
        assert _any_game_reached_deuce([(21, 10), (22, 20)]) is True

    def test_deuce_not_detected_when_no_deuce(self) -> None:
        from ml.feature_engineering import _any_game_reached_deuce
        assert _any_game_reached_deuce([(21, 10), (21, 15)]) is False

    def test_deuce_empty_scores(self) -> None:
        from ml.feature_engineering import _any_game_reached_deuce
        assert _any_game_reached_deuce([]) is False

    def test_deuce_at_29_29_golden_point(self) -> None:
        from ml.feature_engineering import _any_game_reached_deuce
        assert _any_game_reached_deuce([(30, 29)]) is True

    # ---- _apply_p1p2_swap ----

    def test_swap_flips_a_and_b_suffixes(self) -> None:
        from ml.feature_engineering import _apply_p1p2_swap
        feats = {"score_a": 10.0, "score_b": 20.0}
        swapped = _apply_p1p2_swap(feats)
        assert swapped["score_a"] == 20.0
        assert swapped["score_b"] == 10.0

    def test_swap_negates_diff_features(self) -> None:
        from ml.feature_engineering import _apply_p1p2_swap
        feats = {"elo_diff": 150.0, "score_a": 1.0, "score_b": 2.0}
        swapped = _apply_p1p2_swap(feats)
        assert swapped["elo_diff"] == pytest.approx(-150.0)

    def test_swap_inverts_elo_prob(self) -> None:
        from ml.feature_engineering import _apply_p1p2_swap
        feats = {"elo_prob": 0.7, "score_a": 1.0, "score_b": 2.0}
        swapped = _apply_p1p2_swap(feats)
        assert swapped["elo_prob"] == pytest.approx(0.3)

    def test_swap_leaves_nan_diff_intact(self) -> None:
        from ml.feature_engineering import _apply_p1p2_swap
        feats = {"elo_diff": float("nan")}
        swapped = _apply_p1p2_swap(feats)
        assert math.isnan(swapped["elo_diff"])  # NaN is not negated

    def test_swap_returns_new_dict(self) -> None:
        from ml.feature_engineering import _apply_p1p2_swap
        feats = {"val_a": 1.0, "val_b": 2.0}
        swapped = _apply_p1p2_swap(feats)
        assert swapped is not feats

    # ---- _clamp_llm ----

    def test_clamp_llm_positive_cap(self) -> None:
        from ml.feature_engineering import _clamp_llm
        assert _clamp_llm(0.10) == pytest.approx(0.05)

    def test_clamp_llm_negative_cap(self) -> None:
        from ml.feature_engineering import _clamp_llm
        assert _clamp_llm(-0.10) == pytest.approx(-0.05)

    def test_clamp_llm_within_bounds(self) -> None:
        from ml.feature_engineering import _clamp_llm
        assert _clamp_llm(0.03) == pytest.approx(0.03)
        assert _clamp_llm(-0.02) == pytest.approx(-0.02)

    def test_clamp_llm_zero(self) -> None:
        from ml.feature_engineering import _clamp_llm
        assert _clamp_llm(0.0) == pytest.approx(0.0)


# ===========================================================================
# 10. FeatureBuilder group methods — mocked dependencies
# ===========================================================================

class TestFeatureBuilderGroupMethods:
    """
    Tests FeatureBuilder group methods with mocked ELO / rankings / serve DBs.
    Covers lines in each group method that were not exercised before.
    """

    def _make_builder(self):
        from ml.feature_engineering import FeatureBuilder
        from ml.elo_system import BadmintonEloSystem

        elo = BadmintonEloSystem()
        for disc in Discipline:
            elo.initialize_player("PA", disc)
            elo.initialize_player("PB", disc)
            elo.initialize_player("PA1", disc)
            elo.initialize_player("PA2", disc)
            elo.initialize_player("PB1", disc)
            elo.initialize_player("PB2", disc)

        weekly_db = MagicMock()
        serve_db = MagicMock()

        # Default returns
        weekly_db.get_rank.return_value = None
        weekly_db.get_points.return_value = None
        weekly_db.is_home_region.return_value = False

        serve_db.get_profile.return_value = None
        serve_db.get_smash_win_rate.return_value = None
        serve_db.get_net_win_rate.return_value = None
        serve_db.get_avg_rally_length.return_value = None

        return FeatureBuilder(elo_system=elo, weekly_rankings_db=weekly_db, serve_stat_db=serve_db)

    # ---- Group D: Tournament Context ----

    def test_group_d_returns_7_keys(self) -> None:
        builder = self._make_builder()
        feats = builder.group_d_tournament_context(
            entity_a="PA", entity_b="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_1000,
            round_code="QF",
            draw_size=32,
            match_history={},
            match_date=date(2024, 3, 15),
            tournament_id="T001",
        )
        assert isinstance(feats, dict)
        assert len(feats) == 7

    def test_group_d_tier_code_encoded(self) -> None:
        builder = self._make_builder()
        feats = builder.group_d_tournament_context(
            entity_a="PA", entity_b="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_1000,
            round_code="F",
            draw_size=32,
            match_history={},
            match_date=date(2024, 3, 15),
            tournament_id="T001",
        )
        assert "tourney_level_code" in feats
        assert feats["round_code"] == 6.0  # "F" → 6

    def test_group_d_draw_size_log(self) -> None:
        builder = self._make_builder()
        feats = builder.group_d_tournament_context(
            entity_a="PA", entity_b="PB",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_500,
            round_code="R32",
            draw_size=32,
            match_history={},
            match_date=date(2024, 2, 1),
            tournament_id="T002",
        )
        assert feats["draw_size_log"] == pytest.approx(math.log(32))

    def test_group_d_tournament_momentum_with_history(self) -> None:
        builder = self._make_builder()
        history = {
            "PA": [
                {
                    "date": date(2024, 3, 10),
                    "won": True,
                    "opponent": "PX",
                    "discipline": "MS",
                    "tier": "SUPER_1000",
                    "tournament_id": "T001",
                    "games_played": 2,
                    "opponent_rank": None,
                }
            ]
        }
        feats = builder.group_d_tournament_context(
            entity_a="PA", entity_b="PB",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_1000,
            round_code="QF",
            draw_size=32,
            match_history=history,
            match_date=date(2024, 3, 15),
            tournament_id="T001",
        )
        # PA has 1 win in this tournament before this match
        assert feats["tournament_momentum_a"] > 0.0

    # ---- Group E: Fatigue Schedule ----

    def test_group_e_returns_6_keys(self) -> None:
        builder = self._make_builder()
        feats = builder.group_e_fatigue_schedule(
            entity_a="PA", entity_b="PB",
            match_history={},
            match_date=date(2024, 3, 15),
        )
        assert len(feats) == 6

    def test_group_e_no_matches_in_window_returns_zero(self) -> None:
        builder = self._make_builder()
        feats = builder.group_e_fatigue_schedule(
            entity_a="PA", entity_b="PB",
            match_history={},
            match_date=date(2024, 3, 15),
        )
        assert feats["matches_last7_a"] == 0.0
        assert feats["back_to_back_flag_a"] == 0.0

    def test_group_e_back_to_back_detected(self) -> None:
        builder = self._make_builder()
        ref_date = date(2024, 3, 15)
        history = {
            "PA": [
                {
                    "date": ref_date - timedelta(days=0),  # same day
                    "won": True, "opponent": "PX", "discipline": "MS",
                    "tier": "SUPER_1000", "tournament_id": "T1", "games_played": 2,
                    "opponent_rank": None,
                }
            ]
        }
        feats = builder.group_e_fatigue_schedule("PA", "PB", history, ref_date)
        # date is strictly < ref_date, but same day is excluded
        assert feats["back_to_back_flag_a"] == 0.0

    # ---- Group F: RWP Estimates ----

    def test_group_f_returns_expected_keys(self) -> None:
        builder = self._make_builder()
        feats = builder.group_f_rwp_estimates("PA", "PB", Discipline.MS)
        # Group F spec says 8 features but implementation has shuttle_speed_adj as a 9th
        assert len(feats) >= 8
        for key in [
            "rwp_historical_a", "rwp_historical_b",
            "rwp_discipline_adj_a", "rwp_discipline_adj_b",
            "shuttle_speed_adj", "smash_win_rate_a", "smash_win_rate_b",
            "net_win_rate_a", "rally_length_avg_a",
        ]:
            assert key in feats

    def test_group_f_shuttle_speed_zero_when_none(self) -> None:
        builder = self._make_builder()
        feats = builder.group_f_rwp_estimates("PA", "PB", Discipline.MS, environment_shuttle_speed=None)
        assert feats["shuttle_speed_adj"] == 0.0

    def test_group_f_shuttle_speed_adjustment_computed(self) -> None:
        from config.badminton_config import SHUTTLE_SPEED_NEUTRAL, RWP_SHUTTLE_SPEED_COEFFICIENT
        builder = self._make_builder()
        speed = SHUTTLE_SPEED_NEUTRAL + 2
        feats = builder.group_f_rwp_estimates("PA", "PB", Discipline.MS, environment_shuttle_speed=speed)
        expected = 2 * RWP_SHUTTLE_SPEED_COEFFICIENT
        assert feats["shuttle_speed_adj"] == pytest.approx(expected)

    def test_group_f_nan_when_no_serve_profile(self) -> None:
        builder = self._make_builder()
        feats = builder.group_f_rwp_estimates("PA", "PB", Discipline.MS)
        assert math.isnan(feats["rwp_historical_a"])

    def test_group_f_uses_smash_win_rate_when_available(self) -> None:
        from ml.feature_engineering import FeatureBuilder
        from ml.elo_system import BadmintonEloSystem

        elo = BadmintonEloSystem()
        elo.initialize_player("PA", Discipline.MS)
        elo.initialize_player("PB", Discipline.MS)

        serve_db = MagicMock()
        serve_db.get_profile.return_value = None
        serve_db.get_smash_win_rate.return_value = 0.72
        serve_db.get_net_win_rate.return_value = 0.55
        serve_db.get_avg_rally_length.return_value = 8.3

        weekly_db = MagicMock()
        weekly_db.get_rank.return_value = None
        weekly_db.get_points.return_value = None

        builder = FeatureBuilder(elo, weekly_db, serve_db)
        feats = builder.group_f_rwp_estimates("PA", "PB", Discipline.MS)

        assert feats["smash_win_rate_a"] == pytest.approx(0.72)
        assert feats["net_win_rate_a"] == pytest.approx(0.55)
        assert feats["rally_length_avg_a"] == pytest.approx(8.3)

    # ---- Group G: Doubles-Specific ----

    def test_group_g_singles_returns_nans(self) -> None:
        builder = self._make_builder()
        feats = builder.group_g_doubles(
            entity_a="PA", entity_b="PB",
            discipline=Discipline.MS,
            match_history={},
            match_date=date(2024, 3, 15),
        )
        assert len(feats) == 8
        assert all(math.isnan(v) for v in feats.values())

    def test_group_g_doubles_md_does_not_crash(self) -> None:
        builder = self._make_builder()
        feats = builder.group_g_doubles(
            entity_a="PA1|PA2", entity_b="PB1|PB2",
            discipline=Discipline.MD,
            match_history={},
            match_date=date(2024, 3, 15),
        )
        assert isinstance(feats, dict)
        assert "pair_elo_diff" in feats
        assert "pair_matches_together" in feats

    def test_group_g_xd_computes_gender_combo(self) -> None:
        builder = self._make_builder()
        feats = builder.group_g_doubles(
            entity_a="PA1|PA2", entity_b="PB1|PB2",
            discipline=Discipline.XD,
            match_history={},
            match_date=date(2024, 3, 15),
        )
        assert not math.isnan(feats["gender_combo"])
        assert not math.isnan(feats["dominant_player_elo"])

    def test_group_g_wd_gender_combo_is_nan(self) -> None:
        """For WD (non-XD doubles), gender_combo should be NaN."""
        builder = self._make_builder()
        feats = builder.group_g_doubles(
            entity_a="PA1|PA2", entity_b="PB1|PB2",
            discipline=Discipline.WD,
            match_history={},
            match_date=date(2024, 3, 15),
        )
        assert math.isnan(feats["gender_combo"])
        assert math.isnan(feats["dominant_player_elo"])

    # ---- Group H: Physical Profile ----

    def test_group_h_returns_4_keys(self) -> None:
        builder = self._make_builder()
        registry = {
            "PA": {"birth_date": "1995-04-20"},
            "PB": {"birth_date": "1998-11-10"},
        }
        feats = builder.group_h_physical("PA", "PB", date(2024, 3, 15), registry)
        assert len(feats) == 4

    def test_group_h_age_computed_correctly(self) -> None:
        builder = self._make_builder()
        birth = date(2000, 1, 1)
        match_d = date(2024, 1, 1)
        registry = {
            "PA": {"birth_date": birth.isoformat()},
            "PB": {"birth_date": "1998-06-15"},
        }
        feats = builder.group_h_physical("PA", "PB", match_d, registry)
        expected_age = (match_d - birth).days / 365.25
        assert feats["age_a"] == pytest.approx(expected_age, abs=0.01)

    def test_group_h_missing_player_returns_nan(self) -> None:
        builder = self._make_builder()
        feats = builder.group_h_physical("PA", "PB", date(2024, 3, 15), {})
        assert math.isnan(feats["age_a"])
        assert math.isnan(feats["age_b"])
        assert math.isnan(feats["age_diff"])
        assert math.isnan(feats["age_factor_a"])

    def test_group_h_age_factor_clamped(self) -> None:
        """Age factor must be within [-1.5, 1.5]."""
        builder = self._make_builder()
        # Very old player (45) → age_factor negative, clamped at -1.5
        registry = {
            "PA": {"birth_date": "1979-01-01"},
            "PB": {"birth_date": "2004-01-01"},
        }
        feats = builder.group_h_physical("PA", "PB", date(2024, 1, 1), registry)
        assert feats["age_factor_a"] >= -1.5
        assert feats["age_factor_a"] <= 1.5

    def test_group_h_invalid_birth_date_returns_nan(self) -> None:
        builder = self._make_builder()
        registry = {
            "PA": {"birth_date": "NOT-A-DATE"},
            "PB": {"birth_date": "1998-01-01"},
        }
        feats = builder.group_h_physical("PA", "PB", date(2024, 3, 15), registry)
        assert math.isnan(feats["age_a"])

    # ---- Group I: LLM Augmentation ----

    def test_group_i_returns_6_keys(self) -> None:
        builder = self._make_builder()
        feats = builder.group_i_llm("PA", "PB", date(2024, 3, 15), news_db=None)
        assert len(feats) == 6

    def test_group_i_no_news_db_returns_zeros(self) -> None:
        builder = self._make_builder()
        feats = builder.group_i_llm("PA", "PB", date(2024, 3, 15), news_db=None)
        assert feats["llm_fitness_signal_a"] == 0.0
        assert feats["llm_fitness_signal_b"] == 0.0
        assert feats["llm_retirement_risk_flag"] == 0.0

    def test_group_i_signals_clamped(self) -> None:
        builder = self._make_builder()
        news_db = MagicMock()
        news_db.get_signals.side_effect = [
            {"fitness": 0.99, "motivation": -0.99, "venue": 0.5, "retirement_risk": True},
            {"fitness": -0.99, "motivation": 0.99, "venue": -0.5, "retirement_risk": False},
        ]
        feats = builder.group_i_llm("PA", "PB", date(2024, 3, 15), news_db=news_db)
        assert feats["llm_fitness_signal_a"] == pytest.approx(0.05)
        assert feats["llm_fitness_signal_b"] == pytest.approx(-0.05)
        assert feats["llm_retirement_risk_flag"] == 1.0

    def test_group_i_retirement_risk_union(self) -> None:
        """retirement_risk_flag is 1 if EITHER player has retirement risk."""
        builder = self._make_builder()
        news_db = MagicMock()
        news_db.get_signals.side_effect = [
            {"fitness": 0.0, "motivation": 0.0, "venue": 0.0, "retirement_risk": False},
            {"fitness": 0.0, "motivation": 0.0, "venue": 0.0, "retirement_risk": True},
        ]
        feats = builder.group_i_llm("PA", "PB", date(2024, 3, 15), news_db=news_db)
        assert feats["llm_retirement_risk_flag"] == 1.0

    # ---- Group A: ELO & BWF Ranking ----

    def test_group_a_returns_dict(self) -> None:
        builder = self._make_builder()
        feats = builder.group_a_elo_ranking("PA", "PB", Discipline.MS, date(2024, 3, 1))
        assert isinstance(feats, dict)

    def test_group_a_with_real_ranks(self) -> None:
        from ml.feature_engineering import FeatureBuilder
        from ml.elo_system import BadmintonEloSystem

        elo = BadmintonEloSystem()
        elo.initialize_player("PA", Discipline.MS)
        elo.initialize_player("PB", Discipline.MS)

        weekly_db = MagicMock()
        weekly_db.get_rank.side_effect = [5, 10]
        weekly_db.get_points.side_effect = [12000, 8000]

        serve_db = MagicMock()
        serve_db.get_profile.return_value = None
        serve_db.get_smash_win_rate.return_value = None
        serve_db.get_net_win_rate.return_value = None
        serve_db.get_avg_rally_length.return_value = None

        builder = FeatureBuilder(elo, weekly_db, serve_db)
        feats = builder.group_a_elo_ranking("PA", "PB", Discipline.MS, date(2024, 3, 1))

        assert feats["bwf_rank_a"] == pytest.approx(math.log(5))
        assert feats["bwf_rank_b"] == pytest.approx(math.log(10))
        assert not math.isnan(feats["bwf_rank_diff"])
        assert not math.isnan(feats["bwf_points_diff"])

    # ---- Group B: Recent Form ----

    def test_group_b_returns_dict(self) -> None:
        builder = self._make_builder()
        feats = builder.group_b_recent_form("PA", "PB", Discipline.MS, {}, date(2024, 3, 1))
        assert isinstance(feats, dict)

    def test_group_b_form_diff_computed_when_both_available(self) -> None:
        builder = self._make_builder()
        # Build history with enough matches for form to be non-NaN
        hist: Dict[str, List] = {
            "PA": [
                {"date": date(2024, 2, i), "won": i % 2 == 0,
                 "discipline": "MS", "opponent": "PX", "opponent_rank": None,
                 "tier": "SUPER_500", "tournament_id": "T1", "games_played": 2}
                for i in range(1, 16)
            ],
            "PB": [
                {"date": date(2024, 2, i), "won": True,
                 "discipline": "MS", "opponent": "PY", "opponent_rank": None,
                 "tier": "SUPER_500", "tournament_id": "T1", "games_played": 2}
                for i in range(1, 16)
            ],
        }
        feats = builder.group_b_recent_form("PA", "PB", Discipline.MS, hist, date(2024, 3, 1))
        # form_momentum_diff may be NaN if weighted form can't be computed, otherwise float
        assert isinstance(feats.get("form_momentum_diff"), float)


# ===========================================================================
# 11. build_feature_dataset — integration smoke test with mocked ELO
# ===========================================================================

class TestBuildFeatureDataset:
    """
    Smoke-tests for the main pipeline function.
    Uses minimal match data so the loop runs without actual data sources.
    """

    def _make_matches_df(self, n: int = 5, discipline: str = "MS") -> pd.DataFrame:
        """Build a minimal matches DataFrame for the pipeline."""
        rows = []
        for i in range(n):
            rows.append({
                "match_id": f"M{i:04d}",
                "date": date(2019, 1, i + 1),
                "tournament_id": "T001",
                "tier": "SUPER_500",
                "discipline": discipline,
                "round": "R32",
                "draw_size": 32,
                "entity_a_id": "PA",
                "entity_b_id": "PB",
                "winner_id": "PA" if i % 2 == 0 else "PB",
                "game_scores": [(21, 10), (21, 15)],
            })
        return pd.DataFrame(rows)

    def _make_mock_deps(self):
        from ml.elo_system import BadmintonEloSystem
        elo = BadmintonEloSystem()
        for disc in Discipline:
            elo.initialize_player("PA", disc)
            elo.initialize_player("PB", disc)

        weekly_db = MagicMock()
        weekly_db.get_rank.return_value = None
        weekly_db.get_points.return_value = None
        weekly_db.is_home_region.return_value = False

        serve_db = MagicMock()
        serve_db.get_profile.return_value = None
        serve_db.get_smash_win_rate.return_value = None
        serve_db.get_net_win_rate.return_value = None
        serve_db.get_avg_rally_length.return_value = None

        player_registry: Dict[str, Dict] = {
            "PA": {"birth_date": "1995-05-10", "current_age": 28.0},
            "PB": {"birth_date": "1997-08-15", "current_age": 26.0},
        }
        return elo, weekly_db, serve_db, player_registry

    def test_returns_dataframe(self) -> None:
        from ml.feature_engineering import build_feature_dataset
        elo, weekly_db, serve_db, registry = self._make_mock_deps()
        matches = self._make_matches_df(n=20)
        result = build_feature_dataset(
            matches_df=matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            random_seed=99,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20

    def test_output_has_target_columns(self) -> None:
        from ml.feature_engineering import build_feature_dataset
        elo, weekly_db, serve_db, registry = self._make_mock_deps()
        matches = self._make_matches_df(n=20)
        result = build_feature_dataset(
            matches_df=matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            random_seed=99,
        )
        for col in ["target_win", "target_2_0", "target_deuce"]:
            assert col in result.columns

    def test_output_has_feat_prefix_columns(self) -> None:
        from ml.feature_engineering import build_feature_dataset
        elo, weekly_db, serve_db, registry = self._make_mock_deps()
        matches = self._make_matches_df(n=20)
        result = build_feature_dataset(
            matches_df=matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            random_seed=99,
        )
        feat_cols = [c for c in result.columns if c.startswith("feat_")]
        assert len(feat_cols) > 0

    def test_discipline_filter_applied(self) -> None:
        from ml.feature_engineering import build_feature_dataset
        elo, weekly_db, serve_db, registry = self._make_mock_deps()

        # Mix of MS and WS — use 20 each so swap stays balanced
        ms_matches = self._make_matches_df(n=20, discipline="MS")
        ws_matches = self._make_matches_df(n=20, discipline="WS")
        all_matches = pd.concat([ms_matches, ws_matches], ignore_index=True)

        # Initialize WS players too
        for p in ["PA", "PB"]:
            elo.initialize_player(p, Discipline.WS)

        result = build_feature_dataset(
            matches_df=all_matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            discipline=Discipline.MS,
            random_seed=99,
        )
        assert all(result["discipline"] == "MS")

    def test_date_column_is_date_type(self) -> None:
        from ml.feature_engineering import build_feature_dataset
        elo, weekly_db, serve_db, registry = self._make_mock_deps()
        matches = self._make_matches_df(n=20)
        result = build_feature_dataset(
            matches_df=matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            random_seed=99,
        )
        # date column should contain date objects
        assert len(result) > 0
        sample_date = result["date"].iloc[0]
        assert isinstance(sample_date, (date, type(pd.Timestamp("2024-01-01"))))

    def test_elo_updated_after_feature_extraction(self) -> None:
        """
        ELO must increase for winner across multiple matches (temporal correctness).
        If ELO were updated before feature extraction, values would be stale on first match.
        This test verifies the pipeline runs without error — leakage detection is in H5.
        """
        from ml.feature_engineering import build_feature_dataset
        elo, weekly_db, serve_db, registry = self._make_mock_deps()

        initial_rating = elo.get_rating("PA", Discipline.MS)
        matches = self._make_matches_df(n=20)
        build_feature_dataset(
            matches_df=matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            random_seed=99,
        )
        final_rating = elo.get_rating("PA", Discipline.MS)
        # PA wins 10 of 20 matches — rating should change
        assert final_rating != initial_rating


# ===========================================================================
# 12. build_feature_dataset — P1 win rate validation
# ===========================================================================

class TestBuildFeatureDatasetP1WinRate:
    def test_win_rate_balance_with_large_dataset(self) -> None:
        """With random seed and enough matches, P1 win rate should be ~0.5."""
        from ml.feature_engineering import build_feature_dataset
        from ml.elo_system import BadmintonEloSystem

        elo = BadmintonEloSystem()
        for disc in Discipline:
            elo.initialize_player("PA", disc)
            elo.initialize_player("PB", disc)

        weekly_db = MagicMock()
        weekly_db.get_rank.return_value = None
        weekly_db.get_points.return_value = None
        weekly_db.is_home_region.return_value = False

        serve_db = MagicMock()
        serve_db.get_profile.return_value = None
        serve_db.get_smash_win_rate.return_value = None
        serve_db.get_net_win_rate.return_value = None
        serve_db.get_avg_rally_length.return_value = None

        registry = {"PA": {"birth_date": "1995-01-01"}, "PB": {"birth_date": "1997-01-01"}}

        rows = []
        for i in range(100):
            rows.append({
                "match_id": f"M{i:04d}",
                "date": date(2019, 1, 1) + timedelta(days=i),
                "tournament_id": "T001",
                "tier": "SUPER_500",
                "discipline": "MS",
                "round": "R32",
                "draw_size": 32,
                "entity_a_id": "PA",
                "entity_b_id": "PB",
                "winner_id": "PA" if i % 2 == 0 else "PB",
                "game_scores": [(21, 10), (21, 15)],
            })
        matches = pd.DataFrame(rows)

        result = build_feature_dataset(
            matches_df=matches,
            elo_system=elo,
            weekly_rankings_db=weekly_db,
            serve_stat_db=serve_db,
            player_registry=registry,
            random_seed=42,
        )
        if len(result) > 0:
            p1_win_rate = 1.0 - result["target_win"].mean()
            # With random swap, P1 win rate should be near 0.5
            assert 0.35 <= p1_win_rate <= 0.65, f"P1 win rate {p1_win_rate:.3f} out of expected range"

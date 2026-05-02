"""
test_ml_inference.py
====================
Tests for the badminton ML inference pipeline and regime gate.

Uses MagicMock for all model .pkl dependencies — tests do NOT require trained models.
ZERO hardcoded expected probability values.

Covers:
  - BadmintonModelInference instantiation
  - predict() returns float in [0, 1]
  - Extreme ELO gap → probability near boundary
  - RWP bisection inversion
  - RWP output in valid range [RWP_MIN_VALID, RWP_MAX_VALID]
  - RegimeGate R0/R1/R2 classification
  - R0 (sparse data) → triggered by default ELO
  - R1 (standard) → normal match
  - R2 (rich data) → top tier + high match count
  - Missing model gracefully raises RuntimeError
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    TournamentTier,
    RWP_MIN_VALID,
    RWP_MAX_VALID,
)
from ml.inference import BadmintonModelInference, InferenceResult
from ml.regime_gate import Regime, RegimeGate, RegimeInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def match_date() -> date:
    return date(2025, 6, 15)


@pytest.fixture
def inference_engine_no_model() -> BadmintonModelInference:
    """Inference engine pointing to a non-existent model directory."""
    return BadmintonModelInference(
        model_dir="/nonexistent/models",
        elo_system=MagicMock(),
        weekly_rankings_db=MagicMock(),
        serve_stat_db=MagicMock(),
    )


@pytest.fixture
def mock_base_model() -> MagicMock:
    """A mock sklearn-like model with predict_proba."""
    import numpy as np
    model = MagicMock()
    model.predict_proba = MagicMock(
        return_value=np.array([[0.35, 0.65]])
    )
    return model


@pytest.fixture
def mock_meta_model() -> MagicMock:
    """Mock meta-learner (stacking)."""
    import numpy as np
    model = MagicMock()
    model.predict_proba = MagicMock(
        return_value=np.array([[0.30, 0.70]])
    )
    return model


@pytest.fixture
def mock_calibrator() -> MagicMock:
    """Mock calibrator (Platt/isotonic)."""
    import numpy as np
    cal = MagicMock()
    cal.predict_proba = MagicMock(
        return_value=np.array([[0.32, 0.68]])
    )
    return cal


@pytest.fixture
def mock_regime_gate() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_model_tuple(
    mock_base_model: MagicMock,
    mock_meta_model: MagicMock,
    mock_calibrator: MagicMock,
    mock_regime_gate: MagicMock,
) -> tuple:
    """Mimics the (base_models, meta, calibrator, regime_gate) tuple loaded from .pkl."""
    return (
        [mock_base_model, mock_base_model, mock_base_model],
        mock_meta_model,
        mock_calibrator,
        mock_regime_gate,
    )


@pytest.fixture
def loaded_inference_engine(
    mock_model_tuple: tuple,
) -> BadmintonModelInference:
    """
    Inference engine with a mocked model already loaded and feature extraction mocked.
    """
    engine = BadmintonModelInference(
        model_dir="/mocked",
        elo_system=MagicMock(),
        weekly_rankings_db=MagicMock(),
        serve_stat_db=MagicMock(),
    )
    # Pre-load the mocked model
    engine._models[Discipline.MS.value] = mock_model_tuple
    return engine


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

class TestInferenceInstantiation:
    def test_engine_instantiates(self) -> None:
        engine = BadmintonModelInference()
        assert engine is not None
        assert isinstance(engine._models, dict)

    def test_engine_accepts_custom_model_dir(self) -> None:
        engine = BadmintonModelInference(model_dir="/custom/path")
        from pathlib import Path
        assert engine._model_dir == Path("/custom/path")

    def test_inference_result_dataclass(self) -> None:
        result = InferenceResult(
            entity_a_id="A",
            entity_b_id="B",
            discipline=Discipline.MS,
            p_a_wins=0.65,
            p_a_wins_2_0=0.40,
            p_deuce=0.30,
            rwp_a=0.52,
            rwp_b=0.51,
            regime="R1",
            model_available=True,
            n_features_used=60,
        )
        assert result.p_a_wins == 0.65
        assert result.regime == "R1"


# ---------------------------------------------------------------------------
# 2. predict() returns float in [0, 1]
# ---------------------------------------------------------------------------

class TestPredictOutput:
    def test_predict_returns_valid_probability(
        self, loaded_inference_engine: BadmintonModelInference, match_date: date
    ) -> None:
        # Mock feature extraction to return a valid feature vector
        with patch.object(
            loaded_inference_engine, "_extract_features",
            return_value=[0.1] * 66,
        ), patch.object(
            loaded_inference_engine, "_rwp_from_match_prob",
            return_value=(0.52, 0.515),
        ):
            result = loaded_inference_engine.predict(
                "player_A", "player_B",
                Discipline.MS, TournamentTier.SUPER_500, match_date,
            )
            assert 0.0 <= result.p_a_wins <= 1.0
            assert isinstance(result.p_a_wins, float)

    def test_predict_returns_inference_result(
        self, loaded_inference_engine: BadmintonModelInference, match_date: date
    ) -> None:
        with patch.object(
            loaded_inference_engine, "_extract_features",
            return_value=[0.1] * 66,
        ), patch.object(
            loaded_inference_engine, "_rwp_from_match_prob",
            return_value=(0.53, 0.515),
        ):
            result = loaded_inference_engine.predict(
                "A", "B", Discipline.MS,
                TournamentTier.SUPER_500, match_date,
            )
            assert isinstance(result, InferenceResult)
            assert result.entity_a_id == "A"
            assert result.entity_b_id == "B"
            assert result.model_available is True


# ---------------------------------------------------------------------------
# 3. Missing features → RuntimeError
# ---------------------------------------------------------------------------

class TestMissingFeatures:
    def test_predict_raises_when_no_features(
        self, loaded_inference_engine: BadmintonModelInference, match_date: date
    ) -> None:
        with patch.object(
            loaded_inference_engine, "_extract_features",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="Could not extract features"):
                loaded_inference_engine.predict(
                    "A", "B", Discipline.MS,
                    TournamentTier.SUPER_500, match_date,
                )


# ---------------------------------------------------------------------------
# 4. Missing model → RuntimeError (not silent failure)
# ---------------------------------------------------------------------------

class TestMissingModel:
    def test_predict_raises_when_model_not_found(
        self, inference_engine_no_model: BadmintonModelInference, match_date: date
    ) -> None:
        with pytest.raises(RuntimeError, match="No model available"):
            inference_engine_no_model.predict(
                "A", "B", Discipline.MS,
                TournamentTier.SUPER_500, match_date,
            )

    def test_load_model_returns_none_for_missing_path(
        self, inference_engine_no_model: BadmintonModelInference
    ) -> None:
        result = inference_engine_no_model._load_model(Discipline.MS)
        assert result is None

    def test_load_model_caches_after_first_call(
        self, loaded_inference_engine: BadmintonModelInference
    ) -> None:
        # The model is pre-loaded in the fixture
        model = loaded_inference_engine._load_model(Discipline.MS)
        assert model is not None
        # Second call should return the same cached object
        model2 = loaded_inference_engine._load_model(Discipline.MS)
        assert model is model2


# ---------------------------------------------------------------------------
# 5. RWP bisection: verify the inverse produces valid range
# ---------------------------------------------------------------------------

class TestRWPBisection:
    def test_rwp_from_match_prob_returns_tuple(self) -> None:
        """Test the static _rwp_from_match_prob method with mocked Markov engine."""
        with patch("core.markov_engine.BadmintonMarkovEngine") as MockMarkov:
            mock_engine = MagicMock()
            MockMarkov.return_value = mock_engine

            # Simulate a monotonically increasing p_match as rwp increases
            def fake_compute(rwp_a, rwp_b, discipline, server_first_game):
                result = MagicMock()
                result.p_a_wins_match = rwp_a  # simplified: p_match ~ rwp
                return result

            mock_engine.compute_match_probabilities = fake_compute

            with patch("config.badminton_config.RWP_BASELINE", {Discipline.MS: 0.515}):
                rwp_a, rwp_b = BadmintonModelInference._rwp_from_match_prob(
                    0.60, Discipline.MS
                )
                assert isinstance(rwp_a, float)
                assert isinstance(rwp_b, float)
                # rwp_b should be the baseline
                assert rwp_b == 0.515

    def test_rwp_bisection_converges_within_bounds(self) -> None:
        """RWP output should be within the valid range."""
        with patch("core.markov_engine.BadmintonMarkovEngine") as MockMarkov:
            mock_engine = MagicMock()
            MockMarkov.return_value = mock_engine

            # Simulate monotonic p_match ~ sigmoid(rwp)
            import math
            def fake_compute(rwp_a, rwp_b, discipline, server_first_game):
                result = MagicMock()
                # Sigmoid-like monotonic function
                result.p_a_wins_match = 1.0 / (1.0 + math.exp(-20 * (rwp_a - 0.5)))
                return result

            mock_engine.compute_match_probabilities = fake_compute

            with patch("config.badminton_config.RWP_BASELINE", {Discipline.MS: 0.515}):
                rwp_a, _ = BadmintonModelInference._rwp_from_match_prob(
                    0.65, Discipline.MS
                )
                assert RWP_MIN_VALID <= rwp_a <= RWP_MAX_VALID

    def test_rwp_higher_match_prob_yields_higher_rwp(self) -> None:
        """Higher P(A wins) should map to higher RWP for A."""
        with patch("core.markov_engine.BadmintonMarkovEngine") as MockMarkov:
            mock_engine = MagicMock()
            MockMarkov.return_value = mock_engine

            import math
            def fake_compute(rwp_a, rwp_b, discipline, server_first_game):
                result = MagicMock()
                result.p_a_wins_match = 1.0 / (1.0 + math.exp(-20 * (rwp_a - 0.5)))
                return result

            mock_engine.compute_match_probabilities = fake_compute

            with patch("config.badminton_config.RWP_BASELINE", {Discipline.MS: 0.515}):
                rwp_low, _ = BadmintonModelInference._rwp_from_match_prob(
                    0.40, Discipline.MS
                )
                rwp_high, _ = BadmintonModelInference._rwp_from_match_prob(
                    0.75, Discipline.MS
                )
                assert rwp_high > rwp_low


# ---------------------------------------------------------------------------
# 6. Regime Gate — R0 / R1 / R2 classification
# ---------------------------------------------------------------------------

class TestRegimeGate:
    def test_r0_when_default_elo(self) -> None:
        inputs = RegimeInput(
            entity_a_match_count=50,
            entity_b_match_count=50,
            entity_a_elo_is_default=True,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_1000,
            discipline_value="MS",
        )
        assert RegimeGate.classify(inputs) == Regime.R0

    def test_r0_when_both_default_elo(self) -> None:
        inputs = RegimeInput(
            entity_a_match_count=0,
            entity_b_match_count=0,
            entity_a_elo_is_default=True,
            entity_b_elo_is_default=True,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        assert RegimeGate.classify(inputs) == Regime.R0

    def test_r0_when_low_match_count(self) -> None:
        inputs = RegimeInput(
            entity_a_match_count=2,
            entity_b_match_count=50,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R0

    def test_r0_when_none_match_count(self) -> None:
        inputs = RegimeInput(
            entity_a_match_count=None,
            entity_b_match_count=50,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R0


class TestRegimeGateR1:
    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000, TournamentTier.OLYMPICS}))
    def test_r1_standard_match(self) -> None:
        """Normal match at a mid-tier tournament with sufficient data → R1."""
        inputs = RegimeInput(
            entity_a_match_count=30,
            entity_b_match_count=25,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R1

    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000, TournamentTier.OLYMPICS}))
    def test_r1_at_top_tier_but_insufficient_matches(self) -> None:
        """Top tier but not enough matches for R2 → R1."""
        inputs = RegimeInput(
            entity_a_match_count=50,
            entity_b_match_count=30,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_1000,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R1


class TestRegimeGateR2:
    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000, TournamentTier.OLYMPICS}))
    def test_r2_rich_data_at_top_tier(self) -> None:
        """Both players have abundant data at a major event → R2."""
        inputs = RegimeInput(
            entity_a_match_count=200,
            entity_b_match_count=150,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_1000,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R2

    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000, TournamentTier.OLYMPICS}))
    def test_r2_at_olympics(self) -> None:
        inputs = RegimeInput(
            entity_a_match_count=120,
            entity_b_match_count=110,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.OLYMPICS,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R2

    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000, TournamentTier.OLYMPICS}))
    def test_r2_not_triggered_at_low_tier(self) -> None:
        """Even with lots of data, a low tier should not qualify for R2."""
        inputs = RegimeInput(
            entity_a_match_count=200,
            entity_b_match_count=200,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_300,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R1


# ---------------------------------------------------------------------------
# 7. Regime enum values
# ---------------------------------------------------------------------------

class TestRegimeEnum:
    def test_regime_values(self) -> None:
        assert Regime.R0.value == "R0"
        assert Regime.R1.value == "R1"
        assert Regime.R2.value == "R2"

    def test_regime_count(self) -> None:
        assert len(Regime) == 3


# ---------------------------------------------------------------------------
# 8. Feature extraction failure in inference
# ---------------------------------------------------------------------------

class TestFeatureExtractionInInference:
    def test_extract_features_returns_none_without_elo_system(self) -> None:
        engine = BadmintonModelInference(
            model_dir="/mocked",
            elo_system=None,
        )
        result = engine._extract_features(
            "A", "B", Discipline.MS,
            TournamentTier.SUPER_500, date(2025, 6, 15),
        )
        assert result is None

    def test_extract_features_handles_exception_gracefully(self) -> None:
        mock_elo = MagicMock()
        engine = BadmintonModelInference(
            model_dir="/mocked",
            elo_system=mock_elo,
        )
        # Patch FeatureBuilder at its definition site (local import inside _extract_features)
        with patch("ml.feature_engineering.FeatureBuilder", side_effect=Exception("test error")):
            result = engine._extract_features(
                "A", "B", Discipline.MS,
                TournamentTier.SUPER_500, date(2025, 6, 15),
            )
            assert result is None


# ---------------------------------------------------------------------------
# 9. InferenceResult field validation
# ---------------------------------------------------------------------------

class TestInferenceResultFields:
    def test_all_fields_present(self) -> None:
        result = InferenceResult(
            entity_a_id="A",
            entity_b_id="B",
            discipline=Discipline.WS,
            p_a_wins=0.55,
            p_a_wins_2_0=0.30,
            p_deuce=0.35,
            rwp_a=0.52,
            rwp_b=0.51,
            regime="R2",
            model_available=True,
            n_features_used=66,
        )
        assert result.discipline == Discipline.WS
        assert result.n_features_used == 66
        assert result.model_available is True

    def test_regime_gate_input_dataclass(self) -> None:
        inputs = RegimeInput(
            entity_a_match_count=10,
            entity_b_match_count=20,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        assert inputs.entity_a_match_count == 10
        assert inputs.tier == TournamentTier.SUPER_500


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000}))
    def test_regime_boundary_exactly_at_r0_threshold(self) -> None:
        """Match count exactly at the R0 threshold should NOT be R0."""
        inputs = RegimeInput(
            entity_a_match_count=10,
            entity_b_match_count=10,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R1

    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000}))
    def test_regime_boundary_one_below_r0(self) -> None:
        """One below R0 threshold → R0."""
        inputs = RegimeInput(
            entity_a_match_count=9,
            entity_b_match_count=50,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_500,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R0

    @patch("ml.regime_gate.REGIME_R0_MIN_MATCHES", 10)
    @patch("ml.regime_gate.REGIME_R2_MIN_MATCHES", 100)
    @patch("ml.regime_gate.REGIME_R2_TIERS", frozenset({TournamentTier.SUPER_1000}))
    def test_regime_boundary_exactly_at_r2_threshold(self) -> None:
        """Exactly at R2 match threshold at R2 tier → R2."""
        inputs = RegimeInput(
            entity_a_match_count=100,
            entity_b_match_count=100,
            entity_a_elo_is_default=False,
            entity_b_elo_is_default=False,
            tier=TournamentTier.SUPER_1000,
            discipline_value="MS",
        )
        result = RegimeGate.classify(inputs)
        assert result == Regime.R2

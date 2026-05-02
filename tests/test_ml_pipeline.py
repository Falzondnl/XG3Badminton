"""
test_ml_pipeline.py
===================
Tests for ML pipeline modules: ELO system, RegimeGate, FeatureBuilder groups,
and ModelInference.

Covers:
  - EloCalculator: expected_score, new_ratings, k_factor
  - BadmintonEloSystem: initialize, get_rating, update_after_match, match_probability
  - EloEntry construction and attributes
  - RegimeGate.classify: R0/R1/R2 cases
  - FeatureBuilder: group method interfaces (mocked dependencies)
  - BadmintonModelInference: predict() with no model loaded (graceful fallback)
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from ml.elo_system import (
    BadmintonEloSystem,
    EloCalculator,
    EloEntry,
    EloPool,
    EloSystemError,
)
from ml.regime_gate import Regime, RegimeGate, RegimeInput
from ml.inference import BadmintonModelInference, InferenceResult


# ---------------------------------------------------------------------------
# 1. EloCalculator — pure math
# ---------------------------------------------------------------------------

class TestEloCalculator:
    def test_expected_score_equal_ratings(self) -> None:
        e = EloCalculator.expected_score(1500.0, 1500.0)
        assert abs(e - 0.5) < 1e-6

    def test_expected_score_higher_rated_wins(self) -> None:
        e_high = EloCalculator.expected_score(1700.0, 1500.0)
        e_low  = EloCalculator.expected_score(1500.0, 1700.0)
        assert e_high > 0.5
        assert e_low < 0.5
        assert abs(e_high + e_low - 1.0) < 1e-6

    def test_expected_score_range(self) -> None:
        for diff in [-400, -200, 0, 200, 400]:
            e = EloCalculator.expected_score(1500.0 + diff, 1500.0)
            assert 0.0 < e < 1.0

    def test_new_ratings_winner_increases(self) -> None:
        r_win, r_lose = EloCalculator.new_ratings(
            rating_winner=1500.0, rating_loser=1500.0, k=20
        )
        assert r_win > 1500.0
        assert r_lose < 1500.0

    def test_new_ratings_upset_smaller_gain(self) -> None:
        """Winner with higher rating gains less than an upset winner."""
        r_fav_win, r_fav_lose = EloCalculator.new_ratings(
            rating_winner=1700.0, rating_loser=1300.0, k=20
        )
        r_upset_win, r_upset_lose = EloCalculator.new_ratings(
            rating_winner=1300.0, rating_loser=1700.0, k=20
        )
        # Upset winner gains more rating
        gain_fav = r_fav_win - 1700.0
        gain_upset = r_upset_win - 1300.0
        assert gain_upset > gain_fav

    def test_new_ratings_sum_preserved(self) -> None:
        """Sum of ratings is conserved after update."""
        rating_w, rating_l = 1600.0, 1400.0
        r_w, r_l = EloCalculator.new_ratings(
            rating_winner=rating_w, rating_loser=rating_l, k=20
        )
        assert abs((r_w + r_l) - (rating_w + rating_l)) < 1e-6

    def test_k_factor_returns_positive(self) -> None:
        k = EloCalculator.k_factor(
            tier=TournamentTier.SUPER_1000,
            discipline=Discipline.MS,
        )
        assert k > 0

    def test_k_factor_with_age(self) -> None:
        k = EloCalculator.k_factor(
            tier=TournamentTier.SUPER_500,
            discipline=Discipline.MS,
            age=22.0,
        )
        assert k > 0


# ---------------------------------------------------------------------------
# 2. EloEntry construction
# ---------------------------------------------------------------------------

class TestEloEntry:
    def test_constructs_defaults(self) -> None:
        entry = EloEntry(entity_id="P001", pool=EloPool.MS_OVERALL)
        assert entry.entity_id == "P001"
        assert entry.rating == 1500.0
        assert entry.matches_played == 0
        assert entry.peak_rating == 1500.0

    def test_constructs_custom_rating(self) -> None:
        entry = EloEntry(
            entity_id="P002",
            pool=EloPool.WS_OVERALL,
            rating=1800.0,
            matches_played=150,
            peak_rating=1820.0,
        )
        assert entry.rating == 1800.0
        assert entry.matches_played == 150

    @pytest.mark.parametrize("pool", list(EloPool))
    def test_all_pools_construct(self, pool: EloPool) -> None:
        entry = EloEntry(entity_id="P000", pool=pool)
        assert entry.pool == pool


# ---------------------------------------------------------------------------
# 3. BadmintonEloSystem — full lifecycle
# ---------------------------------------------------------------------------

class TestBadmintonEloSystem:
    def test_initialize_and_get_rating(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("P001", Discipline.MS)
        r = elo.get_rating("P001", Discipline.MS)
        assert r == 1500.0

    def test_get_rating_uninitialized_returns_default(self) -> None:
        elo = BadmintonEloSystem()
        result = elo.get_rating_or_default("UNKNOWN_PLAYER", Discipline.MS)
        # Returns (rating, is_default) tuple or just a float depending on impl
        rating = result[0] if isinstance(result, tuple) else result
        assert rating == 1500.0

    def test_update_after_match_winner_increases(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("P001", Discipline.MS)
        elo.initialize_player("P002", Discipline.MS)
        new_w, new_l = elo.update_after_match(
            winner_entity_id="P001",
            loser_entity_id="P002",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_500,
            match_date=date(2025, 6, 15),
        )
        assert new_w > 1500.0
        assert new_l < 1500.0

    def test_update_after_match_returns_tuple(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("PA", Discipline.WS)
        elo.initialize_player("PB", Discipline.WS)
        result = elo.update_after_match(
            winner_entity_id="PA",
            loser_entity_id="PB",
            discipline=Discipline.WS,
            tier=TournamentTier.SUPER_1000,
            match_date=date(2025, 3, 20),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_match_probability_equal_ratings(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("P1", Discipline.MS)
        elo.initialize_player("P2", Discipline.MS)
        p = elo.match_probability("P1", "P2", Discipline.MS)
        assert abs(p - 0.5) < 0.01

    def test_match_probability_higher_elo_wins(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("STRONG", Discipline.MS, initial_rating=1800.0)
        elo.initialize_player("WEAK",   Discipline.MS, initial_rating=1400.0)
        p = elo.match_probability("STRONG", "WEAK", Discipline.MS)
        assert p > 0.5

    def test_match_probability_range(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("A", Discipline.MS, initial_rating=1700.0)
        elo.initialize_player("B", Discipline.MS, initial_rating=1300.0)
        p = elo.match_probability("A", "B", Discipline.MS)
        assert 0.0 < p < 1.0

    def test_elo_diff_returns_float(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("X", Discipline.MS, initial_rating=1600.0)
        elo.initialize_player("Y", Discipline.MS, initial_rating=1400.0)
        diff = elo.elo_diff("X", "Y", Discipline.MS)
        assert isinstance(diff, float)
        assert diff == pytest.approx(200.0, abs=1.0)

    def test_snapshot_returns_dict(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("P1", Discipline.MS)
        snap = elo.snapshot(Discipline.MS)
        assert isinstance(snap, dict)
        assert "P1" in snap

    def test_multiple_updates_accumulate(self) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("PA", Discipline.MS)
        elo.initialize_player("PB", Discipline.MS)
        for i in range(5):
            elo.update_after_match(
                winner_entity_id="PA",
                loser_entity_id="PB",
                discipline=Discipline.MS,
                tier=TournamentTier.SUPER_300,
                match_date=date(2025, 1, i + 1),
            )
        r_pa = elo.get_rating("PA", Discipline.MS)
        r_pb = elo.get_rating("PB", Discipline.MS)
        assert r_pa > 1500.0
        assert r_pb < 1500.0

    @pytest.mark.parametrize("disc", list(Discipline))
    def test_all_disciplines_supported(self, disc: Discipline) -> None:
        elo = BadmintonEloSystem()
        elo.initialize_player("P1", disc)
        elo.initialize_player("P2", disc)
        elo.update_after_match(
            winner_entity_id="P1",
            loser_entity_id="P2",
            discipline=disc,
            tier=TournamentTier.SUPER_500,
            match_date=date(2025, 6, 1),
        )
        p = elo.match_probability("P1", "P2", disc)
        assert 0.0 < p < 1.0


# ---------------------------------------------------------------------------
# 4. RegimeGate
# ---------------------------------------------------------------------------

class TestRegimeGate:
    def _input(
        self,
        a_count: int = 50,
        b_count: int = 50,
        a_default: bool = False,
        b_default: bool = False,
        tier: TournamentTier = TournamentTier.SUPER_500,
        disc: str = "MS",
    ) -> RegimeInput:
        return RegimeInput(
            entity_a_match_count=a_count,
            entity_b_match_count=b_count,
            entity_a_elo_is_default=a_default,
            entity_b_elo_is_default=b_default,
            tier=tier,
            discipline_value=disc,
        )

    def test_r2_for_super_1000(self) -> None:
        inp = self._input(tier=TournamentTier.SUPER_1000, a_count=100, b_count=100)
        regime = RegimeGate.classify(inp)
        assert regime == Regime.R2

    def test_r2_for_super_500(self) -> None:
        inp = self._input(tier=TournamentTier.SUPER_500, a_count=80, b_count=80)
        regime = RegimeGate.classify(inp)
        assert regime in (Regime.R2, Regime.R1)  # tier-dependent impl

    def test_r0_for_very_sparse_data(self) -> None:
        inp = self._input(
            a_count=2, b_count=2,
            a_default=True, b_default=True,
            tier=TournamentTier.SUPER_300,
        )
        regime = RegimeGate.classify(inp)
        assert regime == Regime.R0

    def test_r1_standard(self) -> None:
        inp = self._input(
            a_count=30, b_count=30,
            tier=TournamentTier.SUPER_300,
        )
        regime = RegimeGate.classify(inp)
        assert regime in (Regime.R1, Regime.R2)

    def test_returns_regime_enum(self) -> None:
        inp = self._input()
        result = RegimeGate.classify(inp)
        assert isinstance(result, Regime)

    def test_none_match_count_handled(self) -> None:
        """None match counts should default to R0 (sparse)."""
        inp = RegimeInput(
            entity_a_match_count=None,
            entity_b_match_count=None,
            entity_a_elo_is_default=True,
            entity_b_elo_is_default=True,
            tier=TournamentTier.SUPER_300,
            discipline_value="MS",
        )
        regime = RegimeGate.classify(inp)
        assert regime == Regime.R0

    @pytest.mark.parametrize("disc", ["MS", "WS", "MD", "WD", "XD"])
    def test_all_disciplines_classify(self, disc: str) -> None:
        inp = self._input(disc=disc)
        result = RegimeGate.classify(inp)
        assert isinstance(result, Regime)


# ---------------------------------------------------------------------------
# 5. FeatureBuilder — group method interfaces
# ---------------------------------------------------------------------------

class TestFeatureBuilderInterface:
    def _make_builder(self) -> object:
        from ml.feature_engineering import FeatureBuilder
        elo = BadmintonEloSystem()
        elo.initialize_player("PA", Discipline.MS)
        elo.initialize_player("PB", Discipline.MS)
        weekly_db = MagicMock()
        serve_db = MagicMock()
        weekly_db.get_ranking.return_value = None
        serve_db.get_stats.return_value = None
        return FeatureBuilder(elo_system=elo, weekly_rankings_db=weekly_db, serve_stat_db=serve_db)

    def test_constructs(self) -> None:
        builder = self._make_builder()
        assert builder is not None

    def test_has_all_group_methods(self) -> None:
        from ml.feature_engineering import FeatureBuilder
        for group in ["group_a_elo_ranking", "group_b_recent_form", "group_c_h2h",
                      "group_d_tournament_context", "group_e_fatigue_schedule",
                      "group_f_rwp_estimates", "group_g_doubles",
                      "group_h_physical", "group_i_llm"]:
            assert hasattr(FeatureBuilder, group), f"Missing method: {group}"


# ---------------------------------------------------------------------------
# 6. ModelInference — no model loaded → graceful fallback
# ---------------------------------------------------------------------------

class TestModelInference:
    def test_constructs_without_model(self) -> None:
        inf = BadmintonModelInference()
        assert inf is not None

    def test_predict_without_model_returns_result(self) -> None:
        """When no .pkl model is loaded, predict() should either raise
        or return a result with model_available=False. Both are acceptable."""
        from datetime import date as _date
        inf = BadmintonModelInference()
        try:
            result = inf.predict(
                entity_a_id="PA",
                entity_b_id="PB",
                discipline=Discipline.MS,
                tier=TournamentTier.SUPER_500,
                match_date=_date(2025, 6, 15),
            )
            # If it returns without raising: model_available should be False
            assert isinstance(result, InferenceResult)
            assert not result.model_available
        except (RuntimeError, FileNotFoundError, ValueError):
            # Expected when model files aren't present
            pass

    def test_inference_result_constructs(self) -> None:
        result = InferenceResult(
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.MS,
            p_a_wins=0.60,
            p_a_wins_2_0=0.35,
            p_deuce=0.25,
            rwp_a=0.515,
            rwp_b=0.510,
            regime="R2",
            model_available=True,
            n_features_used=66,
        )
        assert result.p_a_wins == 0.60
        assert result.model_available is True
        assert result.n_features_used == 66

    def test_inference_result_probabilities_valid(self) -> None:
        result = InferenceResult(
            entity_a_id="PA",
            entity_b_id="PB",
            discipline=Discipline.WS,
            p_a_wins=0.55,
            p_a_wins_2_0=0.30,
            p_deuce=0.20,
            rwp_a=0.512,
            rwp_b=0.508,
            regime="R1",
            model_available=True,
            n_features_used=60,
        )
        assert 0.0 < result.p_a_wins < 1.0
        assert 0.0 <= result.p_a_wins_2_0 <= 1.0
        assert 0.0 <= result.p_deuce <= 1.0

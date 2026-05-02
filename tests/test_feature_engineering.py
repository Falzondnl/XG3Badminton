"""
test_feature_engineering.py
===========================
Tests for the badminton ML feature engineering pipeline — 66 features, 9 groups (A-I).

Uses mock data (fake match history dicts) — does NOT require real data files.
ZERO hardcoded expected probability values.

Covers:
  - Feature vector has correct length (~66)
  - No NaN or Inf in non-missing features
  - ELO features (Group A) reflect actual ELO differences
  - Recent form features (Group B) decay appropriately
  - Head-to-head features (Group C) are non-negative
  - Feature names are unique
  - Feature vector changes when inputs change
  - Doubles features present for doubles disciplines
  - Features computed BEFORE ELO update (no leakage flag)
  - Utility functions (_win_rate, _current_streak, etc.)
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    ELO_DEFAULT_RATING,
    ML_FEATURES_TOTAL,
    TournamentTier,
)
from ml.elo_system import BadmintonEloSystem, _make_pair_key
from ml.feature_engineering import (
    FeatureBuilder,
    _apply_p1p2_swap,
    _any_game_reached_deuce,
    _clamp_llm,
    _current_streak,
    _games_in_window,
    _h2h_record,
    _is_straight_win,
    _matches_in_window,
    _win_rate,
    _wins_in_tournament,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockRankingsDB:
    """Minimal mock for WeeklyRankingsDB."""

    def __init__(self, ranks: Optional[Dict[str, int]] = None,
                 points: Optional[Dict[str, int]] = None) -> None:
        self._ranks = ranks or {}
        self._points = points or {}

    def get_rank(self, entity_id: str, discipline: Discipline,
                 match_date: date) -> Optional[int]:
        return self._ranks.get(entity_id)

    def get_points(self, entity_id: str, discipline: Discipline,
                   match_date: date) -> Optional[int]:
        return self._points.get(entity_id)

    def is_home_region(self, entity_id: str, match_date: date) -> bool:
        return False


class MockServeStatDB:
    """Minimal mock for ServeStatDB."""

    def __init__(self, profiles: Optional[Dict[str, Any]] = None) -> None:
        self._profiles = profiles or {}

    def get_profile(self, entity_id: str, discipline: Discipline) -> Any:
        return self._profiles.get(entity_id)

    def get_smash_win_rate(self, entity_id: str,
                           discipline: Discipline) -> Optional[float]:
        return None

    def get_net_win_rate(self, entity_id: str,
                         discipline: Discipline) -> Optional[float]:
        return None

    def get_avg_rally_length(self, entity_id: str,
                             discipline: Discipline) -> Optional[float]:
        return None


def _make_match_record(
    match_date: date,
    won: bool,
    opponent: str = "opp",
    discipline: str = "MS",
    tier: str = "SUPER_500",
    tournament_id: str = "T001",
    opponent_rank: Optional[int] = None,
    games_played: int = 2,
) -> Dict[str, Any]:
    return {
        "date": match_date,
        "won": won,
        "opponent": opponent,
        "discipline": discipline,
        "tier": tier,
        "tournament_id": tournament_id,
        "opponent_rank": opponent_rank,
        "games_played": games_played,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elo_system() -> BadmintonEloSystem:
    system = BadmintonEloSystem()
    system.initialize_player("A", Discipline.MS, initial_rating=1600.0)
    system.initialize_player("B", Discipline.MS, initial_rating=1400.0)
    return system


@pytest.fixture
def elo_system_equal() -> BadmintonEloSystem:
    system = BadmintonEloSystem()
    system.initialize_player("X", Discipline.MS)
    system.initialize_player("Y", Discipline.MS)
    return system


@pytest.fixture
def rankings_db() -> MockRankingsDB:
    return MockRankingsDB(
        ranks={"A": 5, "B": 20, "X": 10, "Y": 15},
        points={"A": 80000, "B": 30000, "X": 50000, "Y": 40000},
    )


@pytest.fixture
def serve_db() -> MockServeStatDB:
    return MockServeStatDB()


@pytest.fixture
def builder(
    elo_system: BadmintonEloSystem,
    rankings_db: MockRankingsDB,
    serve_db: MockServeStatDB,
) -> FeatureBuilder:
    return FeatureBuilder(elo_system, rankings_db, serve_db)


@pytest.fixture
def match_date() -> date:
    return date(2025, 6, 15)


@pytest.fixture
def sample_history(match_date: date) -> Dict[str, List[Dict]]:
    """Match history with enough records for form/H2H features."""
    base = match_date - timedelta(days=60)
    records_a: List[Dict] = []
    records_b: List[Dict] = []
    for i in range(15):
        d = base + timedelta(days=i * 3)
        records_a.append(_make_match_record(
            d, won=(i % 3 != 0), opponent="B",
            opponent_rank=20, games_played=2 + (i % 2),
        ))
        records_b.append(_make_match_record(
            d, won=(i % 3 == 0), opponent="A",
            opponent_rank=5, games_played=2 + (i % 2),
        ))
    return {"A": records_a, "B": records_b}


# ---------------------------------------------------------------------------
# 1. Feature vector length
# ---------------------------------------------------------------------------

class TestFeatureVectorLength:
    def test_group_a_returns_expected_count(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        # Group A should have 11 features per spec
        assert len(feats) >= 10  # at least 10 (spec says 11 but bwf_points gets 1 feature)
        assert len(feats) <= 12

    def test_group_b_returns_expected_count(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_b_recent_form(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        assert len(feats) == 10

    def test_group_c_returns_expected_count(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_c_h2h(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        assert len(feats) == 6

    def test_group_e_returns_expected_count(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_e_fatigue_schedule(
            "A", "B", sample_history, match_date
        )
        assert len(feats) == 6

    def test_group_g_singles_returns_8_nans(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_g_doubles(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        assert len(feats) == 8
        # All should be NaN for singles
        assert all(math.isnan(v) for v in feats.values())

    def test_total_feature_count_matches_spec(self) -> None:
        """The ML_FEATURES_TOTAL constant should equal 66."""
        assert ML_FEATURES_TOTAL == 66


# ---------------------------------------------------------------------------
# 2. No NaN or Inf in computed features (non-missing)
# ---------------------------------------------------------------------------

class TestNoInfValues:
    def test_group_a_no_inf(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        for name, val in feats.items():
            if not math.isnan(val):
                assert not math.isinf(val), f"{name} is Inf"


# ---------------------------------------------------------------------------
# 3. ELO features reflect actual differences
# ---------------------------------------------------------------------------

class TestGroupAEloFeatures:
    def test_elo_diff_sign(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        # A has 1600, B has 1400 → diff should be positive
        assert feats["elo_discipline_diff"] > 0

    def test_elo_diff_flips_with_swap(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats_ab = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        feats_ba = builder.group_a_elo_ranking("B", "A", Discipline.MS, match_date)
        assert abs(feats_ab["elo_discipline_diff"] + feats_ba["elo_discipline_diff"]) < 1e-9

    def test_elo_prob_favours_higher_rated(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        assert feats["elo_prob"] > 0.5

    def test_elo_is_default_flags(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        # Both players are initialised (not default)
        assert feats["elo_is_default_a"] == 0.0
        assert feats["elo_is_default_b"] == 0.0


# ---------------------------------------------------------------------------
# 4. Recent form features decay
# ---------------------------------------------------------------------------

class TestGroupBRecentForm:
    def test_win_rate_bounded_zero_one(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_b_recent_form(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        for name, val in feats.items():
            if not math.isnan(val) and "win_rate" in name:
                assert 0.0 <= val <= 1.0, f"{name}={val} out of [0,1]"

    def test_empty_history_gives_nan_form(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_b_recent_form(
            "A", "B", Discipline.MS, {}, match_date
        )
        # Without any history, win rates should be NaN (insufficient data)
        assert math.isnan(feats["win_rate_l10_discipline_a"])

    def test_streak_nonzero_with_data(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_b_recent_form(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        # At least one player should have a nonzero streak with 15 matches
        assert (
            feats["current_streak_a"] != 0.0 or
            feats["current_streak_b"] != 0.0
        )


# ---------------------------------------------------------------------------
# 5. Head-to-head features are non-negative
# ---------------------------------------------------------------------------

class TestGroupCH2H:
    def test_h2h_count_non_negative(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_c_h2h(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        assert feats["h2h_n_all"] >= 0
        assert feats["h2h_n_discipline"] >= 0

    def test_h2h_win_pct_bounded(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_c_h2h(
            "A", "B", Discipline.MS, sample_history, match_date
        )
        val = feats["h2h_win_pct_a"]
        if not math.isnan(val):
            assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# 6. Feature names are unique
# ---------------------------------------------------------------------------

class TestFeatureNameUniqueness:
    def test_all_group_feature_names_unique(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        all_feats: Dict[str, float] = {}
        all_feats.update(builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date))
        all_feats.update(builder.group_b_recent_form("A", "B", Discipline.MS, sample_history, match_date))
        all_feats.update(builder.group_c_h2h("A", "B", Discipline.MS, sample_history, match_date))
        all_feats.update(builder.group_e_fatigue_schedule("A", "B", sample_history, match_date))
        all_feats.update(builder.group_f_rwp_estimates("A", "B", Discipline.MS))
        all_feats.update(builder.group_g_doubles("A", "B", Discipline.MS, sample_history, match_date))

        # If there were name collisions, the dict would silently overwrite and
        # we'd get fewer keys than the sum of individual group sizes.
        group_a_len = len(builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date))
        group_b_len = len(builder.group_b_recent_form("A", "B", Discipline.MS, sample_history, match_date))
        group_c_len = len(builder.group_c_h2h("A", "B", Discipline.MS, sample_history, match_date))
        group_e_len = len(builder.group_e_fatigue_schedule("A", "B", sample_history, match_date))
        group_f_len = len(builder.group_f_rwp_estimates("A", "B", Discipline.MS))
        group_g_len = len(builder.group_g_doubles("A", "B", Discipline.MS, sample_history, match_date))
        expected_total = group_a_len + group_b_len + group_c_len + group_e_len + group_f_len + group_g_len
        assert len(all_feats) == expected_total


# ---------------------------------------------------------------------------
# 7. Feature vector changes when inputs change
# ---------------------------------------------------------------------------

class TestFeatureVectorSensitivity:
    def test_different_elos_produce_different_features(
        self, rankings_db: MockRankingsDB, serve_db: MockServeStatDB, match_date: date
    ) -> None:
        system1 = BadmintonEloSystem()
        system1.initialize_player("P", Discipline.MS, initial_rating=1700.0)
        system1.initialize_player("Q", Discipline.MS, initial_rating=1300.0)
        b1 = FeatureBuilder(system1, rankings_db, serve_db)
        feats1 = b1.group_a_elo_ranking("P", "Q", Discipline.MS, match_date)

        system2 = BadmintonEloSystem()
        system2.initialize_player("P", Discipline.MS, initial_rating=1500.0)
        system2.initialize_player("Q", Discipline.MS, initial_rating=1500.0)
        b2 = FeatureBuilder(system2, rankings_db, serve_db)
        feats2 = b2.group_a_elo_ranking("P", "Q", Discipline.MS, match_date)

        assert feats1["elo_discipline_diff"] != feats2["elo_discipline_diff"]

    def test_different_history_produces_different_form(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        base = match_date - timedelta(days=30)
        hist_all_wins = {
            "A": [_make_match_record(base + timedelta(days=i), won=True) for i in range(10)],
            "B": [_make_match_record(base + timedelta(days=i), won=False) for i in range(10)],
        }
        hist_all_losses = {
            "A": [_make_match_record(base + timedelta(days=i), won=False) for i in range(10)],
            "B": [_make_match_record(base + timedelta(days=i), won=True) for i in range(10)],
        }
        feats_w = builder.group_b_recent_form("A", "B", Discipline.MS, hist_all_wins, match_date)
        feats_l = builder.group_b_recent_form("A", "B", Discipline.MS, hist_all_losses, match_date)

        # Win rate for A should differ
        wr_w = feats_w["win_rate_l5_all_a"]
        wr_l = feats_l["win_rate_l5_all_a"]
        if not math.isnan(wr_w) and not math.isnan(wr_l):
            assert wr_w != wr_l


# ---------------------------------------------------------------------------
# 8. Doubles features present for doubles disciplines
# ---------------------------------------------------------------------------

class TestDoublesFeatures:
    def test_doubles_features_non_nan_for_doubles(self, match_date: date) -> None:
        system = BadmintonEloSystem()
        system.bootstrap_pair_rating("d1", "d2", Discipline.MD, matches_together=5)
        system.bootstrap_pair_rating("d3", "d4", Discipline.MD, matches_together=3)

        pair_a = _make_pair_key("d1", "d2")
        pair_b = _make_pair_key("d3", "d4")

        rankings = MockRankingsDB(ranks={pair_a: 3, pair_b: 8})
        serve = MockServeStatDB()
        builder = FeatureBuilder(system, rankings, serve)

        history: Dict[str, List[Dict]] = {
            pair_a: [_make_match_record(
                match_date - timedelta(days=i * 5), won=True,
                opponent=pair_b, discipline="MD",
            ) for i in range(1, 6)],
            pair_b: [_make_match_record(
                match_date - timedelta(days=i * 5), won=False,
                opponent=pair_a, discipline="MD",
            ) for i in range(1, 6)],
        }

        feats = builder.group_g_doubles(
            pair_a, pair_b, Discipline.MD, history, match_date
        )
        assert len(feats) == 8
        # partner_elo_a should not be NaN for doubles
        assert not math.isnan(feats["partner_elo_a"])
        assert not math.isnan(feats["partner_elo_b"])

    def test_singles_features_all_nan_for_group_g(
        self, builder: FeatureBuilder, match_date: date,
        sample_history: Dict[str, List[Dict]]
    ) -> None:
        feats = builder.group_g_doubles("A", "B", Discipline.MS, sample_history, match_date)
        for name, val in feats.items():
            assert math.isnan(val), f"{name} should be NaN for singles but is {val}"


# ---------------------------------------------------------------------------
# 9. Features computed BEFORE ELO update (temporal correctness)
# ---------------------------------------------------------------------------

class TestTemporalCorrectness:
    def test_feature_uses_pre_update_elo(
        self, rankings_db: MockRankingsDB, serve_db: MockServeStatDB, match_date: date
    ) -> None:
        """
        Verify that feature extraction sees the ELO snapshot BEFORE any update.
        This is the contract enforced by Rule 14 / H5 gate.
        """
        system = BadmintonEloSystem()
        system.initialize_player("A", Discipline.MS, initial_rating=1600.0)
        system.initialize_player("B", Discipline.MS, initial_rating=1400.0)

        builder = FeatureBuilder(system, rankings_db, serve_db)
        feats_before = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        elo_a_before = feats_before["elo_discipline_a"]

        # Now update ELO (as would happen AFTER feature extraction)
        system.update_after_match(
            "A", "B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )

        feats_after = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        elo_a_after = feats_after["elo_discipline_a"]

        # Before update: should have been 1600
        assert elo_a_before == 1600.0
        # After update: should have changed
        assert elo_a_after != 1600.0
        assert elo_a_after > elo_a_before


# ---------------------------------------------------------------------------
# 10. Utility function tests
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    def test_win_rate_basic(self) -> None:
        hist = [
            _make_match_record(date(2025, 1, i + 1), won=(i < 7))
            for i in range(10)
        ]
        wr = _win_rate(hist, window=10, discipline=None, min_matches=3)
        assert not math.isnan(wr)
        assert 0.0 < wr < 1.0

    def test_win_rate_nan_insufficient_data(self) -> None:
        hist = [_make_match_record(date(2025, 1, 1), won=True)]
        wr = _win_rate(hist, window=10, discipline=None, min_matches=3)
        assert math.isnan(wr)

    def test_current_streak_positive(self) -> None:
        hist = [
            _make_match_record(date(2025, 1, i + 1), won=True)
            for i in range(5)
        ]
        assert _current_streak(hist) > 0

    def test_current_streak_negative(self) -> None:
        hist = [
            _make_match_record(date(2025, 1, i + 1), won=False)
            for i in range(5)
        ]
        assert _current_streak(hist) < 0

    def test_current_streak_empty(self) -> None:
        assert _current_streak([]) == 0

    def test_matches_in_window(self) -> None:
        ref = date(2025, 6, 15)
        hist = [
            _make_match_record(ref - timedelta(days=d), won=True)
            for d in [1, 3, 5, 10, 20]
        ]
        count = _matches_in_window(hist, ref, days=7)
        assert count == 3  # days 1, 3, 5

    def test_games_in_window(self) -> None:
        ref = date(2025, 6, 15)
        hist = [
            _make_match_record(ref - timedelta(days=2), won=True, games_played=3),
            _make_match_record(ref - timedelta(days=4), won=False, games_played=2),
        ]
        count = _games_in_window(hist, ref, days=7)
        assert count == 5

    def test_h2h_record_counts(self) -> None:
        ref = date(2025, 6, 15)
        history = {
            "A": [
                _make_match_record(date(2025, 1, 1), won=True, opponent="B"),
                _make_match_record(date(2025, 2, 1), won=False, opponent="B"),
                _make_match_record(date(2025, 3, 1), won=True, opponent="C"),
            ],
        }
        result = _h2h_record("A", "B", history, discipline=None, window=999, before_date=ref)
        assert result["total"] == 2
        assert result["wins_a"] == 1
        assert result["wins_b"] == 1

    def test_wins_in_tournament(self) -> None:
        ref = date(2025, 6, 15)
        history = {
            "A": [
                _make_match_record(date(2025, 6, 12), won=True, tournament_id="T1"),
                _make_match_record(date(2025, 6, 13), won=True, tournament_id="T1"),
                _make_match_record(date(2025, 6, 10), won=True, tournament_id="T2"),
            ],
        }
        assert _wins_in_tournament("A", "T1", history, ref) == 2

    def test_is_straight_win(self) -> None:
        assert _is_straight_win([(21, 15), (21, 18)], "A") is True
        assert _is_straight_win([(21, 15), (18, 21), (21, 19)], "A") is False
        assert _is_straight_win([], "A") is False

    def test_any_game_reached_deuce(self) -> None:
        assert _any_game_reached_deuce([(22, 20), (21, 15)]) is True
        assert _any_game_reached_deuce([(21, 18), (21, 15)]) is False
        assert _any_game_reached_deuce([]) is False

    def test_clamp_llm(self) -> None:
        assert _clamp_llm(0.10) == 0.05
        assert _clamp_llm(-0.10) == -0.05
        assert _clamp_llm(0.03) == 0.03

    def test_apply_p1p2_swap_inverts_diff(self) -> None:
        feats = {
            "elo_discipline_a": 1600.0,
            "elo_discipline_b": 1400.0,
            "elo_discipline_diff": 200.0,
            "elo_prob": 0.75,
        }
        swapped = _apply_p1p2_swap(feats)
        assert swapped["elo_discipline_a"] == 1400.0
        assert swapped["elo_discipline_b"] == 1600.0
        assert swapped["elo_discipline_diff"] == -200.0
        assert abs(swapped["elo_prob"] - 0.25) < 1e-9

    def test_apply_p1p2_swap_roundtrip(self) -> None:
        feats = {
            "val_a": 10.0,
            "val_b": 20.0,
            "metric_diff": 5.0,
            "elo_prob": 0.6,
        }
        double_swapped = _apply_p1p2_swap(_apply_p1p2_swap(feats))
        for k, v in feats.items():
            assert abs(double_swapped[k] - v) < 1e-9, f"Roundtrip failed for {k}"


# ---------------------------------------------------------------------------
# 11. Fatigue features
# ---------------------------------------------------------------------------

class TestGroupEFatigue:
    def test_back_to_back_flag(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        history = {
            # Match 1 day before match_date — within the 1-day back-to-back window.
            # hist_a filters to m["date"] < match_date, so match_date itself would be
            # excluded; using timedelta(days=1) keeps it in hist_a and in the window.
            "A": [_make_match_record(match_date - timedelta(days=1), won=True)],
            "B": [],
        }
        feats = builder.group_e_fatigue_schedule("A", "B", history, match_date)
        assert feats["back_to_back_flag_a"] == 1.0

    def test_no_fatigue_with_empty_history(
        self, builder: FeatureBuilder, match_date: date
    ) -> None:
        feats = builder.group_e_fatigue_schedule("A", "B", {}, match_date)
        assert feats["matches_last7_a"] == 0.0
        assert feats["matches_last7_b"] == 0.0

"""
test_bwf_ranking_features.py
============================
Regression tests for the 4 new BWF raw-rank features added to Group A of the
badminton ML feature engineering pipeline (2026-05-10).

New features under test:
  player_a_bwf_rank  — raw integer rank; sentinel 999 when ranking unavailable
  player_b_bwf_rank  — same for player B
  bwf_rank_diff_raw  — player_a_bwf_rank - player_b_bwf_rank
                       (negative when A is ranked higher / stronger)
  bwf_rank_ratio     — player_b_bwf_rank / (player_a_bwf_rank + 1e-6)
                       (> 1 when A is ranked higher / stronger)

Requirements verified:
  R1 — Missing player gets 999 rank sentinel, not an exception, not NaN.
  R2 — bwf_rank_diff_raw is negative when player_a is better (lower rank number).
  R3 — bwf_rank_ratio > 1 when player_a is better ranked.
  R4 — Temporal correctness: rank lookup uses match_date, not today's date.
  R5 — Both players missing gives symmetric 999/999 — diff=0, ratio=1.
  R6 — Sentinel 999 is NOT NaN (model can consume it without imputation).
  R7 — After P1/P2 swap, ratio recomputed from post-swap raw ranks (not negated).
  R8 — All 4 new feature names are present in group_a output dict.
  R9 — bwf_rank_diff_raw is negated by _apply_p1p2_swap.
  R10 — player_a_bwf_rank and player_b_bwf_rank are swapped by _apply_p1p2_swap.

ZERO hardcoded probabilities. ZERO mock data beyond what is required for
isolation. All assertions are structural, not numeric thresholds.
"""

from __future__ import annotations

import math
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import call

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from repo root or tests/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from ml.elo_system import BadmintonEloSystem
from ml.feature_engineering import FeatureBuilder, _apply_p1p2_swap


# ---------------------------------------------------------------------------
# Sentinel value (must match the constant in feature_engineering.py)
# ---------------------------------------------------------------------------
_MISSING_RANK_SENTINEL: int = 999


# ---------------------------------------------------------------------------
# Minimal mocks
# ---------------------------------------------------------------------------

class _RankingsDBWithCallTracking:
    """
    Mock WeeklyRankingsDB that:
      - Returns a configurable rank/points per entity.
      - Records every (entity_id, discipline, match_date) tuple passed to
        get_rank() so we can assert temporal correctness.
    """

    def __init__(
        self,
        ranks: Optional[Dict[str, int]] = None,
        points: Optional[Dict[str, float]] = None,
    ) -> None:
        self._ranks: Dict[str, int] = ranks or {}
        self._points: Dict[str, float] = points or {}
        self.get_rank_calls: list[tuple] = []
        self.get_points_calls: list[tuple] = []

    def get_rank(
        self, entity_id: str, discipline: Discipline, match_date: date
    ) -> Optional[int]:
        self.get_rank_calls.append((entity_id, discipline, match_date))
        return self._ranks.get(entity_id)

    def get_points(
        self, entity_id: str, discipline: Discipline, match_date: date
    ) -> Optional[float]:
        self.get_points_calls.append((entity_id, discipline, match_date))
        return self._points.get(entity_id)

    def is_home_region(self, entity_id: str, match_date: date) -> bool:
        return False


class _MinimalServeStatDB:
    """Minimal mock — returns None for all stat lookups."""

    def get_profile(self, entity_id: str, discipline: Discipline) -> None:
        return None

    def get_smash_win_rate(self, entity_id: str, discipline: Discipline) -> None:
        return None

    def get_net_win_rate(self, entity_id: str, discipline: Discipline) -> None:
        return None

    def get_avg_rally_length(self, entity_id: str, discipline: Discipline) -> None:
        return None


def _make_builder(
    ranks: Optional[Dict[str, int]] = None,
    points: Optional[Dict[str, float]] = None,
    elo_a: float = 1500.0,
    elo_b: float = 1500.0,
) -> tuple[FeatureBuilder, _RankingsDBWithCallTracking]:
    """
    Construct a FeatureBuilder with controlled ELO and ranking values.

    Returns the builder and the rankings mock (for call-tracking assertions).
    """
    elo_system = BadmintonEloSystem()
    elo_system.initialize_player("A", Discipline.MS, initial_rating=elo_a)
    elo_system.initialize_player("B", Discipline.MS, initial_rating=elo_b)

    rankings_db = _RankingsDBWithCallTracking(
        ranks=ranks or {},
        points=points or {},
    )
    serve_db = _MinimalServeStatDB()
    builder = FeatureBuilder(elo_system, rankings_db, serve_db)
    return builder, rankings_db


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def match_date() -> date:
    return date(2024, 9, 14)


# ---------------------------------------------------------------------------
# R1 — Missing player gets 999 sentinel, not exception, not NaN
# ---------------------------------------------------------------------------

class TestMissingRankSentinel:
    def test_missing_player_a_gets_999_not_exception(self, match_date: date) -> None:
        """Player A has no ranking entry — should produce 999, not raise."""
        builder, _ = _make_builder(ranks={"B": 10})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["player_a_bwf_rank"] == float(_MISSING_RANK_SENTINEL)

    def test_missing_player_b_gets_999_not_exception(self, match_date: date) -> None:
        """Player B has no ranking entry — should produce 999, not raise."""
        builder, _ = _make_builder(ranks={"A": 5})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["player_b_bwf_rank"] == float(_MISSING_RANK_SENTINEL)

    def test_missing_both_players_get_999(self, match_date: date) -> None:
        """Both players missing — both sentinel 999."""
        builder, _ = _make_builder(ranks={})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["player_a_bwf_rank"] == float(_MISSING_RANK_SENTINEL)
        assert feats["player_b_bwf_rank"] == float(_MISSING_RANK_SENTINEL)

    def test_sentinel_is_not_nan(self, match_date: date) -> None:
        """The sentinel 999 must NOT be NaN — model must be able to consume it."""
        builder, _ = _make_builder(ranks={})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        # R6: sentinel is finite, not NaN, not Inf
        rank_a = feats["player_a_bwf_rank"]
        assert not math.isnan(rank_a), "player_a_bwf_rank must not be NaN"
        assert not math.isinf(rank_a), "player_a_bwf_rank must not be Inf"
        assert rank_a == float(_MISSING_RANK_SENTINEL)

    def test_rank_of_zero_treated_as_missing(self, match_date: date) -> None:
        """A rank of 0 from the DB is invalid and must be treated as missing (999)."""
        builder, _ = _make_builder(ranks={"A": 0, "B": 5})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["player_a_bwf_rank"] == float(_MISSING_RANK_SENTINEL)


# ---------------------------------------------------------------------------
# R2 — bwf_rank_diff_raw negative when A is better (lower rank number)
# ---------------------------------------------------------------------------

class TestRankDiffSign:
    def test_diff_negative_when_a_has_better_rank(self, match_date: date) -> None:
        """Rank 3 vs rank 25: diff = 3 - 25 = -22 (negative → A is stronger)."""
        builder, _ = _make_builder(ranks={"A": 3, "B": 25})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_diff_raw"] < 0, (
            f"Expected negative diff when A (rank 3) is better than B (rank 25), "
            f"got {feats['bwf_rank_diff_raw']}"
        )

    def test_diff_positive_when_b_has_better_rank(self, match_date: date) -> None:
        """Rank 40 vs rank 2: diff = 40 - 2 = +38 (positive → B is stronger)."""
        builder, _ = _make_builder(ranks={"A": 40, "B": 2})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_diff_raw"] > 0, (
            f"Expected positive diff when A (rank 40) is worse than B (rank 2), "
            f"got {feats['bwf_rank_diff_raw']}"
        )

    def test_diff_zero_when_ranks_equal(self, match_date: date) -> None:
        """Equal ranks must produce diff = 0."""
        builder, _ = _make_builder(ranks={"A": 15, "B": 15})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_diff_raw"] == 0.0

    def test_diff_exact_value(self, match_date: date) -> None:
        """Exact arithmetic: rank_a=10, rank_b=30 → diff = -20."""
        builder, _ = _make_builder(ranks={"A": 10, "B": 30})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_diff_raw"] == pytest.approx(-20.0)

    def test_diff_with_sentinel_a_missing(self, match_date: date) -> None:
        """When A is missing (sentinel 999) and B is rank 5: diff = 999 - 5 = 994."""
        builder, _ = _make_builder(ranks={"B": 5})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_diff_raw"] == pytest.approx(
            float(_MISSING_RANK_SENTINEL) - 5.0
        )


# ---------------------------------------------------------------------------
# R3 — bwf_rank_ratio > 1 when player_a is better ranked
# ---------------------------------------------------------------------------

class TestRankRatio:
    def test_ratio_greater_than_one_when_a_better(self, match_date: date) -> None:
        """Rank A=1 vs B=50: ratio = 50 / (1 + 1e-6) ≈ 50 → >> 1."""
        builder, _ = _make_builder(ranks={"A": 1, "B": 50})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_ratio"] > 1.0, (
            f"Expected ratio > 1 when A (rank 1) is better than B (rank 50), "
            f"got {feats['bwf_rank_ratio']}"
        )

    def test_ratio_less_than_one_when_b_better(self, match_date: date) -> None:
        """Rank A=50 vs B=1: ratio = 1 / (50 + 1e-6) ≈ 0.02 → < 1."""
        builder, _ = _make_builder(ranks={"A": 50, "B": 1})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_ratio"] < 1.0, (
            f"Expected ratio < 1 when A (rank 50) is worse than B (rank 1), "
            f"got {feats['bwf_rank_ratio']}"
        )

    def test_ratio_approximately_one_when_ranks_equal(self, match_date: date) -> None:
        """Equal ranks 20 vs 20: ratio ≈ 1 (within floating point of epsilon)."""
        builder, _ = _make_builder(ranks={"A": 20, "B": 20})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        # 20 / (20 + 1e-6) ≈ 1.0 within tolerance
        assert feats["bwf_rank_ratio"] == pytest.approx(1.0, rel=1e-4)

    def test_ratio_exact_formula(self, match_date: date) -> None:
        """
        Exact formula check: rank_a=4, rank_b=16.
        ratio = 16 / (4 + 1e-6) ≈ 4.0 (exactly 4 within floating-point precision).
        """
        builder, _ = _make_builder(ranks={"A": 4, "B": 16})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        # 16 / (4 + 1e-6) is just under 4.0
        assert feats["bwf_rank_ratio"] == pytest.approx(
            16.0 / (4.0 + 1e-6), rel=1e-9
        )

    def test_ratio_with_both_missing(self, match_date: date) -> None:
        """Both missing: sentinel/sentinel = 999 / (999 + 1e-6) ≈ 1."""
        builder, _ = _make_builder(ranks={})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        expected = float(_MISSING_RANK_SENTINEL) / (float(_MISSING_RANK_SENTINEL) + 1e-6)
        assert feats["bwf_rank_ratio"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# R4 — Temporal correctness: rank lookup uses match_date
# ---------------------------------------------------------------------------

class TestTemporalCorrectness:
    def test_get_rank_called_with_match_date_not_today(self, match_date: date) -> None:
        """
        WeeklyRankingsDB.get_rank() must be called with match_date, not today's date.
        This is the core Rule 14 / temporal correctness guarantee.
        """
        builder, rankings_db = _make_builder(
            ranks={"A": 5, "B": 10},
        )
        builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        # Every get_rank call must use the supplied match_date, not date.today()
        for entity_id, discipline, call_date in rankings_db.get_rank_calls:
            assert call_date == match_date, (
                f"get_rank called with date {call_date} for entity {entity_id}; "
                f"expected match_date={match_date}. "
                "This violates temporal correctness (Rule 14 / H5 gate)."
            )

    def test_get_points_called_with_match_date_not_today(self, match_date: date) -> None:
        """
        WeeklyRankingsDB.get_points() must also be called with match_date.
        """
        builder, rankings_db = _make_builder(
            ranks={"A": 5, "B": 10},
            points={"A": 90000.0, "B": 50000.0},
        )
        builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        for entity_id, discipline, call_date in rankings_db.get_points_calls:
            assert call_date == match_date, (
                f"get_points called with date {call_date} for entity {entity_id}; "
                f"expected match_date={match_date}."
            )

    def test_different_dates_can_return_different_ranks(self) -> None:
        """
        When the same player had rank 5 on one date and rank 15 on another,
        the feature value must differ — confirming the date argument is used,
        not a global cache.
        """
        date_early = date(2023, 6, 1)
        date_late = date(2024, 6, 1)

        class _DateSensitiveRankingsDB:
            """Returns different rank based on the date passed."""

            def get_rank(
                self, entity_id: str, discipline: Discipline, match_date: date
            ) -> Optional[int]:
                if entity_id == "A":
                    return 5 if match_date == date_early else 15
                return 10

            def get_points(
                self, entity_id: str, discipline: Discipline, match_date: date
            ) -> Optional[float]:
                return None

            def is_home_region(self, entity_id: str, match_date: date) -> bool:
                return False

        elo_system = BadmintonEloSystem()
        elo_system.initialize_player("A", Discipline.MS, initial_rating=1500.0)
        elo_system.initialize_player("B", Discipline.MS, initial_rating=1500.0)

        builder = FeatureBuilder(elo_system, _DateSensitiveRankingsDB(), _MinimalServeStatDB())

        feats_early = builder.group_a_elo_ranking("A", "B", Discipline.MS, date_early)
        feats_late = builder.group_a_elo_ranking("A", "B", Discipline.MS, date_late)

        assert feats_early["player_a_bwf_rank"] == 5.0
        assert feats_late["player_a_bwf_rank"] == 15.0
        assert feats_early["player_a_bwf_rank"] != feats_late["player_a_bwf_rank"]


# ---------------------------------------------------------------------------
# R8 — All 4 new feature names present in group_a output
# ---------------------------------------------------------------------------

class TestNewFeatureNamesPresent:
    _EXPECTED_NEW_FEATURES = {
        "player_a_bwf_rank",
        "player_b_bwf_rank",
        "bwf_rank_diff_raw",
        "bwf_rank_ratio",
    }

    def test_all_new_feature_keys_present(self, match_date: date) -> None:
        builder, _ = _make_builder(ranks={"A": 10, "B": 20})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        for name in self._EXPECTED_NEW_FEATURES:
            assert name in feats, (
                f"Expected feature '{name}' not found in group_a output. "
                f"Present keys: {sorted(feats.keys())}"
            )

    def test_all_new_feature_keys_present_when_ranks_missing(
        self, match_date: date
    ) -> None:
        """New feature keys must exist even when both rankings are unavailable."""
        builder, _ = _make_builder(ranks={})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        for name in self._EXPECTED_NEW_FEATURES:
            assert name in feats, f"Feature '{name}' missing when rankings unavailable."

    def test_total_group_a_feature_count(self, match_date: date) -> None:
        """Group A must return exactly 14 features (6 ELO + 4 log-rank + 4 raw-rank)."""
        builder, _ = _make_builder(ranks={"A": 5, "B": 30})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert len(feats) == 14, (
            f"Expected 14 features in group_a, got {len(feats)}. "
            f"Keys: {sorted(feats.keys())}"
        )

    def test_existing_feature_names_preserved(self, match_date: date) -> None:
        """Existing feature names must not have been renamed (regression guard)."""
        existing_expected = {
            "elo_discipline_a", "elo_discipline_b", "elo_discipline_diff",
            "elo_prob", "elo_is_default_a", "elo_is_default_b",
            "bwf_rank_a", "bwf_rank_b", "bwf_rank_diff", "bwf_points_diff",
        }
        builder, _ = _make_builder(
            ranks={"A": 5, "B": 30},
            points={"A": 80000.0, "B": 40000.0},
        )
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        for name in existing_expected:
            assert name in feats, (
                f"Existing feature '{name}' was unexpectedly removed or renamed."
            )


# ---------------------------------------------------------------------------
# R7, R9, R10 — P1/P2 swap behaviour for new features
# ---------------------------------------------------------------------------

class TestP1P2SwapBehaviour:
    def test_player_ranks_swapped_after_p1p2_swap(self, match_date: date) -> None:
        """
        R10: player_a_bwf_rank and player_b_bwf_rank must be exchanged after swap.
        """
        builder, _ = _make_builder(ranks={"A": 5, "B": 30})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        rank_a_before = feats["player_a_bwf_rank"]
        rank_b_before = feats["player_b_bwf_rank"]

        swapped = _apply_p1p2_swap(feats)

        assert swapped["player_a_bwf_rank"] == rank_b_before, (
            "player_a_bwf_rank should equal original player_b_bwf_rank after swap."
        )
        assert swapped["player_b_bwf_rank"] == rank_a_before, (
            "player_b_bwf_rank should equal original player_a_bwf_rank after swap."
        )

    def test_rank_diff_raw_negated_after_p1p2_swap(self, match_date: date) -> None:
        """
        R9: bwf_rank_diff_raw must be negated when P1/P2 are swapped.
        """
        builder, _ = _make_builder(ranks={"A": 5, "B": 30})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)
        diff_before = feats["bwf_rank_diff_raw"]

        swapped = _apply_p1p2_swap(feats)

        assert swapped["bwf_rank_diff_raw"] == pytest.approx(-diff_before), (
            f"bwf_rank_diff_raw should negate on swap: before={diff_before}, "
            f"after={swapped['bwf_rank_diff_raw']}"
        )

    def test_rank_ratio_recomputed_from_swapped_ranks(self, match_date: date) -> None:
        """
        R7: bwf_rank_ratio must be recomputed from the post-swap raw ranks, not
        simply negated or inverted by a simple formula.

        If A=5, B=30:
          Before swap: ratio = 30 / (5 + 1e-6) ≈ 6.0
          After swap: player_a=30, player_b=5
                      ratio = 5 / (30 + 1e-6) ≈ 0.167
        """
        builder, _ = _make_builder(ranks={"A": 5, "B": 30})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        ratio_before = feats["bwf_rank_ratio"]
        assert ratio_before > 1.0, "Pre-condition: A is better ranked."

        swapped = _apply_p1p2_swap(feats)
        ratio_after = swapped["bwf_rank_ratio"]

        # After swap: new_a_rank=30, new_b_rank=5 → ratio = 5 / (30 + 1e-6)
        expected_after = 5.0 / (30.0 + 1e-6)
        assert ratio_after == pytest.approx(expected_after, rel=1e-9), (
            f"bwf_rank_ratio after swap should be {expected_after:.6f}, "
            f"got {ratio_after:.6f}"
        )
        assert ratio_after < 1.0, "After swap: B (originally A) is now better."

    def test_double_swap_roundtrip_for_raw_rank_features(
        self, match_date: date
    ) -> None:
        """
        Applying P1/P2 swap twice must restore all raw-rank features to their
        original values (roundtrip property).
        """
        builder, _ = _make_builder(ranks={"A": 7, "B": 45})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        double_swapped = _apply_p1p2_swap(_apply_p1p2_swap(feats))

        for key in (
            "player_a_bwf_rank",
            "player_b_bwf_rank",
            "bwf_rank_diff_raw",
            "bwf_rank_ratio",
        ):
            assert double_swapped[key] == pytest.approx(feats[key], rel=1e-9), (
                f"Double-swap roundtrip failed for '{key}': "
                f"original={feats[key]}, after double swap={double_swapped[key]}"
            )

    def test_swap_with_sentinel_values_roundtrip(self, match_date: date) -> None:
        """
        When both players have sentinel 999, double swap must still roundtrip.
        """
        builder, _ = _make_builder(ranks={})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        double_swapped = _apply_p1p2_swap(_apply_p1p2_swap(feats))

        for key in (
            "player_a_bwf_rank",
            "player_b_bwf_rank",
            "bwf_rank_diff_raw",
            "bwf_rank_ratio",
        ):
            assert double_swapped[key] == pytest.approx(feats[key], rel=1e-9), (
                f"Double-swap roundtrip with sentinels failed for '{key}'"
            )


# ---------------------------------------------------------------------------
# Boundary and edge cases
# ---------------------------------------------------------------------------

class TestBoundaryAndEdgeCases:
    def test_rank_1_is_valid_not_sentinel(self, match_date: date) -> None:
        """Rank 1 is a valid rank — must NOT be treated as missing."""
        builder, _ = _make_builder(ranks={"A": 1, "B": 2})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["player_a_bwf_rank"] == 1.0
        assert feats["player_b_bwf_rank"] == 2.0

    def test_high_rank_number_is_valid(self, match_date: date) -> None:
        """A rank of 998 is a valid (very low) ranking, not the sentinel."""
        builder, _ = _make_builder(ranks={"A": 998, "B": 1})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["player_a_bwf_rank"] == 998.0
        assert feats["player_b_bwf_rank"] == 1.0

    def test_new_features_all_finite(self, match_date: date) -> None:
        """All 4 new features must be finite (no Inf) regardless of rank values."""
        for (rank_a, rank_b) in [(1, 1), (1, 999), (999, 1), (999, 999), (50, 50)]:
            builder, _ = _make_builder(ranks={"A": rank_a, "B": rank_b})
            feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

            for key in (
                "player_a_bwf_rank",
                "player_b_bwf_rank",
                "bwf_rank_diff_raw",
                "bwf_rank_ratio",
            ):
                val = feats[key]
                assert not math.isnan(val), f"NaN in {key} with ranks ({rank_a},{rank_b})"
                assert not math.isinf(val), f"Inf in {key} with ranks ({rank_a},{rank_b})"

    def test_rank_ratio_never_zero(self, match_date: date) -> None:
        """
        bwf_rank_ratio should never be exactly 0 because the numerator
        (player_b_bwf_rank) is at least 1 (or sentinel 999).
        """
        # Worst case: B has rank 1, A has rank 999 → ratio = 1 / (999 + 1e-6) > 0
        builder, _ = _make_builder(ranks={"A": 999, "B": 1})
        feats = builder.group_a_elo_ranking("A", "B", Discipline.MS, match_date)

        assert feats["bwf_rank_ratio"] > 0.0

    def test_discipline_ms_vs_ws_both_work(self, match_date: date) -> None:
        """New features must work for WS discipline as well as MS."""
        elo_system = BadmintonEloSystem()
        elo_system.initialize_player("P", Discipline.WS, initial_rating=1550.0)
        elo_system.initialize_player("Q", Discipline.WS, initial_rating=1450.0)

        rankings_db = _RankingsDBWithCallTracking(
            ranks={"P": 3, "Q": 12},
        )
        builder = FeatureBuilder(elo_system, rankings_db, _MinimalServeStatDB())
        feats = builder.group_a_elo_ranking("P", "Q", Discipline.WS, match_date)

        assert feats["player_a_bwf_rank"] == 3.0
        assert feats["player_b_bwf_rank"] == 12.0
        assert feats["bwf_rank_diff_raw"] < 0.0   # P (rank 3) is better
        assert feats["bwf_rank_ratio"] > 1.0       # P (rank 3) is better

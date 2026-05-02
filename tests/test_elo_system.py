"""
test_elo_system.py
==================
Tests for the BadmintonEloSystem — 8 independent ELO pools.

Covers:
  - Initial ratings for new players
  - ELO update after match (winner gains, loser loses)
  - Zero-sum property of ELO updates
  - Expected outcome → small delta, upset → large delta
  - K-factor variation by match count / tier
  - Pair ELO vs individual ELO separation
  - get_rating / get_rating_or_default for unknown players
  - All 8 pool types exist and are independent

ZERO hardcoded expected ELO values — only relative invariants.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import (
    Discipline,
    EloPool,
    ELO_DEFAULT_RATING,
    TournamentTier,
)
from ml.elo_system import (
    BadmintonEloSystem,
    EloCalculator,
    EloEntry,
    EloSystemError,
    EntityNotFoundError,
    PairEloEntry,
    _make_pair_key,
    _parse_pair_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def elo_system() -> BadmintonEloSystem:
    """Fresh ELO system with no pre-loaded data."""
    return BadmintonEloSystem()


@pytest.fixture
def seeded_singles_system(elo_system: BadmintonEloSystem) -> BadmintonEloSystem:
    """ELO system with two MS players initialised at default."""
    elo_system.initialize_player("player_A", Discipline.MS)
    elo_system.initialize_player("player_B", Discipline.MS)
    return elo_system


@pytest.fixture
def seeded_ws_system(elo_system: BadmintonEloSystem) -> BadmintonEloSystem:
    """ELO system with two WS players initialised at default."""
    elo_system.initialize_player("ws_A", Discipline.WS)
    elo_system.initialize_player("ws_B", Discipline.WS)
    return elo_system


@pytest.fixture
def seeded_doubles_system(elo_system: BadmintonEloSystem) -> BadmintonEloSystem:
    """ELO system with two MD pairs bootstrapped."""
    elo_system.bootstrap_pair_rating("d_A1", "d_A2", Discipline.MD, matches_together=0)
    elo_system.bootstrap_pair_rating("d_B1", "d_B2", Discipline.MD, matches_together=0)
    return elo_system


@pytest.fixture
def match_date() -> date:
    return date(2025, 6, 15)


# ---------------------------------------------------------------------------
# 1. Initial ratings for new players
# ---------------------------------------------------------------------------

class TestInitialRatings:
    def test_new_player_gets_default_rating(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        elo_system.initialize_player("new_p", Discipline.MS)
        rating = elo_system.get_rating("new_p", Discipline.MS)
        assert rating == ELO_DEFAULT_RATING

    def test_new_player_ws_gets_default_rating(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        elo_system.initialize_player("new_ws", Discipline.WS)
        rating = elo_system.get_rating("new_ws", Discipline.WS)
        assert rating == ELO_DEFAULT_RATING

    def test_initialize_player_is_idempotent(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        """Calling initialize_player twice does not overwrite the entry."""
        elo_system.initialize_player("p1", Discipline.MS, initial_rating=1600.0)
        elo_system.initialize_player("p1", Discipline.MS, initial_rating=1400.0)
        # First call wins — rating should remain 1600.0
        rating = elo_system.get_rating("p1", Discipline.MS)
        assert rating == 1600.0


# ---------------------------------------------------------------------------
# 2. ELO update after match — winner gains, loser loses
# ---------------------------------------------------------------------------

class TestEloUpdateDirection:
    def test_winner_gains_rating(
        self, seeded_singles_system: BadmintonEloSystem, match_date: date
    ) -> None:
        rating_before = seeded_singles_system.get_rating("player_A", Discipline.MS)
        new_w, _ = seeded_singles_system.update_after_match(
            "player_A", "player_B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        assert new_w > rating_before

    def test_loser_loses_rating(
        self, seeded_singles_system: BadmintonEloSystem, match_date: date
    ) -> None:
        rating_before = seeded_singles_system.get_rating("player_B", Discipline.MS)
        _, new_l = seeded_singles_system.update_after_match(
            "player_A", "player_B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        assert new_l < rating_before


# ---------------------------------------------------------------------------
# 3. Zero-sum property
# ---------------------------------------------------------------------------

class TestZeroSum:
    def test_elo_update_is_zero_sum(
        self, seeded_singles_system: BadmintonEloSystem, match_date: date
    ) -> None:
        r_a = seeded_singles_system.get_rating("player_A", Discipline.MS)
        r_b = seeded_singles_system.get_rating("player_B", Discipline.MS)
        total_before = r_a + r_b

        new_w, new_l = seeded_singles_system.update_after_match(
            "player_A", "player_B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        total_after = new_w + new_l
        assert abs(total_after - total_before) < 1e-6

    def test_zero_sum_holds_for_unequal_ratings(
        self, elo_system: BadmintonEloSystem, match_date: date
    ) -> None:
        elo_system.initialize_player("strong", Discipline.MS, initial_rating=1700.0)
        elo_system.initialize_player("weak", Discipline.MS, initial_rating=1300.0)
        total_before = 1700.0 + 1300.0

        new_w, new_l = elo_system.update_after_match(
            "strong", "weak", Discipline.MS,
            TournamentTier.SUPER_1000, match_date,
        )
        assert abs((new_w + new_l) - total_before) < 1e-6


# ---------------------------------------------------------------------------
# 4. Expected result → small delta; upset → large delta
# ---------------------------------------------------------------------------

class TestDeltaMagnitude:
    def test_expected_win_small_change(
        self, elo_system: BadmintonEloSystem, match_date: date
    ) -> None:
        elo_system.initialize_player("fav", Discipline.MS, initial_rating=1700.0)
        elo_system.initialize_player("dog", Discipline.MS, initial_rating=1300.0)
        new_w, _ = elo_system.update_after_match(
            "fav", "dog", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        delta_expected = new_w - 1700.0
        assert delta_expected > 0  # winner still gains

        # Now test upset (same rating gap, lower-rated wins)
        elo_system.initialize_player("fav2", Discipline.WS, initial_rating=1700.0)
        elo_system.initialize_player("dog2", Discipline.WS, initial_rating=1300.0)
        new_w_upset, _ = elo_system.update_after_match(
            "dog2", "fav2", Discipline.WS,
            TournamentTier.SUPER_500, match_date,
        )
        delta_upset = new_w_upset - 1300.0
        # Upset winner should gain MORE than the expected-result winner
        assert delta_upset > delta_expected

    def test_equal_ratings_moderate_change(
        self, seeded_singles_system: BadmintonEloSystem, match_date: date
    ) -> None:
        new_w, new_l = seeded_singles_system.update_after_match(
            "player_A", "player_B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        delta = new_w - ELO_DEFAULT_RATING
        # For equal ratings the change should be K/2 — verify it is non-trivial
        assert delta > 1.0
        # And symmetric
        assert abs((new_w - ELO_DEFAULT_RATING) + (new_l - ELO_DEFAULT_RATING)) < 1e-6


# ---------------------------------------------------------------------------
# 5. K-factor decreases with more matches (tier-based proxy)
# ---------------------------------------------------------------------------

class TestKFactor:
    def test_higher_tier_higher_k(self) -> None:
        """Olympics / World Champs should have a higher K than Super 100."""
        k_olympic = EloCalculator.k_factor(
            TournamentTier.OLYMPICS, Discipline.MS
        )
        k_s100 = EloCalculator.k_factor(
            TournamentTier.SUPER_100, Discipline.MS
        )
        assert k_olympic > k_s100

    def test_upset_factor_increases_k(self) -> None:
        k_normal = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS, is_upset=False
        )
        k_upset = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS, is_upset=True
        )
        assert k_upset > k_normal

    def test_young_player_boost(self) -> None:
        k_young = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS, age=20.0
        )
        k_prime = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS, age=27.0
        )
        assert k_young > k_prime

    def test_veteran_decay(self) -> None:
        k_vet = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS, age=35.0
        )
        k_prime = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS, age=27.0
        )
        assert k_vet < k_prime

    def test_doubles_k_multiplier(self) -> None:
        k_singles = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MS
        )
        k_doubles = EloCalculator.k_factor(
            TournamentTier.SUPER_500, Discipline.MD
        )
        # Doubles K is multiplied by a different (lower) factor
        assert k_singles != k_doubles


# ---------------------------------------------------------------------------
# 6. Pair ELO vs individual ELO separation
# ---------------------------------------------------------------------------

class TestPairVsIndividualElo:
    def test_bootstrap_pair_creates_pair_entry(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        bootstrapped = elo_system.bootstrap_pair_rating(
            "p1", "p2", Discipline.MD, matches_together=0
        )
        # Bootstrapped rating should be near the default but discounted
        assert bootstrapped < ELO_DEFAULT_RATING
        assert bootstrapped > 0

    def test_bootstrap_pair_not_allowed_for_singles(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        with pytest.raises(EloSystemError):
            elo_system.bootstrap_pair_rating("p1", "p2", Discipline.MS)

    def test_pair_elo_independent_from_singles(
        self, elo_system: BadmintonEloSystem, match_date: date
    ) -> None:
        """Updating a pair ELO should NOT affect singles ELO for the same player."""
        elo_system.initialize_player("multi_A", Discipline.MS)
        ms_before = elo_system.get_rating("multi_A", Discipline.MS)

        elo_system.bootstrap_pair_rating("multi_A", "partner_X", Discipline.MD)
        elo_system.bootstrap_pair_rating("opp_C", "opp_D", Discipline.MD)

        pair_key_a = _make_pair_key("multi_A", "partner_X")
        pair_key_b = _make_pair_key("opp_C", "opp_D")

        elo_system.update_after_match(
            pair_key_a, pair_key_b, Discipline.MD,
            TournamentTier.SUPER_500, match_date,
        )
        ms_after = elo_system.get_rating("multi_A", Discipline.MS)
        assert ms_after == ms_before

    def test_pair_bootstrap_idempotent(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        r1 = elo_system.bootstrap_pair_rating("a", "b", Discipline.WD)
        r2 = elo_system.bootstrap_pair_rating("a", "b", Discipline.WD)
        assert r1 == r2

    def test_pair_key_order_invariant(self) -> None:
        assert _make_pair_key("alice", "bob") == _make_pair_key("bob", "alice")

    def test_parse_pair_key_roundtrip(self) -> None:
        key = _make_pair_key("x", "y")
        players = _parse_pair_key(key)
        assert set(players) == {"x", "y"}

    def test_familiarity_bonus_grows_with_matches(self) -> None:
        entry_new = PairEloEntry(
            pair_key="a|b", discipline=Discipline.MD,
            pool=EloPool.MD_PAIR, rating=1500.0, matches_together=0,
        )
        entry_exp = PairEloEntry(
            pair_key="a|b", discipline=Discipline.MD,
            pool=EloPool.MD_PAIR, rating=1500.0, matches_together=30,
        )
        assert entry_exp.familiarity_bonus > entry_new.familiarity_bonus

    def test_familiarity_bonus_capped(self) -> None:
        entry = PairEloEntry(
            pair_key="a|b", discipline=Discipline.MD,
            pool=EloPool.MD_PAIR, rating=1500.0, matches_together=100,
        )
        entry_max = PairEloEntry(
            pair_key="a|b", discipline=Discipline.MD,
            pool=EloPool.MD_PAIR, rating=1500.0, matches_together=999,
        )
        assert entry.familiarity_bonus == entry_max.familiarity_bonus


# ---------------------------------------------------------------------------
# 7. get_rating returns default / raises for unknown player
# ---------------------------------------------------------------------------

class TestGetRating:
    def test_get_rating_raises_for_unknown_singles(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        with pytest.raises(EntityNotFoundError):
            elo_system.get_rating("nonexistent", Discipline.MS)

    def test_get_rating_raises_for_unknown_doubles_pair(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        with pytest.raises(EntityNotFoundError):
            elo_system.get_rating("no|pair", Discipline.MD)

    def test_get_rating_or_default_returns_default_and_flag(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        rating, is_default = elo_system.get_rating_or_default(
            "unknown", Discipline.MS
        )
        assert rating == ELO_DEFAULT_RATING
        assert is_default is True

    def test_get_rating_or_default_returns_actual_for_known(
        self, seeded_singles_system: BadmintonEloSystem
    ) -> None:
        rating, is_default = seeded_singles_system.get_rating_or_default(
            "player_A", Discipline.MS
        )
        assert rating == ELO_DEFAULT_RATING
        assert is_default is False


# ---------------------------------------------------------------------------
# 8. ELO history tracked per player (matches_played / peak_rating)
# ---------------------------------------------------------------------------

class TestEloHistory:
    def test_matches_played_increments(
        self, seeded_singles_system: BadmintonEloSystem, match_date: date
    ) -> None:
        key = ("player_A", EloPool.MS_OVERALL)
        assert seeded_singles_system._singles[key].matches_played == 0

        seeded_singles_system.update_after_match(
            "player_A", "player_B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        assert seeded_singles_system._singles[key].matches_played == 1

    def test_peak_rating_tracked_for_winner(
        self, seeded_singles_system: BadmintonEloSystem, match_date: date
    ) -> None:
        key = ("player_A", EloPool.MS_OVERALL)
        seeded_singles_system.update_after_match(
            "player_A", "player_B", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        entry = seeded_singles_system._singles[key]
        assert entry.peak_rating >= entry.rating
        assert entry.peak_rating > ELO_DEFAULT_RATING


# ---------------------------------------------------------------------------
# 9. All 8 pool types exist and are independent
# ---------------------------------------------------------------------------

class TestEightPools:
    ALL_POOLS = list(EloPool)

    def test_eight_pools_defined(self) -> None:
        assert len(self.ALL_POOLS) == 8

    def test_pool_names_match_spec(self) -> None:
        expected_names = {
            "MS_OVERALL", "WS_OVERALL",
            "MD_PAIR", "WD_PAIR", "XD_PAIR",
            "MD_INDIVIDUAL", "WD_INDIVIDUAL", "XD_INDIVIDUAL",
        }
        actual_names = {p.value for p in self.ALL_POOLS}
        assert actual_names == expected_names

    def test_ms_ws_pools_are_independent(
        self, elo_system: BadmintonEloSystem, match_date: date
    ) -> None:
        """Updating MS ELO must NOT affect WS ELO for a different player."""
        elo_system.initialize_player("ms_p", Discipline.MS)
        elo_system.initialize_player("ms_q", Discipline.MS)
        elo_system.initialize_player("ws_p", Discipline.WS)

        ws_before = elo_system.get_rating("ws_p", Discipline.WS)
        elo_system.update_after_match(
            "ms_p", "ms_q", Discipline.MS,
            TournamentTier.SUPER_500, match_date,
        )
        ws_after = elo_system.get_rating("ws_p", Discipline.WS)
        assert ws_after == ws_before

    def test_doubles_pair_and_individual_pools_are_separate(
        self, elo_system: BadmintonEloSystem
    ) -> None:
        """MD_PAIR and MD_INDIVIDUAL are different pools."""
        assert EloPool.MD_PAIR != EloPool.MD_INDIVIDUAL
        assert EloPool.WD_PAIR != EloPool.WD_INDIVIDUAL
        assert EloPool.XD_PAIR != EloPool.XD_INDIVIDUAL


# ---------------------------------------------------------------------------
# 10. EloCalculator static methods
# ---------------------------------------------------------------------------

class TestEloCalculator:
    def test_expected_score_symmetric(self) -> None:
        e_ab = EloCalculator.expected_score(1500.0, 1500.0)
        assert abs(e_ab - 0.5) < 1e-9

    def test_expected_score_sums_to_one(self) -> None:
        e_ab = EloCalculator.expected_score(1600.0, 1400.0)
        e_ba = EloCalculator.expected_score(1400.0, 1600.0)
        assert abs(e_ab + e_ba - 1.0) < 1e-9

    def test_higher_rated_has_higher_expected_score(self) -> None:
        e = EloCalculator.expected_score(1700.0, 1400.0)
        assert e > 0.5

    def test_new_ratings_winner_gains(self) -> None:
        new_w, new_l = EloCalculator.new_ratings(1500.0, 1500.0, k=32.0)
        assert new_w > 1500.0
        assert new_l < 1500.0

    def test_pair_bootstrap_discount_applies(self) -> None:
        bootstrapped = EloCalculator.pair_bootstrap_rating(1500.0, 1500.0, 0)
        assert bootstrapped < 1500.0

    def test_pair_bootstrap_discount_reduces_with_matches(self) -> None:
        r0 = EloCalculator.pair_bootstrap_rating(1500.0, 1500.0, 0)
        r20 = EloCalculator.pair_bootstrap_rating(1500.0, 1500.0, 20)
        assert r20 > r0

    def test_xd_pair_elo_weighted(self) -> None:
        """XD pair ELO should weight man (rear) and woman (front) differently."""
        result = EloCalculator.xd_pair_elo(1600.0, 1400.0)
        # Should not be simple average
        simple_avg = (1600.0 + 1400.0) / 2.0
        assert result != simple_avg

    def test_xd_pair_elo_blends_when_pair_elo_available(self) -> None:
        indiv_only = EloCalculator.xd_pair_elo(1600.0, 1400.0, pair_elo=None)
        blended = EloCalculator.xd_pair_elo(1600.0, 1400.0, pair_elo=1550.0)
        # Blended should differ from individual-only estimate
        assert blended != indiv_only


# ---------------------------------------------------------------------------
# 11. Inactivity decay
# ---------------------------------------------------------------------------

class TestInactivityDecay:
    def test_no_decay_within_threshold(self) -> None:
        entry = EloEntry(
            entity_id="p", pool=EloPool.MS_OVERALL,
            rating=1600.0, last_match_date=date(2025, 3, 1),
        )
        entry.apply_inactivity_decay(date(2025, 5, 1))  # ~8 weeks — within 12
        assert entry.rating == 1600.0

    def test_decay_after_threshold(self) -> None:
        entry = EloEntry(
            entity_id="p", pool=EloPool.MS_OVERALL,
            rating=1700.0, last_match_date=date(2024, 1, 1),
        )
        entry.apply_inactivity_decay(date(2025, 6, 1))  # ~74 weeks
        # Rating should have decayed toward 1500 (default)
        assert entry.rating < 1700.0
        assert entry.rating > ELO_DEFAULT_RATING  # still above mean (was 1700)

    def test_no_decay_without_last_match_date(self) -> None:
        entry = EloEntry(
            entity_id="p", pool=EloPool.MS_OVERALL,
            rating=1600.0, last_match_date=None,
        )
        entry.apply_inactivity_decay(date(2025, 6, 1))
        assert entry.rating == 1600.0


# ---------------------------------------------------------------------------
# 12. Snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_returns_all_players_for_discipline(
        self, seeded_singles_system: BadmintonEloSystem
    ) -> None:
        snap = seeded_singles_system.snapshot(Discipline.MS)
        assert "player_A" in snap
        assert "player_B" in snap
        assert len(snap) == 2

    def test_snapshot_empty_for_unused_discipline(
        self, seeded_singles_system: BadmintonEloSystem
    ) -> None:
        snap = seeded_singles_system.snapshot(Discipline.WS)
        assert len(snap) == 0

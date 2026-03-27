"""
elo_system.py
==============
Badminton ELO rating system — 8 independent rating pools.

Pools:
  1. MS_OVERALL     — Men's Singles
  2. WS_OVERALL     — Women's Singles
  3. MD_PAIR        — Men's Doubles (partnership ELO)
  4. WD_PAIR        — Women's Doubles (partnership ELO)
  5. XD_PAIR        — Mixed Doubles (partnership ELO)
  6. MD_INDIVIDUAL  — Individual performance within Men's Doubles context
  7. WD_INDIVIDUAL  — Individual performance within Women's Doubles context
  8. XD_INDIVIDUAL  — Individual performance within Mixed Doubles context

Design principles:
  - Pools 1-5 are entity-level (player for singles, pair for doubles)
  - Pools 6-8 are used for bootstrapping new partnerships (C-10 correction)
  - K-factor varies by tournament tier, age, and upset factor
  - Inactivity decay toward mean (1500) after 12 weeks
  - Pair chemistry bonus: +ELO% per 10 matches together
  - New partnership bootstrap: (ELO_A + ELO_B)/2 × 0.95 (C-10)
  - Temporal correctness: ELO updated AFTER feature extraction (Rule 14 / H5)

Sources:
  - Arpad Elo, "The Rating of Chessplayers" (1978)
  - BWF ranking structure adapted to ELO framework
  - XG3 tennis ELO system (elo_system_v21.py) — adapted for badminton
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, FrozenSet, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    EloPool,
    EloPool,
    TournamentTier,
    ELO_DEFAULT_RATING,
    ELO_INACTIVITY_DECAY_WEEKS,
    ELO_INACTIVITY_DECAY_RATE,
    ELO_UPSET_FACTOR,
    ELO_AGE_YOUNG_BOOST,
    ELO_AGE_VETERAN_DECAY,
    ELO_PAIR_FAMILIARITY_BONUS,
    ELO_NEW_PAIR_BOOTSTRAP_DISCOUNT,
    ELO_K_FACTORS,
    ELO_K_SINGLES_MULTIPLIER,
    ELO_K_DOUBLES_MULTIPLIER,
    DOUBLES_DISCIPLINES,
    XD_MAN_COURT_POSITION_REAR,
    XD_WOMAN_COURT_POSITION_FRONT,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class EloSystemError(RuntimeError):
    """Raised when ELO system encounters an unrecoverable error."""


class EntityNotFoundError(EloSystemError):
    """Raised when a player or pair entity is not found in the ELO registry."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EloEntry:
    """
    ELO rating entry for a single entity in a single pool.
    """
    entity_id: str           # Player ID (singles) or pair key (doubles)
    pool: EloPool
    rating: float = ELO_DEFAULT_RATING
    matches_played: int = 0
    last_match_date: Optional[date] = None
    peak_rating: float = ELO_DEFAULT_RATING

    def apply_inactivity_decay(self, current_date: date) -> None:
        """
        Apply weekly inactivity decay toward mean (1500) if inactive > 12 weeks.

        Mutates self.rating in place.
        """
        if self.last_match_date is None:
            return

        weeks_inactive = (current_date - self.last_match_date).days // 7
        if weeks_inactive <= ELO_INACTIVITY_DECAY_WEEKS:
            return

        decay_weeks = weeks_inactive - ELO_INACTIVITY_DECAY_WEEKS
        # Decay toward mean: rating = mean + (rating - mean) × rate^n
        self.rating = (
            ELO_DEFAULT_RATING
            + (self.rating - ELO_DEFAULT_RATING) * (ELO_INACTIVITY_DECAY_RATE ** decay_weeks)
        )
        logger.debug(
            "elo_inactivity_decay",
            entity_id=self.entity_id,
            pool=self.pool.value,
            weeks_inactive=weeks_inactive,
            new_rating=round(self.rating, 1),
        )


@dataclass
class PairEloEntry:
    """
    ELO entry for a doubles partnership.

    Stores both the pair-level ELO (partnership chemistry) and
    a match count for familiarity bonus calculation.
    """
    pair_key: str            # frozenset({player_a_id, player_b_id}) as sorted string
    discipline: Discipline   # MD / WD / XD
    pool: EloPool
    rating: float = ELO_DEFAULT_RATING
    matches_together: int = 0
    last_match_date: Optional[date] = None

    @property
    def familiarity_bonus(self) -> float:
        """
        ELO bonus for partnership familiarity.
        +ELO_PAIR_FAMILIARITY_BONUS per 10 matches together, capped at 3×.
        """
        bonus_units = min(self.matches_together // 10, 3)
        return bonus_units * ELO_PAIR_FAMILIARITY_BONUS * self.rating

    def apply_inactivity_decay(self, current_date: date) -> None:
        """
        Apply weekly inactivity decay toward mean for dormant partnerships.
        Mirrors EloEntry.apply_inactivity_decay.
        """
        if self.last_match_date is None:
            return
        weeks_inactive = (current_date - self.last_match_date).days // 7
        if weeks_inactive <= ELO_INACTIVITY_DECAY_WEEKS:
            return
        decay_weeks = weeks_inactive - ELO_INACTIVITY_DECAY_WEEKS
        self.rating = (
            ELO_DEFAULT_RATING
            + (self.rating - ELO_DEFAULT_RATING) * (ELO_INACTIVITY_DECAY_RATE ** decay_weeks)
        )
        logger.debug(
            "elo_pair_inactivity_decay",
            pair_key=self.pair_key,
            pool=self.pool.value,
            weeks_inactive=weeks_inactive,
            new_rating=round(self.rating, 1),
        )


# ---------------------------------------------------------------------------
# ELO Calculator (stateless)
# ---------------------------------------------------------------------------

class EloCalculator:
    """
    Stateless ELO calculation utilities.

    Used by BadmintonEloSystem to compute rating updates.
    """

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """
        E(A) = 1 / (1 + 10^((rating_b - rating_a) / 400))

        Standard ELO expected score formula.
        """
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    @staticmethod
    def k_factor(
        tier: TournamentTier,
        discipline: Discipline,
        age: Optional[float] = None,
        is_upset: bool = False,
    ) -> float:
        """
        Compute K-factor for a match result.

        Modifiers applied:
        - Tournament tier base K
        - Singles vs doubles multiplier
        - Age modifier (young boost / veteran decay)
        - Upset factor (winner was lower-rated)
        """
        base_k = ELO_K_FACTORS.get(tier, 24.0)  # Default if tier not found

        # Discipline multiplier
        if discipline in DOUBLES_DISCIPLINES:
            base_k *= ELO_K_DOUBLES_MULTIPLIER
        else:
            base_k *= ELO_K_SINGLES_MULTIPLIER

        # Age modifier
        if age is not None:
            if age < 23:
                base_k *= ELO_AGE_YOUNG_BOOST
            elif age > 32:
                base_k *= ELO_AGE_VETERAN_DECAY

        # Upset factor
        if is_upset:
            base_k *= ELO_UPSET_FACTOR

        return base_k

    @staticmethod
    def new_ratings(
        rating_winner: float,
        rating_loser: float,
        k: float,
    ) -> Tuple[float, float]:
        """
        Compute new ratings after a match result.

        Returns:
            (new_rating_winner, new_rating_loser)
        """
        e_winner = EloCalculator.expected_score(rating_winner, rating_loser)
        e_loser = 1.0 - e_winner

        new_winner = rating_winner + k * (1.0 - e_winner)
        new_loser = rating_loser + k * (0.0 - e_loser)

        return new_winner, new_loser

    @staticmethod
    def pair_bootstrap_rating(
        individual_elo_a: float,
        individual_elo_b: float,
        matches_together: int = 0,
    ) -> float:
        """
        Bootstrap rating for a new/sparse partnership (C-10 correction).

        Formula: (individual_a + individual_b) / 2 × discount
        Discount applied because pair ELO ≠ sum of individual ELOs.
        Discount reduces as matches_together increases.
        """
        base = (individual_elo_a + individual_elo_b) / 2.0
        # Discount reduces from 0.95 → 1.0 over first 20 matches
        discount = ELO_NEW_PAIR_BOOTSTRAP_DISCOUNT + (
            (1.0 - ELO_NEW_PAIR_BOOTSTRAP_DISCOUNT) * min(matches_together, 20) / 20.0
        )
        return base * discount

    @staticmethod
    def xd_pair_elo(
        man_individual_elo: float,
        woman_individual_elo: float,
        pair_elo: Optional[float] = None,
    ) -> float:
        """
        Compute effective pair ELO for XD discipline.

        If pair ELO exists (sufficient matches together), blend with individual.
        Otherwise, weight individual ELOs by court position.

        XD weighting: 60% man's MS ELO + 40% woman's WS ELO (typical court roles).
        """
        individual_estimate = (
            XD_MAN_COURT_POSITION_REAR * man_individual_elo
            + XD_WOMAN_COURT_POSITION_FRONT * woman_individual_elo
        )
        if pair_elo is None:
            return individual_estimate
        # Blend: 30% individual estimate, 70% pair ELO (when pair ELO is established)
        return 0.30 * individual_estimate + 0.70 * pair_elo


# ---------------------------------------------------------------------------
# ELO System (stateful registry)
# ---------------------------------------------------------------------------

class BadmintonEloSystem:
    """
    Stateful ELO registry for all 8 pools.

    Maintains ratings for:
    - Singles players (MS, WS)
    - Doubles pairs as entities (MD_PAIR, WD_PAIR, XD_PAIR)
    - Individual doubles players (MD_INDIVIDUAL, WD_INDIVIDUAL, XD_INDIVIDUAL)

    Usage:
        elo = BadmintonEloSystem()
        elo.load_from_db(db_session)   # populate from PostgreSQL
        probs = elo.match_probability(entity_a_id, entity_b_id, discipline, tier)
        elo.update_after_match(winner_id, loser_id, discipline, tier, match_date)

    CRITICAL: update_after_match must only be called AFTER feature extraction
    for the current match (temporal correctness — Rule 14).
    """

    def __init__(self) -> None:
        # Singles: keyed by (entity_id, pool)
        self._singles: Dict[Tuple[str, EloPool], EloEntry] = {}
        # Doubles pairs: keyed by (pair_key, pool)
        self._pairs: Dict[Tuple[str, EloPool], PairEloEntry] = {}
        # Individual in doubles context: keyed by (entity_id, pool)
        self._doubles_individual: Dict[Tuple[str, EloPool], EloEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_rating(
        self,
        entity_id: str,
        discipline: Discipline,
        match_date: Optional[date] = None,
    ) -> float:
        """
        Get current ELO rating for an entity in the given discipline.

        For doubles: entity_id is the pair key (sorted player IDs).
        For singles: entity_id is the player ID.

        Applies inactivity decay if match_date is provided.

        Raises:
            EntityNotFoundError if entity not found (we do NOT return default).
            Use get_rating_or_default only when explicitly bootstrapping.
        """
        pool = self._discipline_to_pair_pool(discipline)
        key = (entity_id, pool)

        if discipline in DOUBLES_DISCIPLINES:
            entry = self._pairs.get(key)
            if entry is None:
                raise EntityNotFoundError(
                    f"Pair entity '{entity_id}' not found in pool {pool.value}. "
                    f"Use bootstrap_pair_rating() first."
                )
            if match_date:
                entry.apply_inactivity_decay(match_date)
            return entry.rating + entry.familiarity_bonus
        else:
            entry = self._singles.get(key)
            if entry is None:
                raise EntityNotFoundError(
                    f"Player entity '{entity_id}' not found in pool {pool.value}. "
                    f"Use initialize_player() first."
                )
            if match_date:
                entry.apply_inactivity_decay(match_date)
            return entry.rating

    def get_rating_or_default(
        self,
        entity_id: str,
        discipline: Discipline,
        match_date: Optional[date] = None,
    ) -> Tuple[float, bool]:
        """
        Get ELO rating, returning (rating, is_default) tuple.

        Returns (default_rating, True) if entity not found.
        Used ONLY for bootstrapping new entities — never in production inference.
        """
        try:
            return self.get_rating(entity_id, discipline, match_date), False
        except EntityNotFoundError:
            logger.warning(
                "elo_entity_not_found_using_default",
                entity_id=entity_id,
                discipline=discipline.value,
            )
            return ELO_DEFAULT_RATING, True

    def initialize_player(
        self,
        player_id: str,
        discipline: Discipline,
        initial_rating: float = ELO_DEFAULT_RATING,
    ) -> None:
        """Initialize a new player's ELO entry."""
        pool = self._discipline_to_pair_pool(discipline)
        key = (player_id, pool)
        if key in self._singles:
            logger.warning(
                "elo_player_already_exists",
                player_id=player_id,
                pool=pool.value,
            )
            return
        self._singles[key] = EloEntry(
            entity_id=player_id,
            pool=pool,
            rating=initial_rating,
        )

    def bootstrap_pair_rating(
        self,
        player_a_id: str,
        player_b_id: str,
        discipline: Discipline,
        matches_together: int = 0,
    ) -> float:
        """
        Bootstrap a new doubles pair's ELO from individual ratings.

        Creates the pair entry if it doesn't exist.
        Returns the bootstrapped rating.

        C-10 correction: recency decay + partner-switch transfer handled here.
        """
        if discipline not in DOUBLES_DISCIPLINES:
            raise EloSystemError(
                f"bootstrap_pair_rating only valid for doubles: {discipline}"
            )

        pair_key = _make_pair_key(player_a_id, player_b_id)
        pair_pool = self._discipline_to_pair_pool(discipline)
        pair_entry_key = (pair_key, pair_pool)

        if pair_entry_key in self._pairs:
            return self._pairs[pair_entry_key].rating

        # Get individual ratings in doubles context
        indiv_pool = self._discipline_to_individual_pool(discipline)
        elo_a, _ = self._get_individual_doubles_rating(player_a_id, indiv_pool)
        elo_b, _ = self._get_individual_doubles_rating(player_b_id, indiv_pool)

        if discipline == Discipline.XD:
            # XD: weighted by court position
            # Determine gender — caller must provide or we use weighted average
            bootstrapped = EloCalculator.pair_bootstrap_rating(elo_a, elo_b, matches_together)
        else:
            bootstrapped = EloCalculator.pair_bootstrap_rating(elo_a, elo_b, matches_together)

        self._pairs[pair_entry_key] = PairEloEntry(
            pair_key=pair_key,
            discipline=discipline,
            pool=pair_pool,
            rating=bootstrapped,
            matches_together=matches_together,
        )

        logger.info(
            "elo_pair_bootstrapped",
            pair_key=pair_key,
            discipline=discipline.value,
            elo_a=round(elo_a, 1),
            elo_b=round(elo_b, 1),
            bootstrapped=round(bootstrapped, 1),
        )
        return bootstrapped

    def update_after_match(
        self,
        winner_entity_id: str,
        loser_entity_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        match_date: date,
        winner_age: Optional[float] = None,
        loser_age: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Update ELO ratings after a match result.

        CRITICAL: Call this ONLY after feature extraction for the match
        to maintain temporal correctness (Rule 14, H5 gate).

        Returns:
            (new_winner_rating, new_loser_rating)
        """
        pool = self._discipline_to_pair_pool(discipline)

        if discipline in DOUBLES_DISCIPLINES:
            winner_rating = self._get_pair_rating_for_update(
                winner_entity_id, discipline, pool, match_date
            )
            loser_rating = self._get_pair_rating_for_update(
                loser_entity_id, discipline, pool, match_date
            )
        else:
            winner_entry = self._get_or_create_singles_entry(
                winner_entity_id, pool, match_date
            )
            loser_entry = self._get_or_create_singles_entry(
                loser_entity_id, pool, match_date
            )
            winner_rating = winner_entry.rating
            loser_rating = loser_entry.rating

        is_upset = winner_rating < loser_rating
        # Use lower-rated entity's age for upset factor (winner is the upset-er)
        upset_age = winner_age if is_upset else None

        k = EloCalculator.k_factor(
            tier=tier,
            discipline=discipline,
            age=upset_age,
            is_upset=is_upset,
        )

        new_winner_rating, new_loser_rating = EloCalculator.new_ratings(
            rating_winner=winner_rating,
            rating_loser=loser_rating,
            k=k,
        )

        # Persist updated ratings
        self._set_rating(winner_entity_id, discipline, pool, new_winner_rating, match_date)
        self._set_rating(loser_entity_id, discipline, pool, new_loser_rating, match_date)

        logger.info(
            "elo_updated",
            winner=winner_entity_id,
            loser=loser_entity_id,
            discipline=discipline.value,
            tier=tier.value,
            k=round(k, 1),
            winner_delta=round(new_winner_rating - winner_rating, 1),
            loser_delta=round(new_loser_rating - loser_rating, 1),
            new_winner=round(new_winner_rating, 1),
            new_loser=round(new_loser_rating, 1),
        )

        # Also update individual doubles ELO (for pair bootstrapping)
        if discipline in DOUBLES_DISCIPLINES:
            self._update_individual_doubles_elo(
                winner_entity_id, loser_entity_id, discipline, tier, match_date,
                winner_age, loser_age
            )

        return new_winner_rating, new_loser_rating

    def match_probability(
        self,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        match_date: Optional[date] = None,
    ) -> float:
        """
        P(entity A wins) based on ELO difference alone.

        This is the ELO-only baseline — the full model uses ML features.
        """
        rating_a = self.get_rating(entity_a_id, discipline, match_date)
        rating_b = self.get_rating(entity_b_id, discipline, match_date)
        return EloCalculator.expected_score(rating_a, rating_b)

    def elo_diff(
        self,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        match_date: Optional[date] = None,
    ) -> float:
        """Return ELO_A - ELO_B (signed difference)."""
        rating_a = self.get_rating(entity_a_id, discipline, match_date)
        rating_b = self.get_rating(entity_b_id, discipline, match_date)
        return rating_a - rating_b

    def snapshot(self, discipline: Discipline) -> Dict[str, float]:
        """
        Return snapshot of all ratings for a discipline.

        Used by scripts/train_models.py for serialisation to elo_snapshot.pkl.
        """
        pool = self._discipline_to_pair_pool(discipline)
        result: Dict[str, float] = {}

        if discipline in DOUBLES_DISCIPLINES:
            for (entity_id, p), entry in self._pairs.items():
                if p == pool:
                    result[entity_id] = entry.rating
        else:
            for (entity_id, p), entry in self._singles.items():
                if p == pool:
                    result[entity_id] = entry.rating

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discipline_to_pair_pool(discipline: Discipline) -> EloPool:
        return {
            Discipline.MS: EloPool.MS_OVERALL,
            Discipline.WS: EloPool.WS_OVERALL,
            Discipline.MD: EloPool.MD_PAIR,
            Discipline.WD: EloPool.WD_PAIR,
            Discipline.XD: EloPool.XD_PAIR,
        }[discipline]

    @staticmethod
    def _discipline_to_individual_pool(discipline: Discipline) -> EloPool:
        return {
            Discipline.MD: EloPool.MD_INDIVIDUAL,
            Discipline.WD: EloPool.WD_INDIVIDUAL,
            Discipline.XD: EloPool.XD_INDIVIDUAL,
        }[discipline]

    def _get_individual_doubles_rating(
        self, player_id: str, pool: EloPool
    ) -> Tuple[float, bool]:
        """Get individual doubles ELO, returning default if not found."""
        key = (player_id, pool)
        entry = self._doubles_individual.get(key)
        if entry is None:
            return ELO_DEFAULT_RATING, True
        return entry.rating, False

    def _get_pair_rating_for_update(
        self,
        pair_key: str,
        discipline: Discipline,
        pool: EloPool,
        match_date: date,
    ) -> float:
        """Get or create pair rating, applying inactivity decay."""
        key = (pair_key, pool)
        entry = self._pairs.get(key)
        if entry is None:
            # Should have been bootstrapped — create default
            logger.warning(
                "elo_pair_missing_creating_default",
                pair_key=pair_key,
                pool=pool.value,
            )
            self._pairs[key] = PairEloEntry(
                pair_key=pair_key,
                discipline=discipline,
                pool=pool,
                rating=ELO_DEFAULT_RATING,
            )
            return ELO_DEFAULT_RATING
        entry.apply_inactivity_decay(match_date)
        return entry.rating + entry.familiarity_bonus

    def _get_or_create_singles_entry(
        self, entity_id: str, pool: EloPool, match_date: date
    ) -> EloEntry:
        """Get or create singles entry, applying inactivity decay."""
        key = (entity_id, pool)
        if key not in self._singles:
            logger.warning(
                "elo_player_missing_creating_default",
                entity_id=entity_id,
                pool=pool.value,
            )
            self._singles[key] = EloEntry(entity_id=entity_id, pool=pool)
        entry = self._singles[key]
        entry.apply_inactivity_decay(match_date)
        return entry

    def _set_rating(
        self,
        entity_id: str,
        discipline: Discipline,
        pool: EloPool,
        new_rating: float,
        match_date: date,
    ) -> None:
        """Persist updated rating."""
        if discipline in DOUBLES_DISCIPLINES:
            key = (entity_id, pool)
            if key in self._pairs:
                self._pairs[key].rating = new_rating
                self._pairs[key].last_match_date = match_date
                self._pairs[key].matches_together += 1
        else:
            key = (entity_id, pool)
            if key in self._singles:
                self._singles[key].rating = new_rating
                self._singles[key].last_match_date = match_date
                self._singles[key].matches_played += 1
                if new_rating > self._singles[key].peak_rating:
                    self._singles[key].peak_rating = new_rating

    def _update_individual_doubles_elo(
        self,
        winner_pair_key: str,
        loser_pair_key: str,
        discipline: Discipline,
        tier: TournamentTier,
        match_date: date,
        winner_age: Optional[float],
        loser_age: Optional[float],
    ) -> None:
        """
        Update individual ELO for each player within a pair, for bootstrapping purposes.

        Individual doubles ELO tracks how well a player performs in doubles
        context, independent of their partner. Used when a player changes partners.
        """
        indiv_pool = self._discipline_to_individual_pool(discipline)
        winner_players = _parse_pair_key(winner_pair_key)
        loser_players = _parse_pair_key(loser_pair_key)

        for w_player in winner_players:
            for l_player in loser_players:
                w_entry = self._get_or_create_doubles_individual_entry(
                    w_player, indiv_pool
                )
                l_entry = self._get_or_create_doubles_individual_entry(
                    l_player, indiv_pool
                )
                k = EloCalculator.k_factor(tier, discipline) * 0.5  # Half weight for individual
                new_w, new_l = EloCalculator.new_ratings(w_entry.rating, l_entry.rating, k)
                w_entry.rating = new_w
                l_entry.rating = new_l
                w_entry.last_match_date = match_date
                l_entry.last_match_date = match_date

    def _get_or_create_doubles_individual_entry(
        self, player_id: str, pool: EloPool
    ) -> EloEntry:
        key = (player_id, pool)
        if key not in self._doubles_individual:
            self._doubles_individual[key] = EloEntry(entity_id=player_id, pool=pool)
        return self._doubles_individual[key]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _make_pair_key(player_a_id: str, player_b_id: str) -> str:
    """
    Create a canonical, order-invariant pair key.

    Uses sorted alphabetical order to ensure frozenset semantics.
    """
    return "|".join(sorted([player_a_id, player_b_id]))


def _parse_pair_key(pair_key: str) -> list[str]:
    """Parse a pair key back into individual player IDs."""
    return pair_key.split("|")

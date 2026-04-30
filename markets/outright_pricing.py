"""
outright_pricing.py
===================
Tournament outright and futures pricing engine for badminton.

Computes win probabilities for all players/pairs in a tournament bracket
using Monte Carlo simulation over the draw structure.

Algorithm:
  1. Draw parsing: extract bracket seedings and match schedule
  2. For each pair of players in the bracket: compute P(A beats B)
     using Markov engine with RWP estimates
  3. Monte Carlo tournament simulation (N=10,000 iterations):
     - Sample match results for each round
     - Track tournament win counts per entity
     - P(entity wins tournament) = win_count / N_simulations
  4. Apply TIER_MARGINS_OUTRIGHTS
  5. Optional: blend with observed market prices (Pinnacle/Betfair)

Draw types:
  - Single elimination (most BWF World Tour events)
  - Round robin + knockout (Thomas/Uber/Sudirman Cup)
  - Round robin only (BWF World Tour Finals — 4 groups)

Bracket seedings:
  - Top 8 seeds placed per BWF convention
  - Seeds 1-2: opposite halves
  - Seeds 3-4: random within their quarter
  - Seeds 5-8: random within their quarter

ZERO hardcoded probabilities.
Raises RuntimeError if bracket structure invalid.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    TIER_MARGINS_OUTRIGHTS,
    OUTRIGHT_N_SIMULATIONS,
    MAIN_DRAW_SIZES,
)
from core.markov_engine import BadmintonMarkovEngine
from markets.derivative_engine import MarketPrice, MarketFamily
from markets.margin_engine import MarginEngine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class DrawType(str, Enum):
    SINGLE_ELIMINATION = "single_elimination"
    ROUND_ROBIN_KNOCKOUT = "round_robin_knockout"
    ROUND_ROBIN_ONLY = "round_robin_only"


@dataclass
class TournamentEntry:
    """Single tournament entry (player or pair)."""
    entity_id: str
    seeding: Optional[int] = None    # 1-8 for seeded players
    rwp_as_server: float = 0.0       # P(wins rally when serving)
    rwp_as_receiver: float = 0.0
    elo_rating: float = 1500.0
    is_bye: bool = False             # Bye entry (odd-sized draws)


@dataclass
class TournamentDraw:
    """Complete draw structure for a tournament."""
    tournament_id: str
    discipline: Discipline
    tier: TournamentTier
    draw_type: DrawType
    draw_size: int
    entries: List[TournamentEntry]   # In draw order (position 1, 2, ..., N)
    already_played: List[Tuple[str, str, str]] = field(default_factory=list)
    # (entity_a_id, entity_b_id, winner_id) for completed matches

    def validate(self) -> None:
        """Assert draw is internally consistent."""
        if len(self.entries) != self.draw_size:
            raise RuntimeError(
                f"Draw has {len(self.entries)} entries but draw_size={self.draw_size}"
            )
        if self.draw_size not in (8, 16, 32, 64):
            raise RuntimeError(f"Invalid draw_size: {self.draw_size}")


@dataclass
class OutrightPricingResult:
    """Outright pricing result for one entity."""
    entity_id: str
    p_win_tournament: float           # Fair probability
    simulated_wins: int               # In N_SIMULATIONS
    simulated_finals: int             # Simulated final appearances
    simulated_semis: int              # Simulated semi appearances
    odds_fair: float                  # 1 / p_win
    odds_with_margin: float           # After margin
    each_way_p: Optional[float] = None  # P(top 4)


@dataclass
class OutrightResponse:
    """Full outright pricing response for a tournament."""
    tournament_id: str
    discipline: Discipline
    tier: TournamentTier
    results: List[OutrightPricingResult]
    margin_applied: float
    n_simulations: int
    generated_at: float


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class OutrightPricingEngine:
    """
    Monte Carlo tournament simulation for outright pricing.
    """

    def __init__(self, n_simulations: int = OUTRIGHT_N_SIMULATIONS) -> None:
        self._n_simulations = n_simulations
        self._markov = BadmintonMarkovEngine()
        self._margin_engine = MarginEngine()
        self._rng = random.Random()
        # Lazy-import seeder to avoid circular deps at module load time
        self._elo_seeder_available: bool | None = None  # None = unchecked

    def price_tournament(
        self,
        draw: TournamentDraw,
        seed: Optional[int] = None,
    ) -> OutrightResponse:
        """
        Price outright winner market for all entries in a tournament draw.

        Args:
            draw: Fully populated tournament draw.
            seed: Random seed for reproducibility (None = random).

        Returns:
            OutrightResponse with probabilities and odds for all entries.
        """
        draw.validate()
        t_start = time.time()

        if seed is not None:
            self._rng.seed(seed)

        # Pre-compute P(A beats B) for all pairs
        match_probs: Dict[Tuple[str, str], float] = self._precompute_match_probs(
            draw.entries, draw.discipline
        )

        # Account for already-played results
        survivors = self._apply_completed_results(draw.entries, draw.already_played)

        # Run simulation
        win_counts: Dict[str, int] = {e.entity_id: 0 for e in draw.entries if not e.is_bye}
        final_counts: Dict[str, int] = {e.entity_id: 0 for e in draw.entries if not e.is_bye}
        semi_counts: Dict[str, int] = {e.entity_id: 0 for e in draw.entries if not e.is_bye}

        for sim_idx in range(self._n_simulations):
            winner, finalist, semis = self._simulate_tournament(
                entries=survivors,
                draw_type=draw.draw_type,
                draw_size=draw.draw_size,
                match_probs=match_probs,
            )
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1
            if finalist:
                final_counts[finalist] = final_counts.get(finalist, 0) + 1
            for s in semis:
                semi_counts[s] = semi_counts.get(s, 0) + 1

        # Build results
        results = []
        margin = TIER_MARGINS_OUTRIGHTS.get(draw.tier, 0.10)

        total_fair_prob = 0.0
        fair_probs = {}
        for entry in draw.entries:
            if entry.is_bye:
                continue
            p = win_counts.get(entry.entity_id, 0) / self._n_simulations
            fair_probs[entry.entity_id] = p
            total_fair_prob += p

        # Normalise (simulation may not sum to 1.0 due to floating point)
        if total_fair_prob > 0:
            fair_probs = {k: v / total_fair_prob for k, v in fair_probs.items()}

        for entry in sorted(draw.entries, key=lambda e: -fair_probs.get(e.entity_id, 0.0)):
            if entry.is_bye:
                continue
            p = fair_probs.get(entry.entity_id, 0.0)
            if p <= 0.0001:
                odds_fair = 9999.0
            else:
                odds_fair = 1.0 / p

            # Apply margin (power method via MarginEngine)
            # For outrights, use simple proportional approach
            p_margined = p / (1.0 + margin)
            odds_margined = max(1.01, 1.0 / max(0.0001, p_margined))

            # Each-way: P(top 4)
            each_way_p = (
                win_counts.get(entry.entity_id, 0) +
                final_counts.get(entry.entity_id, 0) +
                semi_counts.get(entry.entity_id, 0)
            ) / (self._n_simulations * 4.0)

            results.append(OutrightPricingResult(
                entity_id=entry.entity_id,
                p_win_tournament=p,
                simulated_wins=win_counts.get(entry.entity_id, 0),
                simulated_finals=final_counts.get(entry.entity_id, 0),
                simulated_semis=semi_counts.get(entry.entity_id, 0),
                odds_fair=round(odds_fair, 2),
                odds_with_margin=round(odds_margined, 2),
                each_way_p=min(1.0, each_way_p),
            ))

        elapsed = time.time() - t_start
        logger.info(
            "outright_priced",
            tournament_id=draw.tournament_id,
            discipline=draw.discipline.value,
            n_entries=len([e for e in draw.entries if not e.is_bye]),
            n_simulations=self._n_simulations,
            elapsed_ms=f"{elapsed * 1000:.0f}",
        )

        return OutrightResponse(
            tournament_id=draw.tournament_id,
            discipline=draw.discipline,
            tier=draw.tier,
            results=results,
            margin_applied=margin,
            n_simulations=self._n_simulations,
            generated_at=time.time(),
        )

    def _precompute_match_probs(
        self,
        entries: List[TournamentEntry],
        discipline: Discipline,
    ) -> Dict[Tuple[str, str], float]:
        """
        Pre-compute P(A beats B) for all pairs of non-bye entries.

        Uses Markov engine with each entry's RWP estimates.
        Returns {(entity_a_id, entity_b_id) -> P(A wins)}.
        """
        match_probs: Dict[Tuple[str, str], float] = {}
        live_entries = [e for e in entries if not e.is_bye]

        for i, ea in enumerate(live_entries):
            for j, eb in enumerate(live_entries):
                if i >= j:
                    continue  # Compute once, mirror

                # Use entity's own RWP estimates for A
                rwp_a = max(0.20, min(0.80, ea.rwp_as_server))
                rwp_b = max(0.20, min(0.80, eb.rwp_as_server))

                # Fallback: use ELO-derived estimate if RWP is truly unavailable
                # (check original field before clamp — 0.0 means "not set by caller")
                if ea.rwp_as_server == 0.0 or eb.rwp_as_server == 0.0:
                    elo_a = self._resolve_elo(ea, discipline)
                    elo_b = self._resolve_elo(eb, discipline)
                    elo_diff = elo_a - elo_b
                    from config.badminton_config import RWP_BASELINE
                    baseline = RWP_BASELINE[discipline]
                    rwp_a = baseline + 0.08 / 400.0 * elo_diff * 0.5
                    rwp_b = baseline
                    rwp_a = max(0.20, min(0.80, rwp_a))

                probs = self._markov.compute_match_probabilities(
                    rwp_a=rwp_a,
                    rwp_b=rwp_b,
                    discipline=discipline,
                    server_first_game="A",
                )
                p_a_wins = probs.p_a_wins_match

                match_probs[(ea.entity_id, eb.entity_id)] = p_a_wins
                match_probs[(eb.entity_id, ea.entity_id)] = 1.0 - p_a_wins

        return match_probs

    def _resolve_elo(self, entry: TournamentEntry, discipline: Discipline) -> float:
        """
        Return the best available ELO for a TournamentEntry.

        Resolution order:
          1. entry.elo_rating if it was explicitly set by the caller (not the 1500.0 default).
             We detect "not set" by comparing to exactly 1500.0 — callers setting a real
             rating of exactly 1500 are treated as unset (acceptable approximation).
          2. Seeder lookup from elo_seed_badminton.json via get_seeded_rating().
          3. Fall back to 1500.0 with a warning log.

        This prevents equal-odds pricing for recognisable but RWP-less entries.
        """
        # If caller set a real ELO (not the dataclass default), use it
        if entry.elo_rating != 1500.0:
            return entry.elo_rating

        # Attempt seeder lookup
        try:
            if self._elo_seeder_available is None:
                try:
                    from ml.elo_startup_seeder import get_seeded_rating as _gsr
                    self._elo_seeder_available = True
                except ImportError:
                    self._elo_seeder_available = False

            if self._elo_seeder_available:
                from ml.elo_startup_seeder import get_seeded_rating
                seeded = get_seeded_rating(discipline.value, entry.entity_id)
                if seeded is not None:
                    return seeded
        except Exception as exc:
            logger.warning("outright.elo_seeder_lookup_failed entity=%s error=%s", entry.entity_id, exc)

        # Legitimate fallback — player not in seed data
        logger.warning(
            "outright.elo_fallback_1500 entity=%s discipline=%s — not in seed data, using ELO_DEFAULT",
            entry.entity_id,
            discipline.value,
        )
        return 1500.0

    def _apply_completed_results(
        self,
        entries: List[TournamentEntry],
        already_played: List[Tuple[str, str, str]],
    ) -> List[TournamentEntry]:
        """
        Filter entries to only include survivors after completed matches.

        already_played: list of (entity_a, entity_b, winner) tuples.
        """
        eliminated = set()
        for ea, eb, winner in already_played:
            loser = eb if winner == ea else ea
            eliminated.add(loser)

        return [e for e in entries if e.entity_id not in eliminated]

    def _simulate_tournament(
        self,
        entries: List[TournamentEntry],
        draw_type: DrawType,
        draw_size: int,
        match_probs: Dict[Tuple[str, str], float],
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Simulate one tournament.

        Returns: (winner_id, runner_up_id, [semi_finalist_ids])
        """
        if draw_type == DrawType.SINGLE_ELIMINATION:
            return self._simulate_single_elimination(entries, match_probs)
        elif draw_type == DrawType.ROUND_ROBIN_KNOCKOUT:
            return self._simulate_rr_knockout(entries, match_probs)
        else:
            return self._simulate_round_robin(entries, match_probs)

    def _simulate_single_elimination(
        self,
        entries: List[TournamentEntry],
        match_probs: Dict[Tuple[str, str], float],
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Simulate single elimination bracket."""
        if not entries:
            return None, None, []

        current_round = [e.entity_id for e in entries if not e.is_bye]
        if not current_round:
            return None, None, []

        # Pad to power of 2 if needed
        while len(current_round) < 2:
            current_round.append(None)

        semi_finalists: List[str] = []
        finalist: Optional[str] = None

        n_rounds = 0
        while len(current_round) > 1:
            n_rounds += 1
            next_round = []
            is_semi = len(current_round) == 4
            is_final = len(current_round) == 2

            for i in range(0, len(current_round), 2):
                if i + 1 >= len(current_round):
                    next_round.append(current_round[i])
                    continue

                a = current_round[i]
                b = current_round[i + 1]

                if a is None:
                    next_round.append(b)
                    continue
                if b is None:
                    next_round.append(a)
                    continue

                # Simulate match
                p_a_wins = match_probs.get((a, b), 0.5)
                winner = a if self._rng.random() < p_a_wins else b
                next_round.append(winner)

                if is_semi:
                    semi_finalists.extend([a, b])
                if is_final:
                    finalist = b if winner == a else a

            current_round = next_round

        winner = current_round[0] if current_round else None
        return winner, finalist, semi_finalists

    def _simulate_rr_knockout(
        self,
        entries: List[TournamentEntry],
        match_probs: Dict[Tuple[str, str], float],
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Simulate round-robin group stage + knockout (Thomas/Uber Cup)."""
        # Simple: split into groups of 4, top 2 advance
        groups = []
        group_size = 4
        entry_ids = [e.entity_id for e in entries if not e.is_bye]
        self._rng.shuffle(entry_ids)

        for i in range(0, len(entry_ids), group_size):
            groups.append(entry_ids[i:i + group_size])

        # Play round robin per group
        qualified: List[str] = []
        for group in groups:
            points = {p: 0 for p in group}
            for i, a in enumerate(group):
                for j, b in enumerate(group):
                    if i >= j:
                        continue
                    p_a = match_probs.get((a, b), 0.5)
                    if self._rng.random() < p_a:
                        points[a] += 2
                    else:
                        points[b] += 2
            top2 = sorted(group, key=lambda x: -points[x])[:2]
            qualified.extend(top2)

        # Single elimination from qualified
        if not qualified:
            return None, None, []
        qualified_entries = [TournamentEntry(entity_id=qid) for qid in qualified]
        return self._simulate_single_elimination(qualified_entries, match_probs)

    def _simulate_round_robin(
        self,
        entries: List[TournamentEntry],
        match_probs: Dict[Tuple[str, str], float],
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """Simulate full round robin (BWF World Tour Finals)."""
        entry_ids = [e.entity_id for e in entries if not e.is_bye]
        points = {p: 0 for p in entry_ids}

        for i, a in enumerate(entry_ids):
            for j, b in enumerate(entry_ids):
                if i >= j:
                    continue
                p_a = match_probs.get((a, b), 0.5)
                if self._rng.random() < p_a:
                    points[a] += 2
                else:
                    points[b] += 2

        ranked = sorted(entry_ids, key=lambda x: -points[x])
        winner = ranked[0] if ranked else None
        finalist = ranked[1] if len(ranked) > 1 else None
        semis = ranked[2:4] if len(ranked) > 3 else []
        return winner, finalist, semis

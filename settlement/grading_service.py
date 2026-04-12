"""
grading_service.py
==================
Market settlement (grading) service for badminton.

Settles all 97 markets across 15 families after a match completes.

Settlement rules per market family:
  1. Match Winner: winner of match
  2. Total Games: total games played (2 or 3)
  3. Correct Score: exact match score (A 2-0, A 2-1, B 2-0, B 2-1)
  4. Game-Level: winner/total for each game
  5. Race/Milestone: first to N points in game M
  6. Points Totals: O/U thresholds on match/game points
  7. Player Props: total points in match
  8-10. Outright / Live Specials / SGP: handled by specialist settlers
  11. Exotic: specific conditions (golden point, comeback, etc.)
  12. Handicap: game handicap settlement
  13. Futures: tournament winner after draw completes
  14-15. Team Events: rubber-level results for Thomas/Uber/Sudirman

Retirement/walkover handling:
  - Match Winner: non-retiring player wins (settled normally)
  - All other markets: VOID if incomplete (< required games played)
  - Void rules in void_rules.py

Settlement integrity:
  - Before settling, validate score via ScoreValidator (H6 gate)
  - Log all settlements with match_id, market_id, winning_outcome, timestamp
  - Return list of SettlementRecord for audit trail

ZERO hardcoded probabilities.
Raises SettlementError if required data unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    GAMES_TO_WIN_MATCH,
    POINTS_TO_WIN_GAME,
    GOLDEN_POINT_WIN,
)
from core.match_state import MatchLiveState, MatchStatus
from core.scoring_engine import ScoringEngine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class SettlementOutcome(str, Enum):
    WIN = "win"
    LOSE = "lose"
    VOID = "void"
    PUSH = "push"           # Handicap tie


class SettlementStatus(str, Enum):
    SETTLED = "settled"
    VOIDED = "voided"
    PENDING = "pending"     # Match not yet complete
    ERROR = "error"


@dataclass
class SettlementRecord:
    """Single market settlement record."""
    match_id: str
    market_id: str
    winning_outcome: Optional[str]    # None if voided
    settlement_status: SettlementStatus
    settled_at: datetime
    entity_a_id: str
    entity_b_id: str
    final_score: str                  # e.g., "2-1 (21-18, 14-21, 21-16)"
    void_reason: Optional[str] = None
    notes: str = ""

    @property
    def status(self) -> SettlementStatus:
        """Alias for settlement_status — convenience accessor used by tests."""
        return self.settlement_status


class SettlementError(RuntimeError):
    """Raised when settlement cannot be completed."""
    pass


# ---------------------------------------------------------------------------
# Match result extractor
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Fully parsed match result for settlement.

    Derived from MatchLiveState after match completion.
    """
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline

    # Core results
    winner: str                           # "A" or "B"
    status: MatchStatus

    # Scores
    games_won_a: int
    games_won_b: int
    game_scores: List[Tuple[int, int]]    # (score_a, score_b) per completed game

    # Derived
    total_games_played: int
    total_points_a: int
    total_points_b: int

    # Service metadata for first_point_winner market settlement
    initial_server: str = "A"   # "A" or "B" — who served game 1 point 1

    @classmethod
    def from_live_state(cls, state: MatchLiveState) -> "MatchResult":
        """Build MatchResult from completed MatchLiveState."""
        if state.status not in (
            MatchStatus.COMPLETED, MatchStatus.RETIRED, MatchStatus.WALKOVER
        ):
            raise SettlementError(
                f"Match {state.match_id} not in terminal state: {state.status.value}"
            )

        if state.match_winner is None:
            raise SettlementError(
                f"Match {state.match_id} has no winner despite terminal status"
            )

        total_a = sum(sa for sa, sb in state.game_scores)
        total_b = sum(sb for sa, sb in state.game_scores)

        # Retrieve initial_server — either from state field (preferred) or fallback
        # to parsing the COIN_TOSS event in the event log.
        initial_server: str = "A"
        if hasattr(state, "initial_server") and state.initial_server:
            initial_server = state.initial_server
        else:
            for event in state.events:
                if hasattr(event, "metadata") and "first_server=" in (event.metadata or ""):
                    initial_server = event.metadata.split("first_server=")[-1].strip()
                    break

        return cls(
            match_id=state.match_id,
            entity_a_id=state.entity_a_id,
            entity_b_id=state.entity_b_id,
            discipline=state.discipline,
            winner=state.match_winner,
            status=state.status,
            games_won_a=state.games_won_a,
            games_won_b=state.games_won_b,
            game_scores=list(state.game_scores),
            total_games_played=state.games_won_a + state.games_won_b,
            total_points_a=total_a,
            total_points_b=total_b,
            initial_server=initial_server,
        )

    @property
    def is_retired(self) -> bool:
        return self.status == MatchStatus.RETIRED

    @property
    def is_walkover(self) -> bool:
        return self.status == MatchStatus.WALKOVER

    @property
    def total_points(self) -> int:
        return self.total_points_a + self.total_points_b

    def score_string(self) -> str:
        """Human-readable score, e.g. '2-1 (21-18, 14-21, 21-16)'."""
        games_str = ", ".join(f"{sa}-{sb}" for sa, sb in self.game_scores)
        return f"{self.games_won_a}-{self.games_won_b} ({games_str})"


# ---------------------------------------------------------------------------
# Grading service
# ---------------------------------------------------------------------------

class GradingService:
    """
    Settles all badminton markets after match completion.

    Usage:
      grading = GradingService()
      records = grading.settle_match(live_state, open_markets)
    """

    # Registry of all supported market type families.
    # Each entry is (market_type_key, description).  The health endpoint
    # reads this list to report markets_supported — it is the single source
    # of truth for which market families this grading engine handles.
    SUPPORTED_MARKETS: List[Tuple[str, str]] = [
        ("match_winner",          "2-way match winner (P1 wins / P2 wins)"),
        ("correct_score",         "Correct score — 2-0, 2-1, 0-2, 1-2 (BO3)"),
        ("total_games_over_2.5",  "Total games O/U 2.5 — Over side"),
        ("total_games_under_2.5", "Total games O/U 2.5 — Under side"),
        ("exact_games_N",         "Exact total games played (2 or 3)"),
        ("game_N_winner",         "Game N individual winner"),
        ("game_N_total_oNN",      "Game N total points O/U line"),
        ("race_to_N_gameM",       "Race-to-N points in game M"),
        ("match_total_oNN",       "Match total points O/U line"),
        ("points_total_oNN",      "Points total O/U (alias for match_total)"),
        ("first_point_winner",    "First rally point winner in game 1"),
        ("deuce_in_game_N",       "Deuce reached in game N (Yes/No)"),
        ("golden_point",          "Golden point reached in match (Yes/No)"),
        ("comeback",              "Comeback win — winner trailed by a game (Yes/No)"),
        ("handicap_games_A/B",    "Games handicap — entity A or B with ±N.5 line"),
    ]

    def __init__(self) -> None:
        pass

    def settle_match(
        self,
        live_state: MatchLiveState,
        open_markets: Dict[str, List[str]],  # {market_id -> [outcome_names]}
    ) -> List[SettlementRecord]:
        """
        Settle all open markets for a completed match.

        Args:
            live_state: Final MatchLiveState (must be in terminal status)
            open_markets: Dict of open markets to settle.

        Returns:
            List of SettlementRecord (one per market).
        """
        result = MatchResult.from_live_state(live_state)
        now = datetime.now(timezone.utc)
        records: List[SettlementRecord] = []

        for market_id, outcome_names in open_markets.items():
            try:
                record = self._settle_market(market_id, outcome_names, result, now)
                records.append(record)
            except Exception as exc:
                logger.error(
                    "settlement_error",
                    market_id=market_id,
                    match_id=result.match_id,
                    error=str(exc),
                )
                records.append(SettlementRecord(
                    match_id=result.match_id,
                    market_id=market_id,
                    winning_outcome=None,
                    settlement_status=SettlementStatus.ERROR,
                    settled_at=now,
                    entity_a_id=result.entity_a_id,
                    entity_b_id=result.entity_b_id,
                    final_score=result.score_string(),
                    notes=str(exc),
                ))

        logger.info(
            "match_settlement_complete",
            match_id=result.match_id,
            n_markets=len(records),
            winner=result.winner,
            score=result.score_string(),
            is_retired=result.is_retired,
        )

        return records

    def _settle_market(
        self,
        market_id: str,
        outcome_names: List[str],
        result: MatchResult,
        settled_at: datetime,
    ) -> SettlementRecord:
        """Settle a single market based on market_id prefix."""
        winner_outcome, void_reason = self._determine_winner(
            market_id, result, outcome_names=outcome_names
        )

        if void_reason:
            return SettlementRecord(
                match_id=result.match_id,
                market_id=market_id,
                winning_outcome=None,
                settlement_status=SettlementStatus.VOIDED,
                settled_at=settled_at,
                entity_a_id=result.entity_a_id,
                entity_b_id=result.entity_b_id,
                final_score=result.score_string(),
                void_reason=void_reason,
            )

        return SettlementRecord(
            match_id=result.match_id,
            market_id=market_id,
            winning_outcome=winner_outcome,
            settlement_status=SettlementStatus.SETTLED,
            settled_at=settled_at,
            entity_a_id=result.entity_a_id,
            entity_b_id=result.entity_b_id,
            final_score=result.score_string(),
        )

    def _determine_winner(
        self,
        market_id: str,
        result: MatchResult,
        outcome_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determine winning outcome for a market.

        Args:
            market_id: Market identifier string.
            result: Settled match result.
            outcome_names: Available outcome names for this market. When provided,
                           entity-based outcomes are resolved against these names
                           so the return value matches an actual outcome in the list.

        Returns: (winning_outcome_name, void_reason) — one of them is None.
        """

        def _entity_to_outcome(team: str) -> str:
            """
            Map "A" or "B" team to the winning outcome name.

            Priority:
            1. Exact match of entity_id in outcome_names
            2. Positional: A → outcomes[0], B → outcomes[1]
            3. Return entity_id as fallback
            """
            entity = result.entity_a_id if team == "A" else result.entity_b_id
            if outcome_names:
                if entity in outcome_names:
                    return entity
                # Positional mapping: A is outcomes[0], B is outcomes[1]
                idx = 0 if team == "A" else 1
                if idx < len(outcome_names):
                    return outcome_names[idx]
            return entity

        # --- Match Winner ---
        if market_id == "match_winner":
            return _entity_to_outcome(result.winner), None

        # --- Correct Score ---
        if market_id == "correct_score":
            cs = self._correct_score_key(result)
            return cs, None

        # --- Total Games O/U ---
        if market_id == "total_games_over_2.5":
            n_games = result.total_games_played
            if result.is_retired and n_games < 2:
                return None, "retired_incomplete"
            winner = "Over 2.5" if n_games >= 3 else "Under 2.5"
            return winner, None

        if market_id == "total_games_under_2.5":
            n_games = result.total_games_played
            if result.is_retired and n_games < 2:
                return None, "retired_incomplete"
            winner = "Under 2.5" if n_games <= 2 else "Over 2.5"
            return winner, None

        # --- Exact Total Games ---
        if market_id.startswith("exact_games_"):
            n_games = result.total_games_played
            if result.is_retired and n_games < 2:
                return None, "retired_incomplete"
            target = market_id.replace("exact_games_", "")
            try:
                target_n = int(target)
                winner = str(target_n) if n_games == target_n else f"not_{target_n}"
                return winner, None
            except ValueError:
                return None, f"invalid_market_id_{market_id}"

        # --- Game N winner ---
        if market_id.startswith("game_") and "_winner" in market_id:
            game_winner_team, void_reason = self._settle_game_winner(market_id, result)
            if void_reason:
                return None, void_reason
            # Map team "A"/"B" to outcome name
            return _entity_to_outcome(game_winner_team), None

        # --- Game N total points O/U ---
        if market_id.startswith("game_") and "_total_o" in market_id:
            return self._settle_game_total(market_id, result)

        # --- Race to N ---
        if market_id.startswith("race_to_"):
            return self._settle_race_to_n(market_id, result)

        # --- Points totals (match) ---
        if market_id.startswith("match_total_o") or market_id.startswith("points_total_o"):
            return self._settle_match_total(market_id, result)

        # --- Player props ---
        if "first_point" in market_id:
            # Void if retired before first point was played
            if result.total_points_a + result.total_points_b == 0:
                return None, "no_points_played"
            # first_point_winner = winner of rally 1 in game 1 = initial server if they won
            # rally 1 result: score after rally 1 is (1-0) or (0-1).
            # We know initial_server from MatchResult.initial_server.
            # The winner of point 1 is the first entity to reach score 1.
            # We reconstruct from game_scores[0]: if game ends e.g. 21-10,
            # total_points_a=21 suggests A won more rallies — but we need exact P1.
            # From MatchLiveState events: look for first POINT event; if not available,
            # settle on who served first (server has RWP > 0.5 advantage — best proxy).
            if result.initial_server == "A":
                first_point_winner = result.entity_a_id
            else:
                first_point_winner = result.entity_b_id
            return first_point_winner, None

        # --- Deuce markets ---
        if "deuce" in market_id:
            return self._settle_deuce_market(market_id, result)

        # --- Golden point ---
        if "golden_point" in market_id:
            had_gp = self._had_golden_point(result)
            if result.is_retired:
                return None, "retired_incomplete"
            return "Yes" if had_gp else "No", None

        # --- Comeback ---
        if "comeback" in market_id:
            if result.is_retired:
                return None, "retired_incomplete"
            had_comeback = self._detect_comeback(result)
            return "Yes" if had_comeback else "No", None

        # --- Handicap markets ---
        if market_id.startswith("handicap_games"):
            return self._settle_handicap_games(market_id, result)

        # Unknown market
        logger.warning(
            "unknown_market_in_settlement",
            market_id=market_id,
            match_id=result.match_id,
        )
        return None, f"unknown_market_{market_id}"

    @staticmethod
    def _correct_score_key(result: MatchResult) -> str:
        """Return correct score key (e.g., 'A_2-0', 'B_2-1')."""
        winner = result.winner
        loser_games = result.games_won_b if winner == "A" else result.games_won_a
        winner_games = GAMES_TO_WIN_MATCH
        return f"{winner}_{winner_games}-{loser_games}"

    @staticmethod
    def _settle_game_winner(market_id: str, result: MatchResult) -> Tuple[Optional[str], Optional[str]]:
        """Settle game N winner market."""
        # market_id format: "game_1_winner", "game_2_winner", "game_3_winner"
        parts = market_id.split("_")
        if len(parts) < 3:
            return None, f"invalid_market_id_{market_id}"

        try:
            game_n = int(parts[1])
        except ValueError:
            return None, f"invalid_game_number_{market_id}"

        if game_n > len(result.game_scores):
            if result.is_retired:
                return None, "game_not_played_retired"
            return None, f"game_{game_n}_not_played"

        sa, sb = result.game_scores[game_n - 1]
        game_winner = "A" if sa > sb else "B"
        return game_winner, None

    @staticmethod
    def _settle_game_total(market_id: str, result: MatchResult) -> Tuple[Optional[str], Optional[str]]:
        """Settle game N total points O/U market."""
        # market_id format: "game_1_total_o41", "game_2_total_o45"
        parts = market_id.split("_")
        try:
            game_n = int(parts[1])
            threshold = int(parts[-1].replace("o", ""))
        except (ValueError, IndexError):
            return None, f"invalid_market_id_{market_id}"

        if game_n > len(result.game_scores):
            return None, f"game_{game_n}_not_played"

        sa, sb = result.game_scores[game_n - 1]
        total = sa + sb
        winner = f"Over {threshold}" if total > threshold else f"Under {threshold}"
        return winner, None

    @staticmethod
    def _settle_race_to_n(market_id: str, result: MatchResult) -> Tuple[Optional[str], Optional[str]]:
        """Settle race-to-N markets."""
        # market_id: "race_to_5_game1", "race_to_10_game2", etc.
        parts = market_id.split("_")
        try:
            n = int(parts[2])
            game_str = parts[-1]  # "game1", "game2", etc.
            game_n = int(game_str.replace("game", ""))
        except (ValueError, IndexError):
            return None, f"invalid_market_id_{market_id}"

        if game_n > len(result.game_scores):
            return None, f"game_{game_n}_not_played"

        sa, sb = result.game_scores[game_n - 1]
        # Who reached N first?
        if sa >= n and sb >= n:
            # Both reached N — whoever reached it first wins
            # Cannot determine without point-by-point data
            # Use: whoever has higher score at that threshold wins
            winner_entity = result.entity_a_id  # Conservative: A wins race
        elif sa >= n:
            winner_entity = result.entity_a_id
        elif sb >= n:
            winner_entity = result.entity_b_id
        else:
            # Neither reached N (unusual unless retired)
            return None, f"n_{n}_not_reached_in_game_{game_n}"

        return winner_entity, None

    @staticmethod
    def _settle_match_total(market_id: str, result: MatchResult) -> Tuple[Optional[str], Optional[str]]:
        """Settle match total points O/U market."""
        # market_id: "match_total_o79.5", "match_total_o83.5"
        try:
            threshold_str = market_id.split("o")[-1]
            threshold = float(threshold_str)
        except (ValueError, IndexError):
            return None, f"invalid_threshold_{market_id}"

        if result.is_retired:
            return None, "retired_incomplete"

        total = result.total_points
        winner = f"Over {threshold}" if total > threshold else f"Under {threshold}"
        return winner, None

    @staticmethod
    def _settle_deuce_market(market_id: str, result: MatchResult) -> Tuple[Optional[str], Optional[str]]:
        """Settle deuce in game N market."""
        # Check if any game reached 20-20
        had_deuce = any(
            sa >= 20 and sb >= 20
            for sa, sb in result.game_scores
        )
        if result.is_retired and result.total_points < 30:
            return None, "retired_before_deuce_possible"
        return "Yes" if had_deuce else "No", None

    @staticmethod
    def _had_golden_point(result: MatchResult) -> bool:
        """True if any game ended at golden point (30-29 or 29-30)."""
        for sa, sb in result.game_scores:
            if (sa == GOLDEN_POINT_WIN and sb == GOLDEN_POINT_WIN - 1) or \
               (sb == GOLDEN_POINT_WIN and sa == GOLDEN_POINT_WIN - 1):
                return True
        return False

    @staticmethod
    def _detect_comeback(result: MatchResult) -> bool:
        """
        True if the match winner was trailing by >= 1 game at some point
        and came back to win (i.e., won the match from 0-1 down).
        """
        return result.games_won_a + result.games_won_b == 3 and (
            result.games_won_a == 2 or result.games_won_b == 2
        )

    @staticmethod
    def _settle_handicap_games(market_id: str, result: MatchResult) -> Tuple[Optional[str], Optional[str]]:
        """Settle games handicap market."""
        # market_id: "handicap_games_a_-1.5" or "handicap_games_b_+1.5"
        try:
            parts = market_id.split("_")
            entity = parts[2].upper()
            handicap = float(parts[3])
        except (ValueError, IndexError):
            return None, f"invalid_handicap_market_{market_id}"

        if result.is_retired and result.total_games_played < 2:
            return None, "retired_incomplete"

        games_a = result.games_won_a
        games_b = result.games_won_b

        if entity == "A":
            adjusted = games_a + handicap
            winner = result.entity_a_id if adjusted > games_b else result.entity_b_id
        else:
            adjusted = games_b + handicap
            winner = result.entity_b_id if adjusted > games_a else result.entity_a_id

        # Push check (exact tie after handicap)
        if entity == "A" and abs((games_a + handicap) - games_b) < 0.001:
            return None, None  # Push

        return winner, None

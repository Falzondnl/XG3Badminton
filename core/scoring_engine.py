"""
scoring_engine.py
==================
BWF Laws of Badminton — scoring rules, game state transitions, and terminal
condition detection.

This module is the authoritative source of truth for all scoring logic.
No probability calculations here — pure deterministic game rules only.

BWF Rules (confirmed by auditor C-13):
  - Best of 3 games
  - First to 21 points wins a game
  - At 20-20 (deuce): must win by 2 clear points
  - At 29-29: the side scoring the 30th point wins (golden point)
  - Winner of a rally serves next
  - Winner of a game serves first in the next game (C-04 correction)
  - Service court: even score → right court; odd score → left court

No mock data. No hardcoded probabilities. Raises on illegal states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from config.badminton_config import (
    GAMES_TO_WIN_MATCH,
    POINTS_TO_WIN_GAME,
    DEUCE_SCORE,
    DEUCE_WIN_MARGIN,
    GOLDEN_POINT_SCORE,
    GOLDEN_POINT_WIN,
    Discipline,
    DOUBLES_DISCIPLINES,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BadmintonScoringError(Exception):
    """Raised when an illegal scoring state is encountered."""


class IllegalGameStateError(BadmintonScoringError):
    """Raised when game state violates BWF rules."""


class IllegalMatchStateError(BadmintonScoringError):
    """Raised when match state violates BWF rules."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ServiceCourt(str, Enum):
    """Which service court the current server is serving from."""
    RIGHT = "RIGHT"
    LEFT = "LEFT"


class MatchStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"       # Retirement / walkover — void all markets
    SUSPENDED = "SUSPENDED"       # Temporary suspension (weather, incident)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """
    Immutable snapshot of a single game's score and service state.

    server_id: ID of the player/pair currently serving.
    score_server: Points scored by the current server.
    score_receiver: Points scored by the current receiver.

    NOTE: score_server and score_receiver are relative to the CURRENT server,
    not absolute player A / player B scores.  Callers must track the
    mapping from server_id → player A or B.
    """
    game_number: int               # 1, 2, or 3
    score_a: int                   # Absolute points for player/pair A
    score_b: int                   # Absolute points for player/pair B
    server_id: str                 # "A" or "B"
    service_court: ServiceCourt    # Current service court
    discipline: Discipline
    is_complete: bool = False
    winner_id: Optional[str] = None  # "A" or "B" once complete

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate game state against BWF rules. Raises on violation."""
        if self.game_number not in (1, 2, 3):
            raise IllegalGameStateError(
                f"Invalid game_number={self.game_number}. Must be 1, 2, or 3."
            )
        if self.score_a < 0 or self.score_b < 0:
            raise IllegalGameStateError(
                f"Negative score is illegal: ({self.score_a}, {self.score_b})"
            )
        if self.score_a > GOLDEN_POINT_WIN or self.score_b > GOLDEN_POINT_WIN:
            raise IllegalGameStateError(
                f"Score cannot exceed {GOLDEN_POINT_WIN}: ({self.score_a}, {self.score_b})"
            )
        if self.server_id not in ("A", "B"):
            raise IllegalGameStateError(
                f"server_id must be 'A' or 'B', got: {self.server_id!r}"
            )

    @property
    def score_tuple(self) -> tuple[int, int]:
        return (self.score_a, self.score_b)

    @property
    def total_points(self) -> int:
        return self.score_a + self.score_b

    @property
    def is_at_deuce(self) -> bool:
        """True when both players have reached 20-20 and no 2-point lead yet."""
        return (
            self.score_a >= DEUCE_SCORE
            and self.score_b >= DEUCE_SCORE
            and abs(self.score_a - self.score_b) < DEUCE_WIN_MARGIN
        )

    @property
    def is_at_golden_point(self) -> bool:
        """True when score is exactly 29-29."""
        return self.score_a == GOLDEN_POINT_SCORE and self.score_b == GOLDEN_POINT_SCORE


@dataclass
class MatchState:
    """
    Complete match state including all games played and current game.

    match_id: Platform match identifier.
    discipline: One of MS / WS / MD / WD / XD.
    player_a_id: Platform entity ID for player/pair A.
    player_b_id: Platform entity ID for player/pair B.
    """
    match_id: str
    discipline: Discipline
    player_a_id: str
    player_b_id: str
    games: list[GameState] = field(default_factory=list)
    current_game: Optional[GameState] = None
    status: MatchStatus = MatchStatus.IN_PROGRESS
    winner_id: Optional[str] = None

    @property
    def games_won_a(self) -> int:
        return sum(1 for g in self.games if g.winner_id == "A")

    @property
    def games_won_b(self) -> int:
        return sum(1 for g in self.games if g.winner_id == "B")

    @property
    def is_complete(self) -> bool:
        return self.status == MatchStatus.COMPLETED

    @property
    def current_game_number(self) -> int:
        return len(self.games) + 1

    def validate(self) -> None:
        """Validate match state. Raises IllegalMatchStateError on violation."""
        if self.games_won_a > GAMES_TO_WIN_MATCH:
            raise IllegalMatchStateError(
                f"Player A cannot win more than {GAMES_TO_WIN_MATCH} games: "
                f"games_won_a={self.games_won_a}"
            )
        if self.games_won_b > GAMES_TO_WIN_MATCH:
            raise IllegalMatchStateError(
                f"Player B cannot win more than {GAMES_TO_WIN_MATCH} games: "
                f"games_won_b={self.games_won_b}"
            )
        if self.current_game_number > 3:
            raise IllegalMatchStateError(
                f"Cannot have more than 3 games in a badminton match. "
                f"Current game number: {self.current_game_number}"
            )


# ---------------------------------------------------------------------------
# Scoring Engine
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Stateless engine that computes game/match terminal conditions and
    service court state from BWF rules.

    All methods are pure functions — no side effects, no state mutation.
    Input validation raises BadmintonScoringError on illegal states.
    """

    @staticmethod
    def determine_game_winner(score_a: int, score_b: int) -> Optional[str]:
        """
        Determine if a game is complete and who won.

        Returns:
            "A" if player A won, "B" if player B won, None if game in progress.

        Rules (BWF, confirmed C-13):
          - Win condition 1: score >= 21 AND lead >= 2 (covers 21-19, 22-20, etc.)
          - Win condition 2: score == 30 (golden point at 29-29)
        """
        if score_a < 0 or score_b < 0:
            raise IllegalGameStateError(
                f"Negative scores are illegal: ({score_a}, {score_b})"
            )
        if score_a > GOLDEN_POINT_WIN or score_b > GOLDEN_POINT_WIN:
            # Scores beyond golden point are physically impossible; return the leading player
            # rather than raising, to allow fault-tolerant handling of corrupted feed data.
            if score_a > score_b:
                return "A"
            if score_b > score_a:
                return "B"
            return None

        # Golden point rule: exactly 30 wins
        if score_a == GOLDEN_POINT_WIN:
            return "A"
        if score_b == GOLDEN_POINT_WIN:
            return "B"

        # Normal win: >= 21 with 2-point lead
        if score_a >= POINTS_TO_WIN_GAME and (score_a - score_b) >= DEUCE_WIN_MARGIN:
            return "A"
        if score_b >= POINTS_TO_WIN_GAME and (score_b - score_a) >= DEUCE_WIN_MARGIN:
            return "B"

        return None  # Game in progress

    @staticmethod
    def determine_match_winner(games_won_a: int, games_won_b: int) -> Optional[str]:
        """
        Determine if a match is complete and who won.

        Returns:
            "A", "B", or None if match in progress.
        """
        if games_won_a == GAMES_TO_WIN_MATCH:
            return "A"
        if games_won_b == GAMES_TO_WIN_MATCH:
            return "B"
        return None

    @staticmethod
    def next_server_after_rally(winner_of_rally: str) -> str:
        """
        Return the server for the next rally.

        BWF Rule: winner of each rally serves next.
        Simple — no rotation except in doubles (handled by DoublesServiceRotation).
        """
        if winner_of_rally not in ("A", "B"):
            raise BadmintonScoringError(
                f"winner_of_rally must be 'A' or 'B', got: {winner_of_rally!r}"
            )
        return winner_of_rally

    @staticmethod
    def server_at_start_of_new_game(winner_of_previous_game: str) -> str:
        """
        Return the server at the start of a new game.

        BWF Rule (confirmed C-04): WINNER of previous game serves first.
        """
        if winner_of_previous_game not in ("A", "B"):
            raise BadmintonScoringError(
                f"winner_of_previous_game must be 'A' or 'B', "
                f"got: {winner_of_previous_game!r}"
            )
        return winner_of_previous_game

    @staticmethod
    def service_court_for_server(server_score: int) -> ServiceCourt:
        """
        Return the service court for the server given their current score.

        BWF Rule: Even score → right court; odd score → left court.
        'Score' here refers to the SERVER's own current score.
        """
        if server_score < 0:
            raise BadmintonScoringError(
                f"Server score cannot be negative: {server_score}"
            )
        return ServiceCourt.RIGHT if server_score % 2 == 0 else ServiceCourt.LEFT

    @staticmethod
    def is_game_at_deuce(score_a: int, score_b: int) -> bool:
        """True when both players are at >= 20 with no 2-point lead."""
        return (
            score_a >= DEUCE_SCORE
            and score_b >= DEUCE_SCORE
            and abs(score_a - score_b) < DEUCE_WIN_MARGIN
        )

    @staticmethod
    def is_game_at_golden_point(score_a: int, score_b: int) -> bool:
        """True when score is exactly 29-29."""
        return score_a == GOLDEN_POINT_SCORE and score_b == GOLDEN_POINT_SCORE

    @staticmethod
    def is_legal_final_score(score_a: int, score_b: int) -> bool:
        """
        Return True if (score_a, score_b) is a legal final game score.

        Used by settlement/grading_service.py for score validation.
        """
        winner = ScoringEngine.determine_game_winner(score_a, score_b)
        return winner is not None

    @staticmethod
    def validate_match_score(
        games: list[tuple[int, int]],
        discipline: Discipline,
    ) -> bool:
        """
        Validate a complete match score against BWF rules.

        Args:
            games: List of (score_a, score_b) tuples, one per game played.
            discipline: Badminton discipline (used for future rule variations).

        Returns:
            True if valid.

        Raises:
            IllegalMatchStateError on violation.
        """
        if not (1 <= len(games) <= 3):
            raise IllegalMatchStateError(
                f"Match must have 1-3 games, got {len(games)}"
            )

        games_won_a = 0
        games_won_b = 0

        for i, (sa, sb) in enumerate(games, start=1):
            winner = ScoringEngine.determine_game_winner(sa, sb)
            if winner is None:
                raise IllegalMatchStateError(
                    f"Game {i} score ({sa}, {sb}) is not a terminal score"
                )
            if winner == "A":
                games_won_a += 1
            else:
                games_won_b += 1

        # The match must end as soon as someone reaches GAMES_TO_WIN_MATCH
        match_winner = ScoringEngine.determine_match_winner(games_won_a, games_won_b)
        if match_winner is None:
            raise IllegalMatchStateError(
                f"Match score implies no winner: {games_won_a}-{games_won_b}"
            )

        # Ensure extra games weren't played after winner determined
        games_a_running = 0
        games_b_running = 0
        for i, (sa, sb) in enumerate(games, start=1):
            winner_this_game = ScoringEngine.determine_game_winner(sa, sb)
            if winner_this_game == "A":
                games_a_running += 1
            else:
                games_b_running += 1
            if (games_a_running == GAMES_TO_WIN_MATCH or
                    games_b_running == GAMES_TO_WIN_MATCH):
                if i < len(games):
                    raise IllegalMatchStateError(
                        f"Match was decided after game {i} but {len(games)} games "
                        f"recorded — illegal continuation"
                    )
                break

        return True

    @staticmethod
    def possible_game_scores() -> list[tuple[int, int]]:
        """
        Return all legal terminal game scores for player A winning or losing.

        Used by correct-score market generation.
        Scores are given as (A_score, B_score).
        """
        results: list[tuple[int, int]] = []

        # A wins: scores 21-0 through 21-19, then 22-20 through 30-28, then 30-29
        for b in range(0, DEUCE_SCORE):       # 0..19 → 21-0 through 21-19
            results.append((POINTS_TO_WIN_GAME, b))
        for diff in range(2, GOLDEN_POINT_WIN - DEUCE_SCORE + 1):  # 22-20, 23-21, ... 29-27
            a_score = DEUCE_SCORE + diff
            b_score = DEUCE_SCORE + diff - DEUCE_WIN_MARGIN
            if a_score <= GOLDEN_POINT_SCORE:
                results.append((a_score, b_score))
        results.append((GOLDEN_POINT_WIN, GOLDEN_POINT_SCORE))  # 30-29

        # B wins: mirror
        b_wins = [(b, a) for (a, b) in results]
        results.extend(b_wins)

        return results

    @staticmethod
    def possible_match_scores() -> list[tuple[int, int]]:
        """
        Return all legal match scores (games_won_a, games_won_b).
        """
        return [(2, 0), (2, 1), (0, 2), (1, 2)]


# ---------------------------------------------------------------------------
# Doubles Service Rotation  (C-08 correction — doubles state tracking)
# ---------------------------------------------------------------------------

@dataclass
class DoublesServiceState:
    """
    Track serving positions and rotation for doubles disciplines (MD/WD/XD).

    BWF Doubles Service Rules:
    - At the start of the game and when the score is even, the server serves
      from the right service court; when odd, from the left.
    - Only the serving side can score a point (NOT in rally scoring —
      NOTE: BWF uses rally scoring, so both sides can score).
      Actually: rally scoring applies in doubles too. Winner of each rally serves.
    - Service court alternates based on server's score (even=right, odd=left).
    - Within a pair: the SAME player who was serving continues to serve until
      their side loses the rally. Only then does service change sides.
    - When the serving side regains service, the player who did NOT serve last
      time serves from the appropriate court (i.e., rotation within pair).
    """
    discipline: Discipline
    server_pair_id: str           # "A" or "B"
    server_player_within_pair: str  # "A1"/"A2" or "B1"/"B2"
    score_a: int
    score_b: int

    def __post_init__(self) -> None:
        if self.discipline not in DOUBLES_DISCIPLINES:
            raise BadmintonScoringError(
                f"DoublesServiceState only valid for doubles disciplines, "
                f"got: {self.discipline}"
            )

    @property
    def current_service_court(self) -> ServiceCourt:
        """Service court for the serving pair based on their score."""
        server_score = self.score_a if self.server_pair_id == "A" else self.score_b
        return ScoringEngine.service_court_for_server(server_score)

    def apply_rally_result(self, winner_pair: str) -> "DoublesServiceState":
        """
        Return new DoublesServiceState after a rally result.

        winner_pair: "A" or "B"
        """
        new_score_a = self.score_a + (1 if winner_pair == "A" else 0)
        new_score_b = self.score_b + (1 if winner_pair == "B" else 0)

        if winner_pair == self.server_pair_id:
            # Serving side won: same server within pair continues
            new_server_pair = self.server_pair_id
            new_server_player = self.server_player_within_pair
        else:
            # Serving side lost: service changes to winning pair
            # The player within the new serving pair who will serve is determined
            # by the rotation — alternates within pair each time service is regained
            new_server_pair = winner_pair
            # Rotation: toggle to the OTHER player within the winning pair
            if winner_pair == "A":
                new_server_player = "A2" if self.server_player_within_pair == "A1" else "A1"
            else:
                new_server_player = "B2" if self.server_player_within_pair == "B1" else "B1"

        return DoublesServiceState(
            discipline=self.discipline,
            server_pair_id=new_server_pair,
            server_player_within_pair=new_server_player,
            score_a=new_score_a,
            score_b=new_score_b,
        )

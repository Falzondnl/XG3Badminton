"""
match_state.py
==============
Live match state machine for badminton.

Tracks the complete authoritative state of an in-progress match:
  - Current game number, scores, server, service court
  - Run tracking (consecutive points won by each side)
  - Timeline of all events for momentum analysis
  - Suspension / retirement / walkover flags

State machine is immutable-update: apply_event() returns a new MatchLiveState.
All public methods are pure — no side effects.

Used by:
  - LiveSupervisorAgent (feeds into Bayesian updater)
  - MomentumDetector (reads run history)
  - MarketTrader (determines valid markets)
  - GradingService (final score validation)

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    DOUBLES_DISCIPLINES,
    GAMES_TO_WIN_MATCH,
    POINTS_TO_WIN_GAME,
    DEUCE_SCORE,
    GOLDEN_POINT_SCORE,
    GOLDEN_POINT_WIN,
)
from core.scoring_engine import ScoringEngine, GameState, MatchState

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class PointWinner(str, Enum):
    A = "A"
    B = "B"


class MatchStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUSPENDED = "suspended"    # Temporary halt (bad shuttle, injury timeout)
    RETIRED = "retired"        # Mid-match retirement
    WALKOVER = "walkover"      # Pre-match walkover
    COMPLETED = "completed"


class EventType(str, Enum):
    POINT = "point"
    GAME_END = "game_end"
    MATCH_END = "match_end"
    SUSPENSION_START = "suspension_start"
    SUSPENSION_END = "suspension_end"
    RETIREMENT = "retirement"
    TIMEOUT = "timeout"        # Mid-game timeout (BWF allows 60s max per game)
    INTERVAL = "interval"      # Interval at 11 in game 3 (BWF rule)
    COIN_TOSS = "coin_toss"    # Match start — determines first server


@dataclass(frozen=True)
class MatchEvent:
    """Single discrete event in match timeline."""
    event_type: EventType
    timestamp: datetime
    winner: Optional[PointWinner] = None       # For POINT events
    game_number: int = 1
    score_a_after: int = 0
    score_b_after: int = 0
    server_after: Optional[str] = None         # "A" or "B"
    metadata: str = ""                         # Free text (e.g., "retired: ankle")

    def __post_init__(self) -> None:
        if self.event_type == EventType.POINT and self.winner is None:
            raise ValueError("POINT events must have a winner")


# ---------------------------------------------------------------------------
# Service state for doubles rotation (C-08)
# ---------------------------------------------------------------------------

@dataclass
class DoublesServiceTracker:
    """
    Tracks service rotation state for doubles disciplines.

    BWF doubles service rules:
      - When server wins rally: server retains serve, swaps sides
      - When receiver wins rally: receiver becomes server, no side swap
      - Side assignment is persistent: right_server, left_server per team
    """
    # Which player in each pair is currently assigned to right service court
    team_a_right_player: str = ""
    team_a_left_player: str = ""
    team_b_right_player: str = ""
    team_b_left_player: str = ""

    # Current server team
    serving_team: str = "A"           # "A" or "B"
    serving_player: str = ""          # specific player ID within team

    def apply_point(self, winner_team: str) -> "DoublesServiceTracker":
        """
        Return updated tracker after a point.

        If server wins: server retains serve, swaps service court positions.
        If receiver wins: receiver becomes server in their current position.
        """
        new = DoublesServiceTracker(
            team_a_right_player=self.team_a_right_player,
            team_a_left_player=self.team_a_left_player,
            team_b_right_player=self.team_b_right_player,
            team_b_left_player=self.team_b_left_player,
            serving_team=self.serving_team,
            serving_player=self.serving_player,
        )

        if winner_team == self.serving_team:
            # Server wins: retain serve, swap court positions within serving team
            if new.serving_team == "A":
                new.team_a_right_player, new.team_a_left_player = (
                    new.team_a_left_player, new.team_a_right_player
                )
            else:
                new.team_b_right_player, new.team_b_left_player = (
                    new.team_b_left_player, new.team_b_right_player
                )
        else:
            # Receiver wins: receiving team becomes server
            # The player in the right court of the receiving team serves
            new.serving_team = winner_team
            if winner_team == "A":
                new.serving_player = new.team_a_right_player
            else:
                new.serving_player = new.team_b_right_player

        return new


# ---------------------------------------------------------------------------
# Main live state dataclass
# ---------------------------------------------------------------------------

@dataclass
class MatchLiveState:
    """
    Complete live state of an in-progress badminton match.

    Immutable-update pattern: use apply_point(), apply_game_end(), etc.
    to get a new MatchLiveState.
    """

    # Match identification
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline

    # Match status
    status: MatchStatus = MatchStatus.NOT_STARTED

    # Current game state
    current_game: int = 1               # 1-indexed
    score_a: int = 0                    # Points in current game
    score_b: int = 0
    games_won_a: int = 0
    games_won_b: int = 0

    # Service state
    server: str = "A"                   # "A" or "B"
    initial_server: str = "A"           # Server in game 1 point 1 (coin-toss winner)
    service_court: str = "RIGHT"        # "RIGHT" or "LEFT" (for service court display)

    # Per-game scores (completed games)
    game_scores: List[Tuple[int, int]] = field(default_factory=list)

    # Run tracking
    current_run_a: int = 0             # Consecutive points by A
    current_run_b: int = 0             # Consecutive points by B
    max_run_a: int = 0                 # Max run in current game by A
    max_run_b: int = 0
    match_run_a: int = 0               # Max run across entire match by A
    match_run_b: int = 0

    # Timeline
    events: List[MatchEvent] = field(default_factory=list)

    # Total points played across match
    total_points_played: int = 0
    total_points_a: int = 0            # Points A won across all games
    total_points_b: int = 0

    # Interval flag (BWF: interval at 11 points in game 3)
    interval_taken_game3: bool = False

    # Suspension metadata
    suspension_reason: str = ""
    suspended_at: Optional[datetime] = None

    # Winner (set when status = COMPLETED)
    match_winner: Optional[str] = None  # "A" or "B"

    # Doubles service tracker (None for singles)
    doubles_tracker: Optional[DoublesServiceTracker] = None

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_event_at: Optional[datetime] = None

    def is_in_deuce(self) -> bool:
        """True if current game is at or past deuce point (20-20)."""
        return self.score_a >= DEUCE_SCORE and self.score_b >= DEUCE_SCORE

    def is_at_golden_point(self) -> bool:
        """True if current game is at 29-29 (golden point)."""
        return self.score_a == GOLDEN_POINT_SCORE and self.score_b == GOLDEN_POINT_SCORE

    def is_game_3(self) -> bool:
        """True if currently in the deciding 3rd game."""
        return self.current_game == 3

    def points_in_current_game(self) -> int:
        """Total points played in current game."""
        return self.score_a + self.score_b

    def total_points_in_match_so_far(self) -> int:
        """Sum of all points across all completed games plus current game."""
        completed = sum(a + b for a, b in self.game_scores)
        return completed + self.score_a + self.score_b

    def lead_a(self) -> int:
        """Current point lead for entity A (negative = B is leading)."""
        return self.score_a - self.score_b

    def as_game_state(self) -> GameState:
        """Convert current game to GameState for Markov engine."""
        return GameState(
            score_a=self.score_a,
            score_b=self.score_b,
            server=self.server,
            game_number=self.current_game,
        )

    def as_match_state(self) -> MatchState:
        """Convert to MatchState for Markov engine."""
        return MatchState(
            games_won_a=self.games_won_a,
            games_won_b=self.games_won_b,
            current_game=self.current_game,
            game_state=self.as_game_state(),
        )

    def validate(self) -> None:
        """
        Assert internal consistency.

        Raises ValueError if state is internally inconsistent.
        """
        if self.score_a < 0 or self.score_b < 0:
            raise ValueError(f"Negative score: {self.score_a}-{self.score_b}")
        if self.games_won_a < 0 or self.games_won_b < 0:
            raise ValueError("Negative games won")
        if self.games_won_a > GAMES_TO_WIN_MATCH or self.games_won_b > GAMES_TO_WIN_MATCH:
            raise ValueError(f"Games won exceeds max: {self.games_won_a}/{self.games_won_b}")
        if len(self.game_scores) != self.games_won_a + self.games_won_b:
            raise ValueError(
                f"game_scores length {len(self.game_scores)} != "
                f"games played {self.games_won_a + self.games_won_b}"
            )
        if self.server not in ("A", "B"):
            raise ValueError(f"Invalid server: {self.server!r}")


# ---------------------------------------------------------------------------
# State machine — pure functions
# ---------------------------------------------------------------------------

class BadmintonMatchStateMachine:
    """
    Pure state machine for applying events to MatchLiveState.

    All methods are static and return new state objects.
    No mutation of existing state.
    """

    @staticmethod
    def initialise(
        match_id: str,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        first_server: str = "A",
        doubles_tracker: Optional[DoublesServiceTracker] = None,
        started_at: Optional[datetime] = None,
    ) -> MatchLiveState:
        """
        Create initial live state for a match that is about to start.

        Args:
            first_server: "A" or "B" — winner of coin toss serves first.
        """
        if first_server not in ("A", "B"):
            raise ValueError(f"first_server must be 'A' or 'B', got {first_server!r}")

        ts = started_at or datetime.now(timezone.utc)

        service_court = ScoringEngine.service_court_for_server(server_score=0)

        return MatchLiveState(
            match_id=match_id,
            entity_a_id=entity_a_id,
            entity_b_id=entity_b_id,
            discipline=discipline,
            status=MatchStatus.IN_PROGRESS,
            current_game=1,
            score_a=0,
            score_b=0,
            games_won_a=0,
            games_won_b=0,
            server=first_server,
            initial_server=first_server,
            service_court=service_court,
            game_scores=[],
            current_run_a=0,
            current_run_b=0,
            max_run_a=0,
            max_run_b=0,
            match_run_a=0,
            match_run_b=0,
            events=[
                MatchEvent(
                    event_type=EventType.COIN_TOSS,
                    timestamp=ts,
                    metadata=f"first_server={first_server}",
                )
            ],
            total_points_played=0,
            total_points_a=0,
            total_points_b=0,
            interval_taken_game3=False,
            doubles_tracker=doubles_tracker,
            started_at=ts,
            last_event_at=ts,
        )

    @staticmethod
    def apply_point(
        state: MatchLiveState,
        winner: PointWinner,
        timestamp: Optional[datetime] = None,
    ) -> MatchLiveState:
        """
        Apply a single point to the state.

        Returns a new MatchLiveState with updated scores, server,
        service court, run tracking, and event log.

        Raises ValueError if match is already completed or not in progress.
        """
        if state.status not in (MatchStatus.IN_PROGRESS,):
            raise ValueError(
                f"Cannot apply point to match in status {state.status.value}"
            )

        ts = timestamp or datetime.now(timezone.utc)
        winner_str = winner.value

        # Compute new scores
        new_score_a = state.score_a + (1 if winner_str == "A" else 0)
        new_score_b = state.score_b + (1 if winner_str == "B" else 0)

        # Determine new server (winner of rally serves next — C-04)
        new_server = ScoringEngine.next_server_after_rally(winner_str)

        # Service court: based on server's own score
        server_score = new_score_a if new_server == "A" else new_score_b
        new_service_court = ScoringEngine.service_court_for_server(server_score)

        # Run tracking
        if winner_str == "A":
            new_run_a = state.current_run_a + 1
            new_run_b = 0
        else:
            new_run_a = 0
            new_run_b = state.current_run_b + 1

        new_max_run_a = max(state.max_run_a, new_run_a)
        new_max_run_b = max(state.max_run_b, new_run_b)
        new_match_run_a = max(state.match_run_a, new_run_a)
        new_match_run_b = max(state.match_run_b, new_run_b)

        # Update doubles tracker if applicable
        new_doubles_tracker = None
        if state.doubles_tracker is not None:
            new_doubles_tracker = state.doubles_tracker.apply_point(winner_str)

        # Build event
        new_event = MatchEvent(
            event_type=EventType.POINT,
            timestamp=ts,
            winner=winner,
            game_number=state.current_game,
            score_a_after=new_score_a,
            score_b_after=new_score_b,
            server_after=new_server,
        )

        new_events = state.events + [new_event]
        new_total_a = state.total_points_a + (1 if winner_str == "A" else 0)
        new_total_b = state.total_points_b + (1 if winner_str == "B" else 0)

        # Check if game ended
        game_winner = ScoringEngine.determine_game_winner(new_score_a, new_score_b)

        if game_winner is not None:
            return BadmintonMatchStateMachine._apply_game_end(
                state=state,
                game_winner=game_winner,
                final_score_a=new_score_a,
                final_score_b=new_score_b,
                new_run_a=new_run_a,
                new_run_b=new_run_b,
                new_max_run_a=new_max_run_a,
                new_max_run_b=new_max_run_b,
                new_match_run_a=new_match_run_a,
                new_match_run_b=new_match_run_b,
                new_total_a=new_total_a,
                new_total_b=new_total_b,
                new_events=new_events,
                ts=ts,
                new_doubles_tracker=new_doubles_tracker,
            )

        # Game not ended — return updated in-progress state
        return MatchLiveState(
            match_id=state.match_id,
            entity_a_id=state.entity_a_id,
            entity_b_id=state.entity_b_id,
            discipline=state.discipline,
            status=MatchStatus.IN_PROGRESS,
            current_game=state.current_game,
            score_a=new_score_a,
            score_b=new_score_b,
            games_won_a=state.games_won_a,
            games_won_b=state.games_won_b,
            server=new_server,
            service_court=new_service_court,
            game_scores=list(state.game_scores),
            current_run_a=new_run_a,
            current_run_b=new_run_b,
            max_run_a=new_max_run_a,
            max_run_b=new_max_run_b,
            match_run_a=new_match_run_a,
            match_run_b=new_match_run_b,
            events=new_events,
            total_points_played=state.total_points_played + 1,
            total_points_a=new_total_a,
            total_points_b=new_total_b,
            interval_taken_game3=state.interval_taken_game3,
            doubles_tracker=new_doubles_tracker,
            started_at=state.started_at,
            last_event_at=ts,
        )

    @staticmethod
    def _apply_game_end(
        state: MatchLiveState,
        game_winner: str,
        final_score_a: int,
        final_score_b: int,
        new_run_a: int,
        new_run_b: int,
        new_max_run_a: int,
        new_max_run_b: int,
        new_match_run_a: int,
        new_match_run_b: int,
        new_total_a: int,
        new_total_b: int,
        new_events: List[MatchEvent],
        ts: datetime,
        new_doubles_tracker: Optional[DoublesServiceTracker],
    ) -> MatchLiveState:
        """Handle game completion — may trigger match completion."""

        new_games_won_a = state.games_won_a + (1 if game_winner == "A" else 0)
        new_games_won_b = state.games_won_b + (1 if game_winner == "B" else 0)
        new_game_scores = list(state.game_scores) + [(final_score_a, final_score_b)]

        # Add game_end event
        game_end_event = MatchEvent(
            event_type=EventType.GAME_END,
            timestamp=ts,
            game_number=state.current_game,
            score_a_after=final_score_a,
            score_b_after=final_score_b,
            metadata=f"winner={game_winner}",
        )
        new_events = new_events + [game_end_event]

        # Check match winner (C-02: first to win 2 games wins match)
        match_winner = ScoringEngine.determine_match_winner(new_games_won_a, new_games_won_b)

        if match_winner is not None:
            # Match complete
            match_end_event = MatchEvent(
                event_type=EventType.MATCH_END,
                timestamp=ts,
                game_number=state.current_game,
                score_a_after=new_games_won_a,
                score_b_after=new_games_won_b,
                metadata=f"winner={match_winner}",
            )
            return MatchLiveState(
                match_id=state.match_id,
                entity_a_id=state.entity_a_id,
                entity_b_id=state.entity_b_id,
                discipline=state.discipline,
                status=MatchStatus.COMPLETED,
                current_game=state.current_game,
                score_a=final_score_a,
                score_b=final_score_b,
                games_won_a=new_games_won_a,
                games_won_b=new_games_won_b,
                server=state.server,  # Irrelevant after match end
                service_court=state.service_court,
                game_scores=new_game_scores,
                current_run_a=new_run_a,
                current_run_b=new_run_b,
                max_run_a=new_max_run_a,
                max_run_b=new_max_run_b,
                match_run_a=new_match_run_a,
                match_run_b=new_match_run_b,
                events=new_events + [match_end_event],
                total_points_played=state.total_points_played + 1,
                total_points_a=new_total_a,
                total_points_b=new_total_b,
                interval_taken_game3=state.interval_taken_game3,
                match_winner=match_winner,
                doubles_tracker=new_doubles_tracker,
                started_at=state.started_at,
                completed_at=ts,
                last_event_at=ts,
            )

        # Game ended but match continues — start new game
        new_game = state.current_game + 1

        # C-04: Winner of game serves first in next game
        first_server_new_game = ScoringEngine.server_at_start_of_new_game(game_winner)
        new_service_court = ScoringEngine.service_court_for_server(0)

        return MatchLiveState(
            match_id=state.match_id,
            entity_a_id=state.entity_a_id,
            entity_b_id=state.entity_b_id,
            discipline=state.discipline,
            status=MatchStatus.IN_PROGRESS,
            current_game=new_game,
            score_a=0,
            score_b=0,
            games_won_a=new_games_won_a,
            games_won_b=new_games_won_b,
            server=first_server_new_game,
            service_court=new_service_court,
            game_scores=new_game_scores,
            current_run_a=0,
            current_run_b=0,
            max_run_a=0,
            max_run_b=0,
            match_run_a=new_match_run_a,
            match_run_b=new_match_run_b,
            events=new_events,
            total_points_played=state.total_points_played + 1,
            total_points_a=new_total_a,
            total_points_b=new_total_b,
            interval_taken_game3=False,
            doubles_tracker=new_doubles_tracker,
            started_at=state.started_at,
            last_event_at=ts,
        )

    @staticmethod
    def apply_suspension(
        state: MatchLiveState,
        reason: str,
        timestamp: Optional[datetime] = None,
    ) -> MatchLiveState:
        """Mark match as suspended (bird change, injury, etc.)."""
        ts = timestamp or datetime.now(timezone.utc)
        evt = MatchEvent(
            event_type=EventType.SUSPENSION_START,
            timestamp=ts,
            game_number=state.current_game,
            score_a_after=state.score_a,
            score_b_after=state.score_b,
            metadata=reason,
        )
        new_state = MatchLiveState(**{
            **state.__dict__,
            "status": MatchStatus.SUSPENDED,
            "suspension_reason": reason,
            "suspended_at": ts,
            "events": state.events + [evt],
            "last_event_at": ts,
        })
        return new_state

    @staticmethod
    def resume_from_suspension(
        state: MatchLiveState,
        timestamp: Optional[datetime] = None,
    ) -> MatchLiveState:
        """Resume a suspended match."""
        if state.status != MatchStatus.SUSPENDED:
            raise ValueError(f"Cannot resume match with status {state.status.value}")

        ts = timestamp or datetime.now(timezone.utc)
        evt = MatchEvent(
            event_type=EventType.SUSPENSION_END,
            timestamp=ts,
            game_number=state.current_game,
            score_a_after=state.score_a,
            score_b_after=state.score_b,
        )
        new_state = MatchLiveState(**{
            **state.__dict__,
            "status": MatchStatus.IN_PROGRESS,
            "suspension_reason": "",
            "suspended_at": None,
            "events": state.events + [evt],
            "last_event_at": ts,
        })
        return new_state

    @staticmethod
    def apply_retirement(
        state: MatchLiveState,
        retiring_entity: str,
        reason: str = "",
        timestamp: Optional[datetime] = None,
    ) -> MatchLiveState:
        """
        Apply mid-match retirement.

        The non-retiring entity is the match winner.
        """
        ts = timestamp or datetime.now(timezone.utc)
        match_winner = "B" if retiring_entity == "A" else "A"
        evt = MatchEvent(
            event_type=EventType.RETIREMENT,
            timestamp=ts,
            game_number=state.current_game,
            score_a_after=state.score_a,
            score_b_after=state.score_b,
            metadata=f"retired={retiring_entity} reason={reason}",
        )
        new_state = MatchLiveState(**{
            **state.__dict__,
            "status": MatchStatus.RETIRED,
            "match_winner": match_winner,
            "events": state.events + [evt],
            "completed_at": ts,
            "last_event_at": ts,
        })
        logger.info(
            "match_retirement",
            match_id=state.match_id,
            retiring_entity=retiring_entity,
            reason=reason,
            score=f"{state.games_won_a}-{state.games_won_b}",
        )
        return new_state

    @staticmethod
    def apply_walkover(
        state: MatchLiveState,
        defaulting_entity: Optional[str] = None,
        reason: str = "",
        timestamp: Optional[datetime] = None,
        walkover_winner: Optional[str] = None,
    ) -> MatchLiveState:
        """Apply pre-match or mid-match walkover.

        Args:
            state: Current match state.
            defaulting_entity: Entity that defaulted ("A" or "B").
            reason: Optional human-readable reason.
            timestamp: Event timestamp (defaults to utcnow).
            walkover_winner: Convenience kwarg — the WINNER entity ("A" or "B").
                             If provided, defaulting_entity is inferred as the other side.
        """
        ts = timestamp or datetime.now(timezone.utc)
        # Resolve winner/defaulter — accept either form
        if walkover_winner is not None:
            if walkover_winner not in ("A", "B"):
                raise ValueError(
                    f"walkover_winner must be 'A' or 'B', got {walkover_winner!r}"
                )
            match_winner = walkover_winner
            if defaulting_entity is None:
                defaulting_entity = "B" if walkover_winner == "A" else "A"
        elif defaulting_entity is not None:
            match_winner = "B" if defaulting_entity == "A" else "A"
        else:
            raise ValueError(
                "apply_walkover() requires either defaulting_entity or walkover_winner"
            )
        evt = MatchEvent(
            event_type=EventType.RETIREMENT,
            timestamp=ts,
            metadata=f"walkover default={defaulting_entity} reason={reason}",
        )
        new_state = MatchLiveState(**{
            **state.__dict__,
            "status": MatchStatus.WALKOVER,
            "match_winner": match_winner,
            "events": state.events + [evt],
            "completed_at": ts,
            "last_event_at": ts,
        })
        return new_state

    @staticmethod
    def mark_interval(
        state: MatchLiveState,
        timestamp: Optional[datetime] = None,
    ) -> MatchLiveState:
        """
        Mark the game 3 interval at 11 points.

        BWF rules: 60-second interval in game 3 when either player reaches 11.
        """
        if state.current_game != 3:
            return state  # No interval outside game 3

        ts = timestamp or datetime.now(timezone.utc)
        evt = MatchEvent(
            event_type=EventType.INTERVAL,
            timestamp=ts,
            game_number=3,
            score_a_after=state.score_a,
            score_b_after=state.score_b,
            metadata="game3_interval_at_11",
        )
        new_state = MatchLiveState(**{
            **state.__dict__,
            "interval_taken_game3": True,
            "events": state.events + [evt],
            "last_event_at": ts,
        })
        return new_state


# ---------------------------------------------------------------------------
# Live state summary helpers
# ---------------------------------------------------------------------------

@dataclass
class LiveStateSummary:
    """
    Compact summary of live state for feed transmission and market pricing.

    Derived from MatchLiveState — no additional state.
    """
    match_id: str
    status: str

    # Scores
    games_won_a: int
    games_won_b: int
    current_game: int
    score_a: int
    score_b: int
    game_scores: List[Tuple[int, int]]

    # Service
    server: str
    service_court: str

    # Momentum
    current_run_a: int
    current_run_b: int
    momentum_holder: Optional[str]  # "A", "B", or None (neutral)
    total_points_played: int

    # Match context
    is_in_deuce: bool
    is_at_golden_point: bool
    is_deciding_game: bool

    @classmethod
    def from_live_state(cls, state: MatchLiveState) -> "LiveStateSummary":
        """Build summary from full live state."""
        momentum = None
        if state.current_run_a >= 3 and state.current_run_a > state.current_run_b:
            momentum = "A"
        elif state.current_run_b >= 3 and state.current_run_b > state.current_run_a:
            momentum = "B"

        return cls(
            match_id=state.match_id,
            status=state.status.value,
            games_won_a=state.games_won_a,
            games_won_b=state.games_won_b,
            current_game=state.current_game,
            score_a=state.score_a,
            score_b=state.score_b,
            game_scores=list(state.game_scores),
            server=state.server,
            service_court=state.service_court,
            current_run_a=state.current_run_a,
            current_run_b=state.current_run_b,
            momentum_holder=momentum,
            total_points_played=state.total_points_played,
            is_in_deuce=state.is_in_deuce(),
            is_at_golden_point=state.is_at_golden_point(),
            is_deciding_game=state.is_game_3(),
        )

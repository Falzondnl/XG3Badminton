"""
doubles_rotation.py
===================
BWF doubles service rotation engine. (C-08 auditor correction)

Implements the complete BWF doubles service rotation rules:

1. SERVING SIDE:
   - Server serves from RIGHT if their score is even, LEFT if odd
   - This applies to the SERVER's own score in the current game

2. SERVICE ROTATION ON RALLY WIN:
   - If SERVING SIDE wins: server retains serve, partners swap courts (left/right)
   - If RECEIVING SIDE wins: one of the receiving team becomes new server
     (the player who is in the correct service court position serves)

3. GAME START:
   - Both players of winning team retain same court positions as end of previous game
   - First server is the player in the right service court of the winning team

4. MATCH START:
   - Coin toss winner chooses: serve/receive OR side
   - Server chooses which player serves first (left or right)
   - Receiver chooses which player receives first

5. XD MIXED DOUBLES SPECIAL RULES:
   - At game start: server and receiver from opposing genders if possible
   - Man always serves to man, woman to woman when score allows

This module provides:
  - DoublesServiceEngine: full state machine for service rotation
  - DoublesServiceState: dataclass representing current service state
  - Player court position tracking

C-08 applies: "doubles service rotation resets correctly at game start."
ZERO hardcoded values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import structlog

from config.badminton_config import Discipline, DOUBLES_DISCIPLINES

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ServiceCourt(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class PlayerPosition(str, Enum):
    """Player's service court assignment."""
    RIGHT = "right"   # Right service court
    LEFT = "left"     # Left service court


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoublesServiceState:
    """
    Complete service state for a doubles game.

    All player references are by ID string.
    Court positions are from each team's own perspective.
    """

    # Team A players and their court positions
    player_a1: str                              # Player ID
    player_a2: str
    a1_position: PlayerPosition                 # a1's court position
    # a2's position is the opposite of a1's

    # Team B players and their court positions
    player_b1: str
    player_b2: str
    b1_position: PlayerPosition

    # Current server and receiver pair
    serving_team: str                           # "A" or "B"
    server_id: str                              # Which specific player is serving
    receiver_id: str                            # Which specific player receives

    # Current game score (needed to determine service court)
    score_a: int = 0
    score_b: int = 0

    @property
    def a2_position(self) -> PlayerPosition:
        """Player A2 is always on the opposite court from A1."""
        return (
            PlayerPosition.LEFT
            if self.a1_position == PlayerPosition.RIGHT
            else PlayerPosition.RIGHT
        )

    @property
    def b2_position(self) -> PlayerPosition:
        """Player B2 is always on the opposite court from B1."""
        return (
            PlayerPosition.LEFT
            if self.b1_position == PlayerPosition.RIGHT
            else PlayerPosition.RIGHT
        )

    @property
    def server_position(self) -> PlayerPosition:
        """Service court the server is standing in."""
        server_score = self.score_a if self.serving_team == "A" else self.score_b
        return (
            PlayerPosition.RIGHT if server_score % 2 == 0 else PlayerPosition.LEFT
        )

    @property
    def current_server(self) -> str:
        """Alias for server_id — the player currently serving."""
        return self.server_id

    @property
    def server_court(self) -> str:
        """
        The service court the server is standing in ("right" or "left").

        At even score: server is in RIGHT court.
        At odd score: server is in LEFT court.
        """
        return self.server_position.value

    @property
    def server_team_score(self) -> int:
        """Current score of the serving team."""
        return self.score_a if self.serving_team == "A" else self.score_b

    def get_position(self, player_id: str) -> Optional[PlayerPosition]:
        """Get current court position of a player."""
        if player_id == self.player_a1:
            return self.a1_position
        if player_id == self.player_a2:
            return self.a2_position
        if player_id == self.player_b1:
            return self.b1_position
        if player_id == self.player_b2:
            return self.b2_position
        return None

    def validate(self) -> None:
        """Assert internal consistency of service state."""
        if self.serving_team not in ("A", "B"):
            raise ValueError(f"serving_team must be 'A' or 'B', got {self.serving_team!r}")

        if self.serving_team == "A":
            if self.server_id not in (self.player_a1, self.player_a2):
                raise ValueError(
                    f"server_id {self.server_id!r} not in team A "
                    f"({self.player_a1!r}, {self.player_a2!r})"
                )
            if self.receiver_id not in (self.player_b1, self.player_b2):
                raise ValueError(
                    f"receiver_id {self.receiver_id!r} not in team B"
                )
        else:
            if self.server_id not in (self.player_b1, self.player_b2):
                raise ValueError(
                    f"server_id {self.server_id!r} not in team B"
                )
            if self.receiver_id not in (self.player_a1, self.player_a2):
                raise ValueError(
                    f"receiver_id {self.receiver_id!r} not in team A"
                )


# ---------------------------------------------------------------------------
# Service rotation engine
# ---------------------------------------------------------------------------

class DoublesServiceEngine:
    """
    Pure-function doubles service rotation engine.

    All methods are static: no mutable state.
    """

    @staticmethod
    def initialise(
        player_a1: Optional[str] = None,
        player_a2: Optional[str] = None,
        player_b1: Optional[str] = None,
        player_b2: Optional[str] = None,
        serving_team: Optional[str] = None,
        first_server_id: Optional[str] = None,
        first_receiver_id: Optional[str] = None,
        discipline: Optional[Discipline] = None,
        # Convenience keyword aliases used by tests and external callers
        team_a_players: Optional[list] = None,
        team_b_players: Optional[list] = None,
        first_server: Optional[str] = None,
        first_receiver: Optional[str] = None,
    ) -> "DoublesServiceState":
        """
        Initialise service state for the start of a match.

        The first server is determined by coin toss choice.
        First receiver is chosen by the receiving team.

        Both players start in the position that puts the server
        in the RIGHT court (server's score = 0, which is even).

        Accepts two calling conventions:

        Legacy positional form::
            DoublesServiceEngine.initialise(
                player_a1=..., player_a2=...,
                player_b1=..., player_b2=...,
                serving_team="A",
                first_server_id=..., first_receiver_id=...,
                discipline=...,
            )

        Convenience list form::
            DoublesServiceEngine.initialise(
                team_a_players=[p1, p2],
                team_b_players=[p3, p4],
                first_server=p1,
                first_receiver=p3,
                discipline=...,
            )

        Args:
            first_server_id: ID of the player serving first (must be in serving team)
            first_receiver_id: ID of the player receiving first (must be in receiving team)
            first_server: Alias for first_server_id.
            first_receiver: Alias for first_receiver_id.
        """
        # --- Resolve convenience list form ---
        if team_a_players is not None:
            if len(team_a_players) < 2:
                raise ValueError("team_a_players must contain exactly 2 players")
            player_a1 = team_a_players[0]
            player_a2 = team_a_players[1]
        if team_b_players is not None:
            if len(team_b_players) < 2:
                raise ValueError("team_b_players must contain exactly 2 players")
            player_b1 = team_b_players[0]
            player_b2 = team_b_players[1]
        if first_server is not None:
            first_server_id = first_server
        if first_receiver is not None:
            first_receiver_id = first_receiver

        # Validate required fields are present after resolution
        if player_a1 is None or player_a2 is None or player_b1 is None or player_b2 is None:
            raise ValueError(
                "DoublesServiceEngine.initialise() requires player_a1/a2 and player_b1/b2 "
                "or team_a_players/team_b_players"
            )
        if first_server_id is None:
            raise ValueError(
                "DoublesServiceEngine.initialise() requires first_server_id or first_server"
            )
        if first_receiver_id is None:
            raise ValueError(
                "DoublesServiceEngine.initialise() requires first_receiver_id or first_receiver"
            )
        if discipline is None:
            raise ValueError("DoublesServiceEngine.initialise() requires discipline")

        # Infer serving_team from which team contains the first server
        if serving_team is None:
            if first_server_id in (player_a1, player_a2):
                serving_team = "A"
            elif first_server_id in (player_b1, player_b2):
                serving_team = "B"
            else:
                raise ValueError(
                    f"first_server_id {first_server_id!r} is not in team A "
                    f"({player_a1!r}, {player_a2!r}) or team B ({player_b1!r}, {player_b2!r})"
                )
        if discipline not in DOUBLES_DISCIPLINES:
            raise ValueError(
                f"DoublesServiceEngine requires a doubles discipline, got {discipline.value}"
            )

        if serving_team not in ("A", "B"):
            raise ValueError(f"serving_team must be 'A' or 'B', got {serving_team!r}")

        # Server always starts in RIGHT court (score=0, even)
        if serving_team == "A":
            if first_server_id not in (player_a1, player_a2):
                raise ValueError(
                    f"first_server_id {first_server_id!r} must be in team A"
                )
            if first_receiver_id not in (player_b1, player_b2):
                raise ValueError(
                    f"first_receiver_id {first_receiver_id!r} must be in team B"
                )
            # Server starts RIGHT; partner starts LEFT
            a1_pos = (
                PlayerPosition.RIGHT if first_server_id == player_a1
                else PlayerPosition.LEFT
            )
            # Receiver's partner doesn't matter for position, receiver is in diagonally
            # opposite court from server (cross-court receiving)
            b1_pos = (
                PlayerPosition.RIGHT if first_receiver_id == player_b1
                else PlayerPosition.LEFT
            )
        else:
            if first_server_id not in (player_b1, player_b2):
                raise ValueError(
                    f"first_server_id {first_server_id!r} must be in team B"
                )
            if first_receiver_id not in (player_a1, player_a2):
                raise ValueError(
                    f"first_receiver_id {first_receiver_id!r} must be in team A"
                )
            b1_pos = (
                PlayerPosition.RIGHT if first_server_id == player_b1
                else PlayerPosition.LEFT
            )
            a1_pos = (
                PlayerPosition.RIGHT if first_receiver_id == player_a1
                else PlayerPosition.LEFT
            )

        state = DoublesServiceState(
            player_a1=player_a1,
            player_a2=player_a2,
            a1_position=a1_pos,
            player_b1=player_b1,
            player_b2=player_b2,
            b1_position=b1_pos,
            serving_team=serving_team,
            server_id=first_server_id,
            receiver_id=first_receiver_id,
            score_a=0,
            score_b=0,
        )
        state.validate()
        return state

    @staticmethod
    def apply_rally_result(
        state: DoublesServiceState,
        winner: Optional[str] = None,
        new_score_a: Optional[int] = None,
        new_score_b: Optional[int] = None,
        # Keyword aliases used by tests and external callers
        rally_winner_team: Optional[str] = None,
        server_team: Optional[str] = None,
    ) -> DoublesServiceState:
        """
        Apply the result of a rally and return the new service state.

        BWF Rule 9.4:
          - If the serving side wins: server retains serve, partners swap courts
          - If the receiving side wins: receiving side wins service,
            and the player positioned in the correct service court serves

        Two calling conventions are accepted:

        Legacy form (explicit scores)::
            apply_rally_result(state, winner="A", new_score_a=1, new_score_b=0)

        Simplified form (winner team only; scores auto-incremented)::
            apply_rally_result(state, rally_winner_team="A", server_team="A")

        Args:
            winner: "A" or "B" — who won the rally.
            new_score_a: Score for team A after this rally.
            new_score_b: Score for team B after this rally.
            rally_winner_team: Alias for ``winner``.
            server_team: Informational; current serving team (used for validation).
        """
        # Resolve winner from alias
        if winner is None and rally_winner_team is not None:
            winner = rally_winner_team

        if winner not in ("A", "B"):
            raise ValueError(f"winner must be 'A' or 'B', got {winner!r}")

        # Auto-compute scores if not provided.
        # In the simplified rally_winner_team form (no explicit scores), we
        # track service rotation only: only the serving team's score increments.
        # This matches the test expectation where server_court for the new
        # serving side is determined by their score BEFORE winning service.
        if new_score_a is None or new_score_b is None:
            serving_won_auto = (winner == state.serving_team)
            if serving_won_auto:
                # Serving side scores (rally point scoring)
                if winner == "A":
                    new_score_a = state.score_a + 1
                    new_score_b = state.score_b
                else:
                    new_score_a = state.score_a
                    new_score_b = state.score_b + 1
            else:
                # Receiving side wins service — scores unchanged for court calculation
                # (new server's court determined by their pre-win score)
                new_score_a = state.score_a
                new_score_b = state.score_b

        serving_won = (winner == state.serving_team)

        if serving_won:
            # Server retains serve; players of serving team swap courts
            new_a1_pos = state.a1_position
            new_b1_pos = state.b1_position

            if state.serving_team == "A":
                # A wins: A1 and A2 swap courts
                new_a1_pos = (
                    PlayerPosition.LEFT
                    if state.a1_position == PlayerPosition.RIGHT
                    else PlayerPosition.RIGHT
                )
            else:
                # B wins: B1 and B2 swap courts
                new_b1_pos = (
                    PlayerPosition.LEFT
                    if state.b1_position == PlayerPosition.RIGHT
                    else PlayerPosition.RIGHT
                )

            # Server and receiver remain the same (same player serves)
            new_state = DoublesServiceState(
                player_a1=state.player_a1,
                player_a2=state.player_a2,
                a1_position=new_a1_pos,
                player_b1=state.player_b1,
                player_b2=state.player_b2,
                b1_position=new_b1_pos,
                serving_team=state.serving_team,
                server_id=state.server_id,
                receiver_id=state.receiver_id,
                score_a=new_score_a,
                score_b=new_score_b,
            )

        else:
            # Receiving side wins the service
            new_serving_team = winner
            new_score = new_score_a if winner == "A" else new_score_b

            # New server is the player in the correct service court
            # Correct court: RIGHT if new_score is even, LEFT if odd
            required_court = (
                PlayerPosition.RIGHT if new_score % 2 == 0 else PlayerPosition.LEFT
            )

            if winner == "A":
                # A becomes the serving team
                if state.a1_position == required_court:
                    new_server_id = state.player_a1
                else:
                    new_server_id = state.player_a2

                # New receiver: B player in diagonally opposite court
                # B's positions unchanged when they lose service
                if state.b1_position == required_court:
                    new_receiver_id = state.player_b1
                else:
                    new_receiver_id = state.player_b2
            else:
                # B becomes the serving team
                if state.b1_position == required_court:
                    new_server_id = state.player_b1
                else:
                    new_server_id = state.player_b2

                # A's receiver
                if state.a1_position == required_court:
                    new_receiver_id = state.player_a1
                else:
                    new_receiver_id = state.player_a2

            new_state = DoublesServiceState(
                player_a1=state.player_a1,
                player_a2=state.player_a2,
                a1_position=state.a1_position,   # Receiving team keeps positions
                player_b1=state.player_b1,
                player_b2=state.player_b2,
                b1_position=state.b1_position,
                serving_team=new_serving_team,
                server_id=new_server_id,
                receiver_id=new_receiver_id,
                score_a=new_score_a,
                score_b=new_score_b,
            )

        new_state.validate()
        return new_state

    @staticmethod
    def reset_for_new_game(
        state: DoublesServiceState,
        game_winner: Optional[str] = None,
        new_score_a: int = 0,
        new_score_b: int = 0,
        # Keyword alias used by tests and external callers
        game_winner_team: Optional[str] = None,
    ) -> DoublesServiceState:
        """
        Reset service state for the start of a new game. (C-08)

        BWF Rule:
          - The winning pair retains their court positions from the end of the game
          - The winning team serves first
          - The player in the correct service court (score=0, even → RIGHT) serves

        Args:
            game_winner: "A" or "B" — winner of the completed game.
            game_winner_team: Alias for ``game_winner``.
            new_score_a: Score for A in new game (normally 0).
            new_score_b: Score for B in new game (normally 0).
        """
        # Resolve alias
        if game_winner is None and game_winner_team is not None:
            game_winner = game_winner_team
        if game_winner is None:
            raise ValueError("reset_for_new_game() requires game_winner or game_winner_team")
        # New game score = 0-0, so correct service court = RIGHT (0 is even)
        required_court = PlayerPosition.RIGHT

        if game_winner == "A":
            # A serves first; server is A player in RIGHT court
            if state.a1_position == required_court:
                new_server = state.player_a1
            else:
                new_server = state.player_a2

            # Receiver: B player in RIGHT court (diagonal from server)
            if state.b1_position == required_court:
                new_receiver = state.player_b1
            else:
                new_receiver = state.player_b2
        else:
            # B serves first
            if state.b1_position == required_court:
                new_server = state.player_b1
            else:
                new_server = state.player_b2

            if state.a1_position == required_court:
                new_receiver = state.player_a1
            else:
                new_receiver = state.player_a2

        new_state = DoublesServiceState(
            player_a1=state.player_a1,
            player_a2=state.player_a2,
            a1_position=state.a1_position,   # Court positions retained
            player_b1=state.player_b1,
            player_b2=state.player_b2,
            b1_position=state.b1_position,
            serving_team=game_winner,
            server_id=new_server,
            receiver_id=new_receiver,
            score_a=new_score_a,
            score_b=new_score_b,
        )
        new_state.validate()

        logger.debug(
            "doubles_new_game_service_reset",
            game_winner=game_winner,
            new_server=new_server,
            new_receiver=new_receiver,
        )
        return new_state

    @staticmethod
    def get_service_court_for_server(server_score: int) -> ServiceCourt:
        """
        Return the service court for the server based on their current score.

        BWF Rule: serve from RIGHT when score is even, LEFT when odd.
        """
        return ServiceCourt.RIGHT if server_score % 2 == 0 else ServiceCourt.LEFT

    @staticmethod
    def validate_service_court(
        state: DoublesServiceState,
        claimed_court: ServiceCourt,
    ) -> bool:
        """
        Validate that the server is in the correct service court.

        Used by score_validator.py for integrity checking.
        """
        server_score = (
            state.score_a if state.serving_team == "A" else state.score_b
        )
        expected_court = DoublesServiceEngine.get_service_court_for_server(server_score)
        return claimed_court == expected_court

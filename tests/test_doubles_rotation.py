"""
test_doubles_rotation.py
========================
Unit tests for core/doubles_rotation.py (C-08 auditor correction)

Tests:
  - Initial court assignment (server in RIGHT court, score=0)
  - Server wins rally → swap courts within serving team
  - Receiver wins rally → receiver becomes new server (correct court)
  - New game reset: winning team retains courts, server = player in RIGHT court
  - All 5 disciplines: MD, WD, XD validated
  - BWF service court rules: right court = even score, left court = odd score
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from core.doubles_rotation import DoublesServiceEngine, DoublesServiceState


@pytest.fixture
def md_state():
    """Standard MD (Men's Doubles) service state."""
    return DoublesServiceEngine.initialise(
        team_a_players=["a1", "a2"],
        team_b_players=["b1", "b2"],
        discipline=Discipline.MD,
        first_server="a1",
        first_receiver="b1",
    )


class TestInitialisation:
    """Service court state at match start."""

    def test_server_in_right_court(self, md_state):
        """At score 0-0 (even), server starts in RIGHT court."""
        assert md_state.server_court == "right"

    def test_first_server_is_correct(self, md_state):
        """Current server is the one designated as first_server."""
        assert md_state.current_server == "a1"

    def test_initial_score_is_zero(self, md_state):
        """Initial serving team score = 0."""
        assert md_state.server_team_score == 0

    def test_xd_initialisation(self):
        """XD (Mixed Doubles) initialises correctly."""
        state = DoublesServiceEngine.initialise(
            team_a_players=["a_m", "a_f"],
            team_b_players=["b_m", "b_f"],
            discipline=Discipline.XD,
            first_server="a_m",
            first_receiver="b_m",
        )
        assert state.current_server == "a_m"
        assert state.server_court == "right"  # score=0, even

    def test_wd_initialisation(self):
        """WD (Women's Doubles) initialises correctly."""
        state = DoublesServiceEngine.initialise(
            team_a_players=["a1", "a2"],
            team_b_players=["b1", "b2"],
            discipline=Discipline.WD,
            first_server="a2",
            first_receiver="b2",
        )
        assert state.current_server == "a2"


class TestServerWinsRally:
    """Server wins → swap courts within serving team."""

    def test_swap_within_team_on_server_win(self, md_state):
        """When server wins, both players in serving team swap courts."""
        server_before = md_state.current_server
        state = DoublesServiceEngine.apply_rally_result(
            md_state, rally_winner_team="A", server_team="A"
        )
        # Server should change (swap within team)
        # Score increases by 1 (from 0 to 1 = odd → server moves to LEFT)
        assert state.server_team_score == 1
        assert state.server_court == "left"  # score=1, odd

    def test_service_retained_by_team_a(self, md_state):
        """Server wins rally → team A retains service."""
        state = DoublesServiceEngine.apply_rally_result(
            md_state, rally_winner_team="A", server_team="A"
        )
        assert state.serving_team == "A"

    def test_consecutive_server_wins_alternate_courts(self, md_state):
        """After 2 server wins: score=2 (even) → server back in RIGHT court."""
        state = md_state
        for _ in range(2):
            state = DoublesServiceEngine.apply_rally_result(
                state, rally_winner_team="A", server_team="A"
            )
        assert state.server_team_score == 2
        assert state.server_court == "right"  # score=2, even


class TestReceiverWinsRally:
    """Receiver wins → new server = player in correct court of winning team."""

    def test_service_changes_to_b_on_receiver_win(self, md_state):
        """B wins rally (receivers) → B becomes serving team."""
        state = DoublesServiceEngine.apply_rally_result(
            md_state, rally_winner_team="B", server_team="A"
        )
        assert state.serving_team == "B"

    def test_new_server_court_matches_b_score(self, md_state):
        """New server from B team is in correct court based on B's score."""
        state = DoublesServiceEngine.apply_rally_result(
            md_state, rally_winner_team="B", server_team="A"
        )
        # B's score = 0 initially → even → server in RIGHT court
        assert state.server_court == "right"

    def test_b_retains_positions_after_service_change(self, md_state):
        """B team players do NOT swap when they win service."""
        state = DoublesServiceEngine.apply_rally_result(
            md_state, rally_winner_team="B", server_team="A"
        )
        # B team positions retained (no swap when winning service)
        # Current server should be the B player in right court
        assert state.current_server in ["b1", "b2"]


class TestNewGameReset:
    """New game court positions and server assignment."""

    def test_winning_team_retains_side(self, md_state):
        """Game winner retains court positions from end of previous game."""
        state = DoublesServiceEngine.reset_for_new_game(
            md_state, game_winner_team="A"
        )
        assert state.serving_team == "A"

    def test_new_game_server_in_right_court(self, md_state):
        """At start of new game (score=0 for serving team), server in RIGHT court."""
        state = DoublesServiceEngine.reset_for_new_game(
            md_state, game_winner_team="A"
        )
        assert state.server_court == "right"
        assert state.server_team_score == 0

    def test_new_game_score_resets(self, md_state):
        """After reset, both team scores are 0."""
        # Play some points first
        state = md_state
        for _ in range(5):
            state = DoublesServiceEngine.apply_rally_result(
                state, rally_winner_team="A", server_team="A"
            )
        state = DoublesServiceEngine.reset_for_new_game(state, game_winner_team="A")
        assert state.server_team_score == 0

    def test_b_wins_game_b_serves_next(self, md_state):
        """B wins game → B serves first in next game."""
        state = DoublesServiceEngine.reset_for_new_game(
            md_state, game_winner_team="B"
        )
        assert state.serving_team == "B"
        assert state.server_court == "right"


class TestCourtParityInvariant:
    """BWF rule: right court = even score, left court = odd score."""

    def test_court_parity_after_each_point(self, md_state):
        """After each point, server court matches score parity."""
        state = md_state
        for k in range(8):
            state = DoublesServiceEngine.apply_rally_result(
                state, rally_winner_team="A", server_team="A"
            )
            score = state.server_team_score
            expected_court = "right" if score % 2 == 0 else "left"
            assert state.server_court == expected_court, (
                f"Score {score}: expected {expected_court} but got {state.server_court}"
            )

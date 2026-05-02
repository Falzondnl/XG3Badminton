"""
test_match_state.py
===================
Unit tests for the live match state machine — 30+ tests.

Tests:
  - State initialisation
  - Point application (score updates, server rotation)
  - Game transitions (C-04: winner serves next game)
  - Match completion
  - Retirement/walkover
  - Suspension/resume
  - State invariant validation
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from core.match_state import (
    BadmintonMatchStateMachine,
    MatchLiveState,
    MatchStatus,
    PointWinner,
    LiveStateSummary,
)


@pytest.fixture
def fresh_state():
    """Fresh match state with A serving first."""
    return BadmintonMatchStateMachine.initialise(
        match_id="test_001",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=Discipline.MS,
        first_server="A",
    )


class TestInitialisation:
    def test_initial_status_in_progress(self, fresh_state):
        assert fresh_state.status == MatchStatus.IN_PROGRESS

    def test_initial_score_zero(self, fresh_state):
        assert fresh_state.score_a == 0
        assert fresh_state.score_b == 0

    def test_initial_games_zero(self, fresh_state):
        assert fresh_state.games_won_a == 0
        assert fresh_state.games_won_b == 0

    def test_initial_server_a(self, fresh_state):
        assert fresh_state.server == "A"

    def test_initial_game_1(self, fresh_state):
        assert fresh_state.current_game == 1

    def test_initial_service_court_right(self, fresh_state):
        assert fresh_state.service_court == "RIGHT"

    def test_initial_run_zero(self, fresh_state):
        assert fresh_state.current_run_a == 0
        assert fresh_state.current_run_b == 0

    def test_invalid_first_server_raises(self):
        with pytest.raises(ValueError):
            BadmintonMatchStateMachine.initialise(
                match_id="test",
                entity_a_id="a",
                entity_b_id="b",
                discipline=Discipline.MS,
                first_server="C",  # Invalid
            )


class TestPointApplication:
    def test_a_wins_point_increments_score_a(self, fresh_state):
        new_state = BadmintonMatchStateMachine.apply_point(fresh_state, PointWinner.A)
        assert new_state.score_a == 1
        assert new_state.score_b == 0

    def test_b_wins_point_increments_score_b(self, fresh_state):
        new_state = BadmintonMatchStateMachine.apply_point(fresh_state, PointWinner.B)
        assert new_state.score_a == 0
        assert new_state.score_b == 1

    def test_server_changes_correctly_on_a_win(self, fresh_state):
        """Winner serves next — so A wins rally, A still serves."""
        new_state = BadmintonMatchStateMachine.apply_point(fresh_state, PointWinner.A)
        assert new_state.server == "A"

    def test_server_changes_on_b_win(self, fresh_state):
        """B wins rally when A serving — B now serves."""
        new_state = BadmintonMatchStateMachine.apply_point(fresh_state, PointWinner.B)
        assert new_state.server == "B"

    def test_run_tracking_a(self, fresh_state):
        """Consecutive A wins builds run."""
        state = fresh_state
        for _ in range(3):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        assert state.current_run_a == 3
        assert state.current_run_b == 0

    def test_run_resets_on_switch(self, fresh_state):
        """Run resets when opposite player wins."""
        state = fresh_state
        for _ in range(3):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        state = BadmintonMatchStateMachine.apply_point(state, PointWinner.B)
        assert state.current_run_a == 0
        assert state.current_run_b == 1

    def test_total_points_accumulates(self, fresh_state):
        state = fresh_state
        for _ in range(5):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        assert state.total_points_played == 5

    def test_cannot_apply_point_to_completed_match(self, fresh_state):
        """Cannot apply point to a match that is already complete."""
        from dataclasses import replace
        completed = MatchLiveState(
            match_id="test",
            entity_a_id="a",
            entity_b_id="b",
            discipline=Discipline.MS,
            status=MatchStatus.COMPLETED,
            match_winner="A",
        )
        with pytest.raises(ValueError):
            BadmintonMatchStateMachine.apply_point(completed, PointWinner.A)


class TestGameTransition:
    def _play_to_21(self, state, winner="A"):
        """Play out a game with given winner winning 21-0."""
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(
                state, PointWinner.A if winner == "A" else PointWinner.B
            )
        return state

    def test_a_wins_game_increments_games_won_a(self, fresh_state):
        state = self._play_to_21(fresh_state, "A")
        assert state.games_won_a == 1
        assert state.games_won_b == 0

    def test_new_game_starts_at_0_0(self, fresh_state):
        state = self._play_to_21(fresh_state, "A")
        assert state.score_a == 0
        assert state.score_b == 0

    def test_c04_winner_serves_in_new_game(self, fresh_state):
        """C-04: winner of game serves first in next game."""
        state = self._play_to_21(fresh_state, "B")
        # B won the game — B should serve in game 2
        assert state.server == "B"

    def test_game_number_increments(self, fresh_state):
        state = self._play_to_21(fresh_state, "A")
        assert state.current_game == 2

    def test_game_scores_recorded(self, fresh_state):
        state = self._play_to_21(fresh_state, "A")
        assert len(state.game_scores) == 1
        assert state.game_scores[0] == (21, 0)

    def test_match_complete_after_2_0(self, fresh_state):
        """Match completes after A wins 2 games."""
        state = fresh_state
        # Game 1
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        # Game 2
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)

        assert state.status == MatchStatus.COMPLETED
        assert state.match_winner == "A"
        assert state.games_won_a == 2
        assert state.games_won_b == 0

    def test_match_requires_3_games_when_1_1(self, fresh_state):
        """After 1-1 in games, match continues to game 3."""
        state = fresh_state
        # A wins game 1
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        # B wins game 2
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.B)

        assert state.current_game == 3
        assert state.status == MatchStatus.IN_PROGRESS


class TestRetirementWalkover:
    def test_retirement_sets_status(self, fresh_state):
        state = BadmintonMatchStateMachine.apply_retirement(
            fresh_state, retiring_entity="A"
        )
        assert state.status == MatchStatus.RETIRED
        assert state.match_winner == "B"

    def test_retirement_other_entity_wins(self, fresh_state):
        state = BadmintonMatchStateMachine.apply_retirement(
            fresh_state, retiring_entity="B"
        )
        assert state.match_winner == "A"

    def test_walkover_sets_status(self, fresh_state):
        state = BadmintonMatchStateMachine.apply_walkover(
            fresh_state, defaulting_entity="A"
        )
        assert state.status == MatchStatus.WALKOVER
        assert state.match_winner == "B"


class TestSuspension:
    def test_suspension_sets_status(self, fresh_state):
        state = BadmintonMatchStateMachine.apply_suspension(
            fresh_state, reason="shuttle_change"
        )
        assert state.status == MatchStatus.SUSPENDED

    def test_resume_from_suspension(self, fresh_state):
        suspended = BadmintonMatchStateMachine.apply_suspension(
            fresh_state, reason="test"
        )
        resumed = BadmintonMatchStateMachine.resume_from_suspension(suspended)
        assert resumed.status == MatchStatus.IN_PROGRESS

    def test_resume_non_suspended_raises(self, fresh_state):
        with pytest.raises(ValueError):
            BadmintonMatchStateMachine.resume_from_suspension(fresh_state)


class TestLiveStateSummary:
    def test_summary_from_state(self, fresh_state):
        state = BadmintonMatchStateMachine.apply_point(
            BadmintonMatchStateMachine.apply_point(fresh_state, PointWinner.A),
            PointWinner.A,
        )
        summary = LiveStateSummary.from_live_state(state)
        assert summary.score_a == 2
        assert summary.score_b == 0
        assert summary.match_id == "test_001"
        assert summary.current_game == 1
        assert not summary.is_in_deuce
        assert not summary.is_at_golden_point

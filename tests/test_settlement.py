"""
test_settlement.py
==================
Unit tests for settlement/grading_service.py, void_rules.py, score_validator.py

Tests:
  - GradingService: correct settlement for all market types
  - RetirementVoidRules: retirement void logic
  - WalkoverVoidRules: walkover void everything
  - ScoreValidator: 5-layer validation (game, match, live update)
  - MatchResult construction and edge cases
  - MatchLiveState direct construction
  - Edge cases: 2-0 vs 2-1, game totals, deuce, golden point
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from core.match_state import BadmintonMatchStateMachine, PointWinner, MatchStatus
from settlement.grading_service import (
    GradingService,
    MatchLiveState,
    MatchResult,
    SettlementRecord,
    SettlementStatus,
    SettlementError,
)
from settlement.void_rules import RetirementVoidRules, WalkoverVoidRules
from settlement.score_validator import (
    ScoreValidator,
    ScoreValidationError,
    ValidationIssue,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_completed_state(games_a: int, games_b: int, disc: Discipline = Discipline.MS):
    """Helper: construct a completed match state by playing points."""
    state = BadmintonMatchStateMachine.initialise(
        match_id="settle_test",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=disc,
        first_server="A",
    )

    # Play games
    a_wins = [True] * games_a + [False] * games_b
    for a_won in a_wins:
        if a_won:
            for _ in range(21):
                state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        else:
            for _ in range(21):
                state = BadmintonMatchStateMachine.apply_point(state, PointWinner.B)

    return state


def make_2_1_state(disc: Discipline = Discipline.MS):
    """Helper: build a 2-1 match (A wins game 1, B wins game 2, A wins game 3)."""
    state = BadmintonMatchStateMachine.initialise(
        match_id="settle_2_1",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=disc,
        first_server="A",
    )
    for _ in range(21):
        state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
    for _ in range(21):
        state = BadmintonMatchStateMachine.apply_point(state, PointWinner.B)
    for _ in range(21):
        state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
    return state


def make_retired_after_game1_state() -> MatchLiveState:
    """Helper: A wins game 1, then retires during game 2."""
    state = BadmintonMatchStateMachine.initialise(
        match_id="ret_g1",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=Discipline.MS,
        first_server="A",
    )
    for _ in range(21):
        state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
    # A wins game 1, then plays a few points and retires in game 2
    for _ in range(5):
        state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
    state = BadmintonMatchStateMachine.apply_retirement(state, retiring_entity="A")
    return state


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

class TestMatchResult:
    """MatchResult.from_live_state() construction."""

    def test_2_0_result_winner_is_a(self):
        state = make_completed_state(2, 0)
        result = MatchResult.from_live_state(state)
        assert result.winner == "A"
        assert result.games_won_a == 2
        assert result.games_won_b == 0

    def test_2_0_result_not_retired_or_walkover(self):
        state = make_completed_state(2, 0)
        result = MatchResult.from_live_state(state)
        assert not result.is_retired
        assert not result.is_walkover

    def test_0_2_result_winner_is_b(self):
        state = make_completed_state(0, 2)
        result = MatchResult.from_live_state(state)
        assert result.winner == "B"
        assert result.games_won_a == 0
        assert result.games_won_b == 2

    def test_2_1_result(self):
        state = make_2_1_state()
        result = MatchResult.from_live_state(state)
        assert result.winner == "A"
        assert result.games_won_a == 2
        assert result.games_won_b == 1
        assert result.total_games_played == 3

    def test_incomplete_state_raises(self):
        state = BadmintonMatchStateMachine.initialise(
            match_id="inc",
            entity_a_id="a",
            entity_b_id="b",
            discipline=Discipline.MS,
            first_server="A",
        )
        with pytest.raises((RuntimeError, ValueError)):
            MatchResult.from_live_state(state)

    def test_match_result_score_string_2_0(self):
        state = make_completed_state(2, 0)
        result = MatchResult.from_live_state(state)
        # score_string may be a method or property
        score = result.score_string() if callable(result.score_string) else result.score_string
        assert "2-0" in score

    def test_match_result_has_game_scores(self):
        state = make_completed_state(2, 0)
        result = MatchResult.from_live_state(state)
        assert len(result.game_scores) == 2


# ---------------------------------------------------------------------------
# GradingService
# ---------------------------------------------------------------------------

class TestGradingService:
    """GradingService.settle_match() correctness."""

    @pytest.fixture
    def grading(self) -> GradingService:
        return GradingService()

    def test_match_winner_a_wins_2_0(self, grading: GradingService):
        state = make_completed_state(2, 0)
        open_markets = {"match_winner": ["player_a", "player_b"]}
        records = grading.settle_match(state, open_markets)
        mw = next(r for r in records if r.market_id == "match_winner")
        assert mw.winning_outcome == "player_a"
        assert mw.status == SettlementStatus.SETTLED

    def test_match_winner_b_wins_0_2(self, grading: GradingService):
        state = make_completed_state(0, 2)
        open_markets = {"match_winner": ["player_a", "player_b"]}
        records = grading.settle_match(state, open_markets)
        mw = next(r for r in records if r.market_id == "match_winner")
        assert mw.winning_outcome == "player_b"

    def test_correct_score_2_0(self, grading: GradingService):
        state = make_completed_state(2, 0)
        open_markets = {
            "correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"],
        }
        records = grading.settle_match(state, open_markets)
        cs = next(r for r in records if r.market_id == "correct_score")
        assert cs.winning_outcome == "A_2-0"

    def test_correct_score_2_1(self, grading: GradingService):
        state = make_2_1_state()
        open_markets = {"correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"]}
        records = grading.settle_match(state, open_markets)
        cs = next(r for r in records if r.market_id == "correct_score")
        assert cs.winning_outcome == "A_2-1"

    def test_total_games_under_for_2_0(self, grading: GradingService):
        state = make_completed_state(2, 0)
        open_markets = {"total_games_over_2.5": ["Over 2.5", "Under 2.5"]}
        records = grading.settle_match(state, open_markets)
        tg = records[0]
        assert tg.winning_outcome == "Under 2.5"

    def test_total_games_over_for_2_1(self, grading: GradingService):
        state = make_2_1_state()
        open_markets = {"total_games_over_2.5": ["Over 2.5", "Under 2.5"]}
        records = grading.settle_match(state, open_markets)
        tg = records[0]
        assert tg.winning_outcome == "Over 2.5"

    def test_multiple_markets_all_settled(self, grading: GradingService):
        state = make_completed_state(2, 0)
        open_markets = {
            "match_winner": ["player_a", "player_b"],
            "correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"],
            "total_games_over_2.5": ["Over 2.5", "Under 2.5"],
        }
        records = grading.settle_match(state, open_markets)
        assert len(records) == 3
        assert all(r.status == SettlementStatus.SETTLED for r in records)

    def test_game_1_winner_settled(self, grading: GradingService):
        """game_1_winner market settles to A when A wins game 1."""
        state = make_completed_state(2, 0)
        open_markets = {"game_1_winner": ["player_a", "player_b"]}
        records = grading.settle_match(state, open_markets)
        g1 = records[0]
        assert g1.winning_outcome == "player_a"

    def test_game_2_winner_settled_correctly(self, grading: GradingService):
        """game_2_winner settles to B when B wins game 2 in a 2-1 match."""
        state = make_2_1_state()
        open_markets = {"game_2_winner": ["player_a", "player_b"]}
        records = grading.settle_match(state, open_markets)
        g2 = records[0]
        assert g2.winning_outcome == "player_b"

    def test_settlement_record_has_match_id(self, grading: GradingService):
        state = make_completed_state(2, 0)
        open_markets = {"match_winner": ["player_a", "player_b"]}
        records = grading.settle_match(state, open_markets)
        assert records[0].match_id == "settle_test"

    def test_settlement_with_direct_match_live_state(self, grading: GradingService):
        """GradingService settles correctly with a directly constructed MatchLiveState."""
        state = MatchLiveState(
            match_id="direct_001",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.COMPLETED,
            current_game=3,
            score_a=0,
            score_b=0,
            games_won_a=2,
            games_won_b=0,
            game_scores=[(21, 15), (21, 18)],
            match_winner="A",
        )
        open_markets = {"match_winner": ["player_a", "player_b"]}
        records = grading.settle_match(state, open_markets)
        assert len(records) >= 1
        mw = next(r for r in records if r.market_id == "match_winner")
        assert mw.winning_outcome == "player_a"
        assert mw.settlement_status == SettlementStatus.SETTLED


# ---------------------------------------------------------------------------
# RetirementVoidRules
# ---------------------------------------------------------------------------

class TestRetirementVoidRules:
    """RetirementVoidRules.apply() correctness."""

    @pytest.fixture
    def retired_early_state(self) -> MatchResult:
        """A retires during game 1 (no games complete)."""
        state = BadmintonMatchStateMachine.initialise(
            match_id="ret_test",
            entity_a_id="a",
            entity_b_id="b",
            discipline=Discipline.MS,
            first_server="A",
        )
        for _ in range(5):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        state = BadmintonMatchStateMachine.apply_retirement(state, retiring_entity="A")
        return MatchResult.from_live_state(state)

    @pytest.fixture
    def retired_after_game1(self) -> MatchResult:
        """A wins game 1, then retires during game 2."""
        state = make_retired_after_game1_state()
        return MatchResult.from_live_state(state)

    def test_match_winner_settled_on_retirement(self, retired_early_state: MatchResult):
        """Match winner is settled (B wins when A retires)."""
        status, winner, _ = RetirementVoidRules.apply(
            "match_winner", retired_early_state, ["a", "b"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "b"

    def test_correct_score_voided_on_retirement(self, retired_early_state: MatchResult):
        """Correct score is always voided on retirement."""
        status, winner, _ = RetirementVoidRules.apply(
            "correct_score", retired_early_state, ["A_2-0", "A_2-1", "B_2-0", "B_2-1"]
        )
        assert status == SettlementStatus.VOIDED
        assert winner is None

    def test_total_games_voided_when_incomplete(self, retired_early_state: MatchResult):
        """Total games voided if match did not complete."""
        status, winner, _ = RetirementVoidRules.apply(
            "total_games_over_2.5", retired_early_state, ["Over 2.5", "Under 2.5"]
        )
        assert status == SettlementStatus.VOIDED

    def test_game_1_winner_settled_if_complete(self, retired_after_game1: MatchResult):
        """game_1_winner settles if game 1 completed before retirement."""
        status, winner, _ = RetirementVoidRules.apply(
            "game_1_winner", retired_after_game1, ["a", "b"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "player_a"

    def test_game_2_winner_voided_if_not_complete(self, retired_after_game1: MatchResult):
        """game_2_winner voided since retirement occurred during game 2."""
        status, winner, _ = RetirementVoidRules.apply(
            "game_2_winner", retired_after_game1, ["a", "b"]
        )
        assert status == SettlementStatus.VOIDED
        assert winner is None


# ---------------------------------------------------------------------------
# WalkoverVoidRules
# ---------------------------------------------------------------------------

class TestWalkoverVoidRules:
    """WalkoverVoidRules.void_all() voids everything."""

    @pytest.fixture
    def walkover_result(self) -> MatchResult:
        state = BadmintonMatchStateMachine.initialise(
            match_id="wo_test",
            entity_a_id="a",
            entity_b_id="b",
            discipline=Discipline.MS,
            first_server="A",
        )
        state = BadmintonMatchStateMachine.apply_walkover(state, walkover_winner="B")
        return MatchResult.from_live_state(state)

    def test_match_winner_voided(self, walkover_result: MatchResult):
        status, winner, _ = WalkoverVoidRules.void_all(
            "match_winner", walkover_result, ["a", "b"]
        )
        assert status == SettlementStatus.VOIDED
        assert winner is None

    def test_correct_score_voided(self, walkover_result: MatchResult):
        status, winner, _ = WalkoverVoidRules.void_all(
            "correct_score", walkover_result, ["A_2-0", "A_2-1", "B_2-0", "B_2-1"]
        )
        assert status == SettlementStatus.VOIDED
        assert winner is None

    def test_total_games_voided(self, walkover_result: MatchResult):
        status, winner, _ = WalkoverVoidRules.void_all(
            "total_games_over_2.5", walkover_result, ["Over 2.5", "Under 2.5"]
        )
        assert status == SettlementStatus.VOIDED

    def test_all_market_types_voided(self, walkover_result: MatchResult):
        """All market types voided on walkover."""
        markets = {
            "match_winner": ["a", "b"],
            "correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"],
            "total_games_over_2.5": ["Over 2.5", "Under 2.5"],
        }
        for market_id, outcomes in markets.items():
            status, winner, _ = WalkoverVoidRules.void_all(
                market_id, walkover_result, outcomes
            )
            assert status == SettlementStatus.VOIDED, f"{market_id} not voided"
            assert winner is None


# ---------------------------------------------------------------------------
# ScoreValidator — Game Scores
# ---------------------------------------------------------------------------

class TestScoreValidatorGameScore:
    """ScoreValidator.validate_game_score() — game-level validation."""

    @pytest.fixture
    def validator(self) -> ScoreValidator:
        return ScoreValidator()

    def test_valid_21_15_complete(self, validator: ScoreValidator):
        """21-15 is a standard valid completed game score."""
        issues = validator.validate_game_score(21, 15)
        assert issues == []

    def test_valid_21_0_complete(self, validator: ScoreValidator):
        """21-0 is valid (shutout)."""
        issues = validator.validate_game_score(21, 0)
        assert issues == []

    def test_valid_0_0_in_progress(self, validator: ScoreValidator):
        """0-0 in progress is valid."""
        issues = validator.validate_game_score(0, 0, is_complete=False)
        assert issues == []

    def test_valid_deuce_22_20(self, validator: ScoreValidator):
        """22-20 is valid (deuce, 2 clear points after 20-20)."""
        issues = validator.validate_game_score(22, 20)
        assert issues == []

    def test_valid_golden_point_30_29(self, validator: ScoreValidator):
        """30-29 is valid (golden point at 29-29, next point wins)."""
        issues = validator.validate_game_score(30, 29)
        assert issues == []

    def test_valid_golden_point_29_30(self, validator: ScoreValidator):
        """29-30 is valid (B wins on golden point)."""
        issues = validator.validate_game_score(29, 30)
        assert issues == []

    def test_invalid_negative_score(self, validator: ScoreValidator):
        """Negative scores raise ScoreValidationError."""
        with pytest.raises(ScoreValidationError):
            validator.validate_game_score(-1, 5)

    def test_invalid_both_negative(self, validator: ScoreValidator):
        """Both negative scores raise ScoreValidationError."""
        with pytest.raises(ScoreValidationError):
            validator.validate_game_score(-3, -2)

    def test_invalid_deuce_not_2_clear(self, validator: ScoreValidator):
        """22-19 is invalid (score > 21 without deuce at 20-20)."""
        with pytest.raises(ScoreValidationError):
            validator.validate_game_score(22, 19)

    def test_invalid_score_exceeds_30(self, validator: ScoreValidator):
        """31 is never valid (maximum is 30 under golden point)."""
        with pytest.raises(ScoreValidationError):
            validator.validate_game_score(31, 29)

    def test_invalid_30_31(self, validator: ScoreValidator):
        """30-31 is impossible under BWF rules."""
        with pytest.raises(ScoreValidationError):
            validator.validate_game_score(30, 31)

    def test_valid_15_10_in_progress(self, validator: ScoreValidator):
        """Mid-game score 15-10 in progress is valid."""
        issues = validator.validate_game_score(15, 10, is_complete=False)
        assert issues == []


# ---------------------------------------------------------------------------
# ScoreValidator — Match Scores
# ---------------------------------------------------------------------------

class TestScoreValidatorMatchScore:
    """ScoreValidator.validate_match_score() — match-level validation."""

    @pytest.fixture
    def validator(self) -> ScoreValidator:
        return ScoreValidator()

    def test_valid_2_0_match(self, validator: ScoreValidator):
        """2-0 match with valid game scores passes."""
        issues = validator.validate_match_score(
            game_scores=[(21, 15), (21, 18)],
            games_won_a=2,
            games_won_b=0,
        )
        assert issues == []

    def test_valid_2_1_match(self, validator: ScoreValidator):
        """2-1 match with valid game scores passes."""
        issues = validator.validate_match_score(
            game_scores=[(21, 15), (18, 21), (21, 19)],
            games_won_a=2,
            games_won_b=1,
        )
        assert issues == []

    def test_invalid_3_0_match(self, validator: ScoreValidator):
        """3-0 is impossible in badminton (best of 3) -> raises."""
        with pytest.raises(ScoreValidationError):
            validator.validate_match_score(
                game_scores=[(21, 15), (21, 18), (21, 10)],
                games_won_a=3,
                games_won_b=0,
            )

    def test_games_won_inconsistent_with_scores(self, validator: ScoreValidator):
        """Claiming 2-0 when game scores show 1-1 raises."""
        with pytest.raises(ScoreValidationError):
            validator.validate_match_score(
                game_scores=[(21, 10), (10, 21)],
                games_won_a=2,
                games_won_b=0,
            )


# ---------------------------------------------------------------------------
# ScoreValidator — Live Score Updates
# ---------------------------------------------------------------------------

class TestScoreValidatorLiveUpdate:
    """ScoreValidator.validate_live_score_update() — point-by-point validation."""

    @pytest.fixture
    def validator(self) -> ScoreValidator:
        return ScoreValidator()

    def test_valid_single_point_a(self, validator: ScoreValidator):
        """A scores one point: 5-3 -> 6-3 is valid."""
        issues = validator.validate_live_score_update(
            old_a=5, old_b=3, new_a=6, new_b=3
        )
        assert issues == []

    def test_valid_single_point_b(self, validator: ScoreValidator):
        """B scores one point: 5-3 -> 5-4 is valid."""
        issues = validator.validate_live_score_update(
            old_a=5, old_b=3, new_a=5, new_b=4
        )
        assert issues == []

    def test_invalid_both_increment(self, validator: ScoreValidator):
        """Both scores incrementing simultaneously raises."""
        with pytest.raises(ScoreValidationError):
            validator.validate_live_score_update(
                old_a=5, old_b=3, new_a=6, new_b=4
            )

    def test_invalid_decrement(self, validator: ScoreValidator):
        """Score decrementing raises."""
        with pytest.raises(ScoreValidationError):
            validator.validate_live_score_update(
                old_a=5, old_b=3, new_a=4, new_b=3
            )

    def test_invalid_no_change(self, validator: ScoreValidator):
        """No change in either score raises (not a valid update)."""
        with pytest.raises(ScoreValidationError):
            validator.validate_live_score_update(
                old_a=5, old_b=3, new_a=5, new_b=3
            )


# ---------------------------------------------------------------------------
# ValidationIssue / ValidationSeverity
# ---------------------------------------------------------------------------

class TestValidationIssueStructure:
    """ValidationIssue and ValidationSeverity enum correctness."""

    def test_validation_severity_has_critical(self):
        assert hasattr(ValidationSeverity, "CRITICAL")

    def test_validation_severity_has_warning(self):
        assert hasattr(ValidationSeverity, "WARNING")

    def test_validation_severity_has_info(self):
        assert hasattr(ValidationSeverity, "INFO")

    def test_validation_issue_has_severity_and_description(self):
        """ValidationIssue has severity, layer, description fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            layer=1,
            description="test issue",
        )
        assert issue.severity == ValidationSeverity.WARNING
        assert issue.description == "test issue"
        assert issue.layer == 1
        assert issue.game_number is None
        assert issue.point_index is None

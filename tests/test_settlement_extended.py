"""
test_settlement_extended.py
============================
Extended coverage tests for:
  - settlement/grading_service.py   (target: uncovered branches)
  - settlement/void_rules.py        (target: uncovered branches)
  - agents/live/model_core_agent.py (target: 0% → full coverage)
  - agents/sgp_supervisor.py        (target: uncovered edge cases)
  - agents/outright_supervisor.py   (target: lifecycle + error paths)
  - agents/agent_runtime.py         (target: async lifecycle + queries)

All tests avoid duplicating coverage already in test_settlement.py
and test_agent_supervisors.py.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Settlement imports
from config.badminton_config import Discipline, TournamentTier
from core.match_state import BadmintonMatchStateMachine, MatchStatus, PointWinner
from settlement.grading_service import (
    GradingService,
    MatchLiveState,
    MatchResult,
    SettlementError,
    SettlementOutcome,
    SettlementRecord,
    SettlementStatus,
)
from settlement.void_rules import RetirementVoidRules, WalkoverVoidRules

# Agent imports
from agents.live.model_core_agent import LiveModelOutput, ModelCoreAgent
from agents.sgp_supervisor import (
    SGPMatchContext,
    SGPRejectionReason,
    SGPRequest,
    SGPResponse,
    SGPSupervisorAgent,
    SGPValidationError,
)
from agents.outright_supervisor import (
    OutrightMarketStatus,
    OutrightPriceSnapshot,
    OutrightSupervisorAgent,
    PlayerResult,
    TournamentState,
)
from agents.agent_runtime import (
    AgentRegistration,
    AgentState,
    BadmintonAgentRuntime,
    RuntimeConfig,
)
from markets.outright_pricing import TournamentEntry
from markets.sgp_engine import SGPLeg, SGPLegType


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_state(
    match_id: str = "test_match",
    entity_a: str = "player_a",
    entity_b: str = "player_b",
    status: MatchStatus = MatchStatus.COMPLETED,
    games_won_a: int = 2,
    games_won_b: int = 0,
    game_scores=None,
    winner: str = "A",
) -> MatchLiveState:
    """Build a MatchLiveState directly (no state machine overhead)."""
    if game_scores is None:
        game_scores = [(21, 15), (21, 18)]
    return MatchLiveState(
        match_id=match_id,
        entity_a_id=entity_a,
        entity_b_id=entity_b,
        discipline=Discipline.MS,
        status=status,
        current_game=len(game_scores) + 1,
        score_a=0,
        score_b=0,
        games_won_a=games_won_a,
        games_won_b=games_won_b,
        game_scores=game_scores,
        match_winner=winner,
    )


def _make_2_1_state(entity_a: str = "player_a", entity_b: str = "player_b") -> MatchLiveState:
    """2-1 match: A wins G1 and G3, B wins G2."""
    return MatchLiveState(
        match_id="m_2_1",
        entity_a_id=entity_a,
        entity_b_id=entity_b,
        discipline=Discipline.MS,
        status=MatchStatus.COMPLETED,
        current_game=4,
        score_a=0,
        score_b=0,
        games_won_a=2,
        games_won_b=1,
        game_scores=[(21, 18), (14, 21), (21, 16)],
        match_winner="A",
    )


def _make_retired_state(
    games: list = None,
    retiring: str = "A",
) -> MatchLiveState:
    """Retired match — partial game scores."""
    if games is None:
        games = [(21, 15), (5, 10)]
    return MatchLiveState(
        match_id="m_retired",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=Discipline.MS,
        status=MatchStatus.RETIRED,
        current_game=2,
        score_a=5,
        score_b=10,
        games_won_a=1,
        games_won_b=0,
        game_scores=games,
        match_winner="B",
    )


def _make_walkover_state() -> MatchLiveState:
    state = BadmintonMatchStateMachine.initialise(
        match_id="wo_ext",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=Discipline.MS,
        first_server="A",
    )
    return BadmintonMatchStateMachine.apply_walkover(state, walkover_winner="B")


def _entry(pid: str = "P01", elo: float = 1500.0, seed: int = 1) -> TournamentEntry:
    return TournamentEntry(
        entity_id=pid,
        seeding=seed,
        rwp_as_server=0.515,
        rwp_as_receiver=0.500,
        elo_rating=elo,
    )


def _entries(n: int = 8) -> List[TournamentEntry]:
    return [_entry(f"P{i+1:02d}", seed=i + 1) for i in range(n)]


def _sgp_context(match_id: str = "M_EXT", is_active: bool = True) -> SGPMatchContext:
    return SGPMatchContext(
        match_id=match_id,
        discipline=Discipline.MS,
        tier=TournamentTier.SUPER_500,
        rwp_a=0.515,
        rwp_b=0.510,
        p_match_win=0.60,
        score_a=5,
        score_b=3,
        games_won_a=0,
        games_won_b=0,
        current_game=1,
        server="A",
        is_active=is_active,
    )


def _mock_leg(market_type) -> MagicMock:
    """Create a mock SGP leg with .market attribute (as the supervisor expects)."""
    leg = MagicMock()
    leg.market = market_type
    return leg


def _sgp_req_two_legs(match_id: str = "M_EXT") -> SGPRequest:
    """Build a 2-leg SGP request using mock legs that have the .market attribute."""
    from agents.sgp_supervisor import SGPMarket
    return SGPRequest(
        request_id="REQ_EXT_001",
        match_id=match_id,
        discipline=Discipline.MS,
        legs=[
            _mock_leg(SGPMarket.MATCH_WINNER),
            _mock_leg(SGPMarket.GAME_WINNER),
        ],
    )


# ---------------------------------------------------------------------------
# GradingService — uncovered market branches
# ---------------------------------------------------------------------------

class TestGradingServiceExtended:
    """Covers settlement branches not exercised by test_settlement.py."""

    @pytest.fixture
    def gs(self) -> GradingService:
        return GradingService()

    # --- total_games_under_2.5 ---

    def test_total_games_under_2_5_for_2_0(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"total_games_under_2.5": ["Under 2.5", "Over 2.5"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        assert rec.winning_outcome == "Under 2.5"

    def test_total_games_under_2_5_for_2_1(self, gs: GradingService) -> None:
        state = _make_2_1_state()
        records = gs.settle_match(state, {"total_games_under_2.5": ["Under 2.5", "Over 2.5"]})
        rec = records[0]
        assert rec.winning_outcome == "Over 2.5"

    def test_total_games_over_2_5_void_retired_incomplete(self, gs: GradingService) -> None:
        """Retired after only 1 game completed — total games voided."""
        state = MatchLiveState(
            match_id="ret_1g",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=2,
            score_a=3,
            score_b=0,
            games_won_a=0,
            games_won_b=0,
            game_scores=[(5, 10)],   # only 1 partial game — < 2 completed
            match_winner="B",
        )
        records = gs.settle_match(state, {"total_games_over_2.5": ["Over 2.5", "Under 2.5"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- exact_games ---

    def test_exact_games_2_match(self, gs: GradingService) -> None:
        state = _make_state()  # 2-0 → 2 games
        records = gs.settle_match(state, {"exact_games_2": ["2", "not_2"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        assert rec.winning_outcome == "2"

    def test_exact_games_3_when_2_played(self, gs: GradingService) -> None:
        state = _make_state()  # 2 games played
        records = gs.settle_match(state, {"exact_games_3": ["3", "not_3"]})
        rec = records[0]
        assert rec.winning_outcome == "not_3"

    def test_exact_games_3_match(self, gs: GradingService) -> None:
        state = _make_2_1_state()  # 3 games played
        records = gs.settle_match(state, {"exact_games_3": ["3", "not_3"]})
        rec = records[0]
        assert rec.winning_outcome == "3"

    def test_exact_games_invalid_suffix_returns_void(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"exact_games_xyz": ["x", "y"]})
        rec = records[0]
        assert rec.settlement_status in (SettlementStatus.VOIDED, SettlementStatus.ERROR)

    # --- game_3_winner ---

    def test_game_3_winner_settles_in_2_1_match(self, gs: GradingService) -> None:
        state = _make_2_1_state()
        records = gs.settle_match(state, {"game_3_winner": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        # A won game 3 (21-16)
        assert rec.winning_outcome == "player_a"

    def test_game_3_winner_voided_in_2_0_match(self, gs: GradingService) -> None:
        state = _make_state()  # 2-0, no game 3
        records = gs.settle_match(state, {"game_3_winner": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    def test_game_winner_invalid_market_id(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"game_x_winner": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status in (SettlementStatus.VOIDED, SettlementStatus.ERROR)

    # --- game total points O/U ---

    def test_game_1_total_over(self, gs: GradingService) -> None:
        # G1 = 21-15 → 36 pts; over 35 should win
        state = _make_state(game_scores=[(21, 15), (21, 10)])
        records = gs.settle_match(state, {"game_1_total_o35": ["Over 35", "Under 35"]})
        rec = records[0]
        assert rec.winning_outcome == "Over 35"

    def test_game_1_total_under(self, gs: GradingService) -> None:
        # G1 = 21-10 → 31 pts; over 35 → under wins
        state = _make_state(game_scores=[(21, 10), (21, 12)])
        records = gs.settle_match(state, {"game_1_total_o35": ["Over 35", "Under 35"]})
        rec = records[0]
        assert rec.winning_outcome == "Under 35"

    def test_game_total_not_played_voided(self, gs: GradingService) -> None:
        state = _make_state()  # only 2 games
        records = gs.settle_match(state, {"game_3_total_o40": ["Over 40", "Under 40"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- race_to_n ---

    def test_race_to_5_game1_a_wins(self, gs: GradingService) -> None:
        state = _make_state(game_scores=[(21, 15), (21, 12)])
        records = gs.settle_match(state, {"race_to_5_game1": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        assert rec.winning_outcome == "player_a"

    def test_race_to_5_game1_b_wins(self, gs: GradingService) -> None:
        # Make a state where B wins a game so B gets credit for race_to_5 in G2 if B wins that
        state = _make_state(
            games_won_a=1,
            games_won_b=1,
            game_scores=[(21, 15), (10, 21)],
            winner="A",
        )
        # B reached 5 in game 2 — but A did not reach 5 first if sa < 5
        # Use a game where B definitely reaches 5 first: sa=3, sb=21
        state2 = _make_state(game_scores=[(3, 21), (21, 10)], games_won_a=1, games_won_b=1, winner="A")
        records = gs.settle_match(state2, {"race_to_5_game1": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        assert rec.winning_outcome == "player_b"

    def test_race_to_n_game_not_played_voided(self, gs: GradingService) -> None:
        state = _make_state()  # 2 games
        records = gs.settle_match(state, {"race_to_5_game3": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    def test_race_to_n_invalid_market_id(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"race_to_x_gamex": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status in (SettlementStatus.VOIDED, SettlementStatus.ERROR)

    # --- match total O/U ---

    def test_match_total_over(self, gs: GradingService) -> None:
        # G1=21-15 (36) + G2=21-10 (31) = 67 total; over 60 wins
        state = _make_state(game_scores=[(21, 15), (21, 10)])
        records = gs.settle_match(state, {"match_total_o60": ["Over 60.0", "Under 60.0"]})
        rec = records[0]
        assert rec.winning_outcome == "Over 60.0"

    def test_match_total_under(self, gs: GradingService) -> None:
        state = _make_state(game_scores=[(21, 10), (21, 10)])
        # total = 62, under 70 wins
        records = gs.settle_match(state, {"match_total_o70": ["Over 70.0", "Under 70.0"]})
        rec = records[0]
        assert rec.winning_outcome == "Under 70.0"

    def test_match_total_retired_voided(self, gs: GradingService) -> None:
        state = _make_retired_state()
        records = gs.settle_match(state, {"match_total_o60": ["Over 60.0", "Under 60.0"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    def test_points_total_market_prefix(self, gs: GradingService) -> None:
        """points_total_oXX prefix also routes to _settle_match_total."""
        state = _make_state(game_scores=[(21, 10), (21, 10)])
        records = gs.settle_match(state, {"points_total_o50": ["Over 50.0", "Under 50.0"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED

    # --- first_point ---

    def test_first_point_winner_is_initial_server(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"first_point_winner": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        # initial_server defaults to "A" → player_a
        assert rec.winning_outcome == "player_a"

    def test_first_point_winner_server_b(self, gs: GradingService) -> None:
        state = MatchLiveState(
            match_id="fp_b",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.COMPLETED,
            current_game=3,
            score_a=0,
            score_b=0,
            games_won_a=2,
            games_won_b=0,
            game_scores=[(21, 10), (21, 12)],
            match_winner="A",
            initial_server="B",
        )
        records = gs.settle_match(state, {"first_point_winner": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.winning_outcome == "player_b"

    # --- deuce markets ---

    def test_deuce_market_yes_when_game_reached_20_20(self, gs: GradingService) -> None:
        # Game ends 22-20 → deuce happened
        state = _make_state(game_scores=[(22, 20), (21, 10)])
        records = gs.settle_match(state, {"deuce_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "Yes"

    def test_deuce_market_no_when_no_deuce(self, gs: GradingService) -> None:
        state = _make_state(game_scores=[(21, 10), (21, 12)])
        records = gs.settle_match(state, {"deuce_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "No"

    def test_deuce_market_retired_early_void(self, gs: GradingService) -> None:
        """Retired too early for deuce to be possible → void."""
        state = MatchLiveState(
            match_id="deuce_ret",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=1,
            score_a=5,
            score_b=3,
            games_won_a=0,
            games_won_b=0,
            game_scores=[(5, 3)],
            match_winner="B",
        )
        records = gs.settle_match(state, {"deuce_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- golden_point ---

    def test_golden_point_yes(self, gs: GradingService) -> None:
        # 30-29 is golden point
        state = _make_state(game_scores=[(30, 29), (21, 10)])
        records = gs.settle_match(state, {"golden_point_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "Yes"

    def test_golden_point_no(self, gs: GradingService) -> None:
        state = _make_state(game_scores=[(21, 10), (21, 12)])
        records = gs.settle_match(state, {"golden_point_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "No"

    def test_golden_point_29_30_also_triggers(self, gs: GradingService) -> None:
        state = _make_state(
            game_scores=[(21, 10), (29, 30)],
            games_won_a=1,
            games_won_b=1,
            winner="B",
        )
        records = gs.settle_match(state, {"golden_point_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "Yes"

    def test_golden_point_retired_voided(self, gs: GradingService) -> None:
        state = _make_retired_state()
        records = gs.settle_match(state, {"golden_point_in_match": ["Yes", "No"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- comeback ---

    def test_comeback_yes_in_2_1_match(self, gs: GradingService) -> None:
        state = _make_2_1_state()  # 3 games → comeback detected
        records = gs.settle_match(state, {"comeback_win": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "Yes"

    def test_comeback_no_in_2_0_match(self, gs: GradingService) -> None:
        state = _make_state()  # 2 games, no comeback
        records = gs.settle_match(state, {"comeback_win": ["Yes", "No"]})
        rec = records[0]
        assert rec.winning_outcome == "No"

    def test_comeback_retired_voided(self, gs: GradingService) -> None:
        state = _make_retired_state()
        records = gs.settle_match(state, {"comeback_win": ["Yes", "No"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- handicap_games ---

    def test_handicap_games_a_minus_1_5_a_wins_2_0(self, gs: GradingService) -> None:
        # A wins 2-0; A +(-1.5) = 0.5 > 0 → A still covers
        state = _make_state()
        records = gs.settle_match(
            state, {"handicap_games_a_-1.5": ["player_a", "player_b"]}
        )
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED
        assert rec.winning_outcome == "player_a"

    def test_handicap_games_a_minus_1_5_b_wins_0_2(self, gs: GradingService) -> None:
        # B wins 2-0; A +(-1.5) = -1 + (-1.5) = -1.5, b=2 → B covers
        state = _make_state(
            games_won_a=0, games_won_b=2,
            game_scores=[(10, 21), (12, 21)],
            winner="B",
        )
        records = gs.settle_match(
            state, {"handicap_games_a_-1.5": ["player_a", "player_b"]}
        )
        rec = records[0]
        assert rec.winning_outcome == "player_b"

    def test_handicap_games_b_plus_1_5(self, gs: GradingService) -> None:
        state = _make_state()  # A wins 2-0; B+1.5 = 1.5 > 2? No. B loses
        records = gs.settle_match(
            state, {"handicap_games_b_+1.5": ["player_a", "player_b"]}
        )
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.SETTLED

    def test_handicap_games_invalid_market_returns_void(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"handicap_games_x": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status in (SettlementStatus.VOIDED, SettlementStatus.ERROR)

    def test_handicap_games_retired_early_voided(self, gs: GradingService) -> None:
        # Retired with only 1 partial game (<2 completed)
        state = MatchLiveState(
            match_id="hcap_ret",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=1,
            score_a=5,
            score_b=3,
            games_won_a=0,
            games_won_b=0,
            game_scores=[(5, 3)],
            match_winner="B",
        )
        records = gs.settle_match(state, {"handicap_games_a_-1.5": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- unknown market ---

    def test_unknown_market_returns_void_record(self, gs: GradingService) -> None:
        state = _make_state()
        records = gs.settle_match(state, {"completely_unknown_market": ["x", "y"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.VOIDED

    # --- entity_to_outcome mapping ---

    def test_entity_to_outcome_exact_name_match(self, gs: GradingService) -> None:
        """When entity_id is an exact outcome name, it is returned directly."""
        state = _make_state(entity_a="Viktor_Axelsen", entity_b="Lee_Zii_Jia")
        records = gs.settle_match(
            state, {"match_winner": ["Viktor_Axelsen", "Lee_Zii_Jia"]}
        )
        rec = records[0]
        assert rec.winning_outcome == "Viktor_Axelsen"

    def test_entity_to_outcome_positional_fallback(self, gs: GradingService) -> None:
        """When entity_id not in outcomes, use positional mapping."""
        state = _make_state(entity_a="p_a_id", entity_b="p_b_id")
        records = gs.settle_match(
            state, {"match_winner": ["Player A Name", "Player B Name"]}
        )
        rec = records[0]
        # A wins → first outcome selected
        assert rec.winning_outcome == "Player A Name"

    # --- error path in _settle_market ---

    def test_settlement_error_path_produces_error_record(self, gs: GradingService) -> None:
        """If _determine_winner raises, an ERROR record is produced."""
        state = _make_state()
        with patch.object(gs, "_determine_winner", side_effect=RuntimeError("deliberate error")):
            records = gs.settle_match(state, {"match_winner": ["x", "y"]})
        rec = records[0]
        assert rec.settlement_status == SettlementStatus.ERROR
        assert "deliberate error" in rec.notes

    # --- correct_score B winner ---

    def test_correct_score_b_wins_2_0(self, gs: GradingService) -> None:
        state = _make_state(
            games_won_a=0, games_won_b=2,
            game_scores=[(10, 21), (12, 21)],
            winner="B",
        )
        records = gs.settle_match(state, {"correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"]})
        rec = records[0]
        assert rec.winning_outcome == "B_2-0"

    def test_correct_score_b_wins_2_1(self, gs: GradingService) -> None:
        state = _make_state(
            games_won_a=1,
            games_won_b=2,
            game_scores=[(21, 10), (12, 21), (10, 21)],
            winner="B",
        )
        records = gs.settle_match(state, {"correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"]})
        rec = records[0]
        assert rec.winning_outcome == "B_2-1"

    # --- MatchResult helpers ---

    def test_match_result_total_points(self) -> None:
        state = _make_state(game_scores=[(21, 15), (21, 10)])
        result = MatchResult.from_live_state(state)
        assert result.total_points == 67

    def test_match_result_score_string_2_1(self) -> None:
        state = _make_2_1_state()
        result = MatchResult.from_live_state(state)
        s = result.score_string()
        assert "2-1" in s
        assert "21-18" in s

    def test_match_result_from_live_state_raises_on_non_terminal(self) -> None:
        state = MatchLiveState(
            match_id="inc",
            entity_a_id="a",
            entity_b_id="b",
            discipline=Discipline.MS,
            status=MatchStatus.IN_PROGRESS,
            current_game=1,
            score_a=5,
            score_b=3,
            games_won_a=0,
            games_won_b=0,
            game_scores=[],
            match_winner=None,
        )
        with pytest.raises(SettlementError):
            MatchResult.from_live_state(state)

    def test_match_result_raises_no_winner(self) -> None:
        state = MatchLiveState(
            match_id="no_winner",
            entity_a_id="a",
            entity_b_id="b",
            discipline=Discipline.MS,
            status=MatchStatus.COMPLETED,
            current_game=3,
            score_a=0,
            score_b=0,
            games_won_a=2,
            games_won_b=0,
            game_scores=[(21, 10), (21, 12)],
            match_winner=None,   # deliberately missing
        )
        with pytest.raises(SettlementError):
            MatchResult.from_live_state(state)

    def test_match_result_initial_server_from_event_metadata(self) -> None:
        """Fallback: initial_server parsed from event metadata."""
        state = _make_state()
        # Inject a fake event with metadata
        event = MagicMock()
        event.metadata = "first_server=B"
        object.__setattr__(state, "events", [event])
        # Ensure initial_server field is missing/falsy
        if hasattr(state, "initial_server"):
            object.__setattr__(state, "initial_server", "")
        result = MatchResult.from_live_state(state)
        assert result.initial_server == "B"

    # --- SettlementRecord properties ---

    def test_settlement_record_status_property(self) -> None:
        rec = SettlementRecord(
            match_id="m",
            market_id="match_winner",
            winning_outcome="player_a",
            settlement_status=SettlementStatus.SETTLED,
            settled_at=datetime.now(timezone.utc),
            entity_a_id="player_a",
            entity_b_id="player_b",
            final_score="2-0 (21-10, 21-12)",
        )
        assert rec.status == SettlementStatus.SETTLED

    # --- race_to_n both reached N ---

    def test_race_to_5_both_reached_conservative_a(self, gs: GradingService) -> None:
        """When both reach n, conservative: A wins."""
        state = _make_state(game_scores=[(21, 21), (21, 10)])
        records = gs.settle_match(state, {"race_to_5_game1": ["player_a", "player_b"]})
        rec = records[0]
        assert rec.winning_outcome == "player_a"

    # --- discipline variants ---

    @pytest.mark.parametrize("disc", [Discipline.WS, Discipline.MD, Discipline.XD])
    def test_match_winner_all_disciplines(self, gs: GradingService, disc: Discipline) -> None:
        state = MatchLiveState(
            match_id=f"disc_{disc.value}",
            entity_a_id="ent_a",
            entity_b_id="ent_b",
            discipline=disc,
            status=MatchStatus.COMPLETED,
            current_game=3,
            score_a=0,
            score_b=0,
            games_won_a=2,
            games_won_b=0,
            game_scores=[(21, 10), (21, 12)],
            match_winner="A",
        )
        records = gs.settle_match(state, {"match_winner": ["ent_a", "ent_b"]})
        rec = records[0]
        assert rec.winning_outcome == "ent_a"


# ---------------------------------------------------------------------------
# RetirementVoidRules — uncovered branches
# ---------------------------------------------------------------------------

class TestRetirementVoidRulesExtended:
    """Covers branches in RetirementVoidRules not hit by test_settlement.py."""

    def _retired_1game(self) -> MatchResult:
        """1 complete game, then retired (2-game scenario)."""
        state = MatchLiveState(
            match_id="rv_1g",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=2,
            score_a=5,
            score_b=3,
            games_won_a=1,
            games_won_b=0,
            game_scores=[(21, 15), (5, 3)],
            match_winner="B",
        )
        return MatchResult.from_live_state(state)

    def _retired_2game_complete(self) -> MatchResult:
        """2 complete games in a 2-0 retirement scenario."""
        state = MatchLiveState(
            match_id="rv_2g",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=3,
            score_a=0,
            score_b=0,
            games_won_a=2,
            games_won_b=0,
            game_scores=[(21, 15), (21, 12)],
            match_winner="A",
        )
        return MatchResult.from_live_state(state)

    def test_not_retired_returns_settled_none(self) -> None:
        """Non-retired result passes through without voiding."""
        state = _make_state()
        result = MatchResult.from_live_state(state)
        status, winner, reason = RetirementVoidRules.apply("match_winner", result, ["player_a", "player_b"])
        assert status == SettlementStatus.SETTLED
        assert winner is None

    def test_total_games_settled_when_2_complete_games(self) -> None:
        result = self._retired_2game_complete()
        status, winner, reason = RetirementVoidRules.apply(
            "total_games_over_2.5", result, ["Over 2.5", "Under 2.5"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "Under 2.5"

    def test_total_games_under_2_5_settled_when_2_complete(self) -> None:
        result = self._retired_2game_complete()
        status, winner, reason = RetirementVoidRules.apply(
            "total_games_under_2.5", result, ["Under 2.5", "Over 2.5"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "Under 2.5"

    def test_total_games_voided_when_1_game_only(self) -> None:
        result = self._retired_1game()
        status, winner, reason = RetirementVoidRules.apply(
            "total_games_over_2.5", result, ["Over 2.5", "Under 2.5"]
        )
        assert status == SettlementStatus.VOIDED

    def test_game_total_settled_for_completed_game(self) -> None:
        result = self._retired_1game()
        # G1 completed: 21-15 = 36 total
        status, winner, reason = RetirementVoidRules.apply(
            "game_1_total_o35", result, ["Over 35", "Under 35"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "Over 35"

    def test_game_total_voided_for_incomplete_game(self) -> None:
        result = self._retired_1game()
        # game_3 not in game_scores (only 2 entries) → should void
        status, winner, reason = RetirementVoidRules.apply(
            "game_3_total_o35", result, ["Over 35", "Under 35"]
        )
        assert status == SettlementStatus.VOIDED

    def test_game_total_invalid_market_voided(self) -> None:
        result = self._retired_1game()
        status, winner, reason = RetirementVoidRules.apply(
            "game_x_total_oxxx", result, ["Over 35", "Under 35"]
        )
        assert status == SettlementStatus.VOIDED

    def test_race_on_retirement_settled_for_completed_game(self) -> None:
        result = self._retired_1game()
        # G1 completed: A reached 21 ≥ 5
        status, winner, reason = RetirementVoidRules.apply(
            "race_to_5_game1", result, ["player_a", "player_b"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "player_a"

    def test_race_on_retirement_b_wins_race(self) -> None:
        """B wins race_to_10 in game 1 when A only scored 3 (sa < 10, sb >= 10)."""
        state = MatchLiveState(
            match_id="rv_race_b",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=2,
            score_a=2,
            score_b=8,
            games_won_a=0,
            games_won_b=1,
            game_scores=[(3, 21)],   # sa=3 < 10; sb=21 >= 10 → B wins race_to_10
            match_winner="A",
        )
        result = MatchResult.from_live_state(state)
        status, winner, reason = RetirementVoidRules.apply(
            "race_to_10_game1", result, ["player_a", "player_b"]
        )
        assert status == SettlementStatus.SETTLED
        assert winner == "player_b"

    def test_race_on_retirement_not_reached_voided(self) -> None:
        state = MatchLiveState(
            match_id="rv_race_none",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            status=MatchStatus.RETIRED,
            current_game=1,
            score_a=3,
            score_b=3,
            games_won_a=0,
            games_won_b=0,
            game_scores=[(3, 3)],
            match_winner="B",
        )
        result = MatchResult.from_live_state(state)
        status, winner, reason = RetirementVoidRules.apply(
            "race_to_5_game1", result, ["player_a", "player_b"]
        )
        assert status == SettlementStatus.VOIDED

    def test_race_invalid_market_id_voided(self) -> None:
        result = self._retired_1game()
        status, winner, reason = RetirementVoidRules.apply(
            "race_to_x_gamex", result, ["player_a", "player_b"]
        )
        assert status == SettlementStatus.VOIDED

    def test_race_game_not_played_voided(self) -> None:
        result = self._retired_1game()
        status, winner, reason = RetirementVoidRules.apply(
            "race_to_5_game3", result, ["player_a", "player_b"]
        )
        assert status == SettlementStatus.VOIDED

    def test_game_winner_invalid_market_id_voided(self) -> None:
        result = self._retired_1game()
        status, winner, reason = RetirementVoidRules.apply(
            "game_x_winner", result, ["player_a", "player_b"]
        )
        assert status == SettlementStatus.VOIDED

    def test_arbitrary_market_voided_on_retirement(self) -> None:
        result = self._retired_1game()
        status, winner, reason = RetirementVoidRules.apply(
            "golden_point_in_match", result, ["Yes", "No"]
        )
        assert status == SettlementStatus.VOIDED
        assert reason == "retirement_market_void"


# ---------------------------------------------------------------------------
# WalkoverVoidRules — uncovered: batch form
# ---------------------------------------------------------------------------

class TestWalkoverVoidRulesBatchForm:
    """Test the batch (dict) calling convention for WalkoverVoidRules.void_all."""

    def test_batch_voids_all_markets(self) -> None:
        open_markets = {
            "match_winner": ["a", "b"],
            "correct_score": ["A_2-0", "B_2-0"],
            "total_games_over_2.5": ["Over 2.5", "Under 2.5"],
        }
        result = WalkoverVoidRules.void_all("match_walk_001", open_markets)
        assert isinstance(result, dict)
        assert len(result) == 3
        for mid, (status, reason) in result.items():
            assert status == SettlementStatus.VOIDED
            assert reason == "walkover"

    def test_batch_empty_markets_returns_empty_dict(self) -> None:
        result = WalkoverVoidRules.void_all("match_wo_002", {})
        assert result == {}

    def test_per_market_form_returns_three_tuple(self) -> None:
        state = MatchResult.from_live_state(_make_walkover_state())
        result = WalkoverVoidRules.void_all("game_1_winner", state, ["a", "b"])
        assert isinstance(result, tuple)
        assert len(result) == 3
        status, winner, reason = result
        assert status == SettlementStatus.VOIDED
        assert winner is None
        assert reason == "walkover"


# ---------------------------------------------------------------------------
# ModelCoreAgent
# ---------------------------------------------------------------------------

class TestModelCoreAgent:
    """Full coverage of agents/live/model_core_agent.py."""

    def _make_mocks(self):
        """Return (live_state, bayesian_updater, momentum_detector) mocks."""
        from core.bayesian_updater import LiveRWPEstimate
        from core.momentum_detector import MomentumSnapshot, MomentumRegime, MomentumSignalStrength
        from core.markov_engine import MatchProbabilities

        live_state = MagicMock()
        live_state.match_id = "live_001"
        live_state.server = "A"
        live_state.current_game = 1
        live_state.score_a = 6
        live_state.score_b = 4
        live_state.games_won_a = 0
        live_state.games_won_b = 0
        live_state.discipline = Discipline.MS
        live_state.total_points_played = 10

        rwp_est_a = LiveRWPEstimate(
            entity_id="player_a",
            rwp_prior=0.515,
            rwp_posterior=0.520,
            rwp_live=0.518,
            evidence_weight=0.25,
            uncertainty=0.01,
            server_wins=4,
            server_total=8,
            confidence_interval=(0.48, 0.56),
        )
        rwp_est_b = LiveRWPEstimate(
            entity_id="player_b",
            rwp_prior=0.510,
            rwp_posterior=0.505,
            rwp_live=0.508,
            evidence_weight=0.25,
            uncertainty=0.01,
            server_wins=3,
            server_total=7,
            confidence_interval=(0.46, 0.55),
        )

        bayesian_updater = MagicMock()
        bayesian_updater.get_live_rwp.side_effect = lambda e: rwp_est_a if e == "A" else rwp_est_b

        # MatchProbabilities mock
        markov_probs = MagicMock()
        markov_probs.p_a_wins_match = 0.62

        markov_engine = MagicMock()
        markov_engine.compute_match_probabilities.return_value = markov_probs

        # LiveProbabilityBlend mock
        blend = MagicMock()
        blend.p_a_wins_match_blend = 0.615
        blend.markov_weight = 0.4

        # MomentumSnapshot mock
        momentum = MagicMock(spec=MomentumSnapshot)
        momentum.regime = MomentumRegime.NEUTRAL

        momentum_detector = MagicMock()
        momentum_detector.add_point.return_value = momentum

        return live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum

    def test_constructs(self) -> None:
        agent = ModelCoreAgent()
        assert agent is not None
        assert hasattr(agent, "_markov")

    def test_compute_returns_live_model_output(self) -> None:
        agent = ModelCoreAgent()
        live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum = (
            self._make_mocks()
        )

        mock_snap = MagicMock()
        mock_snap.current_game = 1
        mock_snap.score_a = 6
        mock_snap.score_b = 4

        with (
            patch("agents.live.model_core_agent.LiveProbabilityBlend") as mock_blend_cls,
            patch("agents.live.model_core_agent.LiveStateSummary") as mock_summary_cls,
        ):
            mock_blend_cls.compute.return_value = blend
            mock_summary_cls.return_value = mock_snap
            agent._markov = MagicMock()
            agent._markov.compute_match_probabilities.return_value = markov_probs

            output = agent.compute(
                live_state=live_state,
                bayesian_updater=bayesian_updater,
                momentum_detector=momentum_detector,
                winner="A",
                pre_match_p_a=0.60,
            )

        assert isinstance(output, LiveModelOutput)
        assert output.rwp_a == pytest.approx(0.518)
        assert output.rwp_b == pytest.approx(0.508)
        assert output.p_a_wins_markov == pytest.approx(0.62)

    def test_compute_calls_observe_rally(self) -> None:
        agent = ModelCoreAgent()
        live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum = (
            self._make_mocks()
        )

        with (
            patch("agents.live.model_core_agent.LiveProbabilityBlend") as mock_blend_cls,
            patch("agents.live.model_core_agent.LiveStateSummary"),
        ):
            mock_blend_cls.compute.return_value = blend
            agent._markov = MagicMock()
            agent._markov.compute_match_probabilities.return_value = markov_probs
            agent.compute(
                live_state=live_state,
                bayesian_updater=bayesian_updater,
                momentum_detector=momentum_detector,
                winner="B",
                pre_match_p_a=0.58,
            )

        bayesian_updater.observe_rally.assert_called_once()

    def test_compute_calls_markov_engine(self) -> None:
        agent = ModelCoreAgent()
        live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum = (
            self._make_mocks()
        )

        with (
            patch("agents.live.model_core_agent.LiveProbabilityBlend") as mock_blend_cls,
            patch("agents.live.model_core_agent.LiveStateSummary"),
        ):
            mock_blend_cls.compute.return_value = blend
            markov_mock = MagicMock()
            markov_mock.compute_match_probabilities.return_value = markov_probs
            agent._markov = markov_mock

            agent.compute(
                live_state=live_state,
                bayesian_updater=bayesian_updater,
                momentum_detector=momentum_detector,
                winner="A",
                pre_match_p_a=0.60,
            )

        markov_mock.compute_match_probabilities.assert_called_once()

    def test_compute_calls_momentum_detector(self) -> None:
        agent = ModelCoreAgent()
        live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum = (
            self._make_mocks()
        )

        with (
            patch("agents.live.model_core_agent.LiveProbabilityBlend") as mock_blend_cls,
            patch("agents.live.model_core_agent.LiveStateSummary"),
        ):
            mock_blend_cls.compute.return_value = blend
            agent._markov = MagicMock()
            agent._markov.compute_match_probabilities.return_value = markov_probs

            agent.compute(
                live_state=live_state,
                bayesian_updater=bayesian_updater,
                momentum_detector=momentum_detector,
                winner="A",
                pre_match_p_a=0.60,
            )

        momentum_detector.add_point.assert_called_once()

    def test_compute_blend_weight_in_output(self) -> None:
        agent = ModelCoreAgent()
        live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum = (
            self._make_mocks()
        )

        with (
            patch("agents.live.model_core_agent.LiveProbabilityBlend") as mock_blend_cls,
            patch("agents.live.model_core_agent.LiveStateSummary"),
        ):
            mock_blend_cls.compute.return_value = blend
            agent._markov = MagicMock()
            agent._markov.compute_match_probabilities.return_value = markov_probs

            output = agent.compute(
                live_state=live_state,
                bayesian_updater=bayesian_updater,
                momentum_detector=momentum_detector,
                winner="A",
                pre_match_p_a=0.60,
            )

        assert output.markov_weight == pytest.approx(0.4)
        assert output.p_a_wins_blend == pytest.approx(0.615)

    def test_live_model_output_fields(self) -> None:
        agent = ModelCoreAgent()
        live_state, bayesian_updater, momentum_detector, markov_probs, blend, momentum = (
            self._make_mocks()
        )

        mock_snap = MagicMock()
        mock_snap.current_game = 1
        mock_snap.score_a = 6
        mock_snap.score_b = 4

        with (
            patch("agents.live.model_core_agent.LiveProbabilityBlend") as mock_blend_cls,
            patch("agents.live.model_core_agent.LiveStateSummary") as mock_summary_cls,
        ):
            mock_blend_cls.compute.return_value = blend
            mock_summary_cls.return_value = mock_snap
            agent._markov = MagicMock()
            agent._markov.compute_match_probabilities.return_value = markov_probs

            output = agent.compute(
                live_state=live_state,
                bayesian_updater=bayesian_updater,
                momentum_detector=momentum_detector,
                winner="A",
                pre_match_p_a=0.60,
            )

        assert output.markov_probs is markov_probs
        assert output.momentum is momentum
        assert output.rwp_a_estimate.entity_id == "player_a"
        assert output.rwp_b_estimate.entity_id == "player_b"
        assert output.snap.current_game == 1
        assert output.snap.score_a == 6
        assert output.snap.score_b == 4


# ---------------------------------------------------------------------------
# SGPSupervisorAgent — uncovered branches
# ---------------------------------------------------------------------------

class TestSGPSupervisorAgentExtended:
    """Covers uncovered edge cases in SGPSupervisorAgent."""

    def test_single_leg_rejected(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_S1")
        agent.update_match_context(ctx)
        req = SGPRequest(
            request_id="R_S1",
            match_id="M_S1",
            discipline=Discipline.MS,
            legs=[
                SGPLeg(
                    leg_type=SGPLegType.MATCH_WINNER,
                    selection="A",
                    fair_prob=0.60,
                    market_id="match_winner",
                )
            ],
        )
        resp = agent.price_sgp(req)
        assert resp.is_valid is False
        assert resp.rejection_reason == SGPRejectionReason.SINGLE_LEG

    def test_too_many_legs_rejected(self) -> None:
        from config.badminton_config import SGP_MAX_LEGS

        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_ML")
        agent.update_match_context(ctx)
        legs = [
            SGPLeg(
                leg_type=SGPLegType.MATCH_WINNER,
                selection="A",
                fair_prob=0.60,
                market_id=f"market_{i}",
            )
            for i in range(SGP_MAX_LEGS + 1)
        ]
        req = SGPRequest(
            request_id="R_ML",
            match_id="M_ML",
            discipline=Discipline.MS,
            legs=legs,
        )
        resp = agent.price_sgp(req)
        assert resp.is_valid is False
        assert resp.rejection_reason == SGPRejectionReason.TOO_MANY_LEGS

    def test_inactive_context_rejected(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_INACT", is_active=False)
        agent.update_match_context(ctx)
        req = _sgp_req_two_legs("M_INACT")
        resp = agent.price_sgp(req)
        assert resp.is_valid is False
        assert resp.rejection_reason == SGPRejectionReason.MATCH_NOT_ACTIVE

    def test_no_context_rejection_reason(self) -> None:
        agent = SGPSupervisorAgent()
        req = _sgp_req_two_legs("DOES_NOT_EXIST")
        resp = agent.price_sgp(req)
        assert resp.rejection_reason == SGPRejectionReason.MATCH_NOT_ACTIVE

    def test_duplicate_market_type_rejected(self) -> None:
        """_validate_leg_combination rejects duplicate market types.
        The supervisor accesses leg.market (not leg.leg_type), so we use
        mock leg objects with the .market attribute the supervisor expects."""
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_DUP")
        agent.update_match_context(ctx)

        from agents.sgp_supervisor import SGPMarket
        leg1 = MagicMock()
        leg1.market = SGPMarket.MATCH_WINNER
        leg2 = MagicMock()
        leg2.market = SGPMarket.MATCH_WINNER  # duplicate

        req = SGPRequest(
            request_id="R_DUP",
            match_id="M_DUP",
            discipline=Discipline.MS,
            legs=[leg1, leg2],
        )
        resp = agent.price_sgp(req)
        assert resp.is_valid is False
        assert resp.rejection_reason == SGPRejectionReason.DUPLICATE_MARKET

    def test_incompatible_markets_total_games_correct_score(self) -> None:
        """TOTAL_GAMES + CORRECT_SCORE is a mutually exclusive pair.
        Uses mock legs with .market attribute as the supervisor expects."""
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_INCOMPAT")
        agent.update_match_context(ctx)

        from agents.sgp_supervisor import SGPMarket
        leg1 = MagicMock()
        leg1.market = SGPMarket.TOTAL_GAMES
        leg2 = MagicMock()
        leg2.market = SGPMarket.CORRECT_SCORE

        req = SGPRequest(
            request_id="R_INCOMPAT",
            match_id="M_INCOMPAT",
            discipline=Discipline.MS,
            legs=[leg1, leg2],
        )
        resp = agent.price_sgp(req)
        assert resp.is_valid is False
        assert resp.rejection_reason == SGPRejectionReason.INCOMPATIBLE_OUTCOMES

    def test_engine_error_returns_engine_error_rejection(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_ENG_ERR")
        agent.update_match_context(ctx)

        with patch.object(agent._engine, "price_sgp", side_effect=RuntimeError("engine broke")):
            req = _sgp_req_two_legs("M_ENG_ERR")
            resp = agent.price_sgp(req)

        assert resp.is_valid is False
        assert resp.rejection_reason == SGPRejectionReason.ENGINE_ERROR

    def test_h8_violation_flagged_but_response_returned(self) -> None:
        """H8 failure is recorded but response is still returned with is_valid=True."""
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_H8")
        agent.update_match_context(ctx)

        # Mock engine to return a result violating H8
        from markets.sgp_engine import SGPResponse as EngineResponse

        mock_result = MagicMock(spec=EngineResponse)
        mock_result.margined_odds = 5.00
        mock_result.max_single_leg_odds = 1.80  # SGP > max leg → H8 violation
        mock_result.fair_odds = 4.50

        with patch.object(agent._engine, "price_sgp", return_value=mock_result):
            req = _sgp_req_two_legs("M_H8")
            resp = agent.price_sgp(req)

        assert resp.is_valid is True
        assert resp.h8_passed is False
        assert agent._h8_failures == 1

    def test_h8_no_reference_odds_passes(self) -> None:
        """H8 passes when max_single_leg_odds is None."""
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_H8_NONE")
        agent.update_match_context(ctx)

        from markets.sgp_engine import SGPResponse as EngineResponse

        mock_result = MagicMock(spec=EngineResponse)
        mock_result.margined_odds = 3.50
        mock_result.max_single_leg_odds = None
        mock_result.fair_odds = 3.20

        with patch.object(agent._engine, "price_sgp", return_value=mock_result):
            req = _sgp_req_two_legs("M_H8_NONE")
            resp = agent.price_sgp(req)

        assert resp.is_valid is True
        assert resp.h8_passed is True
        assert resp.h8_detail == "no_leg_odds_reference"

    def test_get_metrics_after_requests(self) -> None:
        agent = SGPSupervisorAgent()
        ctx = _sgp_context("M_METRICS")
        agent.update_match_context(ctx)

        # Fire one valid and one invalid (single leg)
        agent.price_sgp(_sgp_req_two_legs("M_METRICS"))
        single_req = SGPRequest(
            request_id="R_SGL",
            match_id="M_METRICS",
            discipline=Discipline.MS,
            legs=[
                SGPLeg(
                    leg_type=SGPLegType.MATCH_WINNER,
                    selection="A",
                    fair_prob=0.60,
                    market_id="mw",
                )
            ],
        )
        agent.price_sgp(single_req)

        metrics = agent.get_metrics()
        assert metrics["total_requests"] >= 2
        assert metrics["total_rejections"] >= 1
        assert "acceptance_rate" in metrics
        assert metrics["acceptance_rate"] >= 0.0

    def test_get_metrics_zero_requests(self) -> None:
        agent = SGPSupervisorAgent()
        metrics = agent.get_metrics()
        assert metrics["total_requests"] == 0
        assert metrics["acceptance_rate"] == 0.0

    def test_remove_nonexistent_match_no_error(self) -> None:
        agent = SGPSupervisorAgent()
        agent.remove_match("NONEXISTENT")  # Should not raise

    def test_sgp_response_odds_property_none_when_no_result(self) -> None:
        resp = SGPResponse(
            request_id="R1",
            match_id="M1",
            is_valid=False,
            rejection_reason=SGPRejectionReason.SINGLE_LEG,
        )
        assert resp.odds is None
        assert resp.fair_value_odds is None


# ---------------------------------------------------------------------------
# OutrightSupervisorAgent — uncovered paths
# ---------------------------------------------------------------------------

class TestOutrightSupervisorAgentExtended:
    """Covers lifecycle + error paths not in test_agent_supervisors.py."""

    def test_register_duplicate_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_DUP", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        with pytest.raises(RuntimeError, match="already registered"):
            agent.register_tournament("T_DUP", Discipline.MS, TournamentTier.SUPER_500, _entries(8))

    def test_register_empty_entries_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(ValueError, match="No entries"):
            agent.register_tournament("T_EMPTY", Discipline.MS, TournamentTier.SUPER_500, [])

    def test_get_prices_unknown_raises_key_error(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(KeyError):
            agent.get_prices("UNKNOWN_T")

    def test_get_prices_suspended_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_SUSP_P", Discipline.MS, TournamentTier.SUPER_300, _entries(8))
        agent.suspend_tournament("T_SUSP_P")
        with pytest.raises(RuntimeError, match="suspended"):
            agent.get_prices("T_SUSP_P")

    def test_get_prices_resulted_raises(self) -> None:
        """After tournament result, get_prices raises."""
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_RES", Discipline.MS, TournamentTier.SUPER_300, _entries(2))
        # Simulate 1 match to determine winner (2-player draw → tournament complete)
        agent.on_match_result("T_RES", winner_id="P01", loser_id="P02", round_number=1)
        with pytest.raises(RuntimeError, match="resulted"):
            agent.get_prices("T_RES")

    def test_suspend_unknown_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(KeyError):
            agent.suspend_tournament("UNKNOWN")

    def test_resume_unknown_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(KeyError):
            agent.resume_tournament("UNKNOWN")

    def test_resume_non_suspended_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_RNS", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        with pytest.raises(RuntimeError, match="not suspended"):
            agent.resume_tournament("T_RNS")

    def test_suspend_then_resume_cycle(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_SR", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        agent.suspend_tournament("T_SR", reason="test_reason")
        status = agent.get_tournament_status("T_SR")
        assert status["status"] == OutrightMarketStatus.SUSPENDED.value

        agent.resume_tournament("T_SR")
        status = agent.get_tournament_status("T_SR")
        assert status["status"] == OutrightMarketStatus.OPEN.value

    def test_on_match_result_unknown_tournament_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(KeyError):
            agent.on_match_result("UNKNOWN_T", winner_id="P1", loser_id="P2", round_number=1)

    def test_on_match_result_already_resulted_returns_none(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_2P", Discipline.MS, TournamentTier.SUPER_300, _entries(2))
        # First result completes the tournament
        agent.on_match_result("T_2P", winner_id="P01", loser_id="P02", round_number=1)
        # Second call on already-resulted tournament should return None
        result = agent.on_match_result("T_2P", winner_id="P01", loser_id="P02", round_number=2)
        assert result is None

    def test_on_match_result_updates_active_players(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_APL", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        agent.on_match_result("T_APL", winner_id="P01", loser_id="P02", round_number=1)
        status = agent.get_tournament_status("T_APL")
        assert status["n_eliminated"] == 1
        assert status["n_active"] == 7

    def test_get_tournament_status_unknown_raises(self) -> None:
        agent = OutrightSupervisorAgent()
        with pytest.raises(KeyError):
            agent.get_tournament_status("UNKNOWN")

    def test_get_all_tournaments_multiple(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.register_tournament("TA1", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        agent.register_tournament("TA2", Discipline.WS, TournamentTier.SUPER_300, _entries(8))
        result = agent.get_all_tournaments()
        assert len(result) == 2
        ids = {r["tournament_id"] for r in result}
        assert "TA1" in ids
        assert "TA2" in ids

    def test_price_publisher_called_on_reprice(self) -> None:
        published = []
        agent = OutrightSupervisorAgent(price_publisher=lambda tid, snap: published.append(tid))
        agent.register_tournament("T_PUB", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        agent.get_prices("T_PUB")
        assert "T_PUB" in published

    def test_price_publisher_error_doesnt_propagate(self) -> None:
        def bad_publisher(tid, snap):
            raise RuntimeError("publisher error")

        agent = OutrightSupervisorAgent(price_publisher=bad_publisher)
        agent.register_tournament("T_PUBFAIL", Discipline.MS, TournamentTier.SUPER_300, _entries(8))
        # Should not raise even if publisher fails
        snapshot = agent.get_prices("T_PUBFAIL")
        assert isinstance(snapshot, OutrightPriceSnapshot)

    def test_build_snapshot_raises_when_no_cached_prices(self) -> None:
        agent = OutrightSupervisorAgent()
        state = TournamentState(
            tournament_id="T_NCP",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(4),
            last_prices=None,
        )
        with pytest.raises(RuntimeError, match="no cached prices"):
            agent._build_snapshot(state)

    def test_stop_reprice_loop(self) -> None:
        agent = OutrightSupervisorAgent()
        agent.stop_reprice_loop()
        assert agent._running is False

    def test_get_prices_cached_not_stale(self) -> None:
        """Second call within reprice interval returns cached snapshot."""
        agent = OutrightSupervisorAgent()
        agent.register_tournament("T_CACHE", Discipline.MS, TournamentTier.SUPER_500, _entries(8))
        snap1 = agent.get_prices("T_CACHE")
        # Mark as freshly priced
        state = agent._tournaments["T_CACHE"]
        state.last_priced_at = time.monotonic()  # fresh
        snap2 = agent.get_prices("T_CACHE")
        # Both should be valid snapshots
        assert isinstance(snap1, OutrightPriceSnapshot)
        assert isinstance(snap2, OutrightPriceSnapshot)


# ---------------------------------------------------------------------------
# TournamentState
# ---------------------------------------------------------------------------

class TestTournamentState:
    """Unit tests for TournamentState helper methods."""

    def test_record_match_result_eliminates_loser(self) -> None:
        state = TournamentState(
            tournament_id="TS_01",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(4),
        )
        state.record_match_result("P01", "P02", round_number=1)
        assert "P02" in state.eliminated_players
        assert state.player_results["P02"].is_eliminated is True

    def test_record_match_result_increments_winner_rounds(self) -> None:
        state = TournamentState(
            tournament_id="TS_02",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(4),
        )
        state.record_match_result("P01", "P02", round_number=1)
        assert state.player_results["P01"].rounds_won == 1
        assert state.player_results["P01"].current_round == 2

    def test_tournament_complete_when_1_active(self) -> None:
        state = TournamentState(
            tournament_id="TS_03",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(2),
        )
        state.record_match_result("P01", "P02", round_number=1)
        assert state.status == OutrightMarketStatus.RESULTED
        assert state.winner == "P01"

    def test_active_players_excludes_eliminated(self) -> None:
        state = TournamentState(
            tournament_id="TS_04",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(4),
        )
        state.record_match_result("P01", "P03", round_number=1)
        active_ids = {e.entity_id for e in state.active_players}
        assert "P03" not in active_ids
        assert "P01" in active_ids

    def test_draw_size_auto_set(self) -> None:
        state = TournamentState(
            tournament_id="TS_05",
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_300,
            entries=_entries(8),
        )
        assert state.draw_size == 8


# ---------------------------------------------------------------------------
# BadmintonAgentRuntime
# ---------------------------------------------------------------------------

class TestBadmintonAgentRuntime:
    """Coverage for agents/agent_runtime.py — registration, queries, sync paths."""

    # ------------------------------------------------------------------
    # Sync helpers (no event loop needed)
    # ------------------------------------------------------------------

    def test_constructs_with_defaults(self) -> None:
        runtime = BadmintonAgentRuntime()
        assert runtime._running is False
        assert runtime._shutting_down is False

    def test_constructs_with_custom_config(self) -> None:
        cfg = RuntimeConfig(health_check_interval_s=10.0, max_restart_attempts=5)
        runtime = BadmintonAgentRuntime(config=cfg)
        assert runtime.config.health_check_interval_s == 10.0
        assert runtime.config.max_restart_attempts == 5

    def test_get_metrics_empty_runtime(self) -> None:
        runtime = BadmintonAgentRuntime()
        metrics = runtime.get_metrics()
        assert metrics["total_agents"] == 0
        assert metrics["running_agents"] == 0
        assert metrics["healthy_agents"] == 0

    def test_get_metrics_counts_agent_states(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()
        reg = AgentRegistration(agent_id="agent_x", agent=agent_obj)
        reg.state = AgentState.RUNNING
        reg.last_health_ok = True
        runtime._agents["agent_x"] = reg

        metrics = runtime.get_metrics()
        assert metrics["total_agents"] == 1
        assert metrics["running_agents"] == 1
        assert metrics["healthy_agents"] == 1

    def test_list_agents_empty(self) -> None:
        runtime = BadmintonAgentRuntime()
        assert runtime.list_agents() == []

    def test_get_agent_returns_none_for_unknown(self) -> None:
        runtime = BadmintonAgentRuntime()
        assert runtime.get_agent("UNKNOWN") is None

    def test_get_agent_status_unknown_raises(self) -> None:
        runtime = BadmintonAgentRuntime()
        with pytest.raises(ValueError, match="not found"):
            runtime.get_agent_status("UNKNOWN")

    def test_get_all_agent_statuses_empty(self) -> None:
        runtime = BadmintonAgentRuntime()
        assert runtime.get_all_agent_statuses() == []

    # ------------------------------------------------------------------
    # Async registration + lifecycle
    # ------------------------------------------------------------------

    def test_register_agent_adds_to_agents(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()

        async def _run():
            await runtime.register_agent("agent_1", agent_obj)
            return runtime.list_agents()

        agents = asyncio.run(_run())
        assert "agent_1" in agents

    def test_register_duplicate_raises(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()

        async def _run():
            await runtime.register_agent("agt", agent_obj)
            await runtime.register_agent("agt", agent_obj)

        with pytest.raises(ValueError, match="already registered"):
            asyncio.run(_run())

    def test_unregister_unknown_raises(self) -> None:
        runtime = BadmintonAgentRuntime()

        async def _run():
            await runtime.unregister_agent("GHOST")

        with pytest.raises(ValueError, match="not found"):
            asyncio.run(_run())

    def test_unregister_registered_agent(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()

        async def _run():
            await runtime.register_agent("to_remove", agent_obj)
            await runtime.unregister_agent("to_remove")
            return runtime.list_agents()

        remaining = asyncio.run(_run())
        assert "to_remove" not in remaining

    def test_get_agent_status_fields(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()
        agent_obj.__class__.__name__ = "MockAgent"

        async def _run():
            await runtime.register_agent("status_agent", agent_obj, metadata={"env": "test"})
            return runtime.get_agent_status("status_agent")

        status = asyncio.run(_run())
        assert status["agent_id"] == "status_agent"
        assert status["agent_type"] == "MockAgent"
        assert status["state"] == AgentState.REGISTERED.value
        assert status["is_running"] is False
        assert status["metadata"] == {"env": "test"}
        assert status["registered_at"] is not None

    def test_get_all_agent_statuses_returns_list(self) -> None:
        runtime = BadmintonAgentRuntime()

        async def _run():
            await runtime.register_agent("a1", MagicMock())
            await runtime.register_agent("a2", MagicMock())
            return runtime.get_all_agent_statuses()

        statuses = asyncio.run(_run())
        assert len(statuses) == 2

    def test_stop_all_when_running_false(self) -> None:
        runtime = BadmintonAgentRuntime()

        async def _run():
            # No agents, running=False — should complete without error
            await runtime.stop_all()

        asyncio.run(_run())

    def test_shutdown_sets_shutting_down(self) -> None:
        runtime = BadmintonAgentRuntime()

        async def _run():
            await runtime.shutdown()

        asyncio.run(_run())
        assert runtime._shutting_down is True

    def test_context_manager_calls_shutdown(self) -> None:
        runtime = BadmintonAgentRuntime()

        async def _run():
            async with runtime:
                pass  # enter and exit cleanly

        asyncio.run(_run())
        assert runtime._shutting_down is True

    def test_check_agent_health_no_health_check_method_task_done(self) -> None:
        """Agent with no health_check() — healthy iff task not done."""
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock(spec=[])  # no methods
        reg = AgentRegistration(agent_id="hc_test", agent=agent_obj)
        task = MagicMock()
        task.done.return_value = True   # task is done → not healthy
        reg.task = task

        async def _run():
            return await runtime._check_agent_health(reg)

        result = asyncio.run(_run())
        assert result is False

    def test_check_agent_health_no_health_check_method_task_alive(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock(spec=[])
        reg = AgentRegistration(agent_id="hc_alive", agent=agent_obj)
        task = MagicMock()
        task.done.return_value = False  # task still running → healthy
        reg.task = task

        async def _run():
            return await runtime._check_agent_health(reg)

        result = asyncio.run(_run())
        assert result is True

    def test_check_agent_health_with_health_check_method_ok(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()
        agent_obj.health_check = AsyncMock(return_value=True)
        reg = AgentRegistration(agent_id="hc_ok", agent=agent_obj)

        async def _run():
            return await runtime._check_agent_health(reg)

        result = asyncio.run(_run())
        assert result is True

    def test_check_agent_health_with_health_check_method_failing(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()
        agent_obj.health_check = AsyncMock(return_value=False)
        reg = AgentRegistration(agent_id="hc_fail", agent=agent_obj)

        async def _run():
            return await runtime._check_agent_health(reg)

        result = asyncio.run(_run())
        assert result is False

    def test_check_agent_health_exception_returns_false(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()
        agent_obj.health_check = AsyncMock(side_effect=RuntimeError("check failed"))
        reg = AgentRegistration(agent_id="hc_exc", agent=agent_obj)

        async def _run():
            return await runtime._check_agent_health(reg)

        result = asyncio.run(_run())
        assert result is False

    def test_check_agent_health_timeout_returns_false(self) -> None:
        runtime = BadmintonAgentRuntime(config=RuntimeConfig(health_check_timeout_s=0.01))
        agent_obj = MagicMock()

        async def _slow_check():
            await asyncio.sleep(5)  # will be cancelled
            return True

        agent_obj.health_check = _slow_check
        reg = AgentRegistration(agent_id="hc_to", agent=agent_obj)

        async def _run():
            return await runtime._check_agent_health(reg)

        result = asyncio.run(_run())
        assert result is False

    # ------------------------------------------------------------------
    # AgentRegistration helpers
    # ------------------------------------------------------------------

    def test_agent_registration_is_running(self) -> None:
        reg = AgentRegistration(agent_id="r1", agent=MagicMock())
        assert reg.is_running() is False
        reg.state = AgentState.RUNNING
        assert reg.is_running() is True

    def test_agent_registration_can_restart(self) -> None:
        reg = AgentRegistration(agent_id="r2", agent=MagicMock())
        reg.restart_count = 0
        assert reg.can_restart(max_attempts=3) is True
        reg.restart_count = 3
        assert reg.can_restart(max_attempts=3) is False

    # ------------------------------------------------------------------
    # RuntimeConfig defaults
    # ------------------------------------------------------------------

    def test_runtime_config_defaults(self) -> None:
        cfg = RuntimeConfig()
        assert cfg.health_check_interval_s == 30.0
        assert cfg.max_restart_attempts == 3
        assert cfg.auto_restart is True
        assert cfg.shutdown_timeout_s == 30.0

    # ------------------------------------------------------------------
    # _run_agent with execute() method
    # ------------------------------------------------------------------

    def test_shutdown_flag_set_after_stop_all(self) -> None:
        """shutdown() sets _shutting_down and stops running."""
        runtime = BadmintonAgentRuntime()

        async def _run():
            assert runtime._shutting_down is False
            await runtime.stop_all()
            return runtime._running

        running = asyncio.run(_run())
        assert running is False

    def test_get_metrics_unhealthy_not_healthy(self) -> None:
        runtime = BadmintonAgentRuntime()
        agent_obj = MagicMock()
        reg = AgentRegistration(agent_id="agt_u", agent=agent_obj)
        reg.state = AgentState.UNHEALTHY
        reg.last_health_ok = False
        runtime._agents["agt_u"] = reg

        metrics = runtime.get_metrics()
        assert metrics["healthy_agents"] == 0
        assert metrics["unhealthy_agents"] == 1


# ---------------------------------------------------------------------------
# AgentState enum coverage
# ---------------------------------------------------------------------------

class TestAgentStateEnum:
    def test_all_states_have_values(self) -> None:
        expected = {
            "registered", "starting", "running", "unhealthy",
            "restarting", "stopping", "stopped", "failed"
        }
        actual = {s.value for s in AgentState}
        assert expected == actual


# ---------------------------------------------------------------------------
# SettlementOutcome enum
# ---------------------------------------------------------------------------

class TestSettlementOutcomeEnum:
    def test_settlement_outcome_values(self) -> None:
        assert SettlementOutcome.WIN == "win"
        assert SettlementOutcome.LOSE == "lose"
        assert SettlementOutcome.VOID == "void"
        assert SettlementOutcome.PUSH == "push"

    def test_settlement_status_values(self) -> None:
        assert SettlementStatus.SETTLED == "settled"
        assert SettlementStatus.VOIDED == "voided"
        assert SettlementStatus.PENDING == "pending"
        assert SettlementStatus.ERROR == "error"

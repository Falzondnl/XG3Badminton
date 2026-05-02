"""
test_integration_full_pipeline.py
==================================
End-to-end integration tests for the full badminton pricing pipeline.

Tests the complete flow from match registration through to live pricing:
  1. Match state initialisation
  2. Point-by-point live scoring
  3. Bayesian RWP updates
  4. Momentum detection
  5. Live market generation
  6. Market validation (H7 + H10)
  7. Settlement grading

These tests do NOT require BADMINTON_DATA_ROOT or trained models.
All probabilities are computed from Markov engine using test RWP values.

ZERO hardcoded expected probability values — all invariants are relational.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from core.match_state import BadmintonMatchStateMachine, PointWinner, MatchStatus
from core.bayesian_updater import BayesianRWPUpdater
from core.momentum_detector import MomentumDetector
from core.markov_engine import BadmintonMarkovEngine, clear_markov_cache
from markets.derivative_engine import BadmintonDerivativeEngine


class TestFullLivePipeline:
    """End-to-end live match pricing pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up shared objects for each test."""
        clear_markov_cache()
        self.discipline = Discipline.MS
        self.rwp_a = 0.535
        self.rwp_b = 0.529
        self.match_id = "integ_test_001"

        self.state = BadmintonMatchStateMachine.initialise(
            match_id=self.match_id,
            entity_a_id="axelsen",
            entity_b_id="lee_zii_jia",
            discipline=self.discipline,
            first_server="A",
        )

        self.bayesian = BayesianRWPUpdater(
            match_id=self.match_id,
            entity_a_id="axelsen",
            entity_b_id="lee_zii_jia",
            discipline=self.discipline,
            rwp_prior_a=self.rwp_a,
            rwp_prior_b=self.rwp_b,
        )

        self.momentum = MomentumDetector(
            match_id=self.match_id,
            rwp_a=self.rwp_a,
            rwp_b=self.rwp_b,
            discipline_value=self.discipline.value,
        )

        self.markov = BadmintonMarkovEngine()
        self.derivative_engine = BadmintonDerivativeEngine()

    def _play_sequence(self, winners: str):
        """Play a sequence of points. 'A' or 'B' for each point."""
        for i, w in enumerate(winners):
            winner_enum = PointWinner(w)
            server = self.state.server
            score_a = self.state.score_a + (1 if w == "A" else 0)
            score_b = self.state.score_b + (1 if w == "B" else 0)

            self.state = BadmintonMatchStateMachine.apply_point(
                self.state, winner_enum
            )

            self.bayesian.observe_rally(
                server=server,
                winner=w,
                game_number=self.state.current_game,
                point_index=i,
            )

            self.momentum.add_point(
                winner=w,
                server=server,
                score_a=self.state.score_a,
                score_b=self.state.score_b,
                game_number=self.state.current_game,
            )

    def test_initial_markov_consistent(self):
        """Pre-game Markov probabilities are internally consistent."""
        probs = self.markov.compute_match_probabilities(
            rwp_a=self.rwp_a,
            rwp_b=self.rwp_b,
            discipline=self.discipline,
            server_first_game="A",
        )
        total = (
            probs.p_a_wins_2_0 + probs.p_a_wins_2_1 +
            probs.p_b_wins_2_0 + probs.p_b_wins_2_1
        )
        assert abs(total - 1.0) < 1e-6

    def test_5_points_played_state_correct(self):
        """After 5 points, state reflects correct scores."""
        self._play_sequence("AABAB")
        assert self.state.total_points_played == 5
        assert self.state.score_a + self.state.score_b == 5
        assert self.state.current_game == 1
        assert self.state.status == MatchStatus.IN_PROGRESS

    def test_bayesian_updates_after_points(self):
        """Bayesian updater shows evidence weight after points."""
        self._play_sequence("AAABBB")
        est_a = self.bayesian.get_live_rwp("A")
        assert 0 < est_a.evidence_weight <= 1.0
        assert 0.0 < est_a.rwp_live < 1.0

    def test_momentum_tracking_run(self):
        """Momentum detector tracks a run of 3+ by A."""
        self._play_sequence("AAA")
        snapshot = self.momentum.get_last_snapshot()
        assert snapshot is not None
        assert snapshot.current_run_a >= 3
        assert snapshot.current_run_b == 0

    def test_momentum_resets_after_switch(self):
        """Run resets when B wins a point."""
        self._play_sequence("AAAB")
        snapshot = self.momentum.get_last_snapshot()
        assert snapshot.current_run_a == 0
        assert snapshot.current_run_b == 1

    def test_game_completion_transitions_correctly(self):
        """After A wins 21 points, game transitions correctly."""
        # A wins 21-0 in game 1
        self._play_sequence("A" * 21)
        assert self.state.games_won_a == 1
        assert self.state.games_won_b == 0
        assert self.state.current_game == 2
        assert self.state.score_a == 0
        assert self.state.score_b == 0

    def test_c04_winner_serves_in_game_2(self):
        """C-04: A wins game 1 → A serves first in game 2."""
        self._play_sequence("A" * 21)
        assert self.state.server == "A"

    def test_c04_loser_serves_when_b_wins_game(self):
        """C-04: B wins game 1 → B serves first in game 2."""
        self._play_sequence("B" * 21)
        assert self.state.server == "B"

    def test_full_match_2_0(self):
        """Full 2-0 match completion."""
        self._play_sequence("A" * 21 + "A" * 21)
        assert self.state.status == MatchStatus.COMPLETED
        assert self.state.match_winner == "A"
        assert self.state.games_won_a == 2
        assert self.state.games_won_b == 0

    def test_full_match_2_1(self):
        """Full 2-1 match completion."""
        self._play_sequence("A" * 21 + "B" * 21 + "A" * 21)
        assert self.state.status == MatchStatus.COMPLETED
        assert self.state.match_winner == "A"
        assert self.state.games_won_a == 2
        assert self.state.games_won_b == 1

    def test_live_markov_after_points(self):
        """Live Markov computation with updated score."""
        self._play_sequence("A" * 10)  # A leads 10-0

        est_a = self.bayesian.get_live_rwp("A")
        probs = self.markov.compute_match_probabilities(
            rwp_a=est_a.rwp_live,
            rwp_b=self.bayesian.get_live_rwp("B").rwp_live,
            discipline=self.discipline,
            server_first_game=self.state.server,
            score_a=self.state.score_a,
            score_b=self.state.score_b,
            games_won_a=self.state.games_won_a,
            games_won_b=self.state.games_won_b,
            current_game=self.state.current_game,
        )

        # A is leading 10-0, should have > 50% match win prob
        assert probs.p_a_wins_match > 0.50

        # All invariants still hold
        total = (
            probs.p_a_wins_2_0 + probs.p_a_wins_2_1 +
            probs.p_b_wins_2_0 + probs.p_b_wins_2_1
        )
        assert abs(total - 1.0) < 1e-6

    def test_markets_generated_midgame(self):
        """Derivative markets generate correctly mid-game."""
        self._play_sequence("A" * 10 + "B" * 5)  # Score A:10, B:5

        est_a = self.bayesian.get_live_rwp("A")
        ms = self.derivative_engine.compute_all_markets(
            match_id=self.match_id,
            rwp=est_a.rwp_live,
            discipline=self.discipline,
            tier=TournamentTier.SUPER_500,
            p_match_win=0.60,
            server_first_game=self.state.server,
        )

        assert len(ms.markets) > 0

        # All odds >= 1.01 (H10)
        for market_id, prices in ms.markets.items():
            for p in prices:
                assert p.odds >= 1.01, f"H10 fail: {market_id}: {p.odds}"

        # All markets arbitrage-free (H7)
        for market_id, prices in ms.markets.items():
            if len(prices) >= 2:
                total = sum(p.prob_with_margin for p in prices)
                assert total >= 1.0, f"H7 fail: {market_id}: {total}"


class TestSettlementIntegration:
    """Integration tests for settlement after match completion."""

    def test_settle_2_0_match(self):
        """Settlement of a 2-0 match — match winner and correct score."""
        from core.match_state import BadmintonMatchStateMachine, MatchStatus
        from settlement.grading_service import GradingService, MatchResult

        state = BadmintonMatchStateMachine.initialise(
            match_id="settle_test_001",
            entity_a_id="player_a",
            entity_b_id="player_b",
            discipline=Discipline.MS,
            first_server="A",
        )

        # Play 2-0 match
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        for _ in range(21):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)

        assert state.status == MatchStatus.COMPLETED
        assert state.match_winner == "A"

        result = MatchResult.from_live_state(state)
        assert result.winner == "A"
        assert result.games_won_a == 2
        assert result.games_won_b == 0

        grading = GradingService()
        open_markets = {
            "match_winner": ["player_a", "player_b"],
            "correct_score": ["A_2-0", "A_2-1", "B_2-0", "B_2-1"],
            "total_games_over_2.5": ["Over 2.5", "Under 2.5"],
        }
        records = grading.settle_match(state, open_markets)

        assert len(records) == 3

        mw_record = next(r for r in records if r.market_id == "match_winner")
        assert mw_record.winning_outcome == "player_a"

        cs_record = next(r for r in records if r.market_id == "correct_score")
        assert cs_record.winning_outcome == "A_2-0"

        total_record = next(r for r in records if "total_games" in r.market_id)
        assert total_record.winning_outcome == "Under 2.5"

    def test_retirement_voids_correct_score(self):
        """Retirement during game 1 voids correct score market."""
        from settlement.void_rules import RetirementVoidRules
        from settlement.grading_service import MatchResult, SettlementStatus
        from core.match_state import MatchStatus

        state = BadmintonMatchStateMachine.initialise(
            match_id="retire_test",
            entity_a_id="p_a",
            entity_b_id="p_b",
            discipline=Discipline.MS,
            first_server="A",
        )

        # Play 10 points then A retires
        for _ in range(5):
            state = BadmintonMatchStateMachine.apply_point(state, PointWinner.A)
        state = BadmintonMatchStateMachine.apply_retirement(state, retiring_entity="A")

        result = MatchResult.from_live_state(state)
        assert result.is_retired

        # Correct score should be voided
        status, winner, reason = RetirementVoidRules.apply(
            "correct_score", result, ["A_2-0", "A_2-1", "B_2-0", "B_2-1"]
        )
        assert status == SettlementStatus.VOIDED
        assert winner is None

        # Match winner should be settled
        status_mw, winner_mw, _ = RetirementVoidRules.apply(
            "match_winner", result, ["p_a", "p_b"]
        )
        assert status_mw == SettlementStatus.SETTLED
        assert winner_mw == "p_b"  # B wins since A retired

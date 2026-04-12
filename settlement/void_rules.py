"""
void_rules.py
=============
Void/settlement rules for retirement, walkover, and special cases.

BWF and industry standard void rules:
  1. WALKOVER (pre-match):
     - Match Winner: VOID
     - All other markets: VOID
     - Exception: some operators settle Match Winner at 1/1 (even money)
       to the player who was going to play — we VOID per BWF standard.

  2. RETIREMENT (mid-match):
     - Match Winner: SETTLED to non-retiring player
     - Correct Score: VOID
     - Total Games (O/U 2.5):
       - If match completed at 2-0: SETTLED (no retirement affected 3rd game)
       - If match retired during 3rd game: VOID
     - Game N Winner:
       - Games COMPLETED before retirement: SETTLED
       - Current game (if incomplete): VOID
       - Future games: VOID
     - Points totals: VOID unless match completed normally
     - SGP: VOID if any component can't be settled

  3. ABANDONMENT (external reasons — weather, venue issue):
     - All markets: VOID unless match could be completed as scheduled

  4. SPECIAL CASES:
     - Both players unable to compete: all VOID
     - Wrong player listed (admin error): depends on timing — see notes
     - Player DQ after match: settle as VOID (result reversed)

"Completed" definition:
  A market settles (not voided) if the segment it covers has concluded
  according to BWF rules — regardless of retirement after that segment.

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import GAMES_TO_WIN_MATCH
from core.match_state import MatchStatus
from settlement.grading_service import MatchResult, SettlementStatus

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Void rule engine
# ---------------------------------------------------------------------------

class RetirementVoidRules:
    """
    Determines void status for each market when a match ends by retirement.

    Returns settlement disposition: SETTLED, VOIDED, or PUSHED.
    """

    @staticmethod
    def apply(
        market_id: str,
        result: MatchResult,
        outcomes: List[str],
    ) -> Tuple[SettlementStatus, Optional[str], Optional[str]]:
        """
        Apply void rules to a market.

        Returns: (status, winning_outcome, void_reason)
        """
        if not result.is_retired:
            # Not retired — normal settlement applies
            return SettlementStatus.SETTLED, None, None

        # Match Winner: always settled (non-retiring player wins)
        if market_id == "match_winner":
            winner = result.entity_a_id if result.winner == "A" else result.entity_b_id
            return SettlementStatus.SETTLED, winner, None

        # Correct score: VOID on retirement (match incomplete)
        if market_id == "correct_score":
            return SettlementStatus.VOIDED, None, "retirement_correct_score_void"

        # Total games O/U: SETTLED only if the outcome is unambiguous
        if "total_games" in market_id:
            return RetirementVoidRules._settle_total_games_on_retirement(
                market_id, result
            )

        # Game N winner: SETTLED for completed games, VOID for incomplete
        if market_id.startswith("game_") and "_winner" in market_id:
            return RetirementVoidRules._settle_game_winner_on_retirement(
                market_id, result
            )

        # Game N total points: SETTLED for completed games only
        if market_id.startswith("game_") and "_total" in market_id:
            return RetirementVoidRules._settle_game_total_on_retirement(
                market_id, result
            )

        # Race-to-N: SETTLED for completed games, VOID for others
        if market_id.startswith("race_to_"):
            return RetirementVoidRules._settle_race_on_retirement(
                market_id, result
            )

        # All other markets: VOID on retirement
        return SettlementStatus.VOIDED, None, "retirement_market_void"

    @staticmethod
    def _settle_total_games_on_retirement(
        market_id: str,
        result: MatchResult,
    ) -> Tuple[SettlementStatus, Optional[str], Optional[str]]:
        """Total games market retirement handling."""
        games_played = result.total_games_played

        # If 2-0 completed (both games done): can settle O/U 2.5 (it's Under)
        if games_played == 2 and result.games_won_a + result.games_won_b == 2:
            if "over_2.5" in market_id:
                return SettlementStatus.SETTLED, "Under 2.5", None
            elif "under_2.5" in market_id:
                return SettlementStatus.SETTLED, "Under 2.5", None
            return SettlementStatus.SETTLED, "Under 2.5", None

        # Games in progress or 1 game played: VOID
        return SettlementStatus.VOIDED, None, "retirement_total_games_incomplete"

    @staticmethod
    def _settle_game_winner_on_retirement(
        market_id: str,
        result: MatchResult,
    ) -> Tuple[SettlementStatus, Optional[str], Optional[str]]:
        """Game N winner settlement on retirement."""
        parts = market_id.split("_")
        try:
            game_n = int(parts[1])
        except (ValueError, IndexError):
            return SettlementStatus.VOIDED, None, f"invalid_market_id_{market_id}"

        if game_n <= len(result.game_scores):
            # Game completed — settle normally
            sa, sb = result.game_scores[game_n - 1]
            entity = result.entity_a_id if sa > sb else result.entity_b_id
            return SettlementStatus.SETTLED, entity, None

        # Game not completed
        return SettlementStatus.VOIDED, None, f"game_{game_n}_incomplete_on_retirement"

    @staticmethod
    def _settle_game_total_on_retirement(
        market_id: str,
        result: MatchResult,
    ) -> Tuple[SettlementStatus, Optional[str], Optional[str]]:
        """Game total points settlement on retirement."""
        parts = market_id.split("_")
        try:
            game_n = int(parts[1])
            threshold = int(parts[-1].replace("o", ""))
        except (ValueError, IndexError):
            return SettlementStatus.VOIDED, None, f"invalid_market_id_{market_id}"

        if game_n <= len(result.game_scores):
            sa, sb = result.game_scores[game_n - 1]
            total = sa + sb
            winner = f"Over {threshold}" if total > threshold else f"Under {threshold}"
            return SettlementStatus.SETTLED, winner, None

        return SettlementStatus.VOIDED, None, f"game_{game_n}_not_completed"

    @staticmethod
    def _settle_race_on_retirement(
        market_id: str,
        result: MatchResult,
    ) -> Tuple[SettlementStatus, Optional[str], Optional[str]]:
        """Race-to-N settlement on retirement."""
        parts = market_id.split("_")
        try:
            n = int(parts[2])
            game_str = parts[-1]
            game_n = int(game_str.replace("game", ""))
        except (ValueError, IndexError):
            return SettlementStatus.VOIDED, None, f"invalid_race_market_{market_id}"

        if game_n <= len(result.game_scores):
            sa, sb = result.game_scores[game_n - 1]
            if sa >= n:
                return SettlementStatus.SETTLED, result.entity_a_id, None
            elif sb >= n:
                return SettlementStatus.SETTLED, result.entity_b_id, None

        return SettlementStatus.VOIDED, None, f"race_n_{n}_not_reached"


class WalkoverVoidRules:
    """Void rules for pre-match walkover."""

    @staticmethod
    def void_all(
        market_id_or_match_id,
        result_or_open_markets,
        outcomes: Optional[List[str]] = None,
    ):
        """
        Void all open markets for a walkover.

        Two calling conventions are accepted:

        Batch form (original)::
            void_all(match_id: str, open_markets: Dict[str, List[str]])
            → Dict[str, Tuple[SettlementStatus, Optional[str]]]

        Per-market form (used by tests)::
            void_all(market_id: str, result: MatchResult, outcomes: List[str])
            → Tuple[SettlementStatus, Optional[str], Optional[str]]
               (status, winning_outcome, void_reason)
        """
        # Detect per-market form: second arg is a MatchResult (not a dict)
        if outcomes is not None or not isinstance(result_or_open_markets, dict):
            # Per-market form
            status = SettlementStatus.VOIDED
            void_reason = "walkover"
            logger.info(
                "walkover_void_market",
                market_id=market_id_or_match_id,
            )
            return (status, None, void_reason)

        # Batch form (original behaviour)
        match_id = market_id_or_match_id
        open_markets = result_or_open_markets
        void_result = {}
        for mid in open_markets:
            void_result[mid] = (SettlementStatus.VOIDED, "walkover")

        logger.info(
            "walkover_void_all",
            match_id=match_id,
            n_markets_voided=len(void_result),
        )
        return void_result

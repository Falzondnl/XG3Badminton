"""
risk/cashout_calculator.py
==========================
CashoutCalculator — Live cashout value computation.

Algorithm:
  1. Get current live match probability P(bettor_wins) from Markov engine
     evaluated at the current live state (games, scores, server).
  2. Compare to original bet probability (derived from bet's decimal odds).
  3. Cashout value = stake × (current_prob / original_prob) × (1 - commission).

Commission schedule:
  - Standard bettors: 3%
  - Premium bettors: 1.5%

Edge cases:
  - Match completed → full win payout or £0 (no partial cashout)
  - Suspended markets → cashout FROZEN at last-known value (not stale Markov)
  - Odds-on outcomes where current_prob > original_prob → positive cashout uplift
  - Minimum cashout floor: £0.01 (never negative)

Integrates with:
  - BadmintonMarkovEngine (live probability computation)
  - ExposureManager (bet records carry original stake + odds)
  - MatchLiveState (current game scores/server fed to Markov)

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import structlog

from config.badminton_config import Discipline
from core.markov_engine import BadmintonMarkovEngine
from core.match_state import MatchLiveState, MatchStatus
from risk.exposure_manager import BetRecord

logger = structlog.get_logger(__name__)

# Commission schedule
_STANDARD_COMMISSION = 0.03   # 3%
_PREMIUM_COMMISSION = 0.015   # 1.5%
_MIN_CASHOUT_GBP = 0.01       # Floor — never return negative


class CashoutError(RuntimeError):
    """Raised when cashout cannot be computed."""


@dataclass(frozen=True)
class CashoutResult:
    """Result of a cashout computation."""
    bet_id: str
    cashout_value_gbp: float
    original_stake_gbp: float
    original_odds: float
    original_implied_prob: float
    current_win_prob: float
    commission_rate: float
    is_frozen: bool          # True if match is suspended → last known value
    match_completed: bool    # True if match already ended


class CashoutCalculator:
    """
    Computes live cashout values for accepted bets.

    One instance per match.
    """

    def __init__(
        self,
        match_id: str,
        discipline: Discipline,
        markov: Optional[BadmintonMarkovEngine] = None,
    ) -> None:
        self._match_id = match_id
        self._discipline = discipline
        self._markov = markov or BadmintonMarkovEngine()
        # Cache last known win probability per outcome key for suspended markets
        self._last_known_prob: dict[str, float] = {}

    def compute(
        self,
        bet: BetRecord,
        live_state: MatchLiveState,
        outcome_is_player_a: bool,
        is_premium_bettor: bool = False,
    ) -> CashoutResult:
        """
        Compute cashout value for a single accepted bet.

        Args:
            bet:                  Original BetRecord from ExposureManager.
            live_state:           Current authoritative MatchLiveState.
            outcome_is_player_a:  True if the bet is on Player A winning the match.
            is_premium_bettor:    Apply reduced commission (1.5%).

        Returns:
            CashoutResult with cashout_value_gbp and metadata.

        Raises:
            CashoutError if Markov computation fails.
        """
        if bet.match_id != self._match_id:
            raise CashoutError(
                f"Bet match_id={bet.match_id!r} does not match "
                f"calculator match_id={self._match_id!r}"
            )

        commission = _PREMIUM_COMMISSION if is_premium_bettor else _STANDARD_COMMISSION
        outcome_key = f"{bet.market_id}:{bet.outcome_name}"

        # --- Match completed → binary outcome ---
        if live_state.status == MatchStatus.COMPLETED:
            winner = live_state.match_winner
            if winner is None:
                raise CashoutError(
                    f"match status=COMPLETED but match_winner is None "
                    f"for match_id={self._match_id!r}"
                )
            bettor_won = (outcome_is_player_a and winner == "A") or (
                not outcome_is_player_a and winner == "B"
            )
            cashout_value = bet.potential_payout_gbp if bettor_won else 0.0
            return CashoutResult(
                bet_id=bet.bet_id,
                cashout_value_gbp=round(cashout_value, 2),
                original_stake_gbp=bet.stake_gbp,
                original_odds=bet.decimal_odds,
                original_implied_prob=1.0 / bet.decimal_odds,
                current_win_prob=1.0 if bettor_won else 0.0,
                commission_rate=commission,
                is_frozen=False,
                match_completed=True,
            )

        # --- Suspended → return frozen value ---
        is_frozen = live_state.status == MatchStatus.SUSPENDED
        if is_frozen:
            frozen_prob = self._last_known_prob.get(outcome_key)
            if frozen_prob is None:
                raise CashoutError(
                    f"market suspended and no prior probability cached for "
                    f"outcome_key={outcome_key!r} — cashout unavailable"
                )
            current_prob = frozen_prob
            logger.info(
                "cashout_frozen",
                match_id=self._match_id,
                bet_id=bet.bet_id,
                frozen_prob=round(current_prob, 4),
            )
        else:
            # --- Live pricing via Markov ---
            current_prob = self._compute_live_win_prob(
                live_state=live_state,
                outcome_is_player_a=outcome_is_player_a,
            )
            # Cache for suspension fallback
            self._last_known_prob[outcome_key] = current_prob

        # --- Cashout formula ---
        original_prob = 1.0 / bet.decimal_odds
        cashout_raw = bet.stake_gbp * (current_prob / original_prob) * (1.0 - commission)
        cashout_value = max(_MIN_CASHOUT_GBP, round(cashout_raw, 2))

        logger.debug(
            "cashout_computed",
            match_id=self._match_id,
            bet_id=bet.bet_id,
            original_odds=bet.decimal_odds,
            original_prob=round(original_prob, 4),
            current_prob=round(current_prob, 4),
            stake=bet.stake_gbp,
            cashout_value=cashout_value,
            commission=commission,
            is_frozen=is_frozen,
        )

        return CashoutResult(
            bet_id=bet.bet_id,
            cashout_value_gbp=cashout_value,
            original_stake_gbp=bet.stake_gbp,
            original_odds=bet.decimal_odds,
            original_implied_prob=round(original_prob, 6),
            current_win_prob=round(current_prob, 6),
            commission_rate=commission,
            is_frozen=is_frozen,
            match_completed=False,
        )

    def _compute_live_win_prob(
        self,
        live_state: MatchLiveState,
        outcome_is_player_a: bool,
    ) -> float:
        """
        Run Markov engine from current live state to get P(outcome_wins_match).

        Raises:
            CashoutError on Markov failure.
        """
        if live_state.rwp_a is None or live_state.rwp_b is None:
            raise CashoutError(
                f"live_state missing rwp_a/rwp_b for match_id={self._match_id!r} — "
                "Bayesian updater not initialised"
            )

        try:
            match_probs = self._markov.compute_match_probabilities(
                rwp_a=live_state.rwp_a,
                rwp_b=live_state.rwp_b,
                discipline=self._discipline,
                server_first_game=live_state.current_server,
                games_won_a=live_state.games_won_a,
                games_won_b=live_state.games_won_b,
                score_a_current_game=live_state.score_a,
                score_b_current_game=live_state.score_b,
            )
        except Exception as exc:
            raise CashoutError(
                f"Markov engine failed for match_id={self._match_id!r}: {exc}"
            ) from exc

        return (
            match_probs.p_a_wins_match
            if outcome_is_player_a
            else match_probs.p_b_wins_match
        )

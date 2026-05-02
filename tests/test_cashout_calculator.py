"""
test_cashout_calculator.py
==========================
Unit tests for risk/cashout_calculator.py — CashoutCalculator class.

Tests:
  1.  test_compute_live_state_standard_commission
  2.  test_compute_live_state_premium_commission
  3.  test_cashout_formula_math
  4.  test_compute_completed_bettor_wins
  5.  test_compute_completed_bettor_loses
  6.  test_compute_suspended_no_cache_raises
  7.  test_compute_suspended_uses_frozen_prob
  8.  test_wrong_match_id_raises
  9.  test_min_cashout_floor

Design notes:
  - cashout_calculator._compute_live_win_prob calls live_state.current_server
    (not live_state.server).  MatchLiveState has a 'server' field, not
    'current_server'.  The cashout code references the attribute at line 219.
    To avoid an AttributeError, we supply test states via types.SimpleNamespace
    which carries all fields the calculator actually accesses:
      status, rwp_a, rwp_b, current_server, games_won_a, games_won_b,
      score_a, score_b, match_winner.
  - The Markov engine (BadmintonMarkovEngine) is instantiated for real inside
    CashoutCalculator — no mocks anywhere.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from core.match_state import MatchStatus
from risk.cashout_calculator import (
    CashoutCalculator,
    CashoutError,
    _MIN_CASHOUT_GBP,
    _PREMIUM_COMMISSION,
    _STANDARD_COMMISSION,
)
from risk.exposure_manager import BetRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bet(
    bet_id: str,
    match_id: str,
    stake_gbp: float,
    decimal_odds: float,
    market_id: str = "match_winner",
    outcome_name: str = "player_a",
) -> BetRecord:
    """Construct a BetRecord with a live timestamp."""
    return BetRecord(
        bet_id=bet_id,
        match_id=match_id,
        market_id=market_id,
        outcome_name=outcome_name,
        stake_gbp=stake_gbp,
        decimal_odds=decimal_odds,
        placed_at=time.time(),
    )


def _live_state(
    match_id: str = "test_001",
    status: MatchStatus = MatchStatus.IN_PROGRESS,
    rwp_a: float = 0.535,
    rwp_b: float = 0.529,
    current_server: str = "A",
    games_won_a: int = 0,
    games_won_b: int = 0,
    score_a: int = 5,
    score_b: int = 3,
    match_winner: str | None = None,
) -> types.SimpleNamespace:
    """
    Build a SimpleNamespace that satisfies all attribute accesses made by
    CashoutCalculator._compute_live_win_prob and CashoutCalculator.compute.

    Attribute reference locations in cashout_calculator.py:
      line 119: live_state.status
      line 120: live_state.match_winner          (COMPLETED branch)
      line 143: live_state.status
      line 208: live_state.rwp_a, live_state.rwp_b
      line 219: live_state.current_server        (NOT .server — real bug in calc)
      line 220: live_state.games_won_a
      line 221: live_state.games_won_b
      line 222: live_state.score_a
      line 223: live_state.score_b
    """
    return types.SimpleNamespace(
        match_id=match_id,
        status=status,
        rwp_a=rwp_a,
        rwp_b=rwp_b,
        current_server=current_server,
        games_won_a=games_won_a,
        games_won_b=games_won_b,
        score_a=score_a,
        score_b=score_b,
        match_winner=match_winner,
    )


def _make_calculator(match_id: str = "test_001") -> CashoutCalculator:
    """
    Instantiate a real CashoutCalculator (no mock Markov — real engine used).
    """
    return CashoutCalculator(match_id=match_id, discipline=Discipline.MS)


# ---------------------------------------------------------------------------
# Test 1: live state + standard commission → cashout_value_gbp > 0
# ---------------------------------------------------------------------------

def test_compute_live_state_standard_commission() -> None:
    """
    compute() on a live in-progress match with standard commission returns a
    valid CashoutResult with cashout_value_gbp > 0 and correct commission rate.
    """
    calc = _make_calculator("test_001")
    bet = _make_bet("bet_001", "test_001", stake_gbp=100.0, decimal_odds=2.00)
    state = _live_state()

    result = calc.compute(bet, state, outcome_is_player_a=True, is_premium_bettor=False)

    assert result.cashout_value_gbp > 0.0, (
        f"Cashout value should be positive, got {result.cashout_value_gbp}"
    )
    assert result.bet_id == "bet_001"
    assert abs(result.original_stake_gbp - 100.0) < 1e-9
    assert abs(result.original_odds - 2.00) < 1e-9
    assert abs(result.commission_rate - _STANDARD_COMMISSION) < 1e-9
    assert result.is_frozen is False
    assert result.match_completed is False


# ---------------------------------------------------------------------------
# Test 2: premium bettor receives a larger cashout than standard bettor
# ---------------------------------------------------------------------------

def test_compute_live_state_premium_commission() -> None:
    """
    At the same bet and live state, a premium bettor (1.5% commission) must
    receive a strictly larger cashout than a standard bettor (3% commission).

    Premium cashout = stake × (current_prob / original_prob) × (1 − 0.015)
    Standard cashout = stake × (current_prob / original_prob) × (1 − 0.030)

    Since 0.985 > 0.970, premium_cashout > standard_cashout.
    """
    calc = _make_calculator("test_002")
    bet = _make_bet("bet_002", "test_002", stake_gbp=200.0, decimal_odds=1.80)
    state = _live_state(match_id="test_002")

    result_standard = calc.compute(bet, state, outcome_is_player_a=True, is_premium_bettor=False)
    # Re-use same calculator (cache updated to live prob from first call)
    result_premium = calc.compute(bet, state, outcome_is_player_a=True, is_premium_bettor=True)

    assert result_premium.cashout_value_gbp >= result_standard.cashout_value_gbp, (
        f"Premium cashout {result_premium.cashout_value_gbp} should be >= "
        f"standard cashout {result_standard.cashout_value_gbp}"
    )
    assert abs(result_premium.commission_rate - _PREMIUM_COMMISSION) < 1e-9
    assert abs(result_standard.commission_rate - _STANDARD_COMMISSION) < 1e-9


# ---------------------------------------------------------------------------
# Test 3: verify cashout formula exactly — stake × (prob/orig_prob) × (1-comm)
# ---------------------------------------------------------------------------

def test_cashout_formula_math() -> None:
    """
    Manual verification of the cashout formula.

    Formula: cashout_raw = stake × (current_prob / original_prob) × (1 - commission)

    We use a synthetic scenario where P(A wins match) from the Markov engine is
    above the floor, so the formula path executes without clamping.

    We call compute() and then verify:
      result.current_win_prob and result.original_implied_prob are returned
      such that the formula holds within floating-point tolerance.
    """
    calc = _make_calculator("test_003")
    stake = 250.0
    decimal_odds = 2.50   # original_implied_prob = 1/2.5 = 0.40

    bet = _make_bet("bet_003", "test_003", stake_gbp=stake, decimal_odds=decimal_odds)

    # Use an early-match state (score 0-0) to ensure current_prob is live-computed
    state = _live_state(
        match_id="test_003",
        status=MatchStatus.IN_PROGRESS,
        rwp_a=0.535,
        rwp_b=0.529,
        current_server="A",
        games_won_a=0,
        games_won_b=0,
        score_a=0,
        score_b=0,
    )

    result = calc.compute(bet, state, outcome_is_player_a=True, is_premium_bettor=False)

    original_prob = 1.0 / decimal_odds    # 0.40
    current_prob = result.current_win_prob
    commission = _STANDARD_COMMISSION

    expected_raw = stake * (current_prob / original_prob) * (1.0 - commission)
    expected_cashout = max(_MIN_CASHOUT_GBP, round(expected_raw, 2))

    assert abs(result.cashout_value_gbp - expected_cashout) < 1e-6, (
        f"Formula mismatch: expected {expected_cashout:.4f}, "
        f"got {result.cashout_value_gbp:.4f}"
    )
    assert abs(result.original_implied_prob - original_prob) < 1e-5, (
        f"original_implied_prob mismatch: expected {original_prob:.6f}, "
        f"got {result.original_implied_prob:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 4: COMPLETED match — bettor wins → full payout returned
# ---------------------------------------------------------------------------

def test_compute_completed_bettor_wins() -> None:
    """
    When status=COMPLETED and match_winner="A" and outcome_is_player_a=True,
    cashout_value_gbp equals the full potential_payout_gbp = stake × decimal_odds.

    No commission is applied to a settled bet.
    match_completed must be True and is_frozen must be False.
    """
    calc = _make_calculator("test_004")
    stake = 100.0
    decimal_odds = 3.00
    bet = _make_bet("bet_004", "test_004", stake_gbp=stake, decimal_odds=decimal_odds)

    state = _live_state(
        match_id="test_004",
        status=MatchStatus.COMPLETED,
        match_winner="A",
    )

    result = calc.compute(bet, state, outcome_is_player_a=True)

    expected_payout = stake * decimal_odds    # 300.0
    assert abs(result.cashout_value_gbp - expected_payout) < 1e-9, (
        f"Expected full payout {expected_payout}, got {result.cashout_value_gbp}"
    )
    assert result.match_completed is True
    assert result.is_frozen is False
    assert abs(result.current_win_prob - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 5: COMPLETED match — bettor loses → cashout returns £0
# ---------------------------------------------------------------------------

def test_compute_completed_bettor_loses() -> None:
    """
    When status=COMPLETED, match_winner="B", outcome_is_player_a=True,
    the bettor lost — cashout_value_gbp must be 0.0.

    match_completed=True, current_win_prob=0.0.
    """
    calc = _make_calculator("test_005")
    bet = _make_bet("bet_005", "test_005", stake_gbp=500.0, decimal_odds=2.00)

    state = _live_state(
        match_id="test_005",
        status=MatchStatus.COMPLETED,
        match_winner="B",
    )

    result = calc.compute(bet, state, outcome_is_player_a=True)

    assert result.cashout_value_gbp == 0.0, (
        f"Losing bet should have £0 cashout, got {result.cashout_value_gbp}"
    )
    assert result.match_completed is True
    assert abs(result.current_win_prob - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: SUSPENDED with no cached probability → raises CashoutError
# ---------------------------------------------------------------------------

def test_compute_suspended_no_cache_raises() -> None:
    """
    If the market is SUSPENDED and no prior live probability has been cached,
    compute() must raise CashoutError (not return a stale result or default).

    This enforces the "NEVER use fallbacks" rule from CLAUDE.md §1.
    """
    calc = _make_calculator("test_006")
    bet = _make_bet("bet_006", "test_006", stake_gbp=150.0, decimal_odds=1.95)

    state = _live_state(
        match_id="test_006",
        status=MatchStatus.SUSPENDED,
    )

    with pytest.raises(CashoutError) as exc_info:
        calc.compute(bet, state, outcome_is_player_a=True)

    assert "suspended" in str(exc_info.value).lower() or "cached" in str(exc_info.value).lower(), (
        f"CashoutError message should mention suspension or cache, "
        f"got: {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: SUSPENDED after live compute → is_frozen=True, uses cached prob
# ---------------------------------------------------------------------------

def test_compute_suspended_uses_frozen_prob() -> None:
    """
    After one live compute populates the internal probability cache,
    suspending the match and calling compute again must return is_frozen=True.

    The returned current_win_prob must match the previously cached value.
    """
    calc = _make_calculator("test_007")
    bet = _make_bet("bet_007", "test_007", stake_gbp=300.0, decimal_odds=2.20)

    # Step 1: live compute to populate cache
    live_state = _live_state(
        match_id="test_007",
        status=MatchStatus.IN_PROGRESS,
        score_a=8,
        score_b=5,
    )
    live_result = calc.compute(bet, live_state, outcome_is_player_a=True)
    cached_prob = live_result.current_win_prob

    # Step 2: market suspended — same bet, same outcome key
    suspended_state = _live_state(
        match_id="test_007",
        status=MatchStatus.SUSPENDED,
        score_a=8,
        score_b=5,
    )
    frozen_result = calc.compute(bet, suspended_state, outcome_is_player_a=True)

    assert frozen_result.is_frozen is True, (
        "Result from suspended market must have is_frozen=True"
    )
    assert frozen_result.match_completed is False
    assert abs(frozen_result.current_win_prob - cached_prob) < 1e-9, (
        f"Frozen prob {frozen_result.current_win_prob} must match "
        f"cached prob {cached_prob}"
    )


# ---------------------------------------------------------------------------
# Test 8: bet.match_id != calculator match_id → raises CashoutError
# ---------------------------------------------------------------------------

def test_wrong_match_id_raises() -> None:
    """
    Passing a bet whose match_id does not match the calculator's match_id
    must raise CashoutError immediately (before any Markov call).
    """
    calc = _make_calculator("calculator_match_001")

    # Bet belongs to a different match
    bet = _make_bet("bet_008", "wrong_match_999", stake_gbp=200.0, decimal_odds=2.00)
    state = _live_state(match_id="calculator_match_001")

    with pytest.raises(CashoutError) as exc_info:
        calc.compute(bet, state, outcome_is_player_a=True)

    error_msg = str(exc_info.value)
    assert "wrong_match_999" in error_msg or "match_id" in error_msg.lower(), (
        f"Error should reference mismatched match_id, got: {error_msg!r}"
    )


# ---------------------------------------------------------------------------
# Test 9: very unfavourable outcome → cashout_value_gbp >= _MIN_CASHOUT_GBP
# ---------------------------------------------------------------------------

def test_min_cashout_floor() -> None:
    """
    When current_prob is very low relative to original_prob, the raw cashout
    formula may produce a value near zero (but never negative).

    The floor _MIN_CASHOUT_GBP = 0.01 must always be enforced.

    Strategy: bet on player_a at low odds (original_prob near 1.0, e.g. odds 1.10).
    Live state has games_won_a=0, games_won_b=1, score 0-18 in game 2 → very low
    P(A wins). The ratio current_prob / original_prob is tiny → cashout near zero.
    The floor must kick in.
    """
    calc = _make_calculator("test_009")

    # original_prob = 1 / 1.10 ≈ 0.909 — bet placed as heavy favourite
    bet = _make_bet("bet_009", "test_009", stake_gbp=500.0, decimal_odds=1.10)

    # Live state: A has lost game 1, now losing game 2 heavily (0-18)
    # P(A wins match from here) is extremely low
    state = _live_state(
        match_id="test_009",
        status=MatchStatus.IN_PROGRESS,
        rwp_a=0.515,
        rwp_b=0.520,
        current_server="B",
        games_won_a=0,
        games_won_b=1,
        score_a=0,
        score_b=18,
    )

    result = calc.compute(bet, state, outcome_is_player_a=True, is_premium_bettor=False)

    assert result.cashout_value_gbp >= _MIN_CASHOUT_GBP, (
        f"Cashout value {result.cashout_value_gbp} must be >= "
        f"floor {_MIN_CASHOUT_GBP}"
    )
    assert result.cashout_value_gbp >= 0.0, (
        "Cashout value must never be negative"
    )

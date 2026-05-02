"""
test_exposure_manager.py
========================
Unit tests for risk/exposure_manager.py — ExposureManager class.

Tests:
  1.  test_record_bet_updates_exposure
  2.  test_record_multiple_bets_same_outcome
  3.  test_check_limits_passes_within_limits
  4.  test_check_limits_outcome_breach
  5.  test_check_limits_match_breach
  6.  test_get_match_max_loss
  7.  test_get_market_exposure
  8.  test_get_all_exposure_for_context
  9.  test_portfolio_summary
  10. test_zero_exposure_on_empty

All tests use real ExposureManager — ZERO mocks.
Constants sourced directly from exposure_manager module to avoid drift.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.exposure_manager import (
    BetRecord,
    ExposureManager,
    _GLOBAL_MAX_EXPOSURE_GBP,
    _MATCH_MAX_EXPOSURE_GBP,
    _MARKET_MAX_EXPOSURE_GBP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bet(
    bet_id: str,
    match_id: str,
    market_id: str,
    outcome_name: str,
    stake_gbp: float,
    decimal_odds: float,
) -> BetRecord:
    """Construct a BetRecord with a real placed_at timestamp."""
    return BetRecord(
        bet_id=bet_id,
        match_id=match_id,
        market_id=market_id,
        outcome_name=outcome_name,
        stake_gbp=stake_gbp,
        decimal_odds=decimal_odds,
        placed_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Test 1: record_bet updates exposure to the expected profit amount
# ---------------------------------------------------------------------------

def test_record_bet_updates_exposure() -> None:
    """
    After recording one bet, outcome exposure equals potential_profit_gbp.

    potential_profit_gbp = stake × (decimal_odds - 1)
    net_liability_gbp    = total_potential_profit_gbp
    """
    manager = ExposureManager()

    stake = 500.0
    odds = 2.50
    bet = _make_bet("bet_001", "match_1", "match_winner", "player_a", stake, odds)
    manager.record_bet(bet)

    exposure = manager.get_market_exposure("match_1", "match_winner")

    expected_profit = stake * (odds - 1.0)          # 500 × 1.5 = 750.0
    assert "player_a" in exposure
    assert abs(exposure["player_a"] - expected_profit) < 1e-9, (
        f"Expected liability {expected_profit}, got {exposure['player_a']}"
    )


# ---------------------------------------------------------------------------
# Test 2: two bets on same outcome — exposures accumulate
# ---------------------------------------------------------------------------

def test_record_multiple_bets_same_outcome() -> None:
    """
    Two bets on the same outcome: net liability equals the sum of both profits.
    """
    manager = ExposureManager()

    bet_a = _make_bet("bet_001", "match_2", "mkt_winner", "player_a", 1000.0, 1.80)
    bet_b = _make_bet("bet_002", "match_2", "mkt_winner", "player_a", 2000.0, 2.10)
    manager.record_bet(bet_a)
    manager.record_bet(bet_b)

    exposure = manager.get_market_exposure("match_2", "mkt_winner")

    expected = bet_a.potential_profit_gbp + bet_b.potential_profit_gbp
    # 1000 × 0.80 + 2000 × 1.10 = 800 + 2200 = 3000
    assert abs(exposure["player_a"] - expected) < 1e-9, (
        f"Accumulated liability should be {expected}, got {exposure['player_a']}"
    )


# ---------------------------------------------------------------------------
# Test 3: check_limits passes (returns None) for bets well within limits
# ---------------------------------------------------------------------------

def test_check_limits_passes_within_limits() -> None:
    """
    A £100 bet at 2.00 (profit £100) on an empty book returns None.

    All three limits are massively above £100 profit.
    """
    manager = ExposureManager()

    result = manager.check_limits(
        match_id="match_3",
        market_id="mkt_winner",
        outcome_name="player_a",
        stake_gbp=100.0,
        decimal_odds=2.00,
    )

    assert result is None, f"Expected None for safe bet, got: {result!r}"


# ---------------------------------------------------------------------------
# Test 4: check_limits returns error when outcome exposure would exceed £200 K
# ---------------------------------------------------------------------------

def test_check_limits_outcome_breach() -> None:
    """
    Pre-load outcome exposure near the £200 K limit, then push it over.

    _MARKET_MAX_EXPOSURE_GBP = 200_000 (per-outcome limit).
    We fill to £199 990 profit (stake=200 000, odds=2.00 → profit=200 000 − stake).

    Actually: profit = stake × (odds − 1). For odds 2.00: profit = stake × 1.
    To get £199 990 profit: stake = £199 990, odds = 2.00.
    Adding another bet with profit > £10 → total > 200 000 → breach.
    """
    manager = ExposureManager()

    # Fill outcome to £199 990 profit
    fill_stake = 199_990.0
    fill_odds = 2.00
    fill_bet = _make_bet(
        "fill_001", "match_4", "mkt_winner", "player_a",
        fill_stake, fill_odds,
    )
    manager.record_bet(fill_bet)

    # This bet would add £20 profit → total = 200 010 > 200 000
    result = manager.check_limits(
        match_id="match_4",
        market_id="mkt_winner",
        outcome_name="player_a",
        stake_gbp=20.0,
        decimal_odds=2.00,
    )

    assert result is not None, "Expected an error string for outcome limit breach"
    assert "outcome exposure limit" in result.lower() or "exposure" in result.lower(), (
        f"Error message should mention exposure limit, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 5: check_limits returns error when match max-loss would exceed £1 M
# ---------------------------------------------------------------------------

def test_check_limits_match_breach() -> None:
    """
    Fill the match exposure close to £1 M across multiple markets (so no single
    outcome exceeds the £200 K outcome limit), then verify a further bet triggers
    the match max-loss limit.

    Strategy: use 6 distinct markets, each with a single outcome carrying
    £166 000 profit (well under the £200 K outcome cap).
    6 × £166 000 = £996 000 match exposure.

    A new bet on a seventh market with profit £5 000 would push the total to
    £1 001 000 > £1 000 000 → match limit breach.

    Each fill: stake=166_000, odds=2.00 → profit = 166_000 × 1.0 = £166 000.
    """
    manager = ExposureManager()

    for i in range(6):
        manager.record_bet(_make_bet(
            f"fill_m5_{i}", "match_5", f"mkt_{i:02d}", "player_a",
            166_000.0, 2.00,
        ))

    # Current match max loss: 6 × 166 000 = 996 000
    # New bet profit: 5 000 × 1.0 = £5 000 → total 1 001 000 > 1 000 000
    result = manager.check_limits(
        match_id="match_5",
        market_id="mkt_new",
        outcome_name="player_a",
        stake_gbp=5_000.0,
        decimal_odds=2.00,
    )

    assert result is not None, "Expected an error string for match max-loss breach"
    assert "match" in result.lower() or "limit" in result.lower(), (
        f"Error message should reference match limit, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: get_match_max_loss across two markets
# ---------------------------------------------------------------------------

def test_get_match_max_loss() -> None:
    """
    With two markets each having two outcomes, max_loss is the sum of the
    maximum liability per market.

    Market A outcomes:
      player_a: profit = 300 × 1.5 = 450
      player_b: profit = 200 × 0.8 = 160
      max = 450

    Market B outcomes:
      player_a: profit = 500 × 1.0 = 500
      player_b: profit = 100 × 3.0 = 300
      max = 500

    Expected match max loss = 450 + 500 = 950
    """
    manager = ExposureManager()

    # Market A
    manager.record_bet(_make_bet("b1", "match_6", "mkt_winner", "player_a", 300.0, 2.50))
    manager.record_bet(_make_bet("b2", "match_6", "mkt_winner", "player_b", 200.0, 1.80))

    # Market B
    manager.record_bet(_make_bet("b3", "match_6", "mkt_handicap", "player_a", 500.0, 2.00))
    manager.record_bet(_make_bet("b4", "match_6", "mkt_handicap", "player_b", 100.0, 4.00))

    max_loss = manager.get_match_max_loss("match_6")

    profit_a_winner = 300.0 * (2.50 - 1.0)    # 450.0
    profit_b_winner = 200.0 * (1.80 - 1.0)    # 160.0
    max_mkt_a = max(profit_a_winner, profit_b_winner)  # 450.0

    profit_a_handicap = 500.0 * (2.00 - 1.0)  # 500.0
    profit_b_handicap = 100.0 * (4.00 - 1.0)  # 300.0
    max_mkt_b = max(profit_a_handicap, profit_b_handicap)  # 500.0

    expected_max_loss = max_mkt_a + max_mkt_b   # 950.0

    assert abs(max_loss - expected_max_loss) < 1e-9, (
        f"Expected match max loss {expected_max_loss}, got {max_loss}"
    )


# ---------------------------------------------------------------------------
# Test 7: get_market_exposure returns per-outcome liability dict
# ---------------------------------------------------------------------------

def test_get_market_exposure() -> None:
    """
    get_market_exposure returns {outcome_name: net_liability_gbp} for a market.

    Verifies:
      - Both outcomes are present in the returned dict.
      - Liabilities are correct: profit = stake × (odds - 1).
      - No cross-contamination between markets.
    """
    manager = ExposureManager()

    manager.record_bet(_make_bet("b1", "match_7", "mkt_winner", "player_a", 400.0, 2.00))
    manager.record_bet(_make_bet("b2", "match_7", "mkt_winner", "player_b", 600.0, 3.00))
    manager.record_bet(_make_bet("b3", "match_7", "mkt_other", "player_a", 9999.0, 2.00))

    exposure = manager.get_market_exposure("match_7", "mkt_winner")

    assert set(exposure.keys()) == {"player_a", "player_b"}, (
        f"Expected exactly two outcomes, got {set(exposure.keys())}"
    )
    assert abs(exposure["player_a"] - 400.0 * 1.0) < 1e-9
    assert abs(exposure["player_b"] - 600.0 * 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 8: get_all_exposure_for_context returns flat "market_id:outcome" keys
# ---------------------------------------------------------------------------

def test_get_all_exposure_for_context() -> None:
    """
    get_all_exposure_for_context returns a flat dict keyed by "market_id:outcome".

    With bets across two markets × two outcomes each, we expect four keys.
    """
    manager = ExposureManager()

    bets = [
        _make_bet("b1", "match_8", "mkt_result",   "player_a", 200.0, 1.90),
        _make_bet("b2", "match_8", "mkt_result",   "player_b", 150.0, 2.10),
        _make_bet("b3", "match_8", "mkt_totgames", "over_2.5", 100.0, 1.80),
        _make_bet("b4", "match_8", "mkt_totgames", "under_2.5", 80.0, 2.20),
    ]
    for bet in bets:
        manager.record_bet(bet)

    context = manager.get_all_exposure_for_context("match_8")

    expected_keys = {
        "mkt_result:player_a",
        "mkt_result:player_b",
        "mkt_totgames:over_2.5",
        "mkt_totgames:under_2.5",
    }
    assert set(context.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(context.keys())}"
    )
    # Spot-check one value
    assert abs(context["mkt_result:player_a"] - 200.0 * (1.90 - 1.0)) < 1e-9


# ---------------------------------------------------------------------------
# Test 9: get_portfolio_summary — n_active_matches and total_max_loss_gbp
# ---------------------------------------------------------------------------

def test_portfolio_summary() -> None:
    """
    get_portfolio_summary returns correct n_active_matches and total_max_loss_gbp.

    Two matches:
      match_9a: one outcome with profit 1 000
      match_9b: one outcome with profit 2 000

    n_active_matches = 2
    total_max_loss_gbp = 3 000
    """
    manager = ExposureManager()

    # match_9a: stake 1000, odds 2.00 → profit 1000
    manager.record_bet(_make_bet("b1", "match_9a", "mkt_x", "player_a", 1000.0, 2.00))
    # match_9b: stake 1000, odds 3.00 → profit 2000
    manager.record_bet(_make_bet("b2", "match_9b", "mkt_x", "player_a", 1000.0, 3.00))

    summary = manager.get_portfolio_summary()

    assert summary["n_active_matches"] == 2, (
        f"Expected 2 active matches, got {summary['n_active_matches']}"
    )
    assert abs(summary["total_max_loss_gbp"] - 3000.0) < 1e-6, (
        f"Expected total max loss 3000.0, got {summary['total_max_loss_gbp']}"
    )
    assert "global_limit_gbp" in summary
    assert abs(summary["global_limit_gbp"] - _GLOBAL_MAX_EXPOSURE_GBP) < 1e-9
    assert "utilisation_pct" in summary


# ---------------------------------------------------------------------------
# Test 10: fresh manager returns 0 for any match
# ---------------------------------------------------------------------------

def test_zero_exposure_on_empty() -> None:
    """
    A fresh ExposureManager has no recorded bets.

    All query methods return sensible zero-state responses:
      - get_match_max_loss → 0.0
      - get_market_exposure → {}
      - get_all_exposure_for_context → {}
      - get_portfolio_summary → n_active_matches=0, total_max_loss_gbp=0.0
      - check_limits → None (bet is acceptable)
    """
    manager = ExposureManager()

    assert manager.get_match_max_loss("nonexistent_match") == 0.0, (
        "Empty manager should return 0.0 match max loss"
    )
    assert manager.get_market_exposure("nonexistent_match", "mkt_x") == {}, (
        "Empty manager should return empty market exposure dict"
    )
    assert manager.get_all_exposure_for_context("nonexistent_match") == {}, (
        "Empty manager should return empty context exposure dict"
    )

    summary = manager.get_portfolio_summary()
    assert summary["n_active_matches"] == 0
    assert summary["total_max_loss_gbp"] == 0.0

    # A small safe bet should pass
    result = manager.check_limits(
        match_id="nonexistent_match",
        market_id="mkt_x",
        outcome_name="player_a",
        stake_gbp=50.0,
        decimal_odds=2.00,
    )
    assert result is None, (
        "check_limits on empty manager should return None for any safe bet"
    )

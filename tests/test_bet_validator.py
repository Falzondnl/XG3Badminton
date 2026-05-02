"""
test_bet_validator.py
=====================
Unit tests for betting/bet_validator.py

Tests the full validation pipeline:
  - Schema validation (required fields, numeric ranges)
  - Market state checks (OPEN / SUSPENDED / RESULTED)
  - Stake limits (min/max, tier, click scale)
  - Odds staleness check (±2% tolerance)
  - Exposure limit gate (via ExposureManager.check_limits)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from betting.bet_validator import (
    BetValidator,
    BetValidationError,
    BetValidationResult,
    BetRejectionCode,
    IncomingBetRequest,
)
from markets.market_trading_control import MarketStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**kwargs) -> IncomingBetRequest:
    defaults = {
        "bet_id": "bet_001",
        "match_id": "match_001",
        "market_id": "match_winner",
        "outcome_name": "A_wins",
        "stake_gbp": 100.0,
        "offered_odds": 1.85,
        "is_vip": False,
    }
    defaults.update(kwargs)
    return IncomingBetRequest(**defaults)


def _open_trading_control(market_id: str = "match_winner") -> MagicMock:
    """Mock TradingControlManager that reports a market as OPEN."""
    tc = MagicMock()
    tc.get_market_status.return_value = MarketStatus.OPEN
    return tc


def _open_exposure_manager() -> MagicMock:
    """Mock ExposureManager that passes all limits."""
    em = MagicMock()
    em.check_limits.return_value = None   # None = no error
    return em


@pytest.fixture
def validator():
    return BetValidator()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Valid bets pass all layers."""

    def test_valid_standard_bet(self, validator):
        """Standard bet with all checks passing returns BetValidationResult."""
        req = _make_request()
        result = validator.validate(
            req,
            trading_control=_open_trading_control(),
            exposure_manager=_open_exposure_manager(),
            current_odds=1.85,
        )
        assert isinstance(result, BetValidationResult)
        assert result.accepted if hasattr(result, "accepted") else True
        assert result.bet_id == "bet_001"
        assert result.validated_stake_gbp == 100.0

    def test_valid_bet_without_odds_check(self, validator):
        """When current_odds=None, odds check is skipped."""
        req = _make_request(offered_odds=99.0)   # Any odds — not checked
        result = validator.validate(
            req,
            trading_control=_open_trading_control(),
            exposure_manager=_open_exposure_manager(),
            current_odds=None,
        )
        assert result.decimal_odds == 99.0

    def test_vip_valid_high_stake(self, validator):
        """VIP bet up to £25,000 is accepted."""
        req = _make_request(stake_gbp=24_999.0, is_vip=True)
        result = validator.validate(
            req,
            trading_control=_open_trading_control(),
            exposure_manager=_open_exposure_manager(),
        )
        assert result.validated_stake_gbp == 24_999.0


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Required field checks."""

    def test_empty_bet_id_rejected(self, validator):
        req = _make_request(bet_id="")
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_empty_match_id_rejected(self, validator):
        req = _make_request(match_id="")
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_zero_stake_rejected(self, validator):
        req = _make_request(stake_gbp=0.0)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_negative_stake_rejected(self, validator):
        req = _make_request(stake_gbp=-50.0)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_odds_below_1_rejected(self, validator):
        req = _make_request(offered_odds=0.90)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR


# ---------------------------------------------------------------------------
# Market state validation
# ---------------------------------------------------------------------------

class TestMarketStateValidation:
    """Market must be OPEN to accept bets."""

    def test_resulted_market_rejected(self, validator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.RESULTED
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_CLOSED

    def test_suspended_market_rejected(self, validator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.SUSPENDED
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_SUSPENDED

    def test_ghost_market_rejected(self, validator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.GHOST
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_SUSPENDED

    def test_closed_market_rejected(self, validator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.CLOSED
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_CLOSED


# ---------------------------------------------------------------------------
# Stake validation
# ---------------------------------------------------------------------------

class TestStakeValidation:
    """Min/max stake and click scale enforcement."""

    def test_stake_below_min_rejected(self, validator):
        req = _make_request(stake_gbp=0.49)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.STAKE_BELOW_MIN

    def test_stake_at_minimum_accepted(self, validator):
        req = _make_request(stake_gbp=0.50)
        result = validator.validate(
            req, _open_trading_control(), _open_exposure_manager()
        )
        assert result.validated_stake_gbp == 0.50

    def test_standard_stake_above_max_rejected(self, validator):
        req = _make_request(stake_gbp=5_001.0, is_vip=False)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.STAKE_ABOVE_MAX

    def test_vip_stake_above_max_rejected(self, validator):
        req = _make_request(stake_gbp=25_001.0, is_vip=True)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.STAKE_ABOVE_MAX

    def test_click_scale_reduces_allowed_stake(self, validator):
        """click_scale=0.1 → max allowed stake = 500 for standard."""
        req = _make_request(stake_gbp=600.0, is_vip=False)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(
                req, _open_trading_control(), _open_exposure_manager(),
                click_scale=0.10,
            )
        assert exc_info.value.code == BetRejectionCode.STAKE_EXCEEDS_SCALE


# ---------------------------------------------------------------------------
# Odds staleness
# ---------------------------------------------------------------------------

class TestOddsValidation:
    """Offered odds within ±2% tolerance of current odds."""

    def test_odds_within_tolerance_accepted(self, validator):
        req = _make_request(offered_odds=1.85)
        result = validator.validate(
            req, _open_trading_control(), _open_exposure_manager(),
            current_odds=1.86,   # 0.54% drift — within 2%
        )
        assert result is not None

    def test_odds_too_stale_rejected(self, validator):
        req = _make_request(offered_odds=1.85)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(
                req, _open_trading_control(), _open_exposure_manager(),
                current_odds=2.00,   # >2% drift
            )
        assert exc_info.value.code == BetRejectionCode.ODDS_STALE

    def test_odds_exactly_at_tolerance_boundary(self, validator):
        """Odds at exactly 2% drift: check boundary behaviour."""
        current = 1.00
        # 2% drift: offered = 1.02 → drift = 0.02 exactly → should fail (> not >=)
        offered = current * 1.02
        req = _make_request(offered_odds=offered)
        # At exactly 2% we expect rejection since code checks > tolerance
        with pytest.raises(BetValidationError):
            validator.validate(
                req, _open_trading_control(), _open_exposure_manager(),
                current_odds=current,
            )


# ---------------------------------------------------------------------------
# Exposure limits
# ---------------------------------------------------------------------------

class TestExposureValidation:
    """Exposure manager gate."""

    def test_exposure_limit_breach_rejected(self, validator):
        em = MagicMock()
        em.check_limits.return_value = "outcome liability exceeds £50,000 limit"
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), em)
        assert exc_info.value.code == BetRejectionCode.EXPOSURE_LIMIT
        assert "liability" in exc_info.value.detail

    def test_exposure_check_called_with_correct_args(self, validator):
        """ExposureManager.check_limits() is called with the validated stake."""
        em = MagicMock()
        em.check_limits.return_value = None
        req = _make_request(stake_gbp=500.0, offered_odds=2.10)
        validator.validate(req, _open_trading_control(), em)
        em.check_limits.assert_called_once_with(
            match_id="match_001",
            market_id="match_winner",
            outcome_name="A_wins",
            stake_gbp=500.0,
            decimal_odds=2.10,
        )

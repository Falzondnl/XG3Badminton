"""
test_betting.py
===============
Unit tests for betting/bet_processor.py and betting/bet_validator.py

Tests:
  - BetProcessor: construction, accept_bet, get_bet, list_bets, n_accepted_bets
  - BetProcessor: rejection paths (wrong match, validation failure, exposure failure)
  - BetValidator: validate() happy path and rejection codes
  - IncomingBetRequest: construction and field access
  - BetAcceptanceRecord: field correctness after acceptance
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from betting.bet_processor import (
    BetProcessor,
    IncomingBetRequest,
    BetAcceptanceRecord,
    BetProcessorError,
    BetRecord,
)
from betting.bet_validator import (
    BetValidator,
    BetValidationError,
    BetValidationResult,
    BetRejectionCode,
)
from markets.market_trading_control import MarketStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**kwargs) -> IncomingBetRequest:
    """Build an IncomingBetRequest with sensible defaults."""
    defaults = {
        "bet_id": "bet_001",
        "match_id": "match_001",
        "market_id": "match_winner",
        "outcome_name": "A_wins",
        "stake_gbp": 100.0,
        "offered_odds": 2.10,
        "is_vip": False,
    }
    defaults.update(kwargs)
    return IncomingBetRequest(**defaults)


def _open_trading_control() -> MagicMock:
    """Mock TradingControlManager that reports a market as OPEN."""
    tc = MagicMock()
    tc.get_market_status.return_value = MarketStatus.OPEN
    return tc


def _open_exposure_manager() -> MagicMock:
    """Mock ExposureManager that passes all limits and records bets."""
    em = MagicMock()
    em.check_limits.return_value = None
    em.record_bet.return_value = None
    return em


def _mock_cashout() -> MagicMock:
    """Mock CashoutCalculator."""
    return MagicMock()


def _make_processor(match_id: str = "match_001") -> tuple:
    """Build a BetProcessor with standard mocks. Returns (processor, tc, em, cc)."""
    tc = _open_trading_control()
    em = _open_exposure_manager()
    cc = _mock_cashout()
    bp = BetProcessor(
        match_id=match_id,
        trading_control=tc,
        exposure_manager=em,
        cashout_calculator=cc,
    )
    return bp, tc, em, cc


# ---------------------------------------------------------------------------
# BetProcessor — Construction
# ---------------------------------------------------------------------------

class TestBetProcessorConstruction:
    """BetProcessor construction and initial state."""

    def test_construction_succeeds(self):
        bp, _, _, _ = _make_processor()
        assert bp is not None

    def test_initial_n_accepted_bets_is_zero(self):
        bp, _, _, _ = _make_processor()
        assert bp.n_accepted_bets == 0

    def test_initial_list_bets_empty(self):
        bp, _, _, _ = _make_processor()
        assert bp.list_bets_for_match() == []


# ---------------------------------------------------------------------------
# BetProcessor — accept_bet happy path
# ---------------------------------------------------------------------------

class TestBetProcessorAcceptBet:
    """BetProcessor.accept_bet() happy path."""

    def test_accept_bet_returns_acceptance_record(self):
        bp, _, _, _ = _make_processor()
        req = _make_request()
        result = bp.accept_bet(req, current_odds=2.10)
        assert isinstance(result, BetAcceptanceRecord)

    def test_accepted_bet_has_correct_bet_id(self):
        bp, _, _, _ = _make_processor()
        req = _make_request(bet_id="bet_xyz")
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.bet_id == "bet_xyz"

    def test_accepted_bet_has_correct_match_id(self):
        bp, _, _, _ = _make_processor()
        req = _make_request()
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.match_id == "match_001"

    def test_accepted_bet_has_correct_market_id(self):
        bp, _, _, _ = _make_processor()
        req = _make_request()
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.market_id == "match_winner"

    def test_accepted_bet_has_correct_outcome(self):
        bp, _, _, _ = _make_processor()
        req = _make_request()
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.outcome_name == "A_wins"

    def test_accepted_bet_has_correct_stake(self):
        bp, _, _, _ = _make_processor()
        req = _make_request(stake_gbp=250.0)
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.stake_gbp == 250.0

    def test_accepted_bet_has_correct_odds(self):
        bp, _, _, _ = _make_processor()
        req = _make_request(offered_odds=2.10)
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.decimal_odds == 2.10

    def test_accepted_bet_payout_calculated(self):
        bp, _, _, _ = _make_processor()
        req = _make_request(stake_gbp=100.0, offered_odds=2.10)
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.potential_payout_gbp == pytest.approx(210.0, abs=0.01)

    def test_accepted_bet_has_placed_at_timestamp(self):
        bp, _, _, _ = _make_processor()
        req = _make_request()
        result = bp.accept_bet(req, current_odds=2.10)
        assert result.placed_at > 0

    def test_n_accepted_bets_increments(self):
        bp, _, _, _ = _make_processor()
        assert bp.n_accepted_bets == 0
        bp.accept_bet(_make_request(bet_id="b1"), current_odds=2.10)
        assert bp.n_accepted_bets == 1
        bp.accept_bet(_make_request(bet_id="b2"), current_odds=2.10)
        assert bp.n_accepted_bets == 2

    def test_exposure_record_bet_called(self):
        bp, _, em, _ = _make_processor()
        req = _make_request()
        bp.accept_bet(req, current_odds=2.10)
        em.record_bet.assert_called_once()


# ---------------------------------------------------------------------------
# BetProcessor — get_bet
# ---------------------------------------------------------------------------

class TestBetProcessorGetBet:
    """BetProcessor.get_bet() lookup."""

    def test_get_bet_returns_accepted_record(self):
        bp, _, _, _ = _make_processor()
        req = _make_request(bet_id="bet_lookup")
        bp.accept_bet(req, current_odds=2.10)
        found = bp.get_bet("bet_lookup")
        assert found is not None
        assert found.bet_id == "bet_lookup"

    def test_get_bet_returns_none_for_unknown(self):
        bp, _, _, _ = _make_processor()
        assert bp.get_bet("nonexistent_bet") is None

    def test_get_bet_after_multiple_accepts(self):
        bp, _, _, _ = _make_processor()
        bp.accept_bet(_make_request(bet_id="b1"), current_odds=2.10)
        bp.accept_bet(_make_request(bet_id="b2"), current_odds=2.10)
        assert bp.get_bet("b1").bet_id == "b1"
        assert bp.get_bet("b2").bet_id == "b2"


# ---------------------------------------------------------------------------
# BetProcessor — list_bets_for_match
# ---------------------------------------------------------------------------

class TestBetProcessorListBets:
    """BetProcessor.list_bets_for_match() listing."""

    def test_list_bets_returns_all_accepted(self):
        bp, _, _, _ = _make_processor()
        bp.accept_bet(_make_request(bet_id="b1"), current_odds=2.10)
        bp.accept_bet(_make_request(bet_id="b2"), current_odds=2.10)
        bp.accept_bet(_make_request(bet_id="b3"), current_odds=2.10)
        bets = bp.list_bets_for_match()
        assert len(bets) == 3

    def test_list_bets_contains_correct_ids(self):
        bp, _, _, _ = _make_processor()
        bp.accept_bet(_make_request(bet_id="alpha"), current_odds=2.10)
        bp.accept_bet(_make_request(bet_id="beta"), current_odds=2.10)
        bets = bp.list_bets_for_match()
        ids = {b.bet_id for b in bets}
        assert ids == {"alpha", "beta"}

    def test_list_bets_returns_acceptance_records(self):
        bp, _, _, _ = _make_processor()
        bp.accept_bet(_make_request(bet_id="b1"), current_odds=2.10)
        bets = bp.list_bets_for_match()
        assert all(isinstance(b, BetAcceptanceRecord) for b in bets)


# ---------------------------------------------------------------------------
# BetProcessor — Rejection paths
# ---------------------------------------------------------------------------

class TestBetProcessorRejection:
    """BetProcessor rejects bets correctly."""

    def test_wrong_match_id_raises_processor_error(self):
        """Bet for a different match_id raises BetProcessorError."""
        bp, _, _, _ = _make_processor(match_id="match_001")
        req = _make_request(match_id="match_999")
        with pytest.raises(BetProcessorError):
            bp.accept_bet(req, current_odds=2.10)

    def test_exposure_recording_failure_raises_processor_error(self):
        """If exposure_manager.record_bet() raises, BetProcessorError propagates."""
        bp, _, em, _ = _make_processor()
        em.record_bet.side_effect = RuntimeError("DB down")
        req = _make_request()
        with pytest.raises(BetProcessorError):
            bp.accept_bet(req, current_odds=2.10)

    def test_rejected_bet_not_counted(self):
        """Rejected bets do not increment n_accepted_bets."""
        bp, _, _, _ = _make_processor(match_id="match_001")
        req = _make_request(match_id="match_999")
        with pytest.raises(BetProcessorError):
            bp.accept_bet(req, current_odds=2.10)
        assert bp.n_accepted_bets == 0


# ---------------------------------------------------------------------------
# BetValidator — validate() happy path
# ---------------------------------------------------------------------------

class TestBetValidatorHappyPath:
    """BetValidator.validate() returns BetValidationResult for valid bets."""

    @pytest.fixture
    def validator(self) -> BetValidator:
        return BetValidator()

    def test_valid_bet_returns_result(self, validator: BetValidator):
        req = _make_request()
        result = validator.validate(
            req,
            trading_control=_open_trading_control(),
            exposure_manager=_open_exposure_manager(),
            current_odds=2.10,
        )
        assert isinstance(result, BetValidationResult)
        assert result.bet_id == "bet_001"

    def test_valid_bet_stake_preserved(self, validator: BetValidator):
        req = _make_request(stake_gbp=500.0)
        result = validator.validate(
            req,
            trading_control=_open_trading_control(),
            exposure_manager=_open_exposure_manager(),
            current_odds=2.10,
        )
        assert result.validated_stake_gbp == 500.0

    def test_valid_bet_odds_preserved(self, validator: BetValidator):
        req = _make_request(offered_odds=3.50)
        result = validator.validate(
            req,
            trading_control=_open_trading_control(),
            exposure_manager=_open_exposure_manager(),
            current_odds=3.50,
        )
        assert result.decimal_odds == 3.50


# ---------------------------------------------------------------------------
# BetValidator — Schema rejection
# ---------------------------------------------------------------------------

class TestBetValidatorSchemaRejection:
    """BetValidator.validate() rejects malformed bets."""

    @pytest.fixture
    def validator(self) -> BetValidator:
        return BetValidator()

    def test_empty_bet_id_rejected(self, validator: BetValidator):
        req = _make_request(bet_id="")
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_zero_stake_rejected(self, validator: BetValidator):
        req = _make_request(stake_gbp=0.0)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_negative_stake_rejected(self, validator: BetValidator):
        req = _make_request(stake_gbp=-50.0)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR

    def test_odds_below_1_rejected(self, validator: BetValidator):
        req = _make_request(offered_odds=0.90)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, _open_trading_control(), _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.SCHEMA_ERROR


# ---------------------------------------------------------------------------
# BetValidator — Market state rejection
# ---------------------------------------------------------------------------

class TestBetValidatorMarketState:
    """BetValidator.validate() rejects bets for non-OPEN markets."""

    @pytest.fixture
    def validator(self) -> BetValidator:
        return BetValidator()

    def test_suspended_market_rejected(self, validator: BetValidator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.SUSPENDED
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_SUSPENDED

    def test_resulted_market_rejected(self, validator: BetValidator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.RESULTED
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_CLOSED

    def test_closed_market_rejected(self, validator: BetValidator):
        tc = MagicMock()
        tc.get_market_status.return_value = MarketStatus.CLOSED
        req = _make_request()
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(req, tc, _open_exposure_manager())
        assert exc_info.value.code == BetRejectionCode.MARKET_CLOSED


# ---------------------------------------------------------------------------
# BetValidator — Odds staleness
# ---------------------------------------------------------------------------

class TestBetValidatorOddsStaleness:
    """BetValidator.validate() rejects stale odds."""

    @pytest.fixture
    def validator(self) -> BetValidator:
        return BetValidator()

    def test_odds_within_tolerance_accepted(self, validator: BetValidator):
        req = _make_request(offered_odds=1.85)
        result = validator.validate(
            req, _open_trading_control(), _open_exposure_manager(),
            current_odds=1.86,
        )
        assert result is not None

    def test_odds_too_stale_rejected(self, validator: BetValidator):
        req = _make_request(offered_odds=1.85)
        with pytest.raises(BetValidationError) as exc_info:
            validator.validate(
                req, _open_trading_control(), _open_exposure_manager(),
                current_odds=2.00,
            )
        assert exc_info.value.code == BetRejectionCode.ODDS_STALE


# ---------------------------------------------------------------------------
# IncomingBetRequest — construction
# ---------------------------------------------------------------------------

class TestIncomingBetRequest:
    """IncomingBetRequest dataclass construction."""

    def test_construction_all_fields(self):
        req = IncomingBetRequest(
            bet_id="b1",
            match_id="m1",
            market_id="match_winner",
            outcome_name="A",
            stake_gbp=50.0,
            offered_odds=1.50,
            is_vip=True,
        )
        assert req.bet_id == "b1"
        assert req.match_id == "m1"
        assert req.market_id == "match_winner"
        assert req.outcome_name == "A"
        assert req.stake_gbp == 50.0
        assert req.offered_odds == 1.50
        assert req.is_vip is True

    def test_default_is_vip_false(self):
        req = IncomingBetRequest(
            bet_id="b2",
            match_id="m2",
            market_id="mw",
            outcome_name="B",
            stake_gbp=10.0,
            offered_odds=2.00,
        )
        assert req.is_vip is False

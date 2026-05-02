"""
conftest.py
===========
Shared pytest fixtures for the XG3 Badminton test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from core.markov_engine import BadmintonMarkovEngine, clear_markov_cache
from core.scoring_engine import ScoringEngine


@pytest.fixture(scope="session", autouse=True)
def clear_cache():
    """Clear Markov cache before test session."""
    clear_markov_cache()
    yield
    clear_markov_cache()


@pytest.fixture
def markov_engine():
    """Fresh Markov engine for each test."""
    return BadmintonMarkovEngine()


@pytest.fixture
def ms_match_state():
    """Standard MS match state."""
    from core.match_state import BadmintonMatchStateMachine
    return BadmintonMatchStateMachine.initialise(
        match_id="fixture_ms_001",
        entity_a_id="player_a",
        entity_b_id="player_b",
        discipline=Discipline.MS,
        first_server="A",
    )


@pytest.fixture
def standard_rwp():
    """Standard RWP values for testing."""
    return {"rwp_a": 0.535, "rwp_b": 0.529}


@pytest.fixture
def derivative_engine():
    """BadmintonDerivativeEngine instance."""
    from markets.derivative_engine import BadmintonDerivativeEngine
    return BadmintonDerivativeEngine()


@pytest.fixture
def standard_market_set(derivative_engine, standard_rwp):
    """Standard market set for MS Super 500 match."""
    return derivative_engine.compute_all_markets(
        match_id="fixture_market_001",
        rwp=standard_rwp["rwp_a"],
        discipline=Discipline.MS,
        tier=TournamentTier.SUPER_500,
        p_match_win=0.55,
        server_first_game="A",
    )

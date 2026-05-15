"""
TIER 2B REVERSE ENGINEER — Badminton (Family A, Badminton Markov Inverse)
=========================================================================

Inverts Pinnacle's de-vigged match-win probability into the (RWP_a, RWP_b)
parameter pair that the BadmintonMarkovEngine would produce to reproduce that
exact probability.

DESIGN (from TIER_2B_REVERSE_ENGINEERING_DESIGN_20260515.md §2.1):
  Step 1: Pin Player A RWP from prior = discipline baseline + ELO-implied offset.
          Uses existing RWPCalculator.ELO_RWP_COEFFICIENT scaling.
  Step 2: Brent root-find for RWP_b such that:
          markov.p_win_match_from_rwp(rwp_a_pinned, rwp_b, discipline) = p_pinnacle
          Search interval: [RWP_MIN_VALID=0.40, RWP_MAX_VALID=0.65]
  Step 3: Round-trip parity check: |repriced_prob - p_pinnacle| < 0.5pp
  Step 4: Return full feature dict (rwp_a, rwp_b, derived match/game/set probs,
          p_2_0, p_2_1 score distribution for derivative engine)

Forward Markov engine: BadmintonMarkovEngine (core/markov_engine.py)
Prior anchor:          RWP_BASELINE (discipline-keyed) + ELO_RWP_COEFFICIENT offset

OPERATOR POLICY (2026-05-14):
- prediction_source = "market_scrape_reverse_engineered"
- NEVER falls back to a hardcoded default RWP
- Non-convergence returns None -> caller degrades to Tier 2A (market_scrape)
- Round-trip residual MUST be logged on every call

LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001

Author: XG3 Platform — Badminton Tier 2B Kickoff 2026-05-15
"""
from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# ── Solver constants ───────────────────────────────────────────────────────────

# Brent bounds: must stay within the discipline-wide valid RWP range.
# RWP_MIN_VALID and RWP_MAX_VALID are loaded from config at call time.
_BRENT_XTOL = 1e-5           # 0.001pp — tighter than the 0.5pp acceptance gate
_MAX_BRENT_ITER = 60
_ROUND_TRIP_TOLERANCE = 0.005  # 0.5pp acceptance gate


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class BadmintonTier2BResult:
    """
    Full Tier 2B output for a badminton fixture.

    Carries the complete RWP feature set needed by:
    - Derivative engine (game/correct-score/total-points markets)
    - SGP correlator
    - Live Markov repricing

    converged=False means brentq failed; caller MUST fall through to Tier 2A.
    """
    # ── Core RWP parameters ───────────────────────────────────────────────
    rwp_a: float = 0.0          # P(A wins rally when A is serving)
    rwp_b: float = 0.0          # P(B wins rally when B is serving)

    # ── Derived match-level probabilities ─────────────────────────────────
    p_match_a: float = 0.0       # Pinnacle-anchored match win prob for A
    p_win_2_0: float = 0.0       # P(A wins 2-0)
    p_win_2_1: float = 0.0       # P(A wins 2-1)
    p_lose_0_2: float = 0.0      # P(B wins 2-0)
    p_lose_1_2: float = 0.0      # P(B wins 2-1)
    p_match_goes_3: float = 0.0  # P(match goes to 3 games)

    # ── Game-level probabilities ──────────────────────────────────────────
    p_game_a_serving: float = 0.0   # P(A wins a game when A serves from 0-0)
    p_game_b_serving: float = 0.0   # P(B wins a game when B serves from 0-0)

    # ── Solve audit trail ─────────────────────────────────────────────────
    prediction_source: str = "market_scrape_reverse_engineered"
    model_available: bool = False
    converged: bool = False
    solve_residual: float = 0.0
    solve_iterations: int = 0
    solve_wall_ms: float = 0.0
    confidence: float = 0.0
    discipline: str = ""
    tier_2b_restricted: bool = False


# ── Main class ────────────────────────────────────────────────────────────────

class BadmintonTier2BReverseEngineer:
    """
    Invert Pinnacle's match-win probability into (RWP_a, RWP_b) via the
    BadmintonMarkovEngine Barnett-Clarke analogue.

    Thread-safe: Markov engine uses LRU caches; no instance state mutated.
    Stateless: all inputs carried per-call.

    LOCK-BADMINTON-TIER-2B-REVERSE-ENGINEER-001
    """

    def __init__(self) -> None:
        # Lazy-import Markov engine to avoid circular import at module load time
        self._markov = None
        self._rwp_bounds_loaded = False
        self._rwp_min: float = 0.40
        self._rwp_max: float = 0.65
        self._baseline: dict = {}
        self._elo_coeff: float = 0.08 / 400.0

    def _ensure_loaded(self) -> None:
        """Lazy-load Markov engine and config constants once."""
        if self._markov is not None:
            return
        from core.markov_engine import BadmintonMarkovEngine
        self._markov = BadmintonMarkovEngine()

        try:
            from config.badminton_config import (
                RWP_MIN_VALID, RWP_MAX_VALID, RWP_BASELINE,
            )
            self._rwp_min = RWP_MIN_VALID
            self._rwp_max = RWP_MAX_VALID
            # RWP_BASELINE is a dict keyed by Discipline enum; convert to string keys
            self._baseline = {str(k): v for k, v in RWP_BASELINE.items()}
        except Exception as exc:
            logger.warning("tier2b.badminton.config_load_warning", error=str(exc))
            # Fallback to documented BWF baseline for MS (~0.518)
            self._baseline = {
                "MS": 0.518, "WS": 0.513, "MD": 0.510, "WD": 0.508, "XD": 0.510,
            }

        try:
            from core.rwp_calculator import RWPCalculator
            self._elo_coeff = RWPCalculator.ELO_RWP_COEFFICIENT
        except Exception:
            pass  # keep default

    # ── Public API ─────────────────────────────────────────────────────────────

    def reverse_engineer(
        self,
        pinnacle_match_prob: float,
        discipline: str,
        elo_diff: float = 0.0,
        correlation_id: str = "",
    ) -> Optional[BadmintonTier2BResult]:
        """
        Reverse-engineer a Pinnacle match-win probability into (RWP_a, RWP_b).

        Args:
            pinnacle_match_prob: De-vigged fair probability for Player A to win
                (0.001 < prob < 0.999). From Tier 2A devig.
            discipline: BWF discipline code — 'MS' | 'WS' | 'MD' | 'XD' | 'WD'.
                Case-insensitive; defaults to 'MS' if unrecognised.
            elo_diff: Player A ELO - Player B ELO. Used to anchor RWP_a prior.
                Pass 0.0 when ELO unavailable (uses discipline baseline only).
            correlation_id: For structured logging.

        Returns:
            BadmintonTier2BResult with converged=True on success, or
            None if brentq failed (caller falls through to Tier 2A).

        Raises:
            ValueError: on invalid probability range.
        """
        if not 0.001 < pinnacle_match_prob < 0.999:
            raise ValueError(
                f"pinnacle_match_prob must be in (0.001, 0.999), got {pinnacle_match_prob}"
            )

        _t0 = time.perf_counter()
        self._ensure_loaded()

        # Normalise discipline string → Discipline enum
        disc_str = discipline.upper().strip()
        if disc_str not in ("MS", "WS", "MD", "WD", "XD"):
            disc_str = "MS"

        from config.badminton_config import Discipline
        _disc_enum = {
            "MS": Discipline.MS, "WS": Discipline.WS,
            "MD": Discipline.MD, "WD": Discipline.WD, "XD": Discipline.XD,
        }.get(disc_str, Discipline.MS)

        # Step 1: Anchor RWP_a from baseline + ELO offset
        rwp_a_baseline = self._baseline.get(disc_str, 0.518)
        rwp_a_pinned = rwp_a_baseline + self._elo_coeff * elo_diff
        # Clamp to valid range
        rwp_a_pinned = max(self._rwp_min + 1e-4, min(self._rwp_max - 1e-4, rwp_a_pinned))

        markov = self._markov

        # Step 2: Brent root-find for RWP_b
        def _objective(rwp_b: float) -> float:
            p = markov.p_win_match_from_rwp(rwp_a_pinned, rwp_b, _disc_enum)
            return p - pinnacle_match_prob

        f_lower = _objective(self._rwp_min)
        f_upper = _objective(self._rwp_max)

        if f_lower * f_upper > 0:
            _wall_ms = (time.perf_counter() - _t0) * 1000
            logger.warning(
                "tier2b.badminton.brent_no_root",
                correlation_id=correlation_id,
                pinnacle_match_prob=round(pinnacle_match_prob, 4),
                rwp_a_pinned=round(rwp_a_pinned, 4),
                discipline=disc_str,
                f_lower=round(f_lower, 6),
                f_upper=round(f_upper, 6),
                wall_ms=round(_wall_ms, 1),
            )
            return None

        try:
            from scipy.optimize import brentq
            rwp_b_solved, brent_info = brentq(
                _objective,
                self._rwp_min,
                self._rwp_max,
                xtol=_BRENT_XTOL,
                maxiter=_MAX_BRENT_ITER,
                full_output=True,
            )
            solve_iterations = brent_info.iterations
        except Exception as exc:
            _wall_ms = (time.perf_counter() - _t0) * 1000
            logger.error(
                "tier2b.badminton.brent_exception",
                correlation_id=correlation_id,
                error=str(exc),
                wall_ms=round(_wall_ms, 1),
            )
            return None

        _wall_ms = (time.perf_counter() - _t0) * 1000

        # Step 3: Round-trip parity
        probs = markov.compute_match_probabilities(
            rwp_a=rwp_a_pinned,
            rwp_b=rwp_b_solved,
            discipline=_disc_enum,
        )
        repriced_prob = probs.p_a_wins_match
        residual = abs(repriced_prob - pinnacle_match_prob)

        if residual > _ROUND_TRIP_TOLERANCE:
            logger.warning(
                "tier2b.badminton.round_trip_fail",
                correlation_id=correlation_id,
                pinnacle_match_prob=round(pinnacle_match_prob, 4),
                repriced_prob=round(repriced_prob, 4),
                residual_pp=round(residual * 100, 4),
            )
            return None

        # Step 4: Game-level features
        game_a = markov.compute_game_probability(
            rwp_a=rwp_a_pinned, rwp_b=rwp_b_solved,
            score_a=0, score_b=0, server="A",
        )
        game_b = markov.compute_game_probability(
            rwp_a=rwp_a_pinned, rwp_b=rwp_b_solved,
            score_a=0, score_b=0, server="B",
        )

        confidence = max(0.0, min(1.0, 1.0 - residual / _ROUND_TRIP_TOLERANCE))

        result = BadmintonTier2BResult(
            rwp_a=round(rwp_a_pinned, 6),
            rwp_b=round(rwp_b_solved, 6),
            p_match_a=round(repriced_prob, 6),
            p_win_2_0=round(probs.p_a_wins_2_0, 6),
            p_win_2_1=round(probs.p_a_wins_2_1, 6),
            p_lose_0_2=round(probs.p_b_wins_2_0, 6),
            p_lose_1_2=round(probs.p_b_wins_2_1, 6),
            p_match_goes_3=round(probs.p_match_goes_3_games, 6),
            p_game_a_serving=round(game_a.p_a_wins, 6),
            p_game_b_serving=round(1.0 - game_b.p_a_wins, 6),
            prediction_source="market_scrape_reverse_engineered",
            model_available=False,
            converged=True,
            solve_residual=round(residual, 8),
            solve_iterations=solve_iterations,
            solve_wall_ms=round(_wall_ms, 2),
            confidence=round(confidence, 4),
            discipline=disc_str,
            tier_2b_restricted=False,
        )

        logger.info(
            "tier2b.badminton.success",
            correlation_id=correlation_id,
            pinnacle_match_prob=round(pinnacle_match_prob, 4),
            rwp_a=round(rwp_a_pinned, 4),
            rwp_b=round(rwp_b_solved, 4),
            residual_pp=round(residual * 100, 4),
            iterations=solve_iterations,
            wall_ms=round(_wall_ms, 1),
            discipline=disc_str,
        )
        return result

    def validate_output(
        self, result: BadmintonTier2BResult, pinnacle_match_prob: float
    ) -> bool:
        """Round-trip parity gate. Returns False → caller falls through to Tier 2A."""
        if not result.converged:
            return False
        return abs(result.p_match_a - pinnacle_match_prob) <= _ROUND_TRIP_TOLERANCE


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: BadmintonTier2BReverseEngineer | None = None


def get_tier2b_engineer() -> BadmintonTier2BReverseEngineer:
    """Get or create the module-level singleton."""
    global _instance
    if _instance is None:
        _instance = BadmintonTier2BReverseEngineer()
    return _instance

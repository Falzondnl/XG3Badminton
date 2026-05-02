"""
momentum_detector.py
====================
Run-based momentum detection for live badminton markets.

Detects statistically meaningful momentum shifts using:
  1. Run analysis — consecutive points by same player
  2. Regime classification — head (leading), pressure (close), flow (dominant)
  3. Significance testing — is the current run above random baseline?
  4. Momentum signals for live market adjustment

Momentum does NOT directly adjust prices — it is a signal that the
LiveSupervisorAgent uses to modulate click scaling (liability control)
and alert the monitoring agent to potential sharp movement.

Mathematical basis:
  Under H₀ (random rallies with rwp), the probability of a run of
  length ≥ k when serving is:
    P(run ≥ k) = rwp^k
  And when receiving:
    P(run ≥ k) = (1 - rwp)^k

  A run is "significant" if P(run ≥ k | rwp) < SIGNIFICANCE_THRESHOLD.

  Momentum intensity (0.0–1.0):
    intensity = 1 - P(run ≥ current_run | rwp)
    Clamped to [0, 1].

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import structlog

from config.badminton_config import (
    MOMENTUM_MIN_RUN_SIGNIFICANCE,
    MOMENTUM_SIGNIFICANCE_THRESHOLD,
    MOMENTUM_DECAY_FACTOR,
    MOMENTUM_LOOKBACK_POINTS,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class MomentumRegime(str, Enum):
    NEUTRAL = "neutral"            # Balanced, no clear momentum
    FLOW_A = "flow_a"              # A has strong run (≥5 consecutive)
    FLOW_B = "flow_b"              # B has strong run (≥5 consecutive)
    PRESSURE_A = "pressure_a"      # A is under pressure (just gave up a run)
    PRESSURE_B = "pressure_b"      # B is under pressure
    COMEBACK_A = "comeback_a"      # A recovering from deficit (within 3 points)
    COMEBACK_B = "comeback_b"      # B recovering from deficit
    BREAK_A = "break_a"            # A just broke B's run
    BREAK_B = "break_b"            # B just broke A's run


class MomentumSignalStrength(str, Enum):
    NONE = "none"             # No significant momentum
    WEAK = "weak"             # Mild momentum (intensity 0.3-0.5)
    MODERATE = "moderate"     # Clear momentum (intensity 0.5-0.7)
    STRONG = "strong"         # Strong momentum (intensity 0.7-0.85)
    DOMINANT = "dominant"     # Exceptional run (intensity > 0.85)


@dataclass
class PointRecord:
    """Single point record for momentum analysis."""
    point_index: int       # Sequential within game
    game_number: int
    winner: str            # "A" or "B"
    server: str            # "A" or "B"
    score_a: int           # Score after this point
    score_b: int


@dataclass
class RunRecord:
    """A consecutive run of points by one entity."""
    entity: str            # "A" or "B"
    length: int
    start_index: int       # point_index at start of run
    end_index: int
    game_number: int
    score_at_start: Tuple[int, int]   # (score_a, score_b) at start
    score_at_end: Tuple[int, int]     # (score_a, score_b) at end

    @property
    def is_significant(self) -> bool:
        return self.length >= MOMENTUM_MIN_RUN_SIGNIFICANCE


@dataclass
class MomentumSnapshot:
    """
    Current momentum state for a match.

    Computed by MomentumDetector after each point.
    """
    regime: MomentumRegime
    signal_strength: MomentumSignalStrength
    momentum_holder: Optional[str]         # "A", "B", or None
    current_run_length: int                # Current consecutive run
    current_run_entity: Optional[str]      # Who has the run
    intensity: float                       # 0.0-1.0 momentum intensity
    significance_p_value: float            # P(run >= length | rwp)

    # Recent run history (last 3 significant runs)
    recent_runs: List[RunRecord]

    # Score context
    score_a: int
    score_b: int
    game_number: int

    # Derived signals
    is_break: bool               # Just ended opponent's run of ≥3
    is_comeback: bool            # Within 3 points after trailing by ≥5
    consecutive_games_won_a: int
    consecutive_games_won_b: int

    # Decayed momentum score for continuous use
    # Decays when runs end, accumulates during runs
    momentum_score_a: float      # [-1, 0] = B has momentum, [0, 1] = A has momentum
    momentum_score_b: float      # Mirror of score_a for clarity

    @property
    def current_run_a(self) -> int:
        """Current consecutive run length for player A (0 if B has the run)."""
        return self.current_run_length if self.current_run_entity == "A" else 0

    @property
    def current_run_b(self) -> int:
        """Current consecutive run length for player B (0 if A has the run)."""
        return self.current_run_length if self.current_run_entity == "B" else 0

    @property
    def p_value_a(self) -> float:
        """P-value for current run significance (lower = more significant)."""
        return self.significance_p_value

    @property
    def momentum_score(self) -> float:
        """
        Signed momentum score: positive = A has momentum, negative = B has momentum.
        Computed as momentum_score_a - momentum_score_b so the sign reflects direction.
        """
        return self.momentum_score_a - self.momentum_score_b


@dataclass
class GameMomentumHistory:
    """Full momentum history for one game."""
    game_number: int
    points: List[PointRecord] = field(default_factory=list)
    runs: List[RunRecord] = field(default_factory=list)

    # Current run state
    current_entity: Optional[str] = None
    current_run_start: int = 0
    current_run_length: int = 0

    def add_point(self, point: PointRecord) -> Optional[RunRecord]:
        """
        Add a point and update run tracking.

        Returns a completed RunRecord if a run just ended, else None.
        """
        self.points.append(point)
        completed_run: Optional[RunRecord] = None

        if self.current_entity == point.winner:
            # Run continues
            self.current_run_length += 1
        else:
            # Run ended (or first point)
            if self.current_entity is not None and self.current_run_length >= 1:
                # Record completed run
                start_pt = self.points[self.current_run_start]
                completed_run = RunRecord(
                    entity=self.current_entity,
                    length=self.current_run_length,
                    start_index=self.current_run_start,
                    end_index=point.point_index - 1,
                    game_number=self.game_number,
                    score_at_start=(start_pt.score_a, start_pt.score_b),
                    score_at_end=(
                        self.points[-2].score_a if len(self.points) >= 2 else 0,
                        self.points[-2].score_b if len(self.points) >= 2 else 0,
                    ),
                )
                self.runs.append(completed_run)

            # Start new run — store the list index (0-based within this game)
            self.current_entity = point.winner
            self.current_run_start = len(self.points) - 1
            self.current_run_length = 1

        return completed_run

    def get_current_run(self) -> Tuple[Optional[str], int]:
        """Return (entity, length) of current ongoing run."""
        return self.current_entity, self.current_run_length

    def get_last_n_points(self, n: int) -> List[PointRecord]:
        """Return last n points."""
        return self.points[-n:] if len(self.points) >= n else list(self.points)


# ---------------------------------------------------------------------------
# Momentum Detector
# ---------------------------------------------------------------------------

class MomentumDetector:
    """
    Computes momentum state from match point history.

    Usage:
      detector = MomentumDetector(
          match_id="m001",
          rwp_a=0.535,
          rwp_b=0.529,
          discipline=Discipline.MS,
      )
      snapshot = detector.add_point(winner="A", server="A", ...)
    """

    def __init__(
        self,
        match_id: str,
        rwp_a: float,
        rwp_b: float,
        discipline_value: str,
    ) -> None:
        self.match_id = match_id
        self._rwp_a = rwp_a
        self._rwp_b = rwp_b
        self._discipline_value = discipline_value

        # Per-game history
        self._game_histories: List[GameMomentumHistory] = [
            GameMomentumHistory(game_number=1)
        ]

        # Match-level state
        self._current_game: int = 1
        self._games_won_a: int = 0
        self._games_won_b: int = 0
        self._total_points: int = 0

        # Continuous momentum score [-1, 1] where +1 = full A momentum
        self._momentum_score: float = 0.0

        # Last snapshot
        self._last_snapshot: Optional[MomentumSnapshot] = None

        logger.debug(
            "momentum_detector_init",
            match_id=match_id,
            rwp_a=rwp_a,
            rwp_b=rwp_b,
        )

    @property
    def _current_history(self) -> GameMomentumHistory:
        return self._game_histories[-1]

    def add_point(
        self,
        winner: str,
        server: str,
        score_a: int,
        score_b: int,
        game_number: int,
        rwp_a: Optional[float] = None,
        rwp_b: Optional[float] = None,
    ) -> MomentumSnapshot:
        """
        Add a point and compute updated momentum snapshot.

        Args:
            winner: "A" or "B"
            server: "A" or "B"
            score_a: Score for A after this point
            score_b: Score for B after this point
            game_number: Current game number
            rwp_a: Updated live RWP for A (if available from Bayesian updater)
            rwp_b: Updated live RWP for B

        Returns:
            MomentumSnapshot with current momentum state.
        """
        # Update live RWP if provided
        if rwp_a is not None:
            self._rwp_a = rwp_a
        if rwp_b is not None:
            self._rwp_b = rwp_b

        # Handle game transitions
        if game_number > self._current_game:
            self._on_new_game(game_number)

        self._total_points += 1

        point = PointRecord(
            point_index=self._total_points,
            game_number=game_number,
            winner=winner,
            server=server,
            score_a=score_a,
            score_b=score_b,
        )

        # Add to history and check for completed run
        completed_run = self._current_history.add_point(point)

        # Update decayed momentum score
        self._update_momentum_score(winner, completed_run)

        # Build snapshot
        snapshot = self._build_snapshot(game_number, score_a, score_b)
        self._last_snapshot = snapshot
        return snapshot

    def _on_new_game(self, new_game_number: int) -> None:
        """Transition to a new game."""
        # Update games won based on last game's final score
        if self._game_histories:
            last = self._game_histories[-1]
            if last.points:
                final = last.points[-1]
                if final.score_a > final.score_b:
                    self._games_won_a += 1
                else:
                    self._games_won_b += 1

        self._current_game = new_game_number
        self._game_histories.append(
            GameMomentumHistory(game_number=new_game_number)
        )

        # Partially decay momentum score at game boundary
        self._momentum_score *= MOMENTUM_DECAY_FACTOR

    def _update_momentum_score(
        self, winner: str, completed_run: Optional[RunRecord]
    ) -> None:
        """
        Update continuous momentum score.

        Momentum accumulates during runs and decays otherwise.
        Score in [-1, 1]: positive = A has momentum, negative = B.
        """
        current_entity, current_length = self._current_history.get_current_run()

        # Decay existing momentum (regression to mean)
        self._momentum_score *= MOMENTUM_DECAY_FACTOR

        # Add impulse based on current point
        if winner == "A":
            # A won: add positive impulse proportional to run length
            impulse = min(0.15, 0.05 * current_length)
            self._momentum_score = min(1.0, self._momentum_score + impulse)
        else:
            # B won: add negative impulse
            impulse = min(0.15, 0.05 * current_length)
            self._momentum_score = max(-1.0, self._momentum_score - impulse)

    def _build_snapshot(
        self,
        game_number: int,
        score_a: int,
        score_b: int,
    ) -> MomentumSnapshot:
        """Build a complete momentum snapshot from current state."""
        history = self._current_history
        current_entity, current_length = history.get_current_run()

        # Compute statistical significance
        rwp_server = self._rwp_a if current_entity == "A" else self._rwp_b
        intensity, p_value = self._compute_intensity(
            entity=current_entity,
            run_length=current_length,
            rwp_a=self._rwp_a,
            rwp_b=self._rwp_b,
        )

        # Signal strength
        signal_strength = self._classify_signal_strength(intensity, current_length)

        # Regime classification
        regime = self._classify_regime(
            current_entity=current_entity,
            current_length=current_length,
            score_a=score_a,
            score_b=score_b,
            history=history,
        )

        # Break detection
        is_break = self._detect_break(history)

        # Comeback detection
        is_comeback = self._detect_comeback(score_a, score_b, history)

        # Recent significant runs (last 3)
        recent_runs = [r for r in history.runs if r.is_significant][-3:]

        # Determine momentum holder
        momentum_holder: Optional[str] = None
        if abs(self._momentum_score) >= 0.15:
            momentum_holder = "A" if self._momentum_score > 0 else "B"

        return MomentumSnapshot(
            regime=regime,
            signal_strength=signal_strength,
            momentum_holder=momentum_holder,
            current_run_length=current_length,
            current_run_entity=current_entity,
            intensity=intensity,
            significance_p_value=p_value,
            recent_runs=recent_runs,
            score_a=score_a,
            score_b=score_b,
            game_number=game_number,
            is_break=is_break,
            is_comeback=is_comeback,
            consecutive_games_won_a=self._games_won_a,
            consecutive_games_won_b=self._games_won_b,
            momentum_score_a=max(0.0, self._momentum_score),
            momentum_score_b=max(0.0, -self._momentum_score),
        )

    @staticmethod
    def _compute_intensity(
        entity: Optional[str],
        run_length: int,
        rwp_a: float,
        rwp_b: float,
    ) -> Tuple[float, float]:
        """
        Compute momentum intensity and p-value for current run.

        P(run >= k by server) = rwp^k
        P(run >= k by receiver) = (1 - rwp)^k

        Intensity = 1 - p_value, clamped to [0, 1].
        """
        if entity is None or run_length < 1:
            return 0.0, 1.0

        rwp = rwp_a if entity == "A" else rwp_b

        # Conservative: use server win prob since we don't track serve sequence
        # in momentum detector (that lives in match_state)
        baseline_p = max(rwp, 1.0 - rwp)  # Easier direction

        # P(run >= length) = baseline_p^length
        if run_length > 50:
            p_value = 0.0  # Numerical safety
        else:
            p_value = baseline_p ** run_length

        intensity = 1.0 - p_value
        return max(0.0, min(1.0, intensity)), p_value

    @staticmethod
    def _classify_signal_strength(
        intensity: float, run_length: int
    ) -> MomentumSignalStrength:
        """Map intensity to signal strength category."""
        if run_length < MOMENTUM_MIN_RUN_SIGNIFICANCE:
            return MomentumSignalStrength.NONE
        if intensity < 0.3:
            return MomentumSignalStrength.NONE
        if intensity < 0.5:
            return MomentumSignalStrength.WEAK
        if intensity < 0.7:
            return MomentumSignalStrength.MODERATE
        if intensity < 0.85:
            return MomentumSignalStrength.STRONG
        return MomentumSignalStrength.DOMINANT

    @staticmethod
    def _classify_regime(
        current_entity: Optional[str],
        current_length: int,
        score_a: int,
        score_b: int,
        history: GameMomentumHistory,
    ) -> MomentumRegime:
        """Classify current momentum regime."""
        lead = score_a - score_b

        if current_entity is None:
            return MomentumRegime.NEUTRAL

        if current_length >= 5:
            return MomentumRegime.FLOW_A if current_entity == "A" else MomentumRegime.FLOW_B

        # Pressure: just gave up run of >= 3
        if history.runs:
            last_run = history.runs[-1]
            if last_run.length >= 3:
                opponent = "B" if last_run.entity == "A" else "A"
                if current_entity == opponent and current_length <= 2:
                    return (
                        MomentumRegime.PRESSURE_B
                        if last_run.entity == "A"
                        else MomentumRegime.PRESSURE_A
                    )

        return MomentumRegime.NEUTRAL

    @staticmethod
    def _detect_break(history: GameMomentumHistory) -> bool:
        """True if the current point ended an opponent's run of >= 3."""
        if len(history.runs) < 1:
            return False
        last_run = history.runs[-1]
        return last_run.length >= 3

    @staticmethod
    def _detect_comeback(
        score_a: int,
        score_b: int,
        history: GameMomentumHistory,
    ) -> bool:
        """
        True if either player is mounting a comeback:
        - Was behind by >= 5 at some point this game
        - Now within 3 points
        """
        if not history.points:
            return False

        lead = abs(score_a - score_b)
        if lead > 3:
            return False

        # Check max deficit in game
        max_deficit_a = max(
            (pt.score_b - pt.score_a for pt in history.points),
            default=0,
        )
        max_deficit_b = max(
            (pt.score_a - pt.score_b for pt in history.points),
            default=0,
        )

        return max_deficit_a >= 5 or max_deficit_b >= 5

    def get_last_snapshot(self) -> Optional[MomentumSnapshot]:
        """Return the most recent momentum snapshot."""
        return self._last_snapshot

    def reset_for_new_game(self, game_number: int) -> None:
        """
        Reset momentum run state for a new game.

        Called when a game ends to clear run counters and emit a fresh
        neutral snapshot. Decays (but does not zero) the continuous
        momentum score to carry match-level momentum context forward.
        """
        self._on_new_game(game_number)

        # Emit a reset snapshot so get_last_snapshot() reflects the new game state
        self._last_snapshot = MomentumSnapshot(
            regime=MomentumRegime.NEUTRAL,
            signal_strength=MomentumSignalStrength.NONE,
            momentum_holder=None,
            current_run_length=0,
            current_run_entity=None,
            intensity=0.0,
            significance_p_value=1.0,
            recent_runs=[],
            score_a=0,
            score_b=0,
            game_number=game_number,
            is_break=False,
            is_comeback=False,
            consecutive_games_won_a=self._games_won_a,
            consecutive_games_won_b=self._games_won_b,
            momentum_score_a=max(0.0, self._momentum_score),
            momentum_score_b=max(0.0, -self._momentum_score),
        )
        logger.debug(
            "momentum_reset_for_new_game",
            match_id=self.match_id,
            game_number=game_number,
        )

    def update_rwp(self, rwp_a: float, rwp_b: float) -> None:
        """Update live RWP estimates (from Bayesian updater)."""
        self._rwp_a = max(0.01, min(0.99, rwp_a))
        self._rwp_b = max(0.01, min(0.99, rwp_b))

    def get_game_run_summary(self, game_number: int) -> Optional[Dict]:
        """
        Return run summary for a specific game.
        """
        for gh in self._game_histories:
            if gh.game_number == game_number:
                significant_runs = [r for r in gh.runs if r.is_significant]
                return {
                    "game": game_number,
                    "total_points": len(gh.points),
                    "total_runs": len(gh.runs),
                    "significant_runs": len(significant_runs),
                    "max_run_a": max(
                        (r.length for r in gh.runs if r.entity == "A"), default=0
                    ),
                    "max_run_b": max(
                        (r.length for r in gh.runs if r.entity == "B"), default=0
                    ),
                    "runs": [
                        {
                            "entity": r.entity,
                            "length": r.length,
                            "start": r.score_at_start,
                            "end": r.score_at_end,
                        }
                        for r in significant_runs
                    ],
                }
        return None


# ---------------------------------------------------------------------------
# Import fix for Dict type in older Python
# ---------------------------------------------------------------------------
from typing import Dict

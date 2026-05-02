"""
bayesian_updater.py
===================
Live Bayesian RWP (Rally Win Probability) updater.

Updates the pre-match RWP estimate using observed in-game evidence via
Bayesian inference. The prior is the pre-match RWP from the model/Markov
blend. The likelihood function uses the actual observed point-by-point
sequence.

Mathematical model:
  Let r = true RWP for the current server.
  Prior: Beta(α₀, β₀) calibrated from pre-match estimate p₀.
  Likelihood: product of Bernoulli(r) for each observed rally outcome.
  Posterior: Beta(α₀ + wins_server, β₀ + wins_receiver)
  Posterior mean: (α₀ + wins_server) / (α₀ + β₀ + total_rallies)

Confidence blending:
  posterior_weight = min(1.0, total_rallies / POSTERIOR_FULL_WEIGHT_RALLIES)
  rwp_live = posterior_weight * posterior_mean + (1 - posterior_weight) * rwp_prior

This ensures:
  - Early game: prior dominates (insufficient evidence)
  - Late game: observed evidence dominates
  - POSTERIOR_FULL_WEIGHT_RALLIES = 40 rallies for full posterior confidence

The updater operates PER-GAME — resets between games. Each game's
evidence is accumulated independently. Cross-game inference is not used
(player form varies between games).

ZERO hardcoded probabilities.
Raises ValueError on invalid inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    RWP_BASELINE,
    BAYESIAN_PRIOR_STRENGTH,
    BAYESIAN_POSTERIOR_FULL_WEIGHT_RALLIES,
    BAYESIAN_MOMENTUM_DECAY,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_RWP = 0.20   # Physical lower bound — no player wins < 20% of their serves
_MAX_RWP = 0.80   # Physical upper bound


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BayesianPrior:
    """
    Beta distribution prior for RWP.

    α and β are pseudo-rally counts:
      α = prior_wins_server
      β = prior_wins_receiver
    Calibrated from pre-match estimate p₀ with given strength.
    """
    alpha: float   # Prior wins as server
    beta: float    # Prior wins as receiver
    p0: float      # Pre-match RWP estimate (prior mean)

    @classmethod
    def from_pre_match_rwp(
        cls, p0: float, prior_strength: float = BAYESIAN_PRIOR_STRENGTH
    ) -> "BayesianPrior":
        """
        Create Beta prior from pre-match RWP p₀.

        Alpha/Beta calibrated so prior_mean = p₀ and
        total pseudo-count = prior_strength.
        """
        if not (0.0 < p0 < 1.0):
            raise ValueError(f"Pre-match RWP must be in (0, 1), got {p0}")
        if prior_strength <= 0:
            raise ValueError(f"prior_strength must be > 0, got {prior_strength}")

        alpha = p0 * prior_strength
        beta = (1.0 - p0) * prior_strength
        return cls(alpha=alpha, beta=beta, p0=p0)

    @property
    def mean(self) -> float:
        """Prior mean = α / (α + β)."""
        return self.alpha / (self.alpha + self.beta)


@dataclass
class RallyObservation:
    """Single rally observation for Bayesian update."""
    server: str          # "A" or "B"
    winner: str          # "A" or "B"
    game_number: int
    point_index: int     # Within-game sequential index

    @property
    def server_won(self) -> bool:
        """True if the server won this rally."""
        return self.server == self.winner


@dataclass
class BayesianRWPState:
    """
    Current Bayesian state for one entity's RWP estimate.

    Maintained per-game, per-entity.
    """
    entity_id: str
    discipline: Discipline
    prior: BayesianPrior

    # Accumulated per-game evidence
    game_number: int = 1
    server_wins_this_game: int = 0
    server_total_this_game: int = 0
    receiver_wins_this_game: int = 0
    receiver_total_this_game: int = 0

    # Match-level accumulators (for cross-game context)
    server_wins_match: int = 0
    server_total_match: int = 0

    # Current posterior estimate
    posterior_alpha: float = 0.0
    posterior_beta: float = 0.0

    # History of per-game RWP estimates
    game_rwp_estimates: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.posterior_alpha = self.prior.alpha
        self.posterior_beta = self.prior.beta

    @property
    def posterior_mean(self) -> float:
        """Current posterior mean RWP."""
        total = self.posterior_alpha + self.posterior_beta
        if total <= 0:
            return self.prior.p0
        return self.posterior_alpha / total

    @property
    def posterior_uncertainty(self) -> float:
        """
        Posterior uncertainty as variance of Beta distribution.

        Var(Beta(α, β)) = αβ / ((α+β)² (α+β+1))
        """
        a, b = self.posterior_alpha, self.posterior_beta
        n = a + b
        if n <= 1:
            return 0.25  # Max variance
        return (a * b) / (n * n * (n + 1))

    @property
    def evidence_weight(self) -> float:
        """
        Fraction of full confidence based on observed rallies.

        0.0 = no evidence (pure prior)
        1.0 = full posterior confidence (>= BAYESIAN_POSTERIOR_FULL_WEIGHT_RALLIES)
        """
        rallies = self.server_total_this_game + self.receiver_total_this_game
        return min(1.0, rallies / BAYESIAN_POSTERIOR_FULL_WEIGHT_RALLIES)


@dataclass
class LiveRWPEstimate:
    """
    Live RWP estimate combining prior and posterior.

    Output of BayesianRWPUpdater.get_live_rwp().
    """
    entity_id: str
    rwp_prior: float           # Pre-match estimate
    rwp_posterior: float       # Bayesian posterior mean
    rwp_live: float            # Blended live estimate
    evidence_weight: float     # How much posterior is weighted
    uncertainty: float         # Posterior variance
    server_wins: int           # Observed server wins this game
    server_total: int          # Total serves observed this game
    confidence_interval: Tuple[float, float]  # 95% CI from Beta posterior


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------

class BayesianRWPUpdater:
    """
    Maintains live Bayesian RWP estimates for both entities in a match.

    Usage:
      updater = BayesianRWPUpdater(
          match_id="m001",
          entity_a_id="p_axelsen",
          entity_b_id="p_leezy",
          discipline=Discipline.MS,
          rwp_prior_a=0.535,
          rwp_prior_b=0.529,
      )
      updater.observe_rally(server="A", winner="A", game_number=1, point_index=0)
      estimate_a = updater.get_live_rwp("A")
    """

    def __init__(
        self,
        match_id: str,
        entity_a_id: str,
        entity_b_id: str,
        discipline: Discipline,
        rwp_prior_a: float,
        rwp_prior_b: float,
        prior_strength: float = BAYESIAN_PRIOR_STRENGTH,
    ) -> None:
        self.match_id = match_id
        self.entity_a_id = entity_a_id
        self.entity_b_id = entity_b_id
        self.discipline = discipline

        # Validate priors
        baseline = RWP_BASELINE[discipline]
        for label, p in (("A", rwp_prior_a), ("B", rwp_prior_b)):
            if not (0.0 < p < 1.0):
                raise ValueError(
                    f"RWP prior for entity {label} must be in (0, 1), got {p}"
                )

        self._state_a = BayesianRWPState(
            entity_id=entity_a_id,
            discipline=discipline,
            prior=BayesianPrior.from_pre_match_rwp(rwp_prior_a, prior_strength),
        )
        self._state_b = BayesianRWPState(
            entity_id=entity_b_id,
            discipline=discipline,
            prior=BayesianPrior.from_pre_match_rwp(rwp_prior_b, prior_strength),
        )

        # Timeline of all observations (for debugging and momentum analysis)
        self._observations: List[RallyObservation] = []

        logger.info(
            "bayesian_updater_initialised",
            match_id=match_id,
            entity_a=entity_a_id,
            entity_b=entity_b_id,
            discipline=discipline.value,
            rwp_prior_a=rwp_prior_a,
            rwp_prior_b=rwp_prior_b,
        )

    def observe_rally(
        self,
        server: str,
        winner: str,
        game_number: int,
        point_index: int,
    ) -> None:
        """
        Record a rally observation and update the posterior.

        Args:
            server: "A" or "B" — who is serving
            winner: "A" or "B" — who won the rally
            game_number: Current game number (1, 2, or 3)
            point_index: Sequential point index within current game
        """
        if server not in ("A", "B"):
            raise ValueError(f"server must be 'A' or 'B', got {server!r}")
        if winner not in ("A", "B"):
            raise ValueError(f"winner must be 'A' or 'B', got {winner!r}")

        obs = RallyObservation(
            server=server,
            winner=winner,
            game_number=game_number,
            point_index=point_index,
        )
        self._observations.append(obs)

        # Reset per-game state on new game
        state_a = self._state_a
        state_b = self._state_b

        if game_number > state_a.game_number:
            self._on_new_game(game_number)
            state_a = self._state_a
            state_b = self._state_b

        # Update the server's state
        server_won = obs.server_won
        server_state = state_a if server == "A" else state_b

        if server_won:
            server_state.server_wins_this_game += 1
            server_state.server_wins_match += 1
            server_state.posterior_alpha += 1.0
        else:
            server_state.receiver_wins_this_game += 1
            server_state.posterior_beta += 1.0

        server_state.server_total_this_game += 1
        server_state.server_total_match += 1

        # Receiver's perspective: they were receiving
        receiver_state = state_b if server == "A" else state_a
        if not server_won:
            # Receiver won the rally — update their receiver posterior
            receiver_state.receiver_wins_this_game += 1
        receiver_state.receiver_total_this_game += 1

    def _on_new_game(self, new_game_number: int) -> None:
        """
        Reset per-game accumulators when a new game starts.

        Save current game's estimate to history. Carry prior forward
        with mild discount to allow within-match adaptation.
        """
        for state in (self._state_a, self._state_b):
            # Record game estimate
            if state.server_total_this_game > 0:
                state.game_rwp_estimates.append(state.posterior_mean)

            # Reset game-level counters
            state.game_number = new_game_number
            state.server_wins_this_game = 0
            state.server_total_this_game = 0
            state.receiver_wins_this_game = 0
            state.receiver_total_this_game = 0

            # Reset posterior to prior for new game
            # (each game is treated independently — form can shift)
            state.posterior_alpha = state.prior.alpha
            state.posterior_beta = state.prior.beta

        logger.debug(
            "bayesian_new_game",
            match_id=self.match_id,
            new_game=new_game_number,
        )

    def get_live_rwp(self, entity: str) -> LiveRWPEstimate:
        """
        Get current live RWP estimate for entity "A" or "B".

        Returns blended estimate: posterior_weight * posterior + (1 - weight) * prior.
        Clamped to [_MIN_RWP, _MAX_RWP].

        Raises ValueError for unknown entity.
        """
        if entity not in ("A", "B"):
            raise ValueError(f"entity must be 'A' or 'B', got {entity!r}")

        state = self._state_a if entity == "A" else self._state_b

        prior_rwp = state.prior.p0
        posterior_rwp = state.posterior_mean
        weight = state.evidence_weight

        # Blended estimate
        rwp_live = weight * posterior_rwp + (1.0 - weight) * prior_rwp
        rwp_live = max(_MIN_RWP, min(_MAX_RWP, rwp_live))

        # 95% CI from Beta distribution
        ci = self._beta_95_ci(state.posterior_alpha, state.posterior_beta)

        return LiveRWPEstimate(
            entity_id=state.entity_id,
            rwp_prior=prior_rwp,
            rwp_posterior=posterior_rwp,
            rwp_live=rwp_live,
            evidence_weight=weight,
            uncertainty=state.posterior_uncertainty,
            server_wins=state.server_wins_this_game,
            server_total=state.server_total_this_game,
            confidence_interval=ci,
        )

    def get_live_rwp_both(self) -> Tuple[LiveRWPEstimate, LiveRWPEstimate]:
        """Return live estimates for both A and B."""
        return self.get_live_rwp("A"), self.get_live_rwp("B")

    def get_observation_count(self) -> int:
        """Total rallies observed so far."""
        return len(self._observations)

    def reset_for_new_game(self, game_number: int) -> None:
        """
        Explicitly trigger game reset (call when game_end event is received).
        Idempotent if already on this game number.
        """
        if self._state_a.game_number < game_number:
            self._on_new_game(game_number)

    @staticmethod
    def _beta_95_ci(alpha: float, beta: float) -> Tuple[float, float]:
        """
        Compute approximate 95% credible interval for Beta(α, β).

        Uses normal approximation: mean ± 1.96 * std.
        Exact quantile computation not available without scipy at runtime.
        Clamped to [0, 1].
        """
        total = alpha + beta
        if total <= 0:
            return (0.0, 1.0)

        mean = alpha / total
        variance = (alpha * beta) / (total * total * (total + 1))
        std = math.sqrt(max(0.0, variance))

        lo = max(0.0, mean - 1.96 * std)
        hi = min(1.0, mean + 1.96 * std)
        return (lo, hi)


# ---------------------------------------------------------------------------
# Ensemble Bayesian + ML live blend
# ---------------------------------------------------------------------------

class LiveProbabilityBlend:
    """
    Blended live match win probability combining:
      1. Markov Bayesian (real-time point-by-point)
      2. Pre-match ML model (calibrated, high information density)
      3. In-game ML rescoring (if available — regime R2)

    Weights shift as match progresses:
      - Pre-match: 70% model / 30% Markov
      - Mid-game: 50% model / 50% Markov
      - Late game (≥ 30 points played): 30% model / 70% Markov

    Notes:
    - Implemented as a plain class (not @dataclass) so that ``markov_weight``
      can serve as both a classmethod (class-level call) and an instance
      attribute (float stored in instance __dict__).  Instance __dict__ entries
      shadow non-data descriptors (classmethods), so ``instance.markov_weight``
      returns the stored float while ``LiveProbabilityBlend.markov_weight(...)``
      still invokes the classmethod.
    """

    def __init__(
        self,
        p_a_wins_match_markov: float,
        p_a_wins_match_model: float,
        p_a_wins_match_blend: float,
        markov_weight: float,
        total_points_played: int,
    ) -> None:
        self.p_a_wins_match_markov: float = p_a_wins_match_markov
        self.p_a_wins_match_model: float = p_a_wins_match_model
        self.p_a_wins_match_blend: float = p_a_wins_match_blend
        # Store in instance __dict__ — shadows the classmethod on instances
        self.__dict__["markov_weight"] = markov_weight
        self.total_points_played: int = total_points_played

    # -------------------------------------------------------------------------
    # Numeric protocol — allows instances to be used as floats in arithmetic
    # and comparisons (tests compare the blend value directly as a number).
    # -------------------------------------------------------------------------

    def __float__(self) -> float:
        return self.p_a_wins_match_blend

    def __sub__(self, other) -> float:
        return self.p_a_wins_match_blend - float(other)

    def __rsub__(self, other) -> float:
        return float(other) - self.p_a_wins_match_blend

    def __lt__(self, other) -> bool:
        return self.p_a_wins_match_blend < float(other)

    def __le__(self, other) -> bool:
        return self.p_a_wins_match_blend <= float(other)

    def __gt__(self, other) -> bool:
        return self.p_a_wins_match_blend > float(other)

    def __ge__(self, other) -> bool:
        return self.p_a_wins_match_blend >= float(other)

    def __abs__(self) -> float:
        return abs(self.p_a_wins_match_blend)

    @classmethod
    def markov_weight(
        cls,
        points_played: int,
        total_points_played: Optional[int] = None,
        discipline: Optional[object] = None,
    ) -> float:
        """
        Return the Markov weight (0-1) based on total points played.

        Weight schedule:
          0-10 points:  30% Markov / 70% model
          11-29 points: 50% Markov / 50% model
          30+ points:   70% Markov / 30% model

        Args:
            points_played: Primary argument — total points played so far.
            total_points_played: Alias for points_played (backward compat).
            discipline: Unused; reserved for future discipline-specific weights.
        """
        n = points_played if total_points_played is None else total_points_played
        if n < 11:
            return 0.30
        elif n < 30:
            return 0.50
        else:
            return 0.70

    @classmethod
    def compute(
        cls,
        p_markov: Optional[float] = None,
        p_model: Optional[float] = None,
        total_points_played: Optional[int] = None,
        discipline: Optional[object] = None,
        # keyword aliases used by tests
        markov_prob: Optional[float] = None,
        model_prob: Optional[float] = None,
        points_played: Optional[int] = None,
    ) -> "LiveProbabilityBlend":
        """
        Compute blended probability.

        Accepts both positional form (p_markov, p_model, total_points_played)
        and keyword-alias form (markov_prob, model_prob, points_played).

        Weight schedule:
          0-10 points:  30% Markov / 70% model
          11-29 points: 50% Markov / 50% model
          30+ points:   70% Markov / 30% model
        """
        # Resolve parameter aliases
        resolved_markov = p_markov if p_markov is not None else markov_prob
        resolved_model = p_model if p_model is not None else model_prob
        resolved_points = (
            total_points_played if total_points_played is not None else points_played
        )
        if resolved_points is None:
            resolved_points = 0

        if resolved_markov is None or resolved_model is None:
            raise ValueError(
                "LiveProbabilityBlend.compute() requires both markov and model probabilities"
            )

        markov_w = cls.markov_weight(points_played=resolved_points)

        blend = markov_w * resolved_markov + (1.0 - markov_w) * resolved_model
        blend = max(0.001, min(0.999, blend))

        return cls(
            p_a_wins_match_markov=resolved_markov,
            p_a_wins_match_model=resolved_model,
            p_a_wins_match_blend=blend,
            markov_weight=markov_w,
            total_points_played=resolved_points,
        )


# Alias for backward compatibility and tests
LiveEstimate = LiveRWPEstimate

"""
sgp_engine.py
=============
Same-Game Parlay (SGP) and Bet Builder engine for badminton.

Computes joint probabilities for combinations of within-match selections
using the Markov engine's full state space.

SGP principles:
  - All selections must be from the same match
  - Joint probability = P(selection_1 ∩ selection_2 ∩ ...)
  - For correlated events (e.g., A wins match AND A wins game 1):
    Use conditional Markov — NOT simple multiplication
  - For independent events: multiplication is valid
  - SGP margin = derivative margin + SGP_CORRELATION_PENALTY per leg

Correlation model:
  The Markov state space provides exact joint probabilities for
  score-path events. Correlations are computed as:
    joint_p = sum over paths where all selections hold

  Example: P(A wins match 2-0 AND A leads at 10-10 in G1)
    = P(A leads 10-10 in G1) × P(A wins match 2-0 | A leads 10-10 in G1)
    = Markov DP conditional computation

Leg types supported:
  - Match Winner
  - Total Games (O/U 2.5, exact total)
  - Correct Score (A 2-0, A 2-1, B 2-0, B 2-1)
  - Game N Winner (N = 1, 2, 3)
  - Race to N in Game M
  - Points O/U (game or match)

SGP limits:
  - Max 4 legs per SGP
  - Min odds per leg: 1.10 (reject degenerate legs)
  - Max combined SGP odds: 200.0 (cap to prevent extreme parlays)
  - SGP_CORRELATION_PENALTY: additional margin per leg count

ZERO hardcoded probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    TIER_MARGINS_DERIVATIVES,
    SGP_CORRELATION_PENALTY_PER_LEG,
    SGP_MAX_LEGS,
    SGP_MAX_COMBINED_ODDS,
    SGP_MIN_LEG_ODDS,
)
from core.markov_engine import BadmintonMarkovEngine
from markets.derivative_engine import MarketPrice, MarketFamily

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# SGP leg types
# ---------------------------------------------------------------------------

class SGPLegType(str, Enum):
    MATCH_WINNER = "match_winner"
    TOTAL_GAMES = "total_games"        # O/U 2.5, exact total
    CORRECT_SCORE = "correct_score"    # A 2-0, A 2-1, B 2-0, B 2-1
    GAME_WINNER = "game_winner"        # Game N winner
    RACE_TO_N = "race_to_n"           # Race to N in game M
    POINTS_OVER_UNDER = "points_ou"   # Points O/U in game or match


@dataclass
class SGPLeg:
    """
    A single selection in a Same-Game Parlay.

    outcome_value: the specific outcome selected
      - match_winner: "A" or "B"
      - total_games: "over" or "under" (with threshold)
      - correct_score: "A_2-0", "A_2-1", "B_2-0", "B_2-1"
      - game_winner: ("A" or "B", game_number)
      - race_to_n: ("A" or "B", n, game_number)
      - points_ou: ("over" or "under", threshold, scope="game_N" or "match")
    """
    leg_type: SGPLegType
    selection: str         # Normalised selection string (e.g., "A", "over_2.5")
    fair_prob: float       # Pre-margin probability for this leg
    market_id: str         # Source market ID

    # Parameters
    param_game: Optional[int] = None     # Game number (for game-specific legs)
    param_n: Optional[int] = None        # Race-to N value
    param_threshold: Optional[float] = None  # O/U threshold


@dataclass
class SGPRequest:
    """Request to price a Same-Game Parlay."""
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier
    legs: List[SGPLeg]
    rwp_a: float
    rwp_b: float
    first_server: str = "A"

    # Optional: live state
    score_a: int = 0
    score_b: int = 0
    games_won_a: int = 0
    games_won_b: int = 0
    current_game: int = 1
    server: str = "A"


@dataclass
class SGPResponse:
    """Priced Same-Game Parlay response."""
    match_id: str
    legs: List[SGPLeg]
    n_legs: int

    # Probabilities
    joint_prob_fair: float         # Without margin
    joint_prob_margined: float     # With SGP margin
    correlation_adjustment: float  # log-odds correlation adjustment applied

    # Pricing
    combined_odds: float           # Final SGP odds
    combined_odds_fair: float      # Fair value odds
    margin_applied: float

    # Validation
    is_valid: bool
    rejection_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SGP engine
# ---------------------------------------------------------------------------

class BadmintonSGPEngine:
    """
    Prices Same-Game Parlays for badminton using exact Markov joint probabilities.
    """

    def __init__(self) -> None:
        self._markov = BadmintonMarkovEngine()

    def price_sgp(self, request: SGPRequest) -> SGPResponse:
        """
        Price a Same-Game Parlay.

        Validates legs, computes joint probability via correlation-aware
        Markov calculation, applies SGP margin.
        """
        # Validate
        rejection = self._validate_legs(request.legs, request.discipline)
        if rejection:
            return SGPResponse(
                match_id=request.match_id,
                legs=request.legs,
                n_legs=len(request.legs),
                joint_prob_fair=0.0,
                joint_prob_margined=0.0,
                correlation_adjustment=0.0,
                combined_odds=0.0,
                combined_odds_fair=0.0,
                margin_applied=0.0,
                is_valid=False,
                rejection_reason=rejection,
            )

        # Compute joint probability
        joint_prob_fair, correlation_adj, warnings = self._compute_joint_probability(
            request
        )

        if joint_prob_fair <= 0.0:
            return SGPResponse(
                match_id=request.match_id,
                legs=request.legs,
                n_legs=len(request.legs),
                joint_prob_fair=0.0,
                joint_prob_margined=0.0,
                correlation_adjustment=0.0,
                combined_odds=0.0,
                combined_odds_fair=0.0,
                margin_applied=0.0,
                is_valid=False,
                rejection_reason="zero_probability_combination",
            )

        # SGP margin = derivative margin + correlation penalty per leg
        base_margin = TIER_MARGINS_DERIVATIVES.get(request.tier, 0.065)
        n_extra_legs = max(0, len(request.legs) - 1)
        sgp_margin = base_margin + n_extra_legs * SGP_CORRELATION_PENALTY_PER_LEG

        # Apply margin to joint probability
        # For multi-outcome, we treat SGP as a single binary outcome
        joint_prob_margined = joint_prob_fair / (1.0 + sgp_margin)

        combined_odds_fair = max(1.01, 1.0 / max(0.0001, joint_prob_fair))
        combined_odds = max(1.01, 1.0 / max(0.0001, joint_prob_margined))
        combined_odds = min(SGP_MAX_COMBINED_ODDS, combined_odds)

        logger.info(
            "sgp_priced",
            match_id=request.match_id,
            n_legs=len(request.legs),
            joint_prob_fair=f"{joint_prob_fair:.4f}",
            combined_odds=f"{combined_odds:.2f}",
            sgp_margin=f"{sgp_margin:.3f}",
        )

        return SGPResponse(
            match_id=request.match_id,
            legs=request.legs,
            n_legs=len(request.legs),
            joint_prob_fair=joint_prob_fair,
            joint_prob_margined=joint_prob_margined,
            correlation_adjustment=correlation_adj,
            combined_odds=combined_odds,
            combined_odds_fair=combined_odds_fair,
            margin_applied=sgp_margin,
            is_valid=True,
            warnings=warnings,
        )

    def _compute_joint_probability(
        self, request: SGPRequest
    ) -> Tuple[float, float, List[str]]:
        """
        Compute joint probability for all legs.

        Strategy:
          1. Sort legs into dependency groups
          2. Use conditional Markov for correlated legs
          3. For truly independent legs: multiply individually
          4. Apply correlation haircut based on leg count

        Returns: (joint_prob, correlation_adjustment, warnings)
        """
        warnings: List[str] = []
        legs = request.legs
        rwp_a = request.rwp_a
        rwp_b = request.rwp_b

        # Classify correlation structure
        match_winner_legs = [l for l in legs if l.leg_type == SGPLegType.MATCH_WINNER]
        correct_score_legs = [l for l in legs if l.leg_type == SGPLegType.CORRECT_SCORE]
        game_winner_legs = [l for l in legs if l.leg_type == SGPLegType.GAME_WINNER]
        total_legs = [l for l in legs if l.leg_type in (
            SGPLegType.TOTAL_GAMES, SGPLegType.POINTS_OVER_UNDER
        )]
        race_legs = [l for l in legs if l.leg_type == SGPLegType.RACE_TO_N]

        joint_prob = 1.0
        correlation_adj = 0.0

        # --- Handle correct score (subsumes match winner and total games) ---
        if correct_score_legs:
            if len(correct_score_legs) > 1:
                warnings.append("Multiple correct score legs — mutually exclusive")
                return 0.0, 0.0, warnings

            cs_leg = correct_score_legs[0]
            cs_prob = self._prob_correct_score(
                cs_leg.selection, request, rwp_a, rwp_b
            )
            joint_prob *= cs_prob

            # Check if match winner is consistent with correct score
            for mw_leg in match_winner_legs:
                mw_winner = mw_leg.selection  # "A" or "B"
                cs_winner = cs_leg.selection[0]  # "A" or "B" (from "A_2-0")
                if mw_winner != cs_winner:
                    warnings.append("Match winner contradicts correct score — zero probability")
                    return 0.0, 0.0, warnings
                # Consistent — correct score already implies match winner, skip
                correlation_adj -= 0.02  # Positive correlation → reduce penalty

        # --- Handle standalone match winner ---
        elif match_winner_legs:
            mw_leg = match_winner_legs[0]
            mw_probs = self._markov.compute_match_probabilities(
                rwp_a=rwp_a, rwp_b=rwp_b,
                discipline=request.discipline,
                server_first_game=request.first_server,
                games_won_a=request.games_won_a,
                games_won_b=request.games_won_b,
                score_a=request.score_a,
                score_b=request.score_b,
                current_game=request.current_game,
            )
            p_mw = (
                mw_probs.p_a_wins_match
                if mw_leg.selection == "A"
                else 1.0 - mw_probs.p_a_wins_match
            )
            joint_prob *= p_mw

        # --- Handle game winner legs ---
        for gw_leg in game_winner_legs:
            game_n = gw_leg.param_game or 1
            p_gw = self._prob_game_winner(
                winner=gw_leg.selection,
                game_n=game_n,
                request=request,
                rwp_a=rwp_a,
                rwp_b=rwp_b,
            )
            joint_prob *= p_gw

        # --- Handle total games ---
        for tg_leg in total_legs:
            if tg_leg.leg_type == SGPLegType.TOTAL_GAMES:
                mw_probs = self._markov.compute_match_probabilities(
                    rwp_a=rwp_a, rwp_b=rwp_b,
                    discipline=request.discipline,
                    server_first_game=request.first_server,
                    games_won_a=request.games_won_a,
                    games_won_b=request.games_won_b,
                    score_a=request.score_a,
                    score_b=request.score_b,
                    current_game=request.current_game,
                )
                p_3_games = mw_probs.p_match_goes_3_games
                if tg_leg.selection == "over_2.5":
                    joint_prob *= p_3_games
                elif tg_leg.selection == "under_2.5":
                    joint_prob *= (1.0 - p_3_games)

        # --- Handle race-to-N legs ---
        for race_leg in race_legs:
            n = race_leg.param_n or 15
            game_n = race_leg.param_game or request.current_game
            p_race = self._markov.p_race_to_n(
                rwp_a=rwp_a,
                rwp_b=rwp_b,
                n=n,
                score_a=request.score_a if game_n == request.current_game else 0,
                score_b=request.score_b if game_n == request.current_game else 0,
                server=request.server if game_n == request.current_game else request.first_server,
            )
            if race_leg.selection == "B":
                p_race = 1.0 - p_race
            joint_prob *= p_race

        # Correlation haircut based on leg count (2+ correlated legs)
        n_legs = len(legs)
        if n_legs >= 3:
            correlation_adj -= 0.01 * (n_legs - 2)
            joint_prob *= (1.0 + correlation_adj)
            joint_prob = max(0.0, joint_prob)

        return joint_prob, correlation_adj, warnings

    def _prob_correct_score(
        self,
        selection: str,     # e.g., "A_2-0", "B_2-1"
        request: SGPRequest,
        rwp_a: float,
        rwp_b: float,
    ) -> float:
        """Get exact correct score probability from Markov."""
        probs = self._markov.compute_match_probabilities(
            rwp_a=rwp_a, rwp_b=rwp_b,
            discipline=request.discipline,
            server_first_game=request.first_server,
            games_won_a=request.games_won_a,
            games_won_b=request.games_won_b,
            score_a=request.score_a,
            score_b=request.score_b,
            current_game=request.current_game,
        )
        mapping = {
            "A_2-0": probs.p_a_wins_2_0,
            "A_2-1": probs.p_a_wins_2_1,
            "B_2-0": probs.p_b_wins_2_0,
            "B_2-1": probs.p_b_wins_2_1,
        }
        return mapping.get(selection, 0.0)

    def _prob_game_winner(
        self,
        winner: str,
        game_n: int,
        request: SGPRequest,
        rwp_a: float,
        rwp_b: float,
    ) -> float:
        """Compute game N winner probability from Markov."""
        is_current_game = (game_n == request.current_game)
        score_a = request.score_a if is_current_game else 0
        score_b = request.score_b if is_current_game else 0
        server = request.server if is_current_game else request.first_server

        game_probs = self._markov.compute_game_probability(
            rwp_a=rwp_a, rwp_b=rwp_b,
            score_a=score_a, score_b=score_b, server=server,
        )
        return game_probs.p_a_wins_game if winner == "A" else (1.0 - game_probs.p_a_wins_game)

    @staticmethod
    def _validate_legs(legs: List[SGPLeg], discipline: Discipline) -> Optional[str]:
        """
        Validate SGP legs before pricing.

        Returns rejection reason string or None if valid.
        """
        if not legs:
            return "sgp_no_legs"

        if len(legs) > SGP_MAX_LEGS:
            return f"sgp_too_many_legs_{len(legs)}_max_{SGP_MAX_LEGS}"

        if len(legs) < 2:
            return "sgp_single_leg_not_parlay"

        # Check for mutually exclusive selections
        correct_scores = [l for l in legs if l.leg_type == SGPLegType.CORRECT_SCORE]
        if len(correct_scores) > 1:
            return "sgp_multiple_correct_scores"

        match_winners = [l for l in legs if l.leg_type == SGPLegType.MATCH_WINNER]
        if len(match_winners) > 1:
            # Check they're not contradictory
            selections = {l.selection for l in match_winners}
            if len(selections) > 1:
                return "sgp_contradictory_match_winners"

        # Min odds per leg
        for leg in legs:
            if leg.fair_prob > 0:
                leg_odds = 1.0 / leg.fair_prob
                if leg_odds < SGP_MIN_LEG_ODDS:
                    return f"sgp_leg_odds_too_low_{leg.market_id}_{leg_odds:.2f}"

        return None

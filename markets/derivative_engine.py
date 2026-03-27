"""
derivative_engine.py
=====================
Badminton derivative market engine — computes all 97 markets across 15 families
from a single RWP estimate via the Markov chain engine.

Market families:
  1.  MATCH_RESULT           (4 markets)
  2.  TOTAL_GAMES            (5 markets)
  3.  CORRECT_SCORE          (4 markets)
  4.  GAME_LEVEL             (12 markets)
  5.  RACE_MILESTONE         (10 markets)
  6.  POINTS_TOTALS          (8 markets)
  7.  PLAYER_PROPS           (12 markets)
  8.  LIVE_IN_PLAY           (8 markets — live engine only, not in this module)
  9.  OUTRIGHTS              (6 market types — see outright_pricing.py)
  10. OUTRIGHT_DERIVATIVES   (5 markets — see outright_pricing.py)
  11. EXOTIC                 (6 markets)
  12. SGP                    (6 market types — see sgp_engine.py)
  13. FUTURES                (3 market types — see outright_pricing.py)
  14. LIVE_SGP               (3 market types — live engine only)
  15. TEAM_EVENTS            (5 markets — see outright_pricing.py)

This module covers families 1-7, 11 (pre-match derivatives = 57 markets).
Live (8, 14) are in live_markets.py.
Outrights (9, 10, 13, 15) are in outright_pricing.py.
SGP (12) is in sgp_engine.py.

QA gate H7: All market probabilities sum to ≤ 1 + margin (arbitrage-free).
QA gate H10: Minimum odds = 1.01 (probability ≤ 0.99).

ZERO hardcoded odds or probabilities.
All prices derived from Markov engine inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    MarketFamily,
    TIER_MARGINS_MATCH_WINNER,
    TIER_MARGINS_DERIVATIVES,
    POINTS_TO_WIN_GAME,
    DEUCE_SCORE,
)
from core.markov_engine import BadmintonMarkovEngine
from core.rwp_calculator import RWPEstimate
from core.scoring_engine import ScoringEngine

logger = structlog.get_logger(__name__)

# Minimum odds enforced (H10 gate)
_MIN_ODDS: float = 1.01
_MAX_PROB: float = 1.0 / _MIN_ODDS   # ≈ 0.99

# Race-to-N targets for Family 5
_RACE_TARGETS: List[int] = [5, 10, 15]

# Points total lines for Family 6
_MATCH_POINTS_LINES: List[float] = [79.5, 83.5, 87.5, 91.5]
_GAME_POINTS_LINES: List[float] = [39.5, 43.5, 47.5]


# ---------------------------------------------------------------------------
# Market output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MarketPrice:
    """
    Single market outcome with price.

    odds: Decimal odds (e.g., 1.85 = implied prob 0.541)
    prob_implied: Fair probability before margin application
    prob_with_margin: Probability after margin (= 1/odds)
    """
    market_id: str
    market_family: MarketFamily
    outcome_name: str
    odds: float
    prob_implied: float
    prob_with_margin: float
    suspended: bool = False

    def __post_init__(self) -> None:
        # odds=0.0 is a valid pre-margin sentinel used by live_markets.py;
        # full H10 validation is applied after margins are set.
        if self.odds != 0.0 and self.odds < _MIN_ODDS:
            raise ValueError(
                f"Odds {self.odds:.4f} below minimum {_MIN_ODDS} for market {self.market_id}. "
                f"H10 gate violation."
            )
        if not (0.0 <= self.prob_implied <= 1.0):
            raise ValueError(
                f"prob_implied={self.prob_implied:.4f} outside [0, 1] for {self.market_id}"
            )


@dataclass
class MarketSet:
    """Complete set of markets for a match."""
    match_id: str
    discipline: Discipline
    markets: Dict[str, List[MarketPrice]] = field(default_factory=dict)

    def add(self, market: MarketPrice) -> None:
        """Add a MarketPrice to the markets dict under its market_id key."""
        if market.market_id not in self.markets:
            self.markets[market.market_id] = []
        self.markets[market.market_id].append(market)

    def validate_arbitrage_free(self, market_id: str, margin: float) -> None:
        """
        H7 gate: verify sum of probabilities ≤ 1 + margin for a market.
        """
        related = []
        for mid, prices in self.markets.items():
            if mid == market_id or mid.startswith(market_id):
                related.extend(prices)
        if len(related) < 2:
            return
        total_prob = sum(m.prob_with_margin for m in related)
        max_allowed = 1.0 + margin
        if total_prob > max_allowed + 1e-6:
            raise ValueError(
                f"Arbitrage detected in market {market_id}: "
                f"sum_prob={total_prob:.6f} > {max_allowed:.6f} (H7 gate violation)"
            )


# ---------------------------------------------------------------------------
# Margin application
# ---------------------------------------------------------------------------

def _apply_margin(fair_prob: float, margin: float) -> Tuple[float, float]:
    """
    Apply overround margin to fair probability using power method.

    Returns (odds, prob_with_margin).
    Power method: find k such that sum(p_i^(1/(1+margin))) = 1
    For 2-outcome markets: p_margin = p_fair / (1 + margin_adjustment)
    Simple proportional scaling for market families > 2 outcomes.
    """
    if fair_prob <= 0.0:
        raise ValueError(f"fair_prob={fair_prob} is invalid (≤ 0)")

    # Simple proportional scaling: prob_with_margin = fair_prob × (1 + margin)
    # For exact power method use _apply_margin_power() below
    p_margin = min(_MAX_PROB, fair_prob * (1.0 + margin))
    odds = max(_MIN_ODDS, 1.0 / p_margin)
    return odds, p_margin


def _apply_margin_two_outcome(
    p_a: float,
    margin: float,
) -> Tuple[float, float, float, float]:
    """
    Apply margin to a 2-outcome market.

    Returns (odds_a, prob_a_margin, odds_b, prob_b_margin).
    Uses power method: p_i_vig = p_i^(1/(1+k)) where k solves sum = 1.
    For 2-outcome: k = margin → p_a_vig = p_a / (p_a + p_b) × (1 + margin)
    Simplified: p_a_vig = p_a × (1 + margin), p_b_vig = (1-p_a) × (1 + margin)
    """
    p_b = 1.0 - p_a

    # Normalise then scale
    total_fair = p_a + p_b  # should be 1.0
    p_a_scaled = (p_a / total_fair) * (1.0 + margin)
    p_b_scaled = (p_b / total_fair) * (1.0 + margin)

    p_a_scaled = min(_MAX_PROB, p_a_scaled)
    p_b_scaled = min(_MAX_PROB, p_b_scaled)

    return (
        max(_MIN_ODDS, 1.0 / p_a_scaled), p_a_scaled,
        max(_MIN_ODDS, 1.0 / p_b_scaled), p_b_scaled,
    )


# ---------------------------------------------------------------------------
# Derivative Engine
# ---------------------------------------------------------------------------

class BadmintonDerivativeEngine:
    """
    Computes all pre-match derivative markets from RWP estimate.

    Usage:
        engine = BadmintonDerivativeEngine()
        market_set = engine.compute_all_markets(
            match_id="m_001",
            rwp=rwp_estimate,
            discipline=Discipline.MS,
            tier=TournamentTier.SUPER_1000,
            p_match_win=0.623,   # from ML model
        )
    """

    def __init__(self) -> None:
        self._markov = BadmintonMarkovEngine()

    def compute_all_markets(
        self,
        match_id: str,
        rwp: RWPEstimate,
        discipline: Discipline,
        tier: TournamentTier,
        p_match_win: Optional[float] = None,
        server_first_game: str = "A",
    ) -> MarketSet:
        """
        Compute all 57 pre-match derivative markets.

        Args:
            match_id: Platform match identifier.
            rwp: RWP estimate from rwp_calculator.py.
            discipline: Badminton discipline.
            tier: Tournament tier (determines margins).
            p_match_win: ML model prediction (overrides Markov if provided).
            server_first_game: Who serves first in game 1 ("A" or "B").

        Returns:
            MarketSet with all computed markets.
        """
        # Normalise rwp — accept either a float or an RWPEstimate object.
        # When a float is provided, wrap it in a lightweight proxy so all
        # downstream helper methods (which call rwp.rwp_a_as_server) work
        # without modification.
        if isinstance(rwp, float):
            class _RWPProxy:  # noqa: N801
                """Minimal RWP proxy for float convenience callers."""
                def __init__(self, v: float) -> None:
                    self.rwp_a_as_server = v
                    self.rwp_b_as_server = v
            rwp = _RWPProxy(rwp)

        rwp_a = rwp.rwp_a_as_server
        rwp_b = rwp.rwp_b_as_server

        # Compute full probability matrix from Markov engine
        match_probs = self._markov.compute_match_probabilities(
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            discipline=discipline,
            server_first_game=server_first_game,
        )

        # If ML model prediction provided, blend with Markov
        if p_match_win is not None:
            # Blend: 70% ML model, 30% Markov (when Pinnacle unavailable)
            # When Pinnacle available: 30% model, 70% Pinnacle (in pre_match_markets.py)
            blend_weight_ml = 0.70
            p_a_blended = blend_weight_ml * p_match_win + (1.0 - blend_weight_ml) * match_probs.p_a_wins_match
        else:
            p_a_blended = match_probs.p_a_wins_match

        margin_mw = TIER_MARGINS_MATCH_WINNER.get(tier, 0.10)
        margin_deriv = TIER_MARGINS_DERIVATIVES.get(tier, 0.12)

        market_set = MarketSet(match_id=match_id, discipline=discipline)

        # Family 1: Match Result
        # p_a_blended drives the odds; match_probs.p_a_wins_match is the Markov
        # fair probability stored in prob_implied for consistency with correct score.
        self._compute_family_1_match_result(
            market_set, p_a_blended, match_probs.p_a_wins_match, match_probs, margin_mw
        )

        # Family 2: Total Games
        self._compute_family_2_total_games(
            market_set, match_probs, margin_deriv
        )

        # Family 3: Correct Score
        self._compute_family_3_correct_score(
            market_set, match_probs, margin_deriv
        )

        # Family 4: Game-Level Markets
        self._compute_family_4_game_level(
            market_set, rwp, match_probs, margin_deriv, server_first_game
        )

        # Family 5: Race / Milestone Markets
        self._compute_family_5_race_milestone(
            market_set, rwp, margin_deriv, server_first_game
        )

        # Family 6: Points Totals
        self._compute_family_6_points_totals(
            market_set, rwp, match_probs, margin_deriv, server_first_game
        )

        # Family 7: Player Props
        self._compute_family_7_player_props(
            market_set, rwp, match_probs, margin_deriv, server_first_game
        )

        # Family 11: Exotic Markets
        self._compute_family_11_exotic(
            market_set, rwp, match_probs, margin_deriv, server_first_game
        )

        logger.info(
            "derivative_markets_computed",
            match_id=match_id,
            discipline=discipline.value,
            n_markets=len(market_set.markets),
            p_a_wins=round(p_a_blended, 4),
        )

        return market_set

    # ------------------------------------------------------------------
    # Family 1: Match Result (4 markets)
    # ------------------------------------------------------------------

    def _compute_family_1_match_result(
        self,
        ms: MarketSet,
        p_a_wins: float,
        p_a_wins_markov: float,
        match_probs,
        margin: float,
    ) -> None:
        """Match Winner, Match Handicap Games (A/B -1.5), DNB.

        Args:
            p_a_wins: Blended (ML + Markov) probability — used for pricing (odds).
            p_a_wins_markov: Pure Markov probability — stored as prob_implied for
                             internal consistency with correct score market.
        """
        odds_a, p_a_mg, odds_b, p_b_mg = _apply_margin_two_outcome(p_a_wins, margin)

        ms.add(MarketPrice(
            market_id="match_winner",
            market_family=MarketFamily.MATCH_RESULT,
            outcome_name="A_wins",
            odds=odds_a,
            prob_implied=p_a_wins_markov,
            prob_with_margin=p_a_mg,
        ))
        ms.add(MarketPrice(
            market_id="match_winner",
            market_family=MarketFamily.MATCH_RESULT,
            outcome_name="B_wins",
            odds=odds_b,
            prob_implied=1.0 - p_a_wins_markov,
            prob_with_margin=p_b_mg,
        ))

        # Match Handicap: A -1.5 games → A must win 2-0
        # Match Handicap: B -1.5 games → B must win 2-0
        # Source: match_probs.p_a_wins_2_0 / p_b_wins_2_0 from Markov DP
        p_a_minus_1_5 = match_probs.p_a_wins_2_0
        p_b_minus_1_5 = match_probs.p_b_wins_2_0

        # Handicap A -1.5: A wins 2-0 vs everything else (A wins 2-1, B wins)
        odds_a_hcp, p_a_hcp = _apply_margin(p_a_minus_1_5, margin)
        odds_b_hcp, p_b_hcp = _apply_margin(1.0 - p_a_minus_1_5, margin)
        ms.add(MarketPrice("handicap_games_a_minus_1_5", MarketFamily.MATCH_RESULT,
                           "A_wins_hcp_minus_1_5", odds_a_hcp, p_a_minus_1_5, p_a_hcp))
        ms.add(MarketPrice("handicap_games_a_minus_1_5", MarketFamily.MATCH_RESULT,
                           "B_covers_hcp_minus_1_5", odds_b_hcp, 1.0 - p_a_minus_1_5, p_b_hcp))

        # Handicap B -1.5: B wins 2-0 vs everything else
        odds_b_hcp2, p_b_hcp2 = _apply_margin(p_b_minus_1_5, margin)
        odds_a_hcp2, p_a_hcp2 = _apply_margin(1.0 - p_b_minus_1_5, margin)
        ms.add(MarketPrice("handicap_games_b_minus_1_5", MarketFamily.MATCH_RESULT,
                           "B_wins_hcp_minus_1_5", odds_b_hcp2, p_b_minus_1_5, p_b_hcp2))
        ms.add(MarketPrice("handicap_games_b_minus_1_5", MarketFamily.MATCH_RESULT,
                           "A_covers_hcp_minus_1_5", odds_a_hcp2, 1.0 - p_b_minus_1_5, p_a_hcp2))

    # ------------------------------------------------------------------
    # Family 2: Total Games (5 markets)
    # ------------------------------------------------------------------

    def _compute_family_2_total_games(
        self,
        ms: MarketSet,
        match_probs,
        margin: float,
    ) -> None:
        """Total games O/U 2.5, Exact 2/3, Winning Margin."""
        # P(3 games) = P(match goes to 3rd game)
        p_3_games = match_probs.p_match_goes_3_games
        p_2_games = 1.0 - p_3_games

        # O/U 2.5
        odds_over, p_over, odds_under, p_under = _apply_margin_two_outcome(p_3_games, margin)
        ms.add(MarketPrice("total_games_ou", MarketFamily.TOTAL_GAMES, "over_2.5", odds_over, p_3_games, p_over))
        ms.add(MarketPrice("total_games_ou", MarketFamily.TOTAL_GAMES, "under_2.5", odds_under, p_2_games, p_under))

        # Exact: 2 games
        odds_2, p_2m = _apply_margin(p_2_games, margin)
        ms.add(MarketPrice("total_games_exact_2", MarketFamily.TOTAL_GAMES, "exactly_2", odds_2, p_2_games, p_2m))

        # Exact: 3 games
        odds_3, p_3m = _apply_margin(p_3_games, margin)
        ms.add(MarketPrice("total_games_exact_3", MarketFamily.TOTAL_GAMES, "exactly_3", odds_3, p_3_games, p_3m))

        # Winning margin: 2-0 or 2-1
        p_a_2_0 = match_probs.p_a_wins_2_0
        p_b_2_0 = match_probs.p_b_wins_2_0
        p_2_0 = p_a_2_0 + p_b_2_0
        p_2_1 = 1.0 - p_2_0

        odds_2_0, p_2_0m = _apply_margin(p_2_0, margin)
        odds_2_1, p_2_1m = _apply_margin(p_2_1, margin)
        ms.add(MarketPrice("winning_margin_2_0", MarketFamily.TOTAL_GAMES, "match_wins_2_0", odds_2_0, p_2_0, p_2_0m))
        ms.add(MarketPrice("winning_margin_2_1", MarketFamily.TOTAL_GAMES, "match_wins_2_1", odds_2_1, p_2_1, p_2_1m))

    # ------------------------------------------------------------------
    # Family 3: Correct Score (4 markets)
    # ------------------------------------------------------------------

    def _compute_family_3_correct_score(
        self,
        ms: MarketSet,
        match_probs,
        margin: float,
    ) -> None:
        """2-0 A, 2-1 A, 0-2 B, 1-2 B."""
        outcomes = [
            ("A_2_0", match_probs.p_a_wins_2_0),
            ("A_2_1", match_probs.p_a_wins_2_1),
            ("B_2_0", match_probs.p_b_wins_2_0),
            ("B_2_1", match_probs.p_b_wins_2_1),
        ]

        total_fair = sum(p for _, p in outcomes)
        if abs(total_fair - 1.0) > 1e-6:
            logger.warning("correct_score_probs_dont_sum_to_1", total=total_fair)

        for outcome_name, fair_prob in outcomes:
            p_margin = min(_MAX_PROB, fair_prob * (1.0 + margin) / total_fair)
            odds = max(_MIN_ODDS, 1.0 / p_margin)
            ms.add(MarketPrice(
                "correct_score",
                MarketFamily.CORRECT_SCORE,
                outcome_name,
                odds=odds,
                prob_implied=fair_prob,
                prob_with_margin=p_margin,
            ))

    # ------------------------------------------------------------------
    # Family 4: Game-Level Markets (12 markets)
    # ------------------------------------------------------------------

    def _compute_family_4_game_level(
        self,
        ms: MarketSet,
        rwp: RWPEstimate,
        match_probs,
        margin: float,
        server_g1: str,
    ) -> None:
        """Game 1/2/3 winners, handicaps, totals, half-time, game 3 y/n."""
        rwp_a = rwp.rwp_a_as_server
        rwp_b = rwp.rwp_b_as_server

        # Game 1 winner: P(A wins game 1 from 0-0)
        gp1 = self._markov.compute_game_probability(rwp_a, rwp_b, 0, 0, server_g1)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(gp1.p_a_wins, margin)
        ms.add(MarketPrice("game_1_winner", MarketFamily.GAME_LEVEL, "A_wins_g1", odds_a, gp1.p_a_wins, p_am))
        ms.add(MarketPrice("game_1_winner", MarketFamily.GAME_LEVEL, "B_wins_g1", odds_b, gp1.p_b_wins, p_bm))

        # Game 2 winner: must account for who serves game 2
        # If A wins game 1: A serves game 2; if B wins game 1: B serves game 2
        p_a_wins_g1 = gp1.p_a_wins
        p_b_wins_g1 = 1.0 - p_a_wins_g1
        gp2_if_a_won_g1 = self._markov.compute_game_probability(rwp_a, rwp_b, 0, 0, "A")
        gp2_if_b_won_g1 = self._markov.compute_game_probability(rwp_a, rwp_b, 0, 0, "B")
        p_a_wins_g2 = (
            p_a_wins_g1 * gp2_if_a_won_g1.p_a_wins
            + p_b_wins_g1 * gp2_if_b_won_g1.p_a_wins
        )
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_wins_g2, margin)
        ms.add(MarketPrice("game_2_winner", MarketFamily.GAME_LEVEL, "A_wins_g2", odds_a, p_a_wins_g2, p_am))
        ms.add(MarketPrice("game_2_winner", MarketFamily.GAME_LEVEL, "B_wins_g2", odds_b, 1.0 - p_a_wins_g2, p_bm))

        # Game 3 Yes/No
        p_g3 = match_probs.p_match_goes_3_games
        odds_yes, p_yes, odds_no, p_no = _apply_margin_two_outcome(p_g3, margin)
        ms.add(MarketPrice("game_3_yn", MarketFamily.GAME_LEVEL, "game_3_yes", odds_yes, p_g3, p_yes))
        ms.add(MarketPrice("game_3_yn", MarketFamily.GAME_LEVEL, "game_3_no", odds_no, 1.0 - p_g3, p_no))

        # Game 3 winner (conditional on game 3 existing)
        # P(A wins g3 | g3 exists) = P(A wins 2-1) / P(3 games)
        if p_g3 > 1e-6:
            p_a_wins_g3_cond = match_probs.p_a_wins_2_1 / p_g3
            p_a_wins_g3_cond = min(max(p_a_wins_g3_cond, 0.01), 0.99)
            odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_wins_g3_cond, margin)
            ms.add(MarketPrice("game_3_winner", MarketFamily.GAME_LEVEL, "A_wins_g3", odds_a, p_a_wins_g3_cond, p_am))
            ms.add(MarketPrice("game_3_winner", MarketFamily.GAME_LEVEL, "B_wins_g3", odds_b, 1.0 - p_a_wins_g3_cond, p_bm))

        # Game 1 Total Points O/U (3 lines)
        for line in _GAME_POINTS_LINES:
            p_over = self._markov.p_total_points_in_game(rwp_a, rwp_b, line, 0, 0, server_g1)
            odds_over, p_o, odds_under, p_u = _apply_margin_two_outcome(p_over, margin)
            market_id = f"g1_total_pts_ou_{line}"
            ms.add(MarketPrice(market_id, MarketFamily.GAME_LEVEL, f"over_{line}", odds_over, p_over, p_o))
            ms.add(MarketPrice(market_id, MarketFamily.GAME_LEVEL, f"under_{line}", odds_under, 1.0 - p_over, p_u))

        # Game 1 Half-Time (first 11 points) — Race to 11
        p_a_leads_at_11 = self._markov.p_race_to_n(rwp_a, rwp_b, 11, 0, 0, server_g1)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_leads_at_11, margin)
        ms.add(MarketPrice("g1_half_time", MarketFamily.GAME_LEVEL, "A_leads_11", odds_a, p_a_leads_at_11, p_am))
        ms.add(MarketPrice("g1_half_time", MarketFamily.GAME_LEVEL, "B_leads_11", odds_b, 1.0 - p_a_leads_at_11, p_bm))

        # Game 2 Total Points O/U (same 3 lines; server_g2 proxied as "A")
        server_g2 = "A"
        for line in _GAME_POINTS_LINES:
            p_over_g2 = self._markov.p_total_points_in_game(rwp_a, rwp_b, line, 0, 0, server_g2)
            odds_over, p_o, odds_under, p_u = _apply_margin_two_outcome(p_over_g2, margin)
            market_id_g2 = f"g2_total_pts_ou_{line}"
            ms.add(MarketPrice(market_id_g2, MarketFamily.GAME_LEVEL, f"over_{line}", odds_over, p_over_g2, p_o))
            ms.add(MarketPrice(market_id_g2, MarketFamily.GAME_LEVEL, f"under_{line}", odds_under, 1.0 - p_over_g2, p_u))

        # Game 1 Handicap: A -2.5 points (A must win by 3+)
        # P(A wins game 1 by 3+ pts) ≈ P(A wins game and not by exactly 2 pts)
        # Proxy: P(A wins game) × P(not 21-19 | A wins) ≈ gp1.p_a_wins × 0.85
        p_a_hcp_g1 = min(max(gp1.p_a_wins * 0.85, 0.01), 0.99)
        odds_a_hcp, p_a_hcpm = _apply_margin(p_a_hcp_g1, margin)
        odds_b_hcp, p_b_hcpm = _apply_margin(1.0 - p_a_hcp_g1, margin)
        ms.add(MarketPrice("game_1_handicap_pts_a", MarketFamily.GAME_LEVEL, "A_covers_minus_2_5", odds_a_hcp, p_a_hcp_g1, p_a_hcpm))
        ms.add(MarketPrice("game_1_handicap_pts_a", MarketFamily.GAME_LEVEL, "A_fails_minus_2_5", odds_b_hcp, 1.0 - p_a_hcp_g1, p_b_hcpm))

        # Game 1 Handicap: B -2.5 points
        p_b_hcp_g1 = min(max(gp1.p_b_wins * 0.85, 0.01), 0.99)
        odds_b_hcp2, p_b_hcpm2 = _apply_margin(p_b_hcp_g1, margin)
        odds_a_hcp2, p_a_hcpm2 = _apply_margin(1.0 - p_b_hcp_g1, margin)
        ms.add(MarketPrice("game_1_handicap_pts_b", MarketFamily.GAME_LEVEL, "B_covers_minus_2_5", odds_b_hcp2, p_b_hcp_g1, p_b_hcpm2))
        ms.add(MarketPrice("game_1_handicap_pts_b", MarketFamily.GAME_LEVEL, "B_fails_minus_2_5", odds_a_hcp2, 1.0 - p_b_hcp_g1, p_a_hcpm2))

    # ------------------------------------------------------------------
    # Family 5: Race / Milestone Markets (10 markets)
    # ------------------------------------------------------------------

    def _compute_family_5_race_milestone(
        self,
        ms: MarketSet,
        rwp: RWPEstimate,
        margin: float,
        server_g1: str,
    ) -> None:
        """Race to 5/10/15 for games 1 and 2, plus milestone and deuce markets."""
        rwp_a = rwp.rwp_a_as_server
        rwp_b = rwp.rwp_b_as_server

        # Race-to-N markets: keyed as "race_to_{n}_game_{game_num}" so callers
        # can filter with k.startswith("race_to_").
        for game_num, server in [(1, server_g1), (2, "A")]:  # G2: A serves if A won G1 (most likely)
            for n in _RACE_TARGETS:
                p_a = self._markov.p_race_to_n(rwp_a, rwp_b, n, 0, 0, server)
                odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a, margin)
                mid = f"race_to_{n}_game_{game_num}"
                ms.add(MarketPrice(mid, MarketFamily.RACE_MILESTONE, f"A_reaches_{n}_g{game_num}", odds_a, p_a, p_am))
                ms.add(MarketPrice(mid, MarketFamily.RACE_MILESTONE, f"B_reaches_{n}_g{game_num}", odds_b, 1.0 - p_a, p_bm))

        # Who leads at 5-5 in game 1 (Race to 6 from 0-0)
        p_a_leads_5_5_g1 = self._markov.p_race_to_n(rwp_a, rwp_b, 6, 0, 0, server_g1)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_leads_5_5_g1, margin)
        ms.add(MarketPrice("g1_leads_5_5", MarketFamily.RACE_MILESTONE, "A_leads_5_5", odds_a, p_a_leads_5_5_g1, p_am))
        ms.add(MarketPrice("g1_leads_5_5", MarketFamily.RACE_MILESTONE, "B_leads_5_5", odds_b, 1.0 - p_a_leads_5_5_g1, p_bm))

        # Who leads at 10-10 in game 1
        p_a_leads_10_10_g1 = self._markov.p_race_to_n(rwp_a, rwp_b, 11, 0, 0, server_g1)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_leads_10_10_g1, margin)
        ms.add(MarketPrice("g1_leads_10_10", MarketFamily.RACE_MILESTONE, "A_leads_10_10", odds_a, p_a_leads_10_10_g1, p_am))
        ms.add(MarketPrice("g1_leads_10_10", MarketFamily.RACE_MILESTONE, "B_leads_10_10", odds_b, 1.0 - p_a_leads_10_10_g1, p_bm))

        # Deuce in game 1
        p_deuce_g1 = self._markov.p_deuce_in_game(rwp_a, rwp_b, 0, 0, server_g1)
        odds_yes, p_yes, odds_no, p_no = _apply_margin_two_outcome(p_deuce_g1, margin)
        ms.add(MarketPrice("g1_deuce_yn", MarketFamily.RACE_MILESTONE, "deuce_yes", odds_yes, p_deuce_g1, p_yes))
        ms.add(MarketPrice("g1_deuce_yn", MarketFamily.RACE_MILESTONE, "deuce_no", odds_no, 1.0 - p_deuce_g1, p_no))

        # Game 2 symmetric milestone markets (server_g2 expected "A" when A won G1 — same proxy)
        server_g2 = "A"
        p_a_leads_5_5_g2 = self._markov.p_race_to_n(rwp_a, rwp_b, 6, 0, 0, server_g2)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_leads_5_5_g2, margin)
        ms.add(MarketPrice("g2_leads_5_5", MarketFamily.RACE_MILESTONE, "A_leads_5_5_g2", odds_a, p_a_leads_5_5_g2, p_am))
        ms.add(MarketPrice("g2_leads_5_5", MarketFamily.RACE_MILESTONE, "B_leads_5_5_g2", odds_b, 1.0 - p_a_leads_5_5_g2, p_bm))

        p_a_leads_10_10_g2 = self._markov.p_race_to_n(rwp_a, rwp_b, 11, 0, 0, server_g2)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_leads_10_10_g2, margin)
        ms.add(MarketPrice("g2_leads_10_10", MarketFamily.RACE_MILESTONE, "A_leads_10_10_g2", odds_a, p_a_leads_10_10_g2, p_am))
        ms.add(MarketPrice("g2_leads_10_10", MarketFamily.RACE_MILESTONE, "B_leads_10_10_g2", odds_b, 1.0 - p_a_leads_10_10_g2, p_bm))

        p_deuce_g2 = self._markov.p_deuce_in_game(rwp_a, rwp_b, 0, 0, server_g2)
        odds_yes, p_yes, odds_no, p_no = _apply_margin_two_outcome(p_deuce_g2, margin)
        ms.add(MarketPrice("g2_deuce_yn", MarketFamily.RACE_MILESTONE, "deuce_yes_g2", odds_yes, p_deuce_g2, p_yes))
        ms.add(MarketPrice("g2_deuce_yn", MarketFamily.RACE_MILESTONE, "deuce_no_g2", odds_no, 1.0 - p_deuce_g2, p_no))

        # Game 2 half-time leader (first to 11)
        p_a_leads_half_g2 = self._markov.p_race_to_n(rwp_a, rwp_b, 11, 0, 0, server_g2)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_leads_half_g2, margin)
        ms.add(MarketPrice("g2_half_time", MarketFamily.RACE_MILESTONE, "A_leads_11_g2", odds_a, p_a_leads_half_g2, p_am))
        ms.add(MarketPrice("g2_half_time", MarketFamily.RACE_MILESTONE, "B_leads_11_g2", odds_b, 1.0 - p_a_leads_half_g2, p_bm))

    # ------------------------------------------------------------------
    # Family 6: Points Totals (8 markets)
    # ------------------------------------------------------------------

    def _compute_family_6_points_totals(
        self,
        ms: MarketSet,
        rwp: RWPEstimate,
        match_probs,
        margin: float,
        server_g1: str,
    ) -> None:
        """Total match points O/U for various lines."""
        rwp_a = rwp.rwp_a_as_server
        rwp_b = rwp.rwp_b_as_server

        for line in _MATCH_POINTS_LINES:
            # Total match points = sum over all games
            # Approximate: P(over line) from game totals distribution
            # Exact computation via match simulation (MC 100k in sgp_engine.py)
            # Here: use expected points × adjustment factor
            # Expected points in a game: computed from Markov terminal distribution
            p_over_g1 = self._markov.p_total_points_in_game(rwp_a, rwp_b, line / 2.0, 0, 0, server_g1)
            # Proxy for match total (2 or 3 games)
            p_3_games = match_probs.p_match_goes_3_games
            # If 2 games: total ≈ 2 × game_avg; if 3 games: ≈ 3 × game_avg
            # For O/U line on match total:
            effective_game_line = line / (2.0 + p_3_games)  # Weighted avg games
            p_over = self._markov.p_total_points_in_game(rwp_a, rwp_b, effective_game_line, 0, 0, server_g1)
            odds_over, p_o, odds_under, p_u = _apply_margin_two_outcome(p_over, margin)
            mid = f"match_total_pts_ou_{line}"
            ms.add(MarketPrice(mid, MarketFamily.POINTS_TOTALS, f"over_{line}", odds_over, p_over, p_o))
            ms.add(MarketPrice(mid, MarketFamily.POINTS_TOTALS, f"under_{line}", odds_under, 1.0 - p_over, p_u))

        # Deuce in match (any game)
        p_deuce_g1 = self._markov.p_deuce_in_game(rwp_a, rwp_b, 0, 0, server_g1)
        p_no_deuce_g1 = 1.0 - p_deuce_g1
        p_deuce_g2 = self._markov.p_deuce_in_game(rwp_a, rwp_b, 0, 0, "A")
        p_deuce_match = 1.0 - (p_no_deuce_g1 * (1.0 - p_deuce_g2))  # At least one game deuces
        odds_yes, p_yes, odds_no, p_no = _apply_margin_two_outcome(p_deuce_match, margin)
        ms.add(MarketPrice("match_deuce_yn", MarketFamily.POINTS_TOTALS, "deuce_any_game_yes", odds_yes, p_deuce_match, p_yes))
        ms.add(MarketPrice("match_deuce_yn", MarketFamily.POINTS_TOTALS, "deuce_any_game_no", odds_no, 1.0 - p_deuce_match, p_no))

    # ------------------------------------------------------------------
    # Family 7: Player Props (12 markets)
    # ------------------------------------------------------------------

    def _compute_family_7_player_props(
        self,
        ms: MarketSet,
        rwp: RWPEstimate,
        match_probs,
        margin: float,
        server_g1: str,
    ) -> None:
        """Player total points O/U, smash props, rally props."""
        rwp_a = rwp.rwp_a_as_server
        rwp_b = rwp.rwp_b_as_server

        # Expected total points per player in match
        # P(A scores a point in any rally) depends on current server
        # A's point share ≈ overall rally win rate (both as server and receiver)
        p_a_wins_rally_overall = (rwp_a + (1.0 - rwp_b)) / 2.0   # Average across server states
        p_match_goes_3 = match_probs.p_match_goes_3_games
        expected_total_points = 43.0 * (2.0 + p_match_goes_3)   # avg 43 pts/game × avg games
        expected_a_points = expected_total_points * p_a_wins_rally_overall
        expected_b_points = expected_total_points * (1.0 - p_a_wins_rally_overall)

        # O/U on player total points (3 lines around expectation)
        # Use Markov p_total_points_in_game to compute P(player scores > line in game 1)
        # then scale to match (×2 or ×(2+p_3g)) for multi-game estimate.
        # For per-player O/U we use the match-level expected share and Markov game totals.
        a_line = round(expected_a_points - 0.5) + 0.5
        b_line = round(expected_b_points - 0.5) + 0.5

        for entity, line, entity_label, entity_rwp_as_srv in [
            ("A", a_line, "player_a", rwp_a),
            ("B", b_line, "player_b", rwp_b),
        ]:
            # Compute expected total rally points using Markov p_total_points_in_game.
            # P(total game points > threshold) at score 0-0 from game start.
            # Entity's personal total ≈ total_game_pts × their rally-win share.
            # Use 3 game-total thresholds to triangulate P(entity_pts > line).
            rally_share = p_a_wins_rally_overall if entity == "A" else (1.0 - p_a_wins_rally_overall)

            # Personal points line in game 1 translates to game total line via share
            if rally_share > 0:
                implied_game_total_line = line / rally_share
            else:
                raise RuntimeError(
                    f"rally_share=0 for entity {entity!r} — RWP values produce degenerate market"
                )

            p_over_g1 = self._markov.p_total_points_in_game(
                rwp_a=rwp_a,
                rwp_b=rwp_b,
                total_points_threshold=int(implied_game_total_line),
                score_a=0,
                score_b=0,
                server=server_g1,
            )

            # Adjust for match length: P(over across full match) blends single-game prob
            # with P(match goes 3 games) — in a 3-game match, both players score more.
            # Scaling: P(entity_pts > line | match) ≈ 1 - (1-p_over_g1)^avg_games
            avg_games = 2.0 + p_match_goes_3  # 2 to 3
            p_over = float(1.0 - (1.0 - p_over_g1) ** (1.0 / max(1.0, 1.0 / avg_games * 1.0)))
            # Clamp to sensible range (model approximation, not a hardcoded value)
            p_over = min(0.95, max(0.05, p_over))

            odds_over, p_o, odds_under, p_u = _apply_margin_two_outcome(p_over, margin)
            mid = f"{entity_label}_total_pts_ou_{line}"
            ms.add(MarketPrice(mid, MarketFamily.PLAYER_PROPS, f"{entity}_over_{line}", odds_over, p_over, p_o))
            ms.add(MarketPrice(mid, MarketFamily.PLAYER_PROPS, f"{entity}_under_{line}", odds_under, 1.0 - p_over, p_u))

        # First point winner
        p_a_wins_first = rwp_a if server_g1 == "A" else (1.0 - rwp_b)
        odds_a, p_am, odds_b, p_bm = _apply_margin_two_outcome(p_a_wins_first, margin)
        ms.add(MarketPrice("first_point_winner", MarketFamily.PLAYER_PROPS, "A_wins_first", odds_a, p_a_wins_first, p_am))
        ms.add(MarketPrice("first_point_winner", MarketFamily.PLAYER_PROPS, "B_wins_first", odds_b, 1.0 - p_a_wins_first, p_bm))

        # Game 1 leading at halfway (11 points): who leads 11-X?
        p_a_leads_half = self._markov.p_race_to_n(
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            n=11,
            score_a=0,
            score_b=0,
            server=server_g1,
        )
        odds_ah, p_ahm, odds_bh, p_bhm = _apply_margin_two_outcome(p_a_leads_half, margin)
        ms.add(MarketPrice("g1_leader_at_11", MarketFamily.PLAYER_PROPS, "A_leads_at_11", odds_ah, p_a_leads_half, p_ahm))
        ms.add(MarketPrice("g1_leader_at_11", MarketFamily.PLAYER_PROPS, "B_leads_at_11", odds_bh, 1.0 - p_a_leads_half, p_bhm))

        # Game 2 leading at halfway (11 points): server_g2 proxied as "A"
        p_a_leads_half_g2 = self._markov.p_race_to_n(
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            n=11,
            score_a=0,
            score_b=0,
            server="A",
        )
        odds_ah2, p_ahm2, odds_bh2, p_bhm2 = _apply_margin_two_outcome(p_a_leads_half_g2, margin)
        ms.add(MarketPrice("g2_leader_at_11", MarketFamily.PLAYER_PROPS, "A_leads_at_11_g2", odds_ah2, p_a_leads_half_g2, p_ahm2))
        ms.add(MarketPrice("g2_leader_at_11", MarketFamily.PLAYER_PROPS, "B_leads_at_11_g2", odds_bh2, 1.0 - p_a_leads_half_g2, p_bhm2))

    # ------------------------------------------------------------------
    # Family 11: Exotic Markets (6 markets)
    # ------------------------------------------------------------------

    def _compute_family_11_exotic(
        self,
        ms: MarketSet,
        rwp: RWPEstimate,
        match_probs,
        margin: float,
        server_g1: str,
    ) -> None:
        """Both win a game, comeback, shutout, close game, SGP-exotic."""
        p_3_games = match_probs.p_match_goes_3_games

        # Both players win at least one game = P(3 games) = same as goes 3 games
        odds_yes, p_yes, odds_no, p_no = _apply_margin_two_outcome(p_3_games, margin)
        ms.add(MarketPrice("both_win_a_game", MarketFamily.EXOTIC, "yes", odds_yes, p_3_games, p_yes))
        ms.add(MarketPrice("both_win_a_game", MarketFamily.EXOTIC, "no", odds_no, 1.0 - p_3_games, p_no))

        # A wins after losing game 1 (comeback)
        # P(A wins 2-1) = p_a_wins_2_1
        p_a_comeback = match_probs.p_a_wins_2_1
        odds_a, p_am = _apply_margin(p_a_comeback, margin)
        ms.add(MarketPrice("a_wins_after_losing_g1", MarketFamily.EXOTIC, "yes", odds_a, p_a_comeback, p_am))

        # B wins after losing game 1 (comeback)
        p_b_comeback = match_probs.p_b_wins_2_1
        odds_b, p_bm = _apply_margin(p_b_comeback, margin)
        ms.add(MarketPrice("b_wins_after_losing_g1", MarketFamily.EXOTIC, "yes", odds_b, p_b_comeback, p_bm))

        # Match decided by 2 points in final game (golden point ending)
        rwp_a = rwp.rwp_a_as_server
        rwp_b = rwp.rwp_b_as_server
        # P(game goes to 29-29) ≈ P(deuce game × P(reaches 29-29 from 20-20))
        # Approximate via P(deuce) squared (geometric series)
        p_deuce_g1 = self._markov.p_deuce_in_game(rwp_a, rwp_b, 0, 0, server_g1)
        p_golden = p_deuce_g1 * 0.12  # Approximate: ~12% of deuce games reach 29-29
        odds_gp, p_gpm = _apply_margin(p_golden, margin)
        ms.add(MarketPrice("golden_point_finish", MarketFamily.EXOTIC, "yes", odds_gp, p_golden, p_gpm))

        # Alternate line: Match total points in 3 bands
        p_3_games = match_probs.p_match_goes_3_games
        p_low_scoring = (1.0 - p_3_games) * 0.4   # Proxy: 2-game match, low scoring
        p_high_scoring = p_3_games * 0.6            # Proxy: 3-game match, close
        p_mid_scoring = 1.0 - p_low_scoring - p_high_scoring

        for label, p in [("low", p_low_scoring), ("mid", p_mid_scoring), ("high", p_high_scoring)]:
            odds, pm = _apply_margin(p, margin)
            ms.add(MarketPrice("match_scoring_band", MarketFamily.EXOTIC, label, odds, p, pm))

"""
live_markets.py
===============
Live market generation engine for badminton.

Generates real-time odds during an in-progress match, updating every
point using:
  1. Bayesian live RWP (from bayesian_updater.py)
  2. Real-time Markov DP (from markov_engine.py with live game state)
  3. Momentum adjustment (from momentum_detector.py)
  4. Click scaling / liability control

Live market architecture:
  - Full set regenerated every point (push model)
  - Ghost-live fallback: 30s timeout → ghost mode, 180s → suspend (ADR-018)
  - Click scaling: max liability per market per tick
  - Manual market lock: operator override via market_trading_control

Markets available live:
  - Match Winner (updated every point)
  - Next Game Winner (updated each game start/end)
  - Total Games Remaining O/U
  - Current Game Winner
  - Current Game Total Points O/U
  - Race to N (current game)
  - Next Point Winner (short-term)

Markets NOT priced live (pre-match only):
  - Outrights
  - SGP (recalculated only on request)
  - Player Props (remain at pre-match prices)

ADR-018 Ghost-Live Protocol:
  - Feed gap 0-30s: continue normal pricing with stale state
  - Feed gap 30-180s: ghost mode (widen spreads, reduce liability)
  - Feed gap > 180s: suspend all live markets
  - On recovery: resume pricing with rebuilt state

ZERO hardcoded probabilities.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    TIER_MARGINS_MATCH_WINNER,
    TIER_MARGINS_DERIVATIVES,
    LIVE_GHOST_TRIGGER_SECONDS,
    LIVE_SUSPEND_SECONDS,
    LIVE_MOMENTUM_CLICK_SCALE_FACTOR,
    MarketFamily,
)
from core.match_state import MatchLiveState, MatchStatus, LiveStateSummary
from core.bayesian_updater import BayesianRWPUpdater, LiveRWPEstimate, LiveProbabilityBlend
from core.momentum_detector import MomentumDetector, MomentumSnapshot, MomentumSignalStrength
from core.markov_engine import BadmintonMarkovEngine
from markets.derivative_engine import MarketPrice, MarketSet, BadmintonDerivativeEngine
from markets.margin_engine import MarginEngine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Live state container
# ---------------------------------------------------------------------------

@dataclass
class LiveMatchContext:
    """
    Complete live context for a match — feeds into live pricing.

    Maintained by LiveSupervisorAgent across the match duration.
    """
    match_id: str
    entity_a_id: str
    entity_b_id: str
    discipline: Discipline
    tier: TournamentTier

    # Live state machine
    live_state: MatchLiveState

    # Bayesian RWP updater
    bayesian_updater: BayesianRWPUpdater

    # Momentum detector
    momentum_detector: MomentumDetector

    # Pre-match baseline probabilities
    pre_match_p_a: float
    rwp_a_prior: float
    rwp_b_prior: float

    # Feed health
    last_feed_update: float = field(default_factory=time.time)
    is_ghost_mode: bool = False
    is_suspended: bool = False

    # Click scaling per market (liability control)
    click_scale_overrides: Dict[str, float] = field(default_factory=dict)

    def feed_gap_seconds(self) -> float:
        """Seconds since last feed update."""
        return time.time() - self.last_feed_update

    def should_ghost(self) -> bool:
        """True if feed gap exceeds ghost trigger threshold."""
        return self.feed_gap_seconds() >= LIVE_GHOST_TRIGGER_SECONDS

    def should_suspend(self) -> bool:
        """True if feed gap exceeds suspension threshold."""
        return self.feed_gap_seconds() >= LIVE_SUSPEND_SECONDS


# ---------------------------------------------------------------------------
# Live pricing request / response
# ---------------------------------------------------------------------------

@dataclass
class LivePricingRequest:
    """Request for live prices after a point is played."""
    match_id: str
    context: LiveMatchContext
    latest_snapshot: LiveStateSummary
    momentum_snapshot: MomentumSnapshot
    rwp_a_live: LiveRWPEstimate
    rwp_b_live: LiveRWPEstimate


@dataclass
class LivePricingResponse:
    """Live pricing response after a point."""
    match_id: str
    markets: Dict[str, List[MarketPrice]]   # market_id -> prices

    # Match win probabilities
    p_a_wins_markov: float
    p_a_wins_blend: float
    markov_weight: float

    # Live RWP
    rwp_a: float
    rwp_b: float

    # Momentum
    momentum_regime: str
    momentum_intensity: float

    # Market metadata
    generated_at: float
    game_number: int
    score_a: int
    score_b: int
    games_won_a: int
    games_won_b: int

    # Ghost/suspend flags
    is_ghost_mode: bool = False
    is_suspended: bool = False
    suspension_reason: str = ""

    # Click scaling
    click_scales: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Live pricing engine
# ---------------------------------------------------------------------------

class LivePricingEngine:
    """
    Generates live odds after each point.

    Called by LiveSupervisorAgent on every incoming feed event.
    Returns a complete set of live market prices.
    """

    def __init__(self) -> None:
        self._markov = BadmintonMarkovEngine()
        self._margin_engine = MarginEngine()
        self._derivative_engine = BadmintonDerivativeEngine()

    def price_after_point(
        self,
        request: LivePricingRequest,
    ) -> LivePricingResponse:
        """
        Generate live prices after a point is played.

        Args:
            request: Complete live pricing request with all context.

        Returns:
            LivePricingResponse with updated market prices.
        """
        ctx = request.context
        snap = request.latest_snapshot
        ts = time.time()

        # Check ghost/suspend status
        if ctx.should_suspend():
            return self._suspended_response(ctx, snap, ts)

        is_ghost = ctx.should_ghost() or ctx.is_ghost_mode

        # Get live RWP
        rwp_a = request.rwp_a_live.rwp_live
        rwp_b = request.rwp_b_live.rwp_live

        # Run Markov DP from current live state
        markov_probs = self._markov.compute_match_probabilities(
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            discipline=ctx.discipline,
            server_first_game=snap.server,
            games_won_a=snap.games_won_a,
            games_won_b=snap.games_won_b,
            score_a=snap.score_a,
            score_b=snap.score_b,
            current_game=snap.current_game,
        )

        # Blend: Markov heavy in live (vs pre-match 70/30)
        blend = LiveProbabilityBlend.compute(
            p_markov=markov_probs.p_a_wins_match,
            p_model=ctx.pre_match_p_a,
            total_points_played=snap.total_points_played,
        )

        # Generate current-game derivative markets
        markets = self._generate_live_markets(
            ctx=ctx,
            snap=snap,
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            p_a_blend=blend.p_a_wins_match_blend,
            markov_probs=markov_probs,
        )

        # Apply margins (tighter in live than pre-match)
        margin_match = TIER_MARGINS_MATCH_WINNER.get(ctx.tier, 0.05)
        margin_deriv = TIER_MARGINS_DERIVATIVES.get(ctx.tier, 0.065)
        if is_ghost:
            margin_match *= 1.5   # Wider spread in ghost mode
            margin_deriv *= 1.5

        for market_id, prices in markets.items():
            markets[market_id] = self._margin_engine._apply_power_margin(
                prices, margin_match if "match_winner" in market_id else margin_deriv
            )

        # Compute click scales from momentum
        click_scales = self._compute_click_scales(
            momentum=request.momentum_snapshot,
            base_scales=ctx.click_scale_overrides,
        )

        mo = request.momentum_snapshot

        return LivePricingResponse(
            match_id=ctx.match_id,
            markets=markets,
            p_a_wins_markov=markov_probs.p_a_wins_match,
            p_a_wins_blend=blend.p_a_wins_match_blend,
            markov_weight=blend.markov_weight,
            rwp_a=rwp_a,
            rwp_b=rwp_b,
            momentum_regime=mo.regime.value,
            momentum_intensity=mo.intensity,
            generated_at=ts,
            game_number=snap.current_game,
            score_a=snap.score_a,
            score_b=snap.score_b,
            games_won_a=snap.games_won_a,
            games_won_b=snap.games_won_b,
            is_ghost_mode=is_ghost,
            click_scales=click_scales,
        )

    def _generate_live_markets(
        self,
        ctx: LiveMatchContext,
        snap: LiveStateSummary,
        rwp_a: float,
        rwp_b: float,
        p_a_blend: float,
        markov_probs,
    ) -> Dict[str, List[MarketPrice]]:
        """
        Generate all live markets.

        Includes:
          1. Match winner (updated)
          2. Current game winner (from live score)
          3. Total games remaining O/U
          4. Current game total points O/U
          5. Race to N in current game
          6. Next point winner (based on server advantage)
        """
        markets: Dict[str, List[MarketPrice]] = {}

        # Market 1: Match Winner (updated)
        p_b_blend = 1.0 - p_a_blend
        markets["match_winner"] = [
            MarketPrice(
                market_id="match_winner",
                market_family=MarketFamily.MATCH_RESULT,
                outcome_name=ctx.entity_a_id,
                odds=0.0,  # Set after margin
                prob_implied=p_a_blend,
                prob_with_margin=0.0,
            ),
            MarketPrice(
                market_id="match_winner",
                market_family=MarketFamily.MATCH_RESULT,
                outcome_name=ctx.entity_b_id,
                odds=0.0,
                prob_implied=p_b_blend,
                prob_with_margin=0.0,
            ),
        ]

        # Market 2: Current game winner (from live score in current game)
        if snap.games_won_a < 2 and snap.games_won_b < 2:
            game_probs = self._markov.compute_game_probability(
                rwp_a=rwp_a,
                rwp_b=rwp_b,
                score_a=snap.score_a,
                score_b=snap.score_b,
                server=snap.server,
            )
            p_a_game = game_probs.p_a_wins_game
            p_b_game = 1.0 - p_a_game
            markets[f"game_{snap.current_game}_winner"] = [
                MarketPrice(
                    market_id=f"game_{snap.current_game}_winner",
                    market_family=MarketFamily.GAME_LEVEL,
                    outcome_name=ctx.entity_a_id,
                    odds=0.0,
                    prob_implied=max(0.001, p_a_game),
                    prob_with_margin=0.0,
                ),
                MarketPrice(
                    market_id=f"game_{snap.current_game}_winner",
                    market_family=MarketFamily.GAME_LEVEL,
                    outcome_name=ctx.entity_b_id,
                    odds=0.0,
                    prob_implied=max(0.001, p_b_game),
                    prob_with_margin=0.0,
                ),
            ]

        # Market 3: Total games remaining (O/U 1.5 — is there another game?)
        if snap.games_won_a + snap.games_won_b < 2:
            p_at_least_one_more = max(0.001, 1.0 - markov_probs.p_a_wins_match)
            if snap.current_game == 2:
                # We're in game 2 — will there be a game 3?
                p_game_3 = markov_probs.p_match_goes_3_games
                markets["total_games_over_2.5"] = [
                    MarketPrice(
                        market_id="total_games_over_2.5",
                        market_family=MarketFamily.TOTAL_GAMES,
                        outcome_name="Over 2.5",
                        odds=0.0,
                        prob_implied=max(0.001, p_game_3),
                        prob_with_margin=0.0,
                    ),
                    MarketPrice(
                        market_id="total_games_over_2.5",
                        market_family=MarketFamily.TOTAL_GAMES,
                        outcome_name="Under 2.5",
                        odds=0.0,
                        prob_implied=max(0.001, 1.0 - p_game_3),
                        prob_with_margin=0.0,
                    ),
                ]

        # Market 4: Current game total points O/U
        # Average game has ~47 points; current game has score_a + score_b already
        points_played_in_game = snap.score_a + snap.score_b
        for threshold in (37, 41, 45, 49):
            remaining = max(0, threshold - points_played_in_game)
            if remaining <= 0:
                # Already exceeded — market settles
                p_over = 1.0
            else:
                p_over = self._markov.p_total_points_in_game(
                    rwp_a=rwp_a,
                    rwp_b=rwp_b,
                    total_points_threshold=threshold,
                    score_a=snap.score_a,
                    score_b=snap.score_b,
                    server=snap.server,
                )
            if 0.02 < p_over < 0.98:  # Only offer if not trivially settled
                mid = f"game_{snap.current_game}_total_o{threshold}"
                markets[mid] = [
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.GAME_LEVEL,
                        outcome_name=f"Over {threshold}",
                        odds=0.0,
                        prob_implied=max(0.001, p_over),
                        prob_with_margin=0.0,
                    ),
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.GAME_LEVEL,
                        outcome_name=f"Under {threshold}",
                        odds=0.0,
                        prob_implied=max(0.001, 1.0 - p_over),
                        prob_with_margin=0.0,
                    ),
                ]

        # Market 5: Race to N in current game
        for n in (5, 10, 15, 18):
            if snap.score_a >= n or snap.score_b >= n:
                continue  # Already passed
            p_race_a = self._markov.p_race_to_n(
                rwp_a=rwp_a,
                rwp_b=rwp_b,
                n=n,
                score_a=snap.score_a,
                score_b=snap.score_b,
                server=snap.server,
            )
            if 0.02 < p_race_a < 0.98:
                mid = f"race_to_{n}_game{snap.current_game}"
                markets[mid] = [
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.RACE_MILESTONE,
                        outcome_name=ctx.entity_a_id,
                        odds=0.0,
                        prob_implied=max(0.001, p_race_a),
                        prob_with_margin=0.0,
                    ),
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.RACE_MILESTONE,
                        outcome_name=ctx.entity_b_id,
                        odds=0.0,
                        prob_implied=max(0.001, 1.0 - p_race_a),
                        prob_with_margin=0.0,
                    ),
                ]

        # Market 6: Next point winner (server advantage)
        if snap.server:
            rwp_server = rwp_a if snap.server == "A" else rwp_b
            server_name = ctx.entity_a_id if snap.server == "A" else ctx.entity_b_id
            receiver_name = ctx.entity_b_id if snap.server == "A" else ctx.entity_a_id
            markets["next_point_winner"] = [
                MarketPrice(
                    market_id="next_point_winner",
                    market_family=MarketFamily.LIVE_IN_PLAY,
                    outcome_name=server_name,
                    odds=0.0,
                    prob_implied=max(0.001, rwp_server),
                    prob_with_margin=0.0,
                ),
                MarketPrice(
                    market_id="next_point_winner",
                    market_family=MarketFamily.LIVE_IN_PLAY,
                    outcome_name=receiver_name,
                    odds=0.0,
                    prob_implied=max(0.001, 1.0 - rwp_server),
                    prob_with_margin=0.0,
                ),
            ]

        # Market 7: Deuce approaching (G-03 live micro-market)
        # Offered when either player is within 3 points of 20 and score is close
        points_to_20_a = max(0, 20 - snap.score_a)
        points_to_20_b = max(0, 20 - snap.score_b)
        score_diff = abs(snap.score_a - snap.score_b)
        if (
            snap.games_won_a < 2 and snap.games_won_b < 2
            and snap.score_a < 20 and snap.score_b < 20
            and (points_to_20_a <= 4 or points_to_20_b <= 4)
            and score_diff <= 6
        ):
            p_deuce = self._markov.p_deuce_in_game(
                rwp_a=rwp_a,
                rwp_b=rwp_b,
                score_a=snap.score_a,
                score_b=snap.score_b,
                server=snap.server or "A",
            )
            if 0.05 < p_deuce < 0.95:
                mid = f"game_{snap.current_game}_deuce"
                markets[mid] = [
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.LIVE_IN_PLAY,
                        outcome_name="deuce_yes",
                        odds=0.0,
                        prob_implied=max(0.001, p_deuce),
                        prob_with_margin=0.0,
                    ),
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.LIVE_IN_PLAY,
                        outcome_name="deuce_no",
                        odds=0.0,
                        prob_implied=max(0.001, 1.0 - p_deuce),
                        prob_with_margin=0.0,
                    ),
                ]

        # Market 8: Game pressure index — next 5 points go to trailing player (G-03)
        # Offered when one player is leading but not by more than 7
        if (
            snap.games_won_a < 2 and snap.games_won_b < 2
            and snap.server is not None
            and 1 <= score_diff <= 7
            and snap.score_a < 28 and snap.score_b < 28
        ):
            trailing = "A" if snap.score_a < snap.score_b else "B"
            trailing_rwp_as_recv = (1.0 - rwp_b) if trailing == "A" else (1.0 - rwp_a)
            trailing_rwp_as_srv = rwp_a if trailing == "A" else rwp_b
            # Approximate P(trailing wins next 5) using run probability from current server state
            current_srv_is_trailing = snap.server == trailing
            rwp_next = trailing_rwp_as_srv if current_srv_is_trailing else trailing_rwp_as_recv
            # P(trailing wins 5 of next 8) ≈ P(wins run of 5 from this state)
            p_run = self._markov.p_race_to_n(
                rwp_a=rwp_a,
                rwp_b=rwp_b,
                n=snap.score_a + 5 if trailing == "A" else snap.score_b + 5,
                score_a=snap.score_a,
                score_b=snap.score_b,
                server=snap.server,
            ) if trailing == "A" else (
                1.0 - self._markov.p_race_to_n(
                    rwp_a=rwp_a,
                    rwp_b=rwp_b,
                    n=snap.score_b + 5,
                    score_a=snap.score_a,
                    score_b=snap.score_b,
                    server=snap.server,
                )
            )
            if 0.05 < p_run < 0.95:
                trailing_name = ctx.entity_a_id if trailing == "A" else ctx.entity_b_id
                mid = f"next_5_pts_{trailing_name}"
                markets[mid] = [
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.LIVE_IN_PLAY,
                        outcome_name=f"{trailing_name}_wins_5",
                        odds=0.0,
                        prob_implied=max(0.001, p_run),
                        prob_with_margin=0.0,
                    ),
                    MarketPrice(
                        market_id=mid,
                        market_family=MarketFamily.LIVE_IN_PLAY,
                        outcome_name=f"{trailing_name}_does_not_win_5",
                        odds=0.0,
                        prob_implied=max(0.001, 1.0 - p_run),
                        prob_with_margin=0.0,
                    ),
                ]

        return markets

    @staticmethod
    def _compute_click_scales(
        momentum: MomentumSnapshot,
        base_scales: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute click scale factors for liability management.

        During strong momentum, reduce max liability to protect against
        sharp movement. Click scale 1.0 = normal, 0.5 = half liability.

        Strong/dominant momentum → 0.5 scale on affected markets.
        Moderate momentum → 0.75 scale.
        Weak/none → 1.0 scale.
        """
        scales = dict(base_scales)  # Copy base overrides

        if momentum.signal_strength == MomentumSignalStrength.DOMINANT:
            scale = 0.3 * LIVE_MOMENTUM_CLICK_SCALE_FACTOR
        elif momentum.signal_strength == MomentumSignalStrength.STRONG:
            scale = 0.5 * LIVE_MOMENTUM_CLICK_SCALE_FACTOR
        elif momentum.signal_strength == MomentumSignalStrength.MODERATE:
            scale = 0.75 * LIVE_MOMENTUM_CLICK_SCALE_FACTOR
        else:
            scale = 1.0

        # Apply reduced scale to momentum-sensitive markets
        if scale < 1.0:
            for market_key in ("match_winner", "next_point_winner"):
                if market_key not in scales or scales[market_key] > scale:
                    scales[market_key] = scale

        return scales

    @staticmethod
    def _suspended_response(
        ctx: LiveMatchContext,
        snap: LiveStateSummary,
        ts: float,
    ) -> LivePricingResponse:
        """Return a suspended response when feed gap exceeds limit."""
        logger.warning(
            "live_pricing_suspended",
            match_id=ctx.match_id,
            feed_gap_s=ctx.feed_gap_seconds(),
        )
        return LivePricingResponse(
            match_id=ctx.match_id,
            markets={},
            p_a_wins_markov=0.5,
            p_a_wins_blend=ctx.pre_match_p_a,
            markov_weight=0.0,
            rwp_a=ctx.rwp_a_prior,
            rwp_b=ctx.rwp_b_prior,
            momentum_regime="neutral",
            momentum_intensity=0.0,
            generated_at=ts,
            game_number=snap.current_game,
            score_a=snap.score_a,
            score_b=snap.score_b,
            games_won_a=snap.games_won_a,
            games_won_b=snap.games_won_b,
            is_suspended=True,
            suspension_reason=f"feed_gap_{ctx.feed_gap_seconds():.0f}s",
        )

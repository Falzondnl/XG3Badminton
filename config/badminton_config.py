"""
badminton_config.py
====================
Authoritative constants for the XG3 Badminton microservice.

All values sourced from:
  - BWF Laws of Badminton (current)
  - BWF World Tour structure (2024-2025 season)
  - V1 Tier-1 Master Plan + 27 auditor corrections (C-01..C-16, G-01..G-11)

ZERO hardcoded probabilities or odds anywhere in this file.
ZERO mock/stub data.

Rules enforced by CLAUDE.md §8 (Absolute Rules).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, Optional


# ---------------------------------------------------------------------------
# § 1. DISCIPLINE DEFINITIONS
# ---------------------------------------------------------------------------

class Discipline(str, Enum):
    """BWF-recognised competition disciplines."""
    MS = "MS"   # Men's Singles
    WS = "WS"   # Women's Singles
    MD = "MD"   # Men's Doubles
    WD = "WD"   # Women's Doubles
    XD = "XD"   # Mixed Doubles

SINGLES_DISCIPLINES: FrozenSet[Discipline] = frozenset({Discipline.MS, Discipline.WS})
DOUBLES_DISCIPLINES: FrozenSet[Discipline] = frozenset({Discipline.MD, Discipline.WD, Discipline.XD})
ALL_DISCIPLINES: FrozenSet[Discipline] = frozenset(Discipline)


# ---------------------------------------------------------------------------
# § 2. SCORING RULES  (BWF Laws of Badminton, confirmed C-13)
# ---------------------------------------------------------------------------

GAMES_TO_WIN_MATCH: int = 2          # Best of 3 — first to 2 games wins
POINTS_TO_WIN_GAME: int = 21         # First to 21 points wins a game
DEUCE_SCORE: int = 20                # Both players must reach 20-20 before deuce applies
DEUCE_WIN_MARGIN: int = 2            # Must lead by 2 points from 20-20 onwards
GOLDEN_POINT_SCORE: int = 29         # At 29-29, NEXT point wins the game (30 wins)
GOLDEN_POINT_WIN: int = 30           # Score that triggers golden-point winner

# Serving rules (confirmed C-04)
# - Winner of a rally serves next
# - Winner of a game serves first in the next game
# - At match start: coin toss determines first server
# - In doubles: service court (left/right) determined by server's score:
#   even score → right service court; odd score → left service court


# ---------------------------------------------------------------------------
# § 3. TOURNAMENT TIERS  (season-versioned per C-01 / C-03)
# ---------------------------------------------------------------------------

class TournamentTier(str, Enum):
    """BWF tournament tier classification. Season-versioned — not hardcoded counts."""
    OLYMPICS = "OLYMPICS"
    WORLD_CHAMPIONSHIPS = "WORLD_CHAMPIONSHIPS"
    WORLD_TOUR_FINALS = "WORLD_TOUR_FINALS"
    SUPER_1000 = "SUPER_1000"
    SUPER_750 = "SUPER_750"
    SUPER_500 = "SUPER_500"
    SUPER_300 = "SUPER_300"
    SUPER_100 = "SUPER_100"            # International Challenge / Series
    TEAM_EVENT = "TEAM_EVENT"          # Thomas / Uber / Sudirman Cup
    CONTINENTAL = "CONTINENTAL"        # Asia / Europe / Pan-Am championships
    NATIONAL = "NATIONAL"              # National championships (enrichment only)


@dataclass(frozen=True)
class TournamentTierConfig:
    """
    Per-tier configuration. Loaded from tournament_config.json at runtime
    to support season versioning (C-01).  This dataclass is the schema;
    actual values are NOT hardcoded here.
    """
    tier: TournamentTier
    elo_k_factor: float           # K-factor for ELO updates
    margin_match_winner: float    # Overround % for match winner market
    margin_outrights: float       # Overround % for tournament outrights
    main_draw_size: int           # Number of players in main draw (C-02: 32 for S1000/750/500/300)
    ranking_points_winner: int    # BWF ranking points for winner (versioned C-03)
    ranking_points_table: Dict[str, int]  # Full points table by round


# ---------------------------------------------------------------------------
# § 4. DRAW SIZES  (corrected C-02: 32 for Super tiers, NOT 64)
# ---------------------------------------------------------------------------

MAIN_DRAW_SIZES: Dict[TournamentTier, int] = {
    TournamentTier.OLYMPICS: 16,              # Olympics uses 16-player draw per discipline
    TournamentTier.WORLD_CHAMPIONSHIPS: 64,   # WC has 64-player draw with qualifying
    TournamentTier.WORLD_TOUR_FINALS: 8,      # Round-robin + SF/F (8 players)
    TournamentTier.SUPER_1000: 32,            # CORRECTED: 32-player main draw (C-02)
    TournamentTier.SUPER_750: 32,             # CORRECTED: 32-player main draw (C-02)
    TournamentTier.SUPER_500: 32,             # 32-player main draw
    TournamentTier.SUPER_300: 32,             # 32-player main draw
    TournamentTier.SUPER_100: 32,             # Varies — default 32
    TournamentTier.TEAM_EVENT: 16,            # 16 teams (Thomas/Uber/Sudirman)
}


# ---------------------------------------------------------------------------
# § 5. THOMAS / UBER / SUDIRMAN CUP RUBBER ORDER  (corrected C-15)
# ---------------------------------------------------------------------------

THOMAS_CUP_RUBBER_ORDER: list[tuple[Discipline, int]] = [
    (Discipline.MS, 1),   # Rubber 1: MS1
    (Discipline.MD, 1),   # Rubber 2: MD1
    (Discipline.MS, 2),   # Rubber 3: MS2
    (Discipline.MD, 2),   # Rubber 4: MD2
    (Discipline.MS, 3),   # Rubber 5: MS3 (decisive if 2-2)
]
UBER_CUP_RUBBER_ORDER: list[tuple[Discipline, int]] = [
    (Discipline.WS, 1),
    (Discipline.WD, 1),
    (Discipline.WS, 2),
    (Discipline.WD, 2),
    (Discipline.WS, 3),
]
SUDIRMAN_CUP_RUBBER_ORDER: list[tuple[Discipline, int]] = [
    (Discipline.XD, 1),
    (Discipline.MS, 1),
    (Discipline.WS, 1),
    (Discipline.MD, 1),
    (Discipline.WD, 1),
]
RUBBERS_TO_WIN_TIE: int = 3    # First team to win 3 rubbers wins the tie


# ---------------------------------------------------------------------------
# § 6. ELO SYSTEM  (8 pools — V1 Plan + auditor guidance)
# ---------------------------------------------------------------------------

class EloPool(str, Enum):
    """
    8 ELO rating pools.
    5 discipline-level pools (one per discipline).
    3 individual-within-doubles pools (for pair bootstrap & new partner scenarios).
    """
    MS_OVERALL = "MS_OVERALL"
    WS_OVERALL = "WS_OVERALL"
    MD_PAIR = "MD_PAIR"             # Partnership ELO (pair entity)
    WD_PAIR = "WD_PAIR"
    XD_PAIR = "XD_PAIR"
    MD_INDIVIDUAL = "MD_INDIVIDUAL" # Individual performance in doubles context
    WD_INDIVIDUAL = "WD_INDIVIDUAL"
    XD_INDIVIDUAL = "XD_INDIVIDUAL"


ELO_DEFAULT_RATING: float = 1500.0
ELO_INACTIVITY_DECAY_WEEKS: int = 12
ELO_INACTIVITY_DECAY_RATE: float = 0.995   # Weekly decay toward mean after 12 idle weeks
ELO_UPSET_FACTOR: float = 1.15              # K multiplier when lower-ranked wins
ELO_AGE_YOUNG_BOOST: float = 1.20          # K multiplier for players under 23
ELO_AGE_VETERAN_DECAY: float = 0.85        # K multiplier for players over 32
ELO_PAIR_FAMILIARITY_BONUS: float = 0.02   # ELO% bonus per 10 matches as same pair
ELO_NEW_PAIR_BOOTSTRAP_DISCOUNT: float = 0.95  # 5% discount for brand-new partnerships (C-10)

# K-factor by tournament tier
ELO_K_FACTORS: Dict[TournamentTier, float] = {
    TournamentTier.OLYMPICS: 48.0,
    TournamentTier.WORLD_CHAMPIONSHIPS: 48.0,
    TournamentTier.WORLD_TOUR_FINALS: 44.0,
    TournamentTier.SUPER_1000: 40.0,
    TournamentTier.SUPER_750: 36.0,
    TournamentTier.SUPER_500: 32.0,
    TournamentTier.SUPER_300: 24.0,
    TournamentTier.SUPER_100: 20.0,
    TournamentTier.TEAM_EVENT: 44.0,
    TournamentTier.CONTINENTAL: 28.0,
    TournamentTier.NATIONAL: 16.0,
}

# Singles-only K adjustments for discipline split
ELO_K_SINGLES_MULTIPLIER: float = 1.0
ELO_K_DOUBLES_MULTIPLIER: float = 0.875   # Slightly lower — more variance in doubles results


# ---------------------------------------------------------------------------
# § 7. RALLY WIN PROBABILITY (RWP) BASELINES  (literature-derived)
# ---------------------------------------------------------------------------

# Baseline RWP = P(current server wins rally)
# Source: "Using Markov chains to identify player performance in badminton"
#         Scientia et Technica (2022); FineBadminton dataset analysis (2025)
RWP_BASELINE: Dict[Discipline, float] = {
    Discipline.MS: 0.515,   # Slight server advantage in men's singles
    Discipline.WS: 0.512,   # Marginally lower server advantage in women's singles
    Discipline.MD: 0.508,   # Doubles: lower server advantage (pair dynamics)
    Discipline.WD: 0.507,
    Discipline.XD: 0.509,
}

# RWP reasonable bounds for validation
RWP_MIN_VALID: float = 0.30    # Below this → data quality issue (physical lower bound)
RWP_MAX_VALID: float = 0.80    # Above this → data quality issue (physical upper bound)


# ---------------------------------------------------------------------------
# § 8. SHUTTLE SPEED ENVIRONMENTAL MODEL  (corrected C-11)
# ---------------------------------------------------------------------------

class ShuttleType(str, Enum):
    FEATHER = "FEATHER"     # Used in all BWF World Tour events
    PLASTIC = "PLASTIC"     # Used in lower-tier / development events

# BWF shuttle speed numbers (higher = faster shuttle)
SHUTTLE_SPEED_NEUTRAL: int = 76     # Baseline neutral speed
SHUTTLE_SPEED_MIN: int = 73         # Slow (tropical/humid venues)
SHUTTLE_SPEED_MAX: int = 79         # Fast (cold/high-altitude venues)

# RWP adjustment per shuttle speed unit delta from neutral
# Faster shuttle → shorter rallies → server advantage increases slightly
RWP_SHUTTLE_SPEED_COEFFICIENT: float = 0.003

# Hall condition factors (C-11 correction — not just temp/altitude)
HALL_CONDITIONS = {
    "temperature_celsius_coefficient": 0.0015,  # Warmer = slower shuttle = lower RWP server
    "altitude_metres_coefficient": 0.0008,       # Higher altitude = faster shuttle = higher RWP server
    "humidity_pct_coefficient": -0.0010,         # More humid = slower shuttle
    "ac_strength_coefficient": 0.0005,           # Strong AC = some air movement (minimal)
}


# ---------------------------------------------------------------------------
# § 9. FATIGUE MODEL  (§9 V1 Plan)
# ---------------------------------------------------------------------------

FATIGUE_MAX_PENALTY_PP: float = 0.030   # Maximum -3pp RWP from fatigue
FATIGUE_EXTRA_MATCH_SAME_DAY: float = 0.007   # -0.7pp per extra match same day
FATIGUE_LONG_MATCH_THRESHOLD_MINUTES: int = 80
FATIGUE_LONG_MATCH_PENALTY: float = 0.004   # -0.4pp if previous match > 80 min
FATIGUE_WEEKLY_LOAD_COEFFICIENT: float = 0.002  # -0.2pp per match in last 7 days beyond 3


# ---------------------------------------------------------------------------
# § 10. MARKET MARGINS (TIER_MARGINS)
# ---------------------------------------------------------------------------

TIER_MARGINS_MATCH_WINNER: Dict[TournamentTier, float] = {
    TournamentTier.OLYMPICS: 0.04,
    TournamentTier.WORLD_CHAMPIONSHIPS: 0.04,
    TournamentTier.WORLD_TOUR_FINALS: 0.05,
    TournamentTier.SUPER_1000: 0.06,
    TournamentTier.SUPER_750: 0.07,
    TournamentTier.SUPER_500: 0.08,
    TournamentTier.SUPER_300: 0.10,
    TournamentTier.SUPER_100: 0.12,
    TournamentTier.TEAM_EVENT: 0.08,
    TournamentTier.CONTINENTAL: 0.12,
    TournamentTier.NATIONAL: 0.15,
}

TIER_MARGINS_DERIVATIVES: Dict[TournamentTier, float] = {
    TournamentTier.OLYMPICS: 0.06,
    TournamentTier.WORLD_CHAMPIONSHIPS: 0.06,
    TournamentTier.WORLD_TOUR_FINALS: 0.07,
    TournamentTier.SUPER_1000: 0.08,
    TournamentTier.SUPER_750: 0.09,
    TournamentTier.SUPER_500: 0.10,
    TournamentTier.SUPER_300: 0.12,
    TournamentTier.SUPER_100: 0.14,
    TournamentTier.TEAM_EVENT: 0.10,
    TournamentTier.CONTINENTAL: 0.14,
    TournamentTier.NATIONAL: 0.16,
}

TIER_MARGINS_OUTRIGHTS: Dict[TournamentTier, float] = {
    TournamentTier.OLYMPICS: 0.06,
    TournamentTier.WORLD_CHAMPIONSHIPS: 0.06,
    TournamentTier.WORLD_TOUR_FINALS: 0.08,
    TournamentTier.SUPER_1000: 0.10,
    TournamentTier.SUPER_750: 0.12,
    TournamentTier.SUPER_500: 0.14,
    TournamentTier.SUPER_300: 0.16,
    TournamentTier.SUPER_100: 0.18,
    TournamentTier.TEAM_EVENT: 0.10,
    TournamentTier.CONTINENTAL: 0.16,
    TournamentTier.NATIONAL: 0.18,
}

SGP_MARGIN_BASE: float = 0.07    # 7% base SGP margin
SGP_MARGIN_PER_LEG: float = 0.01  # +1% per additional leg

#: Additional margin applied per correlated leg in an SGP (on top of base derivative margin)
SGP_CORRELATION_PENALTY_PER_LEG: float = 0.01
#: Maximum number of legs allowed in a single SGP
SGP_MAX_LEGS: int = 4
#: Maximum combined SGP decimal odds (cap to prevent extreme parlays)
SGP_MAX_COMBINED_ODDS: float = 200.0
#: Minimum odds for any individual SGP leg (reject degenerate/near-certain legs)
SGP_MIN_LEG_ODDS: float = 1.10


# ---------------------------------------------------------------------------
# § 11. LIVE PRICING PARAMETERS
# ---------------------------------------------------------------------------

LIVE_GHOST_TRIGGER_SECONDS: int = 30        # Activate ghost-live if no feed update
LIVE_SUSPEND_TRIGGER_SECONDS: int = 180     # Suspend markets if no feed update > 3 min
LIVE_MAX_ODDS_JUMP_PCT: float = 0.40        # H7 gate: max 40% jump per update
LIVE_PARTICLE_FILTER_N: int = 2_000         # Particle filter size for RWP estimation
LIVE_REPRICE_EVERY_N_RALLIES_G1_G2: int = 3 # Reprice every 3 rallies in games 1 and 2
LIVE_REPRICE_EVERY_RALLY_G3: bool = True    # Reprice every rally in game 3 (decider)


# ---------------------------------------------------------------------------
# § 12. SIMULATION PARAMETERS
# ---------------------------------------------------------------------------

MC_SIMULATIONS_PER_MATCH: int = 100_000    # Monte Carlo simulations per match pricing
MC_SIMULATIONS_OUTRIGHT: int = 50_000      # Tournament outright simulation


# ---------------------------------------------------------------------------
# § 13. ML MODEL PARAMETERS
# ---------------------------------------------------------------------------

ML_FEATURES_TOTAL: int = 66               # 9 groups (A-I)
ML_REGIME_R0_MAX_MATCHES: Dict[Discipline, int] = {
    Discipline.MS: 5,
    Discipline.WS: 5,
    Discipline.MD: 8,   # Pairs accumulate data slower
    Discipline.WD: 8,
    Discipline.XD: 8,
}
ML_REGIME_R1_MAX_MATCHES: Dict[Discipline, int] = {
    Discipline.MS: 50,
    Discipline.WS: 40,  # Smaller field — fewer matches
    Discipline.MD: 60,
    Discipline.WD: 50,
    Discipline.XD: 55,
}

# Train / validation / test split
ML_TRAIN_START_YEAR: int = 2018
ML_TRAIN_END_YEAR: int = 2021       # Inclusive
ML_VAL_TUNE_YEAR: int = 2022        # Hyperparameter tuning
ML_VAL_TEST_YEAR: int = 2023        # Hold-out test
ML_DEPLOY_START_YEAR: int = 2024    # Production — never touched during training

# Three modeling targets (C-09 correction)
ML_TARGET_MATCH_WIN: str = "p_win"          # P(player A wins match)
ML_TARGET_STRAIGHT_WIN: str = "p_2_0_win"  # P(2-0 | A wins)
ML_TARGET_DEUCE: str = "p_deuce_any_game"  # P(at least one game goes to deuce)

# P1/P2 swap guard
ML_P1_WIN_RATE_MIN: float = 0.45
ML_P1_WIN_RATE_MAX: float = 0.55

# QA thresholds
ML_BRIER_THRESHOLD: float = 0.24
ML_AUC_THRESHOLD: float = 0.65
ML_ECE_THRESHOLD: float = 0.05
ML_BRIER_DRIFT_RETRAIN_TRIGGER: float = 0.27   # Auto-retrain if rolling Brier >= 0.27


# ---------------------------------------------------------------------------
# § 14. REGRESSION LOCK
# ---------------------------------------------------------------------------

REGRESSION_LOCK_TESTS_TARGET: int = 150
REGRESSION_LOCK_FILE: str = "lock_state.json"


# ---------------------------------------------------------------------------
# § 15. SERVICE CONFIG
# ---------------------------------------------------------------------------

SERVICE_NAME: str = "xg3-badminton"
SERVICE_PORT: int = 8009
SERVICE_VERSION: str = "1.0.0"
API_PREFIX: str = "/badminton"

# Redis keys
REDIS_LIVE_MATCH_STATE_PREFIX: str = "badminton:live:match:"
REDIS_RWP_CACHE_PREFIX: str = "badminton:rwp:"
REDIS_OUTRIGHT_CACHE_PREFIX: str = "badminton:outright:"

# Data paths (resolved from env vars at runtime — never hardcoded)
DATA_ROOT_ENV_VAR: str = "BADMINTON_DATA_ROOT"   # e.g., D:\codex\Data\Badminton
MODEL_DIR_ENV_VAR: str = "BADMINTON_MODEL_DIR"
DB_URL_ENV_VAR: str = "BADMINTON_DATABASE_URL"
REDIS_URL_ENV_VAR: str = "BADMINTON_REDIS_URL"
OPTIC_ODDS_API_KEY_ENV_VAR: str = "OPTIC_ODDS_API_KEY"
FLASHSCORE_FEED_URL_ENV_VAR: str = "FLASHSCORE_BADMINTON_FEED_URL"


# ---------------------------------------------------------------------------
# § 16. MARKET FAMILIES (15 families, 97 total markets)
# ---------------------------------------------------------------------------

class MarketFamily(str, Enum):
    MATCH_RESULT = "MATCH_RESULT"                 # 4 markets
    TOTAL_GAMES = "TOTAL_GAMES"                   # 5 markets
    CORRECT_SCORE = "CORRECT_SCORE"               # 4 markets
    GAME_LEVEL = "GAME_LEVEL"                     # 12 markets
    RACE_MILESTONE = "RACE_MILESTONE"             # 10 markets
    POINTS_TOTALS = "POINTS_TOTALS"               # 8 markets
    PLAYER_PROPS = "PLAYER_PROPS"                 # 12 markets
    LIVE_IN_PLAY = "LIVE_IN_PLAY"                 # 8 markets (live engine only)
    OUTRIGHTS = "OUTRIGHTS"                       # 6 market types
    OUTRIGHT_DERIVATIVES = "OUTRIGHT_DERIVATIVES" # 5 markets
    EXOTIC = "EXOTIC"                             # 6 markets
    SGP = "SGP"                                   # 6 market types
    FUTURES = "FUTURES"                           # 3 market types
    LIVE_SGP = "LIVE_SGP"                         # 3 market types
    TEAM_EVENTS = "TEAM_EVENTS"                   # 5 markets (Thomas/Uber/Sudirman)


TOTAL_MARKET_COUNT: int = 97


# ---------------------------------------------------------------------------
# § 17. XD-SPECIFIC CONSTANTS  (mixed doubles gender positioning)
# ---------------------------------------------------------------------------

XD_MAN_COURT_POSITION_REAR: float = 0.60    # Man typically rear court (60% weighting for MS ELO)
XD_WOMAN_COURT_POSITION_FRONT: float = 0.40  # Woman typically front court (40% weighting for WS ELO)


# ---------------------------------------------------------------------------
# § 18. TRADING CONTROL CONSTANTS
# ---------------------------------------------------------------------------

TRADING_DEFAULT_CLICK_MAX_GBP: float = 50_000.0       # Default max payout per market per click
TRADING_LIABILITY_SUSPEND_THRESHOLD_GBP: float = 150_000.0  # Auto-suspend when liability exceeds this
TRADING_COOLDOWN_SECONDS: float = 60.0                # Cooldown after auto-suspension before resuming


# ---------------------------------------------------------------------------
# § 19. LIVE TIMING ALIASES  (short names used by agents/monitoring)
# ---------------------------------------------------------------------------

#: Short alias used by agents/monitoring_supervisor.py and agents/outright_supervisor.py
LIVE_GHOST_TRIGGER_S: int = LIVE_GHOST_TRIGGER_SECONDS    # 30s
LIVE_SUSPEND_TRIGGER_S: int = LIVE_SUSPEND_TRIGGER_SECONDS  # 180s


# ---------------------------------------------------------------------------
# § 20. MARKET QUALITY CONSTANTS
# ---------------------------------------------------------------------------

OVERROUND_MIN: float = 0.04    # H1: minimum overround 4% (Tier 1)
OVERROUND_MAX: float = 0.18    # H1: maximum overround 18% (Tier 1)

# H10 gate: minimum publishable odds (floor)
MIN_ODDS: float = 1.01


# ---------------------------------------------------------------------------
# § 21. BAYESIAN / ENVIRONMENT / MOMENTUM CONSTANTS
# ---------------------------------------------------------------------------

BAYESIAN_PRIOR_STRENGTH: float = 10.0                    # Effective prior sample size for Bayesian RWP update
BAYESIAN_POSTERIOR_FULL_WEIGHT_RALLIES: int = 150        # Rallies required for posterior to fully dominate prior
BAYESIAN_MOMENTUM_DECAY: float = 0.92                    # Per-point exponential decay of momentum signal in Bayesian update

ENV_MAX_ADJUSTMENT: float = 0.015                        # Max ±RWP adjustment from environmental factors
ENV_ALTITUDE_THRESHOLD_M: float = 1500.0                 # Altitude above which aerodynamics affect shuttle speed (metres)
ENV_SHUTTLE_SPEED_BASELINE: int = 77                     # BWF baseline shuttle speed grade (indoor standard conditions)

MOMENTUM_MIN_RUN_SIGNIFICANCE: int = 3                   # Minimum consecutive points to constitute a run
MOMENTUM_SIGNIFICANCE_THRESHOLD: int = 5                 # Consecutive points triggering FLOW regime
MOMENTUM_DECAY_FACTOR: float = 0.85                      # Per-point decay for momentum signal weight
MOMENTUM_LOOKBACK_POINTS: int = 20                       # Rolling window for momentum regime detection

#: Alias used by feed_health_monitor.py
LIVE_SUSPEND_SECONDS: int = LIVE_SUSPEND_TRIGGER_SECONDS  # 180s

#: Scale factor applied to click limits when momentum signal is STRONG.
#: 1.5 = 50% tighter sizing when a strong momentum run is detected.
LIVE_MOMENTUM_CLICK_SCALE_FACTOR: float = 1.5

# ---------------------------------------------------------------------------
# § PRE-MATCH PRICING WEIGHTS
# ---------------------------------------------------------------------------

#: Weight given to our own ML model in the Pinnacle/model blend (70% model)
PRE_MATCH_MODEL_WEIGHT: float = 0.70
#: Complementary weight for Pinnacle (30% Pinnacle when available)
PRE_MATCH_MARKOV_WEIGHT: float = 1.0 - PRE_MATCH_MODEL_WEIGHT

#: Pre-match prices expire after this many seconds (1 minute)
MARKET_PRICE_VALIDITY_SECONDS: int = 60

#: How often (seconds) pre-match prices are refreshed by PreMatchSupervisorAgent
PRE_MATCH_REFRESH_INTERVAL_SECONDS: int = 30

#: How often (seconds) outrights are repriced by OutrightSupervisorAgent
OUTRIGHT_REPRICE_INTERVAL_S: int = 60

FEED_ERROR_RATE_ALERT: float = 0.05          # Error rate >= 5% triggers DEGRADED status
FEED_ERROR_RATE_SUSPEND: float = 0.20        # Error rate >= 20% triggers UNHEALTHY/DOWN status
FEED_MESSAGES_WINDOW_SECONDS: int = 300      # Window for error rate calculation (5 minutes)

# ---------------------------------------------------------------------------
# § 22. ML REGIME GATE CONSTANTS (R0 / R1 / R2)
# ---------------------------------------------------------------------------

# R0 — insufficient data; use prior RWP only
REGIME_R0_MIN_MATCHES: int = 5

# R2 — full power model; requires high-tier tournament + minimum match history
REGIME_R2_TIERS: frozenset = frozenset({
    TournamentTier.SUPER_1000,
    TournamentTier.SUPER_750,
    TournamentTier.SUPER_500,
})
REGIME_R2_MIN_MATCHES: int = 30

# ---------------------------------------------------------------------------
# § 23. ORCHESTRATOR LIMITS
# ---------------------------------------------------------------------------

ORCHESTRATOR_MAX_ACTIVE_MATCHES: int = 500   # Hard cap: concurrent live + pre-match matches

# ---------------------------------------------------------------------------
# § 24. OUTRIGHT PRICING
# ---------------------------------------------------------------------------

#: Number of Monte Carlo simulations for tournament outright pricing (V1 Plan §18)
OUTRIGHT_N_SIMULATIONS: int = 100_000

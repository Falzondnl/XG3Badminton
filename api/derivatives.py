"""
Badminton Derivatives Engine — POST /api/v1/badminton/derivatives/generate

Generates derivative markets for a badminton match from match-level win
probabilities supplied by the caller (model output).

Badminton is set-based (best-of-3 or best-of-5), directly analogous to
tennis / table tennis / volleyball set markets.  All combinatorial
probabilities are derived from the input p_player1 / p_player2.

No hardcoded outcome probabilities.  All markets computed from request inputs.
"""
from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/badminton/derivatives", tags=["Badminton Derivatives"])

# ---------------------------------------------------------------------------
# Vig parameters
# ---------------------------------------------------------------------------
_VIG_ML: float = 0.045       # 4.5 % moneyline
_VIG_HANDICAP: float = 0.055 # 5.5 % handicap
_VIG_TOTAL: float = 0.055    # 5.5 % totals
_VIG_CORRECT: float = 0.07   # 7 % correct score
_VIG_GAME1: float = 0.05     # 5 % game 1 winner
_VIG_CLEAN: float = 0.06     # 6 % clean win

# Game-1 regression factor toward 0.5 (accounts for pre-match uncertainty)
_GAME1_REGRESSION: float = 0.05


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class DerivativesRequest(BaseModel):
    match_id: str = Field(..., description="Unique match identifier")
    p_player1: float = Field(..., ge=0.01, le=0.99, description="Calibrated P(Player1 wins match)")
    p_player2: float = Field(..., ge=0.01, le=0.99, description="Calibrated P(Player2 wins match)")
    player1_name: str = Field(default="Player1")
    player2_name: str = Field(default="Player2")
    format: str = Field(
        default="bo3",
        description="Match format: 'bo3' (best-of-3) or 'bo5' (best-of-5)",
    )

    @model_validator(mode="after")
    def _validate(self) -> "DerivativesRequest":
        total = self.p_player1 + self.p_player2
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"p_player1 + p_player2 must sum to ~1.0 (got {total:.4f})")
        if self.format not in ("bo3", "bo5"):
            raise ValueError(f"format must be 'bo3' or 'bo5', got '{self.format}'")
        return self


# ---------------------------------------------------------------------------
# Combinatorial helpers
# ---------------------------------------------------------------------------

def _nCr(n: int, r: int) -> int:
    return math.comb(n, r)


def _p_wins_series(p_game: float, games_to_win: int) -> float:
    """
    P(player wins best-of-(2*games_to_win - 1) series) given per-game win prob p_game.

    For Bo3: games_to_win=2.  P = P(2-0) + P(2-1)
    For Bo5: games_to_win=3.  P = P(3-0) + P(3-1) + P(3-2)
    """
    n_games = games_to_win
    p = 0.0
    q = 1.0 - p_game
    for wins in range(n_games, 2 * n_games):  # games played ranges from n_games to 2*n_games-1
        # Player must win the last game, and have exactly n_games-1 wins in previous (games-1) games
        games_played = wins
        p_path = _nCr(games_played - 1, n_games - 1) * (p_game ** n_games) * (q ** (games_played - n_games))
        p += p_path
    return p


def _infer_game_prob(p_match: float, games_to_win: int) -> float:
    """
    Infer per-game win probability from match win probability by binary search.
    The match win probability is a strictly increasing function of the game win probability.
    """
    lo, hi = 0.001, 0.999
    for _ in range(60):  # 60 bisection iterations → < 1e-18 precision
        mid = (lo + hi) / 2.0
        if _p_wins_series(mid, games_to_win) < p_match:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _correct_score_probs(p_game: float, games_to_win: int) -> Dict[str, float]:
    """
    Exact score probabilities for the series.

    Bo3: {2-0, 2-1, 0-2, 1-2}
    Bo5: {3-0, 3-1, 3-2, 0-3, 1-3, 2-3}
    """
    q = 1.0 - p_game
    scores: Dict[str, float] = {}
    n = games_to_win

    for winner_wins in range(n, 2 * n):
        loser_wins = winner_wins - n
        games_played = winner_wins + loser_wins
        # Winner must win the last game: C(games_played-1, n-1) paths for the first (games_played-1) games
        # where winner got (n-1) wins and loser got all their wins
        # P1 wins with score (winner_wins - loser_wins):
        p1_score = _nCr(games_played - 1, n - 1) * (p_game ** n) * (q ** loser_wins)
        p2_score = _nCr(games_played - 1, n - 1) * (q ** n) * (p_game ** loser_wins)
        scores[f"{n}-{loser_wins}"] = p1_score
        scores[f"{loser_wins}-{n}"] = p2_score

    return scores


def _to_decimal(p: float, vig: float) -> float:
    if p <= 0.0:
        return 9999.0
    implied = min(p * (1.0 + vig), 0.9999)
    return round(1.0 / implied, 4)


def _market(name: str, outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"market_name": name, "outcomes": outcomes}


def _outcome(label: str, probability: float, decimal_odds: float) -> Dict[str, Any]:
    return {"label": label, "probability": round(probability, 6), "decimal_odds": decimal_odds}


# ---------------------------------------------------------------------------
# Market builders
# ---------------------------------------------------------------------------

def _build_moneyline(p1: float, p2: float, n1: str, n2: str) -> Dict[str, Any]:
    return _market(
        "Match Winner",
        [
            _outcome(n1, p1, _to_decimal(p1, _VIG_ML)),
            _outcome(n2, p2, _to_decimal(p2, _VIG_ML)),
        ],
    )


def _build_handicap_games(
    p_game: float, games_to_win: int, n1: str, n2: str
) -> List[Dict[str, Any]]:
    """
    Handicap games markets.

    For Bo3 (games_to_win=2): handicaps -1.5, -0.5, +0.5, +1.5
    For Bo5 (games_to_win=3): handicaps -2.5, -1.5, -0.5, +0.5, +1.5, +2.5

    P(P1 covers -k.5 handicap) = P(P1 wins by > k games)
    """
    scores = _correct_score_probs(p_game, games_to_win)
    max_games = 2 * games_to_win - 1

    markets = []
    if games_to_win == 2:
        handicaps = [-1.5, -0.5, 0.5, 1.5]
    else:
        handicaps = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    for hcap in handicaps:
        p1_covers = 0.0
        for score_str, prob in scores.items():
            parts = score_str.split("-")
            s1, s2 = int(parts[0]), int(parts[1])
            margin = s1 - s2  # positive if P1 wins by that margin
            if margin + hcap > 0:
                p1_covers += prob
        p2_covers = 1.0 - p1_covers

        label_sign = f"{'+' if hcap > 0 else ''}{hcap}"
        markets.append(
            _market(
                f"Games Handicap {label_sign}",
                [
                    _outcome(f"{n1} {label_sign}", p1_covers, _to_decimal(p1_covers, _VIG_HANDICAP)),
                    _outcome(f"{n2} {'+' if -hcap > 0 else ''}{-hcap}", p2_covers, _to_decimal(p2_covers, _VIG_HANDICAP)),
                ],
            )
        )
    return markets


def _build_total_games(p_game: float, games_to_win: int, n1: str, n2: str) -> List[Dict[str, Any]]:
    """
    Total Games O/U.
    Bo3: O/U 2.5  (3 games happens only when score is 1-1 after 2)
    Bo5: O/U 3.5, O/U 4.5
    """
    scores = _correct_score_probs(p_game, games_to_win)

    def total_games_from_score(score_str: str) -> int:
        parts = score_str.split("-")
        return int(parts[0]) + int(parts[1])

    markets = []
    if games_to_win == 2:
        lines = [2.5]
    else:
        lines = [3.5, 4.5]

    for line in lines:
        p_over = sum(prob for s, prob in scores.items() if total_games_from_score(s) > line)
        p_under = 1.0 - p_over
        markets.append(
            _market(
                f"Total Games O/U {line}",
                [
                    _outcome(f"Over {line}", p_over, _to_decimal(p_over, _VIG_TOTAL)),
                    _outcome(f"Under {line}", p_under, _to_decimal(p_under, _VIG_TOTAL)),
                ],
            )
        )
    return markets


def _build_correct_score(
    p_game: float, games_to_win: int, n1: str, n2: str
) -> Dict[str, Any]:
    scores = _correct_score_probs(p_game, games_to_win)
    outcomes = []
    # Sort: P1 wins first (descending game count), then P2 wins
    for score_str in sorted(scores.keys()):
        parts = score_str.split("-")
        s1, s2 = int(parts[0]), int(parts[1])
        if s1 > s2:
            label = f"{n1} {s1}-{s2}"
        else:
            label = f"{n2} {s2}-{s1}"
        prob = scores[score_str]
        outcomes.append(_outcome(label, prob, _to_decimal(prob, _VIG_CORRECT)))
    return _market("Correct Score", outcomes)


def _build_game1_winner(p_game: float, n1: str, n2: str) -> Dict[str, Any]:
    """
    Game 1 Winner — apply regression toward 0.5.

    p_game_1 = p_game * (1 - regression) + 0.5 * regression
    """
    p1 = p_game * (1.0 - _GAME1_REGRESSION) + 0.5 * _GAME1_REGRESSION
    p2 = 1.0 - p1
    return _market(
        "Game 1 Winner",
        [
            _outcome(n1, p1, _to_decimal(p1, _VIG_GAME1)),
            _outcome(n2, p2, _to_decimal(p2, _VIG_GAME1)),
        ],
    )


def _build_clean_win(
    p_game: float, games_to_win: int, n1: str, n2: str
) -> Dict[str, Any]:
    """
    Clean Win (2-0 for Bo3 / 3-0 for Bo5).

    P(2-0) = p_game^2
    P(3-0) = p_game^3
    Similarly for opponent.
    """
    p1_clean = p_game ** games_to_win
    p2_clean = (1.0 - p_game) ** games_to_win
    p_other = 1.0 - p1_clean - p2_clean
    outcomes = [
        _outcome(f"{n1} {games_to_win}-0 (Clean)", p1_clean, _to_decimal(p1_clean, _VIG_CLEAN)),
        _outcome(f"{n2} {games_to_win}-0 (Clean)", p2_clean, _to_decimal(p2_clean, _VIG_CLEAN)),
        _outcome("Decisive (Not Clean)", p_other, _to_decimal(p_other, _VIG_CLEAN)),
    ]
    return _market(f"Clean Win ({games_to_win}-0)", outcomes)


# ---------------------------------------------------------------------------
# Derivatives cache
# ---------------------------------------------------------------------------
_cache: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate", summary="Generate all derivative markets for a badminton match")
async def generate_derivatives(req: DerivativesRequest) -> Dict[str, Any]:
    """
    Generate the full badminton derivative market set from win probabilities.

    Markets:
    - Match Winner (4.5 % vig)
    - Games Handicap (2 lines for Bo3 / 3 lines for Bo5)
    - Total Games O/U (1 line for Bo3 / 2 lines for Bo5)
    - Correct Score (4 outcomes Bo3 / 6 outcomes Bo5)
    - Game 1 Winner (with regression to 0.5)
    - Clean Win (X-0)
    """
    t0 = time.perf_counter()
    rid = str(uuid.uuid4())

    games_to_win = 2 if req.format == "bo3" else 3
    p1 = req.p_player1
    p2 = req.p_player2
    n1 = req.player1_name
    n2 = req.player2_name

    # Infer per-game probability from match probability via bisection
    p_game = _infer_game_prob(p1, games_to_win)

    markets: List[Dict[str, Any]] = []
    markets.append(_build_moneyline(p1, p2, n1, n2))
    markets.extend(_build_handicap_games(p_game, games_to_win, n1, n2))
    markets.extend(_build_total_games(p_game, games_to_win, n1, n2))
    markets.append(_build_correct_score(p_game, games_to_win, n1, n2))
    markets.append(_build_game1_winner(p_game, n1, n2))
    markets.append(_build_clean_win(p_game, games_to_win, n1, n2))

    result = {
        "success": True,
        "data": {
            "match_id": req.match_id,
            "player1": n1,
            "player2": n2,
            "p_player1": p1,
            "p_player2": p2,
            "format": req.format,
            "inferred_per_game_prob_p1": round(p_game, 6),
            "market_count": len(markets),
            "markets": markets,
        },
        "meta": {
            "request_id": rid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
        },
    }

    _cache[req.match_id] = result
    logger.info(
        "badminton_derivatives_generated",
        match_id=req.match_id,
        format=req.format,
        markets=len(markets),
        elapsed_ms=result["meta"]["elapsed_ms"],
    )
    return result


@router.get("/cached/{match_id}", summary="Retrieve cached badminton derivative markets")
async def get_cached_derivatives(match_id: str) -> Dict[str, Any]:
    if match_id not in _cache:
        raise HTTPException(
            status_code=404,
            detail=f"No cached derivatives for match_id '{match_id}'. Call POST .../generate first.",
        )
    return _cache[match_id]


@router.get("/health", summary="Badminton derivatives health check")
async def derivatives_health() -> Dict[str, Any]:
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "service": "badminton-derivatives",
            "cached_matches": len(_cache),
            "supported_formats": ["bo3", "bo5"],
            "market_types": [
                "Match Winner", "Games Handicap", "Total Games O/U",
                "Correct Score", "Game 1 Winner", "Clean Win",
            ],
        },
        "meta": {"timestamp": datetime.now(timezone.utc).isoformat()},
    }

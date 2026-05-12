"""
Badminton Same-Game Parlay (SGP) router — bet365-grade.

POST /api/v1/badminton/sgp/price

Wires the existing BadmintonSGPEngine into a production-routable API. Uses
the BadmintonPredictor singleton from main.py to compute rwp_a/rwp_b via
the Markov-inversion utility, then delegates pricing to the SGP engine.

Refuse-to-price doctrine: structured 503 with code + correlation_id when the
predictor is unavailable. Structured 422 for invalid leg types or duplicate
players. Structured 400 when SGP engine rejects the request (legs/odds/etc.).

Standard response envelope:
    {"success": true, "data": {...}, "meta": {"request_id": "uuid", "timestamp": "ISO8601"}}
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from config.badminton_config import Discipline, TournamentTier
from markets.sgp_engine import (
    BadmintonSGPEngine,
    SGPLeg,
    SGPLegType,
    SGPRequest as SGPEngineRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/badminton", tags=["SGP"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SGPLegBody(BaseModel):
    """A single leg in the SGP request body."""
    leg_type: str = Field(..., description="One of: match_winner, total_games, correct_score, game_winner, race_to_n, points_ou")
    selection: str = Field(..., description="Normalised selection string (e.g., 'A', 'over_2.5')")
    fair_prob: float = Field(..., ge=0.0, le=1.0, description="Pre-margin probability for this leg")
    market_id: str = Field(..., description="Source market identifier")
    param_game: Optional[int] = Field(default=None, description="Game number (for game-specific legs)")
    param_n: Optional[int] = Field(default=None, description="Race-to N value")
    param_threshold: Optional[float] = Field(default=None, description="O/U threshold")


class BadmintonSGPRequestBody(BaseModel):
    match_id: str
    player1: str
    player2: str
    discipline: str = Field(..., description="One of: MS, WS, MD, XD, WD")
    tier: str = Field(..., description="TournamentTier name (e.g., SUPER_500, WORLD_TOUR_FINALS)")
    legs: List[SGPLegBody]
    round: str = Field(default="R32", description="Tournament round (used to resolve rwp via predictor)")
    tournament_type: str = Field(default="Super 500", description="Tournament type (used for predictor features)")
    first_server: str = Field(default="A", description="'A' or 'B'")
    partner1: Optional[str] = None
    partner2: Optional[str] = None
    nationality1: str = ""
    nationality2: str = ""
    country: str = ""


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def _meta(request_id: str) -> Dict[str, str]:
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _ok(data: Any, request_id: str) -> Dict[str, Any]:
    return {"success": True, "data": data, "meta": _meta(request_id)}


def _err(
    code: str,
    message: str,
    request_id: str,
    http_status: int,
    extra: Optional[Dict[str, Any]] = None,
) -> ORJSONResponse:
    body = {
        "success": False,
        "error": {
            "code": code,
            "message": message,
            "correlation_id": request_id,
        },
        "meta": _meta(request_id),
    }
    if extra:
        body["error"].update(extra)
    return ORJSONResponse(content=body, status_code=http_status)


# ---------------------------------------------------------------------------
# RWP resolution — Markov inversion using predictor's calibrated match prob
# ---------------------------------------------------------------------------


def _resolve_rwp(predictor: Any, body: BadmintonSGPRequestBody, discipline: Discipline) -> tuple[float, float]:
    """Resolve rwp_a / rwp_b from the predictor's calibrated match probability.

    Uses BadmintonPredictor.predict() → p_a_wins_match → invert via Markov
    bisection (same utility as ml/inference.py:_rwp_from_match_prob).
    """
    pred = predictor.predict(
        player1=body.player1,
        player2=body.player2,
        discipline=body.discipline,
        round_str=body.round,
        tournament_type=body.tournament_type,
        partner1=body.partner1,
        partner2=body.partner2,
        nationality1=body.nationality1,
        nationality2=body.nationality2,
        country=body.country,
    )
    p_a = float(pred["p1_win_prob"])

    from core.markov_engine import BadmintonMarkovEngine
    from config.badminton_config import RWP_BASELINE

    markov = BadmintonMarkovEngine()
    baseline = RWP_BASELINE[discipline]

    def get_p_match(rwp_trial: float) -> float:
        probs = markov.compute_match_probabilities(
            rwp_a=rwp_trial,
            rwp_b=baseline,
            discipline=discipline,
            server_first_game=body.first_server,
        )
        return probs.p_a_wins_match

    lo, hi = 0.20, 0.80
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if get_p_match(mid) < p_a:
            lo = mid
        else:
            hi = mid

    rwp_a = (lo + hi) / 2.0
    rwp_b = baseline
    return rwp_a, rwp_b


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "/sgp/price",
    summary="Price a Badminton Same-Game Parlay (Markov joint-probability)",
    response_class=ORJSONResponse,
)
async def price_sgp(body: BadmintonSGPRequestBody) -> ORJSONResponse:
    """Price a Same-Game Parlay with Markov-correlated joint probability.

    The engine uses the BadmintonMarkovEngine state space to compute exact
    joint probabilities for correlated leg combinations (e.g., match winner
    AND total games). For independent legs, multiplication is used.

    SGP_CORRELATION_PENALTY_PER_LEG is added per extra leg on top of the
    derivative margin to compensate for correlation residual.

    HTTP 200 — priced SGP.
    HTTP 400 — invalid leg type, leg below SGP_MIN_LEG_ODDS, or combined
               odds exceed SGP_MAX_COMBINED_ODDS — see error.code.
    HTTP 422 — duplicate players or unknown discipline/tier.
    HTTP 503 — predictor not loaded; SGP cannot resolve RWP without it.
    """
    rid = str(uuid.uuid4())

    # Input validation
    if body.player1.strip().lower() == body.player2.strip().lower():
        return _err(
            "INVALID_INPUT",
            "player1 and player2 must be different players",
            rid,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    try:
        discipline = Discipline(body.discipline)
    except ValueError:
        return _err(
            "INVALID_DISCIPLINE",
            f"Unknown discipline '{body.discipline}'. Valid: {[d.value for d in Discipline]}",
            rid,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    try:
        tier = TournamentTier(body.tier)
    except ValueError:
        return _err(
            "INVALID_TIER",
            f"Unknown tier '{body.tier}'. Valid: {[t.value for t in TournamentTier]}",
            rid,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    if not body.legs:
        return _err(
            "EMPTY_SGP",
            "At least one leg required",
            rid,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # Coerce legs
    try:
        legs: List[SGPLeg] = [
            SGPLeg(
                leg_type=SGPLegType(l.leg_type),
                selection=l.selection,
                fair_prob=l.fair_prob,
                market_id=l.market_id,
                param_game=l.param_game,
                param_n=l.param_n,
                param_threshold=l.param_threshold,
            )
            for l in body.legs
        ]
    except ValueError as exc:
        return _err(
            "INVALID_LEG_TYPE",
            str(exc),
            rid,
            status.HTTP_400_BAD_REQUEST,
        )

    # Resolve predictor singleton from main.py
    try:
        import main as _main_module
        predictor = getattr(_main_module, "_predictor", None)
    except Exception as exc:
        logger.error("badminton_sgp.predictor_import_failed: %s", exc)
        predictor = None

    if predictor is None or not predictor.is_ready:
        return _err(
            "MODEL_NOT_LOADED",
            "BadmintonPredictor is not loaded. SGP cannot resolve RWP without the calibrated match probability.",
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # Resolve rwp_a / rwp_b via Markov bisection on calibrated p_a_wins_match
    try:
        rwp_a, rwp_b = _resolve_rwp(predictor, body, discipline)
    except RuntimeError as exc:
        return _err(
            "RWP_RESOLUTION_FAILED",
            str(exc),
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        logger.error("badminton_sgp.rwp_resolution_error: %s", exc, exc_info=True)
        return _err(
            "INTERNAL_ERROR",
            f"RWP resolution failed: {exc}",
            rid,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    # Price SGP
    engine = BadmintonSGPEngine()
    sgp_req = SGPEngineRequest(
        match_id=body.match_id,
        entity_a_id=body.player1,
        entity_b_id=body.player2,
        discipline=discipline,
        tier=tier,
        legs=legs,
        rwp_a=rwp_a,
        rwp_b=rwp_b,
        first_server=body.first_server,
    )

    try:
        result = engine.price_sgp(sgp_req)
    except Exception as exc:
        logger.error("badminton_sgp.engine_error: %s", exc, exc_info=True)
        return _err(
            "INTERNAL_ERROR",
            f"SGP pricing failed: {exc}",
            rid,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    if not result.is_valid:
        return _err(
            "SGP_REJECTED",
            result.rejection_reason or "SGP rejected by engine",
            rid,
            status.HTTP_400_BAD_REQUEST,
            extra={"warnings": result.warnings},
        )

    payload = {
        "match_id": body.match_id,
        "n_legs": result.n_legs,
        "combined_odds": round(result.combined_odds, 4),
        "combined_odds_fair": round(result.combined_odds_fair, 4),
        "joint_prob": round(result.joint_prob_margined, 6),
        "joint_prob_fair": round(result.joint_prob_fair, 6),
        "correlation_adjustment": round(result.correlation_adjustment, 4),
        "margin": round(result.margin_applied, 4),
        "rwp_a": round(rwp_a, 4),
        "rwp_b": round(rwp_b, 4),
        "prediction_source": "model",
        "warnings": result.warnings,
    }

    return ORJSONResponse(content=_ok(payload, rid), status_code=status.HTTP_200_OK)

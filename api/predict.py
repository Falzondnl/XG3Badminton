"""
Badminton /predict endpoint — GAP-A-07.

POST /api/v1/badminton/predict

Returns ML ensemble win probabilities for a badminton match defined by
its core pre-match attributes.  This endpoint:

  - Accepts singles (MS/WS) and doubles (MD/XD/WD) disciplines.
  - Uses the BadmintonPredictor (R0=MS, R1=WS, R2=doubles stacking ensembles
    + BetaCalibrator per regime) loaded at startup.
  - NEVER returns a hardcoded default.  If models are not loaded → HTTP 503.

Standard response envelope:
    {"success": true, "data": {...}, "meta": {"request_id": "uuid", "timestamp": "ISO8601"}}
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, field_validator

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/badminton", tags=["Predict"])

# ---------------------------------------------------------------------------
# Valid disciplines — mirrors config.ALL_DISCIPLINES
# ---------------------------------------------------------------------------

_VALID_DISCIPLINES = {"MS", "WS", "MD", "XD", "WD"}


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


def _error(
    code: str,
    message: str,
    request_id: str,
    http_status: int = 400,
) -> ORJSONResponse:
    return ORJSONResponse(
        content={
            "success": False,
            "error": {"code": code, "message": message},
            "meta": _meta(request_id),
        },
        status_code=http_status,
    )


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class BadmintonPredictRequest(BaseModel):
    """
    Ad-hoc match definition for badminton win probability prediction.

    Singles disciplines (MS, WS): supply player1 and player2.
    Doubles disciplines (MD, XD, WD): also supply partner1 and partner2.
    """

    player1: str = Field(
        ...,
        description="Player/team-one name.  Singles: individual player. Doubles: first player.",
        min_length=1,
        max_length=128,
    )
    player2: str = Field(
        ...,
        description="Player/team-two name.  Singles: individual player. Doubles: first player.",
        min_length=1,
        max_length=128,
    )
    discipline: str = Field(
        ...,
        description="BWF discipline code: MS | WS | MD | XD | WD",
    )
    round: str = Field(
        default="Round of 16",
        description="Tournament round string (e.g. 'Quarter final', 'Final')",
    )
    tournament_type: str = Field(
        default="HSBC BWF World Tour Super 300",
        description="BWF tournament tier label",
    )
    partner1: Optional[str] = Field(
        None,
        description="Partner of player1.  Required for doubles disciplines (MD/XD/WD).",
        max_length=128,
    )
    partner2: Optional[str] = Field(
        None,
        description="Partner of player2.  Required for doubles disciplines (MD/XD/WD).",
        max_length=128,
    )
    nationality1: str = Field(
        default="",
        description="Nationality / country code for player1 team",
        max_length=8,
    )
    nationality2: str = Field(
        default="",
        description="Nationality / country code for player2 team",
        max_length=8,
    )
    country: str = Field(
        default="",
        description="Host country of the tournament",
        max_length=64,
    )
    market_prob_player1: Optional[float] = Field(
        None,
        description=(
            "Market-implied probability for player1 (de-vigged). "
            "When supplied, a logit-space 20%/80% model/market blend is applied."
        ),
        gt=0.0,
        lt=1.0,
    )

    @field_validator("discipline")
    @classmethod
    def validate_discipline(cls, v: str) -> str:
        upper = v.upper()
        if upper not in _VALID_DISCIPLINES:
            raise ValueError(
                f"discipline must be one of {sorted(_VALID_DISCIPLINES)}, got {v!r}"
            )
        return upper


class BadmintonPredictResponseData(BaseModel):
    """Badminton prediction output payload."""

    player1: str
    player2: str
    discipline: str
    round: str
    tournament_type: str
    # Probabilities
    p1_win_prob: float = Field(description="P(player1 wins), clipped to [0.02, 0.98]")
    p2_win_prob: float = Field(description="P(player2 wins) = 1 - p1_win_prob")
    raw_prob: float = Field(description="Raw ensemble probability before calibration")
    # Diagnostics
    regime: str = Field(description="Regime used: MS | WS | doubles")
    market_blend_applied: bool = Field(
        description="True if market_prob_player1 was supplied and logit blend was applied"
    )


# ---------------------------------------------------------------------------
# Logit-space market blend helper
# ---------------------------------------------------------------------------


def _logit_blend(p_model: float, p_market: float, weight_model: float = 0.20) -> float:
    """Blend two probabilities in logit space."""
    import math
    eps = 1e-6
    p_model = max(eps, min(1 - eps, p_model))
    p_market = max(eps, min(1 - eps, p_market))
    l_m = math.log(p_model / (1 - p_model))
    l_k = math.log(p_market / (1 - p_market))
    l_b = weight_model * l_m + (1 - weight_model) * l_k
    return 1.0 / (1.0 + math.exp(-l_b))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    summary="Predict badminton match win probabilities (ML ensemble)",
    response_class=ORJSONResponse,
)
async def predict_match(body: BadmintonPredictRequest) -> ORJSONResponse:
    """
    Return win probabilities for an ad-hoc badminton match.

    The regime-appropriate stacking ensemble (R0=MS, R1=WS, R2=doubles)
    with BetaCalibrator is used.  The global BadmintonPredictor singleton
    (loaded at startup) serves predictions.

    HTTP 503 is returned when the predictor is not loaded.
    HTTP 422 is returned when player1 == player2.
    """
    rid = str(uuid.uuid4())
    logger.info(
        "badminton_predict.requested player1=%s player2=%s discipline=%s rid=%s",
        body.player1, body.player2, body.discipline, rid,
    )

    if body.player1.strip().lower() == body.player2.strip().lower():
        return _error(
            "INVALID_INPUT",
            "player1 and player2 must be different players",
            rid,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    # Resolve the module-level predictor singleton from main.py
    try:
        import main as _main_module
        predictor = getattr(_main_module, "_predictor", None)
    except Exception as exc:
        logger.error("badminton_predict.predictor_import_failed: %s", exc)
        predictor = None

    if predictor is None or not predictor.is_ready:
        return _error(
            "MODEL_NOT_LOADED",
            (
                "BadmintonPredictor is not loaded. "
                "Ensure R0/R1/R2 model artefacts exist under models/ and the service "
                "started successfully.  Check /health for predictor_ready status."
            ),
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # ── Run prediction ────────────────────────────────────────────────────────
    try:
        result = predictor.predict(
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
    except RuntimeError as exc:
        logger.warning("badminton_predict.predictor_runtime_error: %s", exc)
        return _error(
            "MODEL_NOT_LOADED",
            str(exc),
            rid,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    except Exception as exc:
        logger.error("badminton_predict.predictor_error: %s", exc, exc_info=True)
        return _error(
            "INTERNAL_ERROR",
            f"Prediction failed: {exc}",
            rid,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    p1_win = float(result["p1_win_prob"])
    p1_win = max(0.02, min(0.98, p1_win))
    raw_prob = float(result.get("raw_prob", p1_win))

    # ── Optional market logit blend ───────────────────────────────────────────
    market_blend_applied = False
    if body.market_prob_player1 is not None:
        try:
            p1_win = _logit_blend(
                p_model=p1_win,
                p_market=body.market_prob_player1,
                weight_model=0.20,
            )
            p1_win = max(0.02, min(0.98, p1_win))
            market_blend_applied = True
            logger.info(
                "badminton_predict.market_blend_applied market=%s blended=%s rid=%s",
                body.market_prob_player1, round(p1_win, 4), rid,
            )
        except Exception as exc:
            logger.warning("badminton_predict.market_blend_failed: %s", exc)

    p1_win_final = round(p1_win, 6)
    p2_win_final = round(1.0 - p1_win, 6)

    response_data = BadmintonPredictResponseData(
        player1=body.player1,
        player2=body.player2,
        discipline=body.discipline,
        round=body.round,
        tournament_type=body.tournament_type,
        p1_win_prob=p1_win_final,
        p2_win_prob=p2_win_final,
        raw_prob=round(raw_prob, 6),
        regime=result.get("regime", body.discipline if body.discipline in ("MS", "WS") else "doubles"),
        market_blend_applied=market_blend_applied,
    )

    logger.info(
        "badminton_predict.complete p1_win=%s regime=%s market_blend=%s rid=%s",
        p1_win_final, response_data.regime, market_blend_applied, rid,
    )
    return ORJSONResponse(content=_ok(response_data.model_dump(), rid))

"""
main.py
=======
XG3 Badminton Microservice — Port 8034
BWF World Tour match prediction and sportsbook pricing — Tier 1.

Startup sequence:
  1. Load BadmintonPredictor (R0/R1/R2 stacking ensembles + calibrators)
  2. Mount pricing routes under /api/v1/badminton/
  3. Expose /health, /health/ready, /health/live endpoints

Environment variables:
  PORT        — HTTP port (default: 8034)
  SERVICE_ENV — "production" | "development" (default: development)
  LOG_LEVEL   — log level (default: INFO)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import ALL_DISCIPLINES, PORT
from ml.predictor import BadmintonPredictor
from pricing.markets import BadmintonPricer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

_SERVICE_ENV = os.environ.get("SERVICE_ENV", "development")
_IS_PRODUCTION = _SERVICE_ENV == "production"

# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #

class PriceMatchRequest(BaseModel):
    player1: str = Field(..., description="Player / team-one name (singles: individual player)")
    player2: str = Field(..., description="Player / team-two name (singles: individual player)")
    discipline: str = Field(..., description="Discipline code: MS | WS | MD | XD | WD")
    round: str = Field(default="Round of 16", description="Tournament round")
    tournament_type: str = Field(
        default="HSBC BWF World Tour Super 300",
        description="BWF tournament tier label",
    )
    partner1: str | None = Field(
        None, description="Partner of player1 — required for doubles disciplines"
    )
    partner2: str | None = Field(
        None, description="Partner of player2 — required for doubles disciplines"
    )
    nationality1: str = Field(default="", description="Nationality/country code for player1 team")
    nationality2: str = Field(default="", description="Nationality/country code for player2 team")
    country: str = Field(default="", description="Host country of the tournament")


class TrainRequest(BaseModel):
    confirm: bool = Field(
        default=False,
        description="Must be true to confirm the training run (runs synchronously, several minutes)",
    )


# --------------------------------------------------------------------------- #
# Global singletons
# --------------------------------------------------------------------------- #

_predictor: BadmintonPredictor | None = None
_pricer: BadmintonPricer = BadmintonPricer()
_startup_time: float = 0.0


# --------------------------------------------------------------------------- #
# App lifespan
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _startup_time
    _startup_time = time.time()
    logger.info("xg3_badminton_starting env=%s port=%d", _SERVICE_ENV, PORT)

    predictor = BadmintonPredictor()
    try:
        predictor.load()
        _predictor = predictor
        logger.info("xg3_badminton_predictor_ready regimes=%s", predictor.health_detail()["regimes_loaded"])
    except Exception as exc:
        logger.error("xg3_badminton_predictor_load_failed: %s", exc)
        _predictor = None

    yield

    uptime = round(time.time() - _startup_time, 1)
    logger.info("xg3_badminton_shutdown uptime_s=%s", uptime)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="XG3 Badminton Microservice",
    version="1.0.0",
    description=(
        "BWF World Tour match prediction and sportsbook pricing — Tier 1. "
        "5 disciplines: MS, WS, MD, XD, WD. "
        "Markets: Match Winner, Set Handicap, Total Games, Correct Score, "
        "Games Handicap, Correct Score, Clean Win, Trading Controls."
    ),
    docs_url="/docs" if not _IS_PRODUCTION else None,
    redoc_url="/redoc" if not _IS_PRODUCTION else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not _IS_PRODUCTION else [os.environ.get("CORS_ORIGIN", "")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register derivatives, trading controls, H2H, and form routers
from api.derivatives import router as _derivatives_router
from api.trading_controls import router as _trading_controls_router
from api.h2h import router as _h2h_router
from api.form import router as _form_router

app.include_router(_derivatives_router)
app.include_router(_trading_controls_router)
app.include_router(_h2h_router)
app.include_router(_form_router)


# --------------------------------------------------------------------------- #
# Health routes
# --------------------------------------------------------------------------- #

@app.get("/health", tags=["health"])
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "badminton",
        "version": "1.0.0",
        "env": _SERVICE_ENV,
        "port": PORT,
        "uptime_s": round(time.time() - _startup_time, 1) if _startup_time else 0,
    }


@app.get("/health/ready", tags=["health"])
async def health_ready() -> dict[str, Any]:
    ready = _predictor is not None and _predictor.is_ready
    detail = _predictor.health_detail() if _predictor else {"is_ready": False}
    if not ready:
        raise HTTPException(status_code=503, detail={"ready": False, **detail})
    return {"ready": True, **detail}


@app.get("/health/live", tags=["health"])
async def health_live() -> dict[str, str]:
    return {"status": "alive"}


# --------------------------------------------------------------------------- #
# Pricing routes
# --------------------------------------------------------------------------- #

@app.post("/api/v1/badminton/matches/price", tags=["pricing"])
async def price_match(req: PriceMatchRequest) -> dict[str, Any]:
    """
    Price all sportsbook markets for a given badminton match.

    Accepts singles (player1 vs player2) and doubles (player1+partner1 vs player2+partner2).
    Discipline must be one of: MS, WS, MD, XD, WD.

    Returns:
      - ML win probabilities (calibrated)
      - Match Winner market (5% margin)
      - Set Handicap market (−1.5 / +1.5 sets)
      - Total Games O/U 2.5
      - Correct Score (2-0, 2-1, 0-2, 1-2)
    """
    discipline = req.discipline.upper()
    if discipline not in ALL_DISCIPLINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown discipline '{discipline}'. Valid options: {ALL_DISCIPLINES}",
        )

    if _predictor is None or not _predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Train first: "
                "cd E:/DF/XG3V10/badminton && "
                "python -c \"from ml.trainer import BadmintonTrainer; BadmintonTrainer().train_all()\""
            ),
        )

    try:
        prediction = _predictor.predict(
            player1=req.player1,
            player2=req.player2,
            discipline=discipline,
            round_str=req.round,
            tournament_type=req.tournament_type,
            partner1=req.partner1,
            partner2=req.partner2,
            nationality1=req.nationality1,
            nationality2=req.nationality2,
            country=req.country,
        )
    except Exception as exc:
        logger.error("prediction_failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    priced = _pricer.price_match(
        player1=req.player1,
        player2=req.player2,
        p1_win_prob=prediction["p1_win_prob"],
        p2_win_prob=prediction["p2_win_prob"],
    )

    return {
        "success": True,
        "discipline": discipline,
        "round": req.round,
        "tournament_type": req.tournament_type,
        "model_regime": prediction["regime"],
        "raw_prob": prediction["raw_prob"],
        **priced,
    }


# --------------------------------------------------------------------------- #
# Admin routes
# --------------------------------------------------------------------------- #

@app.get("/api/v1/badminton/admin/elo-ratings", tags=["admin"])
async def get_elo_ratings(discipline: str = "MS") -> dict[str, Any]:
    """Return all ELO ratings for a given discipline (MS/WS/MD/XD/WD)."""
    discipline = discipline.upper()
    if discipline not in ALL_DISCIPLINES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown discipline '{discipline}'. Valid: {ALL_DISCIPLINES}",
        )
    if _predictor is None or not _predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ratings = _predictor.get_elo_ratings(discipline)
    sorted_ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)

    return {
        "discipline": discipline,
        "count": len(ratings),
        "top_20": [{"player": p, "elo": round(e, 1)} for p, e in sorted_ratings[:20]],
        "all_ratings": {p: round(e, 1) for p, e in ratings.items()},
    }


@app.post("/api/v1/badminton/admin/train", tags=["admin"])
async def trigger_training(req: TrainRequest) -> dict[str, Any]:
    """
    Trigger full model training for all disciplines.
    confirm=true required. Runs synchronously — takes several minutes.
    """
    if not req.confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true to trigger training. This runs synchronously and takes minutes.",
        )
    try:
        from ml.trainer import BadmintonTrainer
        trainer = BadmintonTrainer()
        trainer.train_all()

        global _predictor
        new_predictor = BadmintonPredictor()
        new_predictor.load()
        _predictor = new_predictor

        return {"success": True, "message": "Training complete. Models reloaded."}
    except Exception as exc:
        logger.error("training_failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")


@app.get("/api/v1/badminton/admin/status", tags=["admin"])
async def admin_status() -> dict[str, Any]:
    """Return detailed service status and predictor health."""
    predictor_detail = _predictor.health_detail() if _predictor else {"is_ready": False}
    return {
        "service": "badminton",
        "version": "1.0.0",
        "port": PORT,
        "disciplines": list(ALL_DISCIPLINES),
        "uptime_s": round(time.time() - _startup_time, 1) if _startup_time else 0,
        "predictor": predictor_detail,
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    port = int(os.environ.get("PORT", PORT))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=(_SERVICE_ENV == "development"),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )

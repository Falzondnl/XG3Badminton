"""
elo_startup_seeder.py
=====================
Load BWF ELO seed ratings into BadmintonFeatureExtractor instances at startup.

Architecture note
-----------------
The active prediction path uses BadmintonFeatureExtractor._elo, a plain
dict[discipline_code][player_key] -> float.  This is populated at training
time from real match history and persisted inside extractor.pkl.

When the extractor.pkl predates this seeder (or was trained on limited data),
unknown players fall back to ELO_DEFAULT = 1500.0, causing equal-odds pricing
for all unrecognised players.

This seeder injects real BWF-ranked ELO values directly into the loaded
extractor instances so that any recognisable player name resolves to a real
rating immediately — without requiring a full model retrain.

Injection is non-destructive: existing entries (learned from match history) are
NEVER overwritten.  The seed provides a floor for players the model hasn't seen.

Regime layout (mirrors predictor.py):
  "MS"      → extractor covers discipline "MS"
  "WS"      → extractor covers discipline "WS"
  "doubles" → extractor covers disciplines "MD", "XD", "WD"

Doubles pair key format  (matches features.py _pair_key()):
  "{PlayerA}|{PlayerB}"  where names are sorted alphabetically.

Usage
-----
Called once from main.py lifespan after predictor.load():

    from ml.elo_startup_seeder import seed_elo_into_predictor
    n = seed_elo_into_predictor(predictor)
    logger.info("badminton_elo_seeded", n_injected=n)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml.predictor import BadmintonPredictor

logger = logging.getLogger(__name__)

_SEED_PATH = Path(__file__).parent.parent / "data" / "elo_seed_badminton.json"

# Map from seed JSON pool name → (extractor_regime_key, discipline_code)
# "doubles" regime covers MD, WD, XD — each has its own discipline key inside
# the extractor's _elo dict.
_POOL_TO_EXTRACTOR: dict[str, tuple[str, str]] = {
    "MS": ("MS", "MS"),
    "WS": ("WS", "WS"),
    "MD": ("doubles", "MD"),
    "WD": ("doubles", "WD"),
    "XD": ("doubles", "XD"),
}


def seed_elo_into_predictor(predictor: "BadmintonPredictor") -> int:
    """
    Inject BWF seed ratings into all loaded BadmintonFeatureExtractor instances.

    Only injects where the player/pair key is NOT already present in the
    extractor's _elo dict (training-derived values take precedence).

    Returns the total count of new entries injected across all pools.

    Raises:
        RuntimeError if seed file is missing.
    """
    if not _SEED_PATH.exists():
        raise RuntimeError(
            f"ELO seed file missing: {_SEED_PATH}. "
            "Cannot seed BWF ratings — all unknown players will use ELO_DEFAULT=1500."
        )

    data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    pools: dict[str, dict[str, float]] = data.get("pools", {})

    total_injected = 0
    per_pool: dict[str, int] = {}

    for pool_name, entities in pools.items():
        mapping = _POOL_TO_EXTRACTOR.get(pool_name)
        if mapping is None:
            logger.warning("elo_seed.unknown_pool pool=%s — skipping", pool_name)
            continue

        regime_key, disc_code = mapping
        extractor = predictor._extractors.get(regime_key)

        if extractor is None:
            # Extractor not loaded (model pkl absent) — log and continue
            logger.warning(
                "elo_seed.extractor_missing regime=%s pool=%s — pkl not loaded, skipping",
                regime_key,
                pool_name,
            )
            per_pool[pool_name] = 0
            continue

        disc_elo: dict[str, float] = extractor._elo[disc_code]
        injected = 0

        for entity_name, rating in entities.items():
            entity_name = entity_name.strip()
            if not entity_name:
                continue
            rating = float(rating)

            if entity_name not in disc_elo:
                disc_elo[entity_name] = rating
                injected += 1
            # If already present (training-derived), leave it — no overwrite.

        per_pool[pool_name] = injected
        total_injected += injected
        logger.info(
            "elo_seed.pool_done pool=%s regime=%s disc=%s injected=%d existing_preserved=%d",
            pool_name,
            regime_key,
            disc_code,
            injected,
            len(entities) - injected,
        )

    logger.info(
        "elo_seed.complete total_injected=%d per_pool=%s source=%s",
        total_injected,
        per_pool,
        str(_SEED_PATH),
    )
    return total_injected


def get_seeded_rating(discipline: str, player_key: str) -> float | None:
    """
    Look up a player/pair's seed rating from the JSON file directly.

    Used by outright_pricing callers to populate TournamentEntry.elo_rating
    without needing access to the predictor singleton.

    Returns None if the seed file is absent or the player is unknown.
    Callers should fall back to 1500.0 explicitly and log a warning.
    """
    if not _SEED_PATH.exists():
        return None

    try:
        data = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
        pools: dict[str, dict[str, float]] = data.get("pools", {})

        disc_upper = discipline.upper()
        pool = pools.get(disc_upper, {})

        # Direct lookup
        rating = pool.get(player_key.strip())
        if rating is not None:
            return float(rating)

        # For doubles: try reversed order key
        if "|" in player_key:
            parts = player_key.split("|", 1)
            reversed_key = f"{parts[1].strip()}|{parts[0].strip()}"
            rating = pool.get(reversed_key)
            if rating is not None:
                return float(rating)

        return None
    except Exception as exc:
        logger.warning("elo_seed.lookup_failed discipline=%s key=%s error=%s", discipline, player_key, exc)
        return None

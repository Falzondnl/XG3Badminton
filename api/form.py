"""
Badminton Player Form — POST /api/v1/badminton/form

Recent form lookup for a single player.

Data discovery order:
  1. env var BADMINTON_DATA_DIR (any .csv / .jsonl inside)
  2. <project_root>/data/*.csv or *.jsonl
  3. <project_root>/matches.csv

If no historical match data is found the endpoint returns an honest
empty response with data_source="no_historical_data".

No fake win/loss counts are ever returned.  Zeros and null only when no data.

Expected CSV columns (when data is present):
  player1, player2, winner, date, score

  where `winner` is the value of either `player1` or `player2` for that row.
"""
from __future__ import annotations

import csv
import json
import os
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/badminton", tags=["badminton-form"])

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class FormRequest(BaseModel):
    player_name: str = Field(..., description="Player name to look up")
    last_n: int = Field(default=10, ge=1, le=100, description="Number of most-recent matches to include")


# ---------------------------------------------------------------------------
# Data discovery (mirrors h2h.py discovery logic — same project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_data_files() -> List[Path]:
    """Return candidate data file paths in priority order."""
    candidates: List[Path] = []

    env_dir = os.environ.get("BADMINTON_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.is_dir():
            candidates.extend(sorted(env_path.glob("*.csv")))
            candidates.extend(sorted(env_path.glob("*.jsonl")))

    data_dir = _PROJECT_ROOT / "data"
    if data_dir.is_dir():
        candidates.extend(sorted(data_dir.glob("*.csv")))
        candidates.extend(sorted(data_dir.glob("*.jsonl")))

    candidates.append(_PROJECT_ROOT / "matches.csv")
    return candidates


def _first_existing_data_file() -> Optional[Path]:
    for path in _candidate_data_files():
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_bytes = nfkd.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii").lower().strip()


def _name_matches(row_name: str, query: str) -> bool:
    return _normalise(row_name) == _normalise(query)


# ---------------------------------------------------------------------------
# Form string helper
# ---------------------------------------------------------------------------

def _build_form_string(results: List[str]) -> str:
    """
    Convert an ordered list of 'W' / 'L' strings (most-recent first) to a
    compact form string such as 'WWLWL'.
    """
    return "".join(r[0].upper() for r in results if r and r[0].upper() in ("W", "L"))


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_csv_form(path: Path, player_name: str, last_n: int) -> Dict[str, Any]:
    """
    Scan a CSV for all matches involving `player_name` and return form data.

    Expected columns: player1, player2, winner, date, score
    """
    raw_matches: List[Dict[str, Any]] = []

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        if reader.fieldnames is None:
            return _no_data_response(player_name)

        fieldmap: Dict[str, str] = {
            f.strip().lower(): f for f in reader.fieldnames if f
        }

        col_p1 = fieldmap.get("player1")
        col_p2 = fieldmap.get("player2")
        col_winner = fieldmap.get("winner")

        if not (col_p1 and col_p2 and col_winner):
            logger.warning(
                "form_csv_missing_required_columns",
                path=str(path),
                found=list(fieldmap.keys()),
            )
            return _no_data_response(player_name)

        col_date = fieldmap.get("date")
        col_score = fieldmap.get("score")

        for row in reader:
            row_p1 = row.get(col_p1, "").strip()
            row_p2 = row.get(col_p2, "").strip()

            player_is_p1 = _name_matches(row_p1, player_name)
            player_is_p2 = _name_matches(row_p2, player_name)

            if not (player_is_p1 or player_is_p2):
                continue

            winner_raw = row.get(col_winner, "").strip()
            if player_is_p1:
                won = _name_matches(winner_raw, row_p1)
                opponent = row_p2
            else:
                won = _name_matches(winner_raw, row_p2)
                opponent = row_p1

            if not _name_matches(winner_raw, row_p1) and not _name_matches(winner_raw, row_p2):
                # Winner value doesn't match either player in this row — skip
                logger.warning(
                    "form_unrecognised_winner",
                    winner=winner_raw,
                    p1=row_p1,
                    p2=row_p2,
                )
                continue

            entry: Dict[str, Any] = {
                "result": "W" if won else "L",
                "opponent": opponent,
            }
            if col_date:
                entry["date"] = row.get(col_date, "")
            if col_score:
                entry["score"] = row.get(col_score, "")

            raw_matches.append(entry)

    # Sort descending by date (most recent first)
    if raw_matches and "date" in raw_matches[0]:
        raw_matches.sort(key=lambda m: m.get("date", ""), reverse=True)

    recent = raw_matches[:last_n]
    wins = sum(1 for m in recent if m["result"] == "W")
    losses = sum(1 for m in recent if m["result"] == "L")
    total = wins + losses
    win_rate = round(wins / total, 4) if total > 0 else None
    form_string = _build_form_string([m["result"] for m in recent])

    return {
        "player_name": player_name,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "form_string": form_string,
        "last_n_requested": last_n,
        "matches_found": len(raw_matches),
        "matches": recent,
        "data_source": str(path),
        "message": f"{len(raw_matches)} matches found for {player_name} in {path.name}",
    }


# ---------------------------------------------------------------------------
# JSONL parser
# ---------------------------------------------------------------------------

def _parse_jsonl_form(path: Path, player_name: str, last_n: int) -> Dict[str, Any]:
    """
    Scan a JSONL file for all matches involving `player_name`.

    Expected keys: player1, player2, winner, date (opt), score (opt)
    """
    raw_matches: List[Dict[str, Any]] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("form_jsonl_parse_error", path=str(path), line=lineno)
                continue

            row_p1 = str(record.get("player1", "")).strip()
            row_p2 = str(record.get("player2", "")).strip()

            player_is_p1 = _name_matches(row_p1, player_name)
            player_is_p2 = _name_matches(row_p2, player_name)

            if not (player_is_p1 or player_is_p2):
                continue

            winner_raw = str(record.get("winner", "")).strip()
            if player_is_p1:
                won = _name_matches(winner_raw, row_p1)
                opponent = row_p2
            else:
                won = _name_matches(winner_raw, row_p2)
                opponent = row_p1

            if not _name_matches(winner_raw, row_p1) and not _name_matches(winner_raw, row_p2):
                logger.warning(
                    "form_unrecognised_winner",
                    winner=winner_raw,
                    p1=row_p1,
                    p2=row_p2,
                )
                continue

            entry: Dict[str, Any] = {
                "result": "W" if won else "L",
                "opponent": opponent,
            }
            for optional_key in ("date", "score", "tournament", "round"):
                if optional_key in record:
                    entry[optional_key] = record[optional_key]

            raw_matches.append(entry)

    if raw_matches and "date" in raw_matches[0]:
        raw_matches.sort(key=lambda m: m.get("date", ""), reverse=True)

    recent = raw_matches[:last_n]
    wins = sum(1 for m in recent if m["result"] == "W")
    losses = sum(1 for m in recent if m["result"] == "L")
    total = wins + losses
    win_rate = round(wins / total, 4) if total > 0 else None
    form_string = _build_form_string([m["result"] for m in recent])

    return {
        "player_name": player_name,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "form_string": form_string,
        "last_n_requested": last_n,
        "matches_found": len(raw_matches),
        "matches": recent,
        "data_source": str(path),
        "message": f"{len(raw_matches)} matches found for {player_name} in {path.name}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_data_response(player_name: str) -> Dict[str, Any]:
    return {
        "player_name": player_name,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "form_string": "",
        "last_n_requested": None,
        "matches_found": 0,
        "matches": [],
        "data_source": "no_historical_data",
        "message": "No historical match data available for badminton",
    }


def _meta(rid: str) -> Dict[str, str]:
    return {
        "request_id": rid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/form", summary="Badminton recent form for a single player")
async def get_form(req: FormRequest) -> Dict[str, Any]:
    """
    Return recent form (win/loss record and form string) for a badminton player.

    Data is sourced from local match history files (CSV or JSONL).
    If no data file is found, an honest empty response is returned —
    no fake statistics are ever fabricated.

    The `form_string` field contains up to `last_n` characters of 'W' or 'L'
    in chronological order, most-recent first (e.g. "WWLWL").

    Discovery order:
      1. BADMINTON_DATA_DIR env var (if set and directory exists)
      2. <project_root>/data/*.csv and *.jsonl
      3. <project_root>/matches.csv
    """
    rid = str(uuid.uuid4())
    player_name = req.player_name.strip()

    data_file = _first_existing_data_file()

    if data_file is None:
        logger.info(
            "form_no_data_file",
            player_name=player_name,
            searched=str(_PROJECT_ROOT),
        )
        payload = _no_data_response(player_name)
        payload["last_n_requested"] = req.last_n
        return {"success": True, "data": payload, "meta": _meta(rid)}

    logger.info(
        "form_data_file_found",
        path=str(data_file),
        player_name=player_name,
    )

    suffix = data_file.suffix.lower()
    try:
        if suffix == ".csv":
            payload = _parse_csv_form(data_file, player_name, req.last_n)
        elif suffix == ".jsonl":
            payload = _parse_jsonl_form(data_file, player_name, req.last_n)
        else:
            logger.warning("form_unsupported_file_format", path=str(data_file))
            payload = _no_data_response(player_name)
            payload["last_n_requested"] = req.last_n
    except Exception as exc:
        logger.error("form_parse_error", path=str(data_file), error=str(exc), exc_info=True)
        payload = _no_data_response(player_name)
        payload["last_n_requested"] = req.last_n

    return {"success": True, "data": payload, "meta": _meta(rid)}


@router.get("/form/health", summary="Form service health check")
async def form_health() -> Dict[str, Any]:
    data_file = _first_existing_data_file()
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "service": "badminton-form",
            "data_available": data_file is not None,
            "data_file": str(data_file) if data_file else None,
            "project_root": str(_PROJECT_ROOT),
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

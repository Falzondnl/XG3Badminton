"""
Badminton H2H — POST /api/v1/badminton/h2h

Head-to-head record lookup between two players.

Data discovery order:
  1. env var BADMINTON_DATA_DIR (any .csv / .jsonl inside)
  2. <project_root>/data/*.csv or *.jsonl
  3. <project_root>/matches.csv

If no historical match data is found, the endpoint returns an honest
zero-count response with data_source="no_historical_data".

No fake win/loss counts are ever returned.  Zeros only when no data.

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

router = APIRouter(prefix="/api/v1/badminton", tags=["badminton-h2h"])

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class H2HRequest(BaseModel):
    player1: str = Field(..., description="First player name")
    player2: str = Field(..., description="Second player name")
    max_results: int = Field(default=20, ge=1, le=200, description="Maximum number of matches to return")


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _candidate_data_files() -> List[Path]:
    """
    Return a list of candidate data file paths to check, in priority order.
    None of these are guaranteed to exist.
    """
    candidates: List[Path] = []

    # 1. Env var override directory
    env_dir = os.environ.get("BADMINTON_DATA_DIR")
    if env_dir:
        env_path = Path(env_dir)
        if env_path.is_dir():
            candidates.extend(sorted(env_path.glob("*.csv")))
            candidates.extend(sorted(env_path.glob("*.jsonl")))

    # 2. <project_root>/data/ directory
    data_dir = _PROJECT_ROOT / "data"
    if data_dir.is_dir():
        candidates.extend(sorted(data_dir.glob("*.csv")))
        candidates.extend(sorted(data_dir.glob("*.jsonl")))

    # 3. <project_root>/matches.csv
    candidates.append(_PROJECT_ROOT / "matches.csv")

    return candidates


def _first_existing_data_file() -> Optional[Path]:
    """Return the first candidate data file that actually exists on disk."""
    for path in _candidate_data_files():
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


# ---------------------------------------------------------------------------
# Name normalisation (simple fold for accent-insensitive matching)
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """Fold to ASCII-lower for loose name comparison."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_bytes = nfkd.encode("ascii", errors="ignore")
    return ascii_bytes.decode("ascii").lower().strip()


def _names_match(a: str, b: str) -> bool:
    return _normalise(a) == _normalise(b)


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_csv_h2h(
    path: Path,
    player1: str,
    player2: str,
    max_results: int,
) -> Dict[str, Any]:
    """
    Parse a CSV file and extract H2H records between player1 and player2.

    Expected columns: player1, player2, winner, date, score
    Column names are matched case-insensitively.
    """
    p1_wins = 0
    p2_wins = 0
    matches: List[Dict[str, Any]] = []

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)

        if reader.fieldnames is None:
            return _no_data_response(player1, player2)

        # Normalise fieldnames for lookup
        fieldmap: Dict[str, str] = {
            f.strip().lower(): f for f in reader.fieldnames if f
        }

        col_p1 = fieldmap.get("player1")
        col_p2 = fieldmap.get("player2")
        col_winner = fieldmap.get("winner")

        if not (col_p1 and col_p2 and col_winner):
            logger.warning(
                "h2h_csv_missing_required_columns",
                path=str(path),
                found=list(fieldmap.keys()),
            )
            return _no_data_response(player1, player2)

        col_date = fieldmap.get("date")
        col_score = fieldmap.get("score")

        for row in reader:
            row_p1 = row.get(col_p1, "").strip()
            row_p2 = row.get(col_p2, "").strip()

            is_h2h = (
                _names_match(row_p1, player1) and _names_match(row_p2, player2)
            ) or (
                _names_match(row_p1, player2) and _names_match(row_p2, player1)
            )
            if not is_h2h:
                continue

            winner_raw = row.get(col_winner, "").strip()
            if _names_match(winner_raw, player1):
                p1_wins += 1
                winner_label = player1
            elif _names_match(winner_raw, player2):
                p2_wins += 1
                winner_label = player2
            else:
                # Winner field value doesn't match either player name — skip
                logger.warning(
                    "h2h_unrecognised_winner",
                    winner=winner_raw,
                    p1=player1,
                    p2=player2,
                )
                continue

            match_entry: Dict[str, Any] = {
                "player1": row_p1,
                "player2": row_p2,
                "winner": winner_label,
            }
            if col_date:
                match_entry["date"] = row.get(col_date, "")
            if col_score:
                match_entry["score"] = row.get(col_score, "")

            matches.append(match_entry)

    # Sort descending by date if available
    if matches and "date" in matches[0]:
        matches.sort(key=lambda m: m.get("date", ""), reverse=True)

    total = p1_wins + p2_wins
    matches = matches[:max_results]

    return {
        "player1": player1,
        "player2": player2,
        "player1_wins": p1_wins,
        "player2_wins": p2_wins,
        "total_matches": total,
        "matches": matches,
        "data_source": str(path),
        "message": f"{total} H2H matches found in {path.name}",
    }


# ---------------------------------------------------------------------------
# JSONL parser
# ---------------------------------------------------------------------------

def _parse_jsonl_h2h(
    path: Path,
    player1: str,
    player2: str,
    max_results: int,
) -> Dict[str, Any]:
    """
    Parse a JSONL file (one JSON object per line) for H2H records.

    Expected keys per record: player1, player2, winner, date (opt), score (opt)
    """
    p1_wins = 0
    p2_wins = 0
    matches: List[Dict[str, Any]] = []

    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("h2h_jsonl_parse_error", path=str(path), line=lineno)
                continue

            row_p1 = str(record.get("player1", "")).strip()
            row_p2 = str(record.get("player2", "")).strip()

            is_h2h = (
                _names_match(row_p1, player1) and _names_match(row_p2, player2)
            ) or (
                _names_match(row_p1, player2) and _names_match(row_p2, player1)
            )
            if not is_h2h:
                continue

            winner_raw = str(record.get("winner", "")).strip()
            if _names_match(winner_raw, player1):
                p1_wins += 1
                winner_label = player1
            elif _names_match(winner_raw, player2):
                p2_wins += 1
                winner_label = player2
            else:
                logger.warning(
                    "h2h_unrecognised_winner",
                    winner=winner_raw,
                    p1=player1,
                    p2=player2,
                )
                continue

            match_entry: Dict[str, Any] = {
                "player1": row_p1,
                "player2": row_p2,
                "winner": winner_label,
            }
            for optional_key in ("date", "score", "tournament", "round"):
                if optional_key in record:
                    match_entry[optional_key] = record[optional_key]

            matches.append(match_entry)

    if matches and "date" in matches[0]:
        matches.sort(key=lambda m: m.get("date", ""), reverse=True)

    total = p1_wins + p2_wins
    matches = matches[:max_results]

    return {
        "player1": player1,
        "player2": player2,
        "player1_wins": p1_wins,
        "player2_wins": p2_wins,
        "total_matches": total,
        "matches": matches,
        "data_source": str(path),
        "message": f"{total} H2H matches found in {path.name}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _no_data_response(player1: str, player2: str) -> Dict[str, Any]:
    return {
        "player1": player1,
        "player2": player2,
        "player1_wins": 0,
        "player2_wins": 0,
        "total_matches": 0,
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

@router.post("/h2h", summary="Badminton head-to-head record between two players")
async def get_h2h(req: H2HRequest) -> Dict[str, Any]:
    """
    Return the head-to-head record between two badminton players.

    Data is sourced from local match history files (CSV or JSONL).
    If no data file is found, an honest zero-count response is returned —
    no fake statistics are ever fabricated.

    Discovery order:
      1. BADMINTON_DATA_DIR env var (if set and directory exists)
      2. <project_root>/data/*.csv and *.jsonl
      3. <project_root>/matches.csv
    """
    rid = str(uuid.uuid4())
    player1 = req.player1.strip()
    player2 = req.player2.strip()

    data_file = _first_existing_data_file()

    if data_file is None:
        logger.info(
            "h2h_no_data_file",
            player1=player1,
            player2=player2,
            searched=str(_PROJECT_ROOT),
        )
        payload = _no_data_response(player1, player2)
        return {"success": True, "data": payload, "meta": _meta(rid)}

    logger.info(
        "h2h_data_file_found",
        path=str(data_file),
        player1=player1,
        player2=player2,
    )

    suffix = data_file.suffix.lower()
    try:
        if suffix == ".csv":
            payload = _parse_csv_h2h(data_file, player1, player2, req.max_results)
        elif suffix == ".jsonl":
            payload = _parse_jsonl_h2h(data_file, player1, player2, req.max_results)
        else:
            logger.warning("h2h_unsupported_file_format", path=str(data_file))
            payload = _no_data_response(player1, player2)
    except Exception as exc:
        logger.error("h2h_parse_error", path=str(data_file), error=str(exc), exc_info=True)
        payload = _no_data_response(player1, player2)

    return {"success": True, "data": payload, "meta": _meta(rid)}


@router.get("/h2h/health", summary="H2H service health check")
async def h2h_health() -> Dict[str, Any]:
    data_file = _first_existing_data_file()
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "service": "badminton-h2h",
            "data_available": data_file is not None,
            "data_file": str(data_file) if data_file else None,
            "project_root": str(_PROJECT_ROOT),
        },
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

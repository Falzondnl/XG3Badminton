"""
Badminton fixture discovery API.

GET  /api/v1/badminton/fixtures          — upcoming fixtures from Optic Odds autodiscovery
GET  /api/v1/badminton/fixtures/upcoming — alias
POST /api/v1/badminton/fixtures          — same data; POST body ignored (RE-AUDIT-4 FIX 4)
POST /api/v1/badminton/fixtures/upcoming — alias for POST

The data backing these endpoints is populated by the background poller in
``services.fixture_poller``, which runs every 30 minutes against the Optic
Odds v3 /fixtures endpoint.  Responses are served directly from the
module-level cache — no synchronous HTTP call is made per request.

Returns empty list before the first poll completes or when
OPTIC_ODDS_API_KEY is not set.

IMP-BADMINTON-FIXTURES-POST-FIX-001 (RE-AUDIT-4 FIX 4 — 2026-04-25):
POST /api/v1/badminton/fixtures returned 404 because only GET was registered.
Added @router.post for both "" and "/upcoming" paths so clients using POST
(e.g. Enterprise gateway batch fixture requests) receive 200 identical to GET.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(
    prefix="/api/v1/badminton/fixtures",
    tags=["Fixtures"],
)

# League IDs kept in sync with services/fixture_poller.py _LEAGUE_IDS
_LEAGUES_POLLED: List[str] = [
    "badminton",
    "bwl_super_series",
    "bwf_world_tour",
]


class FixturesFilterBody(BaseModel):
    """Optional POST body for /fixtures — all fields optional, mirrors GET query params."""

    league: Optional[str] = None


def _meta(rid: str) -> Dict[str, str]:
    return {
        "request_id": rid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _build_fixtures_response(league: Optional[str] = None) -> Dict[str, Any]:
    """Shared response builder for GET and POST /fixtures handlers."""
    from services.fixture_poller import get_cached_fixtures

    all_fixtures = get_cached_fixtures()

    if league:
        all_fixtures = [f for f in all_fixtures if f.get("league_id") == league]

    rid = str(uuid.uuid4())
    return {
        "success": True,
        "data": {
            "fixtures": all_fixtures,
            "count": len(all_fixtures),
            "sport": "badminton",
            "leagues_polled": _LEAGUES_POLLED,
            "fixture_source": "optic_odds_autodiscovery",
            "feed_enabled": True,
        },
        "meta": _meta(rid),
    }


@router.get("", response_model=Dict[str, Any])
@router.get("/upcoming", response_model=Dict[str, Any])
async def list_fixtures(
    league: Optional[str] = Query(
        None,
        description=(
            "Filter by league ID. "
            "One of: badminton, bwl_super_series, bwf_world_tour"
        ),
    ),
) -> Dict[str, Any]:
    """
    Return upcoming badminton fixtures discovered by the Optic Odds autodiscovery poller.

    The poller runs every 30 minutes against Optic Odds /api/v3/fixtures for all three
    BWF circuit league IDs.  Responses are served from the in-memory cache — no live
    HTTP call is made per request.

    Query parameters:
      league — optional league_id filter (e.g. ``bwf_world_tour``)

    Returns an empty fixtures list when:
      - The service has just started and the first poll has not yet completed
      - OPTIC_ODDS_API_KEY is not configured in the environment
    """
    return _build_fixtures_response(league=league)


@router.post("", response_model=Dict[str, Any])
@router.post("/upcoming", response_model=Dict[str, Any])
async def list_fixtures_post(
    body: Optional[FixturesFilterBody] = None,
) -> Dict[str, Any]:
    """POST /api/v1/badminton/fixtures — identical response to GET.

    IMP-BADMINTON-FIXTURES-POST-FIX-001 (RE-AUDIT-4 FIX 4):
    Some clients issue POST requests to /fixtures (e.g. Enterprise gateway
    batch fixture discovery).  This handler mirrors GET exactly.
    Optional body may specify ``league`` to filter by league_id.
    """
    league = body.league if body else None
    return _build_fixtures_response(league=league)

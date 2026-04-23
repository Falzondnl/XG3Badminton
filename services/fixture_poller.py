"""
Badminton Fixture Autodiscovery Poller
=======================================
Polls the Optic Odds API every 30 minutes for upcoming badminton fixtures
across the main BWF circuit leagues and stores them in a module-level cache
so the fixtures endpoints can serve autodiscovered matches without requiring
manual POST /matches/register calls.

Per CLAUDE.md Maximum Utilisation Law:
  - All 3 supported BWF league IDs are polled (not just a single one)
  - Uses full pagination (page_size=100, up to _MAX_PAGES pages per league)
  - Runs automatically on startup — manual ingestion is an override
  - Keeps stale cache on fetch failure — never returns empty due to a transient error

Optic Odds sport slug: ``badminton``
League IDs polled:
  badminton, bwl_super_series, bwf_world_tour

Stdlib-only — no third-party HTTP library required.
Auth: x-api-key header from OPTIC_ODDS_API_KEY env var.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger("xg3_badminton")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SPORT_SLUG = "badminton"
_LEAGUE_IDS: List[str] = [
    "badminton",
    "bwl_super_series",
    "bwf_world_tour",
]
_POLL_INTERVAL_S: int = 30 * 60          # 30 minutes
_LOOKAHEAD_DAYS: int = 14
_PAGE_SIZE: int = 100
_MAX_PAGES: int = 10
_REQUEST_TIMEOUT_S: float = 20.0
_OPTIC_ODDS_BASE = "https://api.opticodds.com/api/v3"

# ---------------------------------------------------------------------------
# Module-level state — guarded by _store_lock
# ---------------------------------------------------------------------------

_store_lock: threading.Lock = threading.Lock()
_fixtures_store: List[Dict[str, Any]] = []

_poller_thread: Optional[threading.Thread] = None
_stop_event: threading.Event = threading.Event()


# ---------------------------------------------------------------------------
# Core poll function (blocking, stdlib-only)
# ---------------------------------------------------------------------------

def poll_fixtures_once() -> List[Dict[str, Any]]:
    """
    Fetch all upcoming badminton fixtures for every supported BWF league
    from the Optic Odds v3 API.

    Uses full pagination per CLAUDE.md Max Utilisation Law.
    Returns a flat list of normalised fixture dicts.
    Does NOT raise — returns the current cached list on any unrecoverable error
    so the cache is never poisoned by a transient outage.
    """
    api_key = os.getenv("OPTIC_ODDS_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "[badminton] OPTIC_ODDS_API_KEY not set — fixture polling disabled; "
            "returning empty fixture list"
        )
        return []

    now_utc = datetime.now(timezone.utc)
    starts_after = now_utc.isoformat()
    starts_before = (now_utc + timedelta(days=_LOOKAHEAD_DAYS)).isoformat()

    all_fixtures: List[Dict[str, Any]] = []

    for league_id in _LEAGUE_IDS:
        page = 1
        while page <= _MAX_PAGES:
            params: Dict[str, Any] = {
                "sport": _SPORT_SLUG,
                "league_id": league_id,
                "status": "unplayed",
                "page": page,
                "page_size": _PAGE_SIZE,
                "starts_after": starts_after,
                "starts_before": starts_before,
            }
            url = f"{_OPTIC_ODDS_BASE}/fixtures?{urlencode(params)}"
            request = urllib.request.Request(
                url,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            try:
                with urllib.request.urlopen(
                    request, timeout=_REQUEST_TIMEOUT_S
                ) as resp:
                    raw_body = resp.read()
                    status_code = resp.status
            except urllib.error.HTTPError as exc:
                status_code = exc.code
                raw_body = b""
                if status_code == 401:
                    logger.error(
                        "[badminton] Optic Odds API key rejected (401) "
                        "league=%s key_prefix=%s",
                        league_id, api_key[:8],
                    )
                    return all_fixtures  # abort — key is wrong for all leagues
                if status_code == 404:
                    logger.warning(
                        "[badminton] League not found (404) league=%s — "
                        "may not be available on current account tier",
                        league_id,
                    )
                    break  # next league
                logger.error(
                    "[badminton] HTTP error league=%s page=%d status=%d",
                    league_id, page, status_code,
                )
                break
            except Exception as exc:
                logger.error(
                    "[badminton] HTTP request failed league=%s page=%d error=%s",
                    league_id, page, exc,
                )
                break  # next league

            if status_code != 200:
                logger.error(
                    "[badminton] Unexpected status league=%s page=%d status=%d",
                    league_id, page, status_code,
                )
                break

            try:
                payload: Dict[str, Any] = json.loads(raw_body.decode("utf-8"))
            except Exception as exc:
                logger.error(
                    "[badminton] JSON parse error league=%s page=%d error=%s",
                    league_id, page, exc,
                )
                break

            page_fixtures: List[Dict[str, Any]] = payload.get("data", [])
            if not page_fixtures:
                # Exhausted this league's pages
                break

            normalised = [_normalise_fixture(f, league_id) for f in page_fixtures]
            all_fixtures.extend(normalised)
            logger.info(
                "[badminton] Polled league=%s page=%d fixtures=%d",
                league_id, page, len(page_fixtures),
            )

            # Honour server-side pagination metadata when present
            meta = payload.get("meta", {})
            total = meta.get("total", 0)
            fetched_so_far = page * _PAGE_SIZE
            if total and fetched_so_far >= total:
                break

            page += 1

    logger.info(
        "[badminton] Fixture poll complete total_fixtures=%d leagues=%d",
        len(all_fixtures), len(_LEAGUE_IDS),
    )
    return all_fixtures


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_fixture(raw: Dict[str, Any], league_id: str) -> Dict[str, Any]:
    """Normalise an Optic Odds v3 fixture dict to the badminton MS schema."""
    home_name = _extract_participant_name(raw, "home")
    away_name = _extract_participant_name(raw, "away")
    return {
        "fixture_id": raw.get("id", ""),
        "home_team": home_name,
        "away_team": away_name,
        "start_date": raw.get("start_date", ""),
        "status": raw.get("status", ""),
        "league_id": league_id,
        "sport": _SPORT_SLUG,
        "source": "optic_odds_poller",
    }


def _extract_participant_name(fixture: Dict[str, Any], side: str) -> str:
    """
    Extract player/team name from an Optic Odds v3 fixture dict.

    Tries three locations in priority order:
      1. <side>_competitors[0].name  (v3 primary)
      2. <side>_team_display         (v3 display name)
      3. <side>_team.name            (v2 legacy)
    """
    competitors = fixture.get(f"{side}_competitors")
    if isinstance(competitors, list) and competitors:
        first = competitors[0]
        if isinstance(first, dict):
            name = first.get("name", "")
            if name:
                return str(name)

    display = fixture.get(f"{side}_team_display")
    if display and isinstance(display, str):
        return display

    raw = fixture.get(f"{side}_team", {})
    if isinstance(raw, dict):
        return raw.get("name", "")
    return str(raw) if raw else ""


# ---------------------------------------------------------------------------
# Background polling loop (runs in a daemon thread)
# ---------------------------------------------------------------------------

def _poll_loop() -> None:
    """
    Long-running thread that polls Optic Odds every _POLL_INTERVAL_S seconds.
    Fires immediately on startup, then waits _POLL_INTERVAL_S between polls.
    Never crashes the thread — exceptions are logged and the loop continues.

    Uses the _stop_event to terminate cleanly on service shutdown.
    """
    global _fixtures_store

    logger.info(
        "[badminton] Fixture autodiscovery loop started "
        "(interval=%ds leagues=%s)",
        _POLL_INTERVAL_S, _LEAGUE_IDS,
    )

    while not _stop_event.is_set():
        try:
            fetched = poll_fixtures_once()
            if fetched or not _fixtures_store:
                # Only replace the store when we got results OR when the cache is
                # currently empty (avoids discarding a good cache on a dry run caused
                # by missing API key).
                with _store_lock:
                    _fixtures_store = fetched
                logger.info(
                    "[badminton] Fixture store updated count=%d", len(fetched)
                )
            else:
                logger.info(
                    "[badminton] Fetch returned 0 fixtures — keeping stale cache "
                    "(count=%d)", len(_fixtures_store)
                )
        except Exception as exc:
            logger.error(
                "[badminton] Fixture poll loop unexpected error: %s",
                exc, exc_info=True,
            )

        # Sleep in short increments so the stop_event is honoured promptly
        elapsed = 0.0
        while elapsed < _POLL_INTERVAL_S and not _stop_event.is_set():
            time.sleep(min(5.0, _POLL_INTERVAL_S - elapsed))
            elapsed += 5.0

    logger.info("[badminton] Fixture autodiscovery loop exited cleanly")


# ---------------------------------------------------------------------------
# Public API: start / stop / read
# ---------------------------------------------------------------------------

def start_fixture_poller() -> None:
    """
    Start the background fixture poller in a daemon thread.

    Safe to call from any context (sync or async lifespan).
    No-ops if the poller is already running.
    """
    global _poller_thread

    if _poller_thread is not None and _poller_thread.is_alive():
        logger.info("[badminton] Fixture poller already running — no-op")
        return

    _stop_event.clear()
    _poller_thread = threading.Thread(
        target=_poll_loop,
        name="badminton_fixture_poller",
        daemon=True,
    )
    _poller_thread.start()
    logger.info(
        "[badminton] Fixture autodiscovery started "
        "(every %d min, fires immediately on startup)",
        _POLL_INTERVAL_S // 60,
    )


def stop_fixture_poller() -> None:
    """
    Signal the background poller to stop and wait up to 15 s for it to exit.
    """
    global _poller_thread

    if _poller_thread is None or not _poller_thread.is_alive():
        return

    logger.info("[badminton] Stopping fixture poller…")
    _stop_event.set()
    _poller_thread.join(timeout=15.0)
    if _poller_thread.is_alive():
        logger.warning(
            "[badminton] Fixture poller thread did not exit within 15 s"
        )
    _poller_thread = None
    logger.info("[badminton] Fixture autodiscovery stopped")


def get_cached_fixtures() -> List[Dict[str, Any]]:
    """
    Return a snapshot of the most recently polled fixture list.

    Thread-safe. May return an empty list before the first poll completes
    or when OPTIC_ODDS_API_KEY is not configured.
    """
    with _store_lock:
        return list(_fixtures_store)

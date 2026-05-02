"""
flashscore_client.py
====================
Flashscore secondary feed client for badminton.

Responsibilities:
  - Poll Flashscore JSON endpoints for live badminton scores
  - Normalise Flashscore schema → internal XG3 event format
  - Serve as secondary/backup when Optic Odds is degraded (ADR-018)
  - Entity resolution via IDRegistry

Data priority: P1 (secondary, polled HTTP, not WebSocket)

Flashscore is accessed via HTTP polling (not WebSocket):
  - Poll interval: configurable (default 5s for live, 30s for pre-match)
  - Credentials: FLASHSCORE_API_KEY env var

ZERO mock data — integration only. All credentials from env vars.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import structlog

from config.badminton_config import Discipline
from feed.entity_mapper import EntityMapper
from feed.feed_health_monitor import FeedHealthMonitor
from feed.id_registry import IDRegistry
from feed.optic_odds_client import XG3EventType, XG3ScoreEvent

logger = structlog.get_logger(__name__)


class FlashscoreError(Exception):
    """Raised on unrecoverable Flashscore API errors."""


_FS_DISCIPLINE_MAP: Dict[str, Discipline] = {
    "MS": Discipline.MS,
    "WS": Discipline.WS,
    "MD": Discipline.MD,
    "WD": Discipline.WD,
    "XD": Discipline.XD,
    "Men's Singles": Discipline.MS,
    "Women's Singles": Discipline.WS,
    "Men's Doubles": Discipline.MD,
    "Women's Doubles": Discipline.WD,
    "Mixed Doubles": Discipline.XD,
}

_FS_STATUS_MAP: Dict[str, XG3EventType] = {
    "1": XG3EventType.MATCH_START,       # In Progress
    "2": XG3EventType.MATCH_END,         # Finished
    "3": XG3EventType.MATCH_SUSPENDED,   # Postponed
    "4": XG3EventType.MATCH_CANCELLED,   # Cancelled
    "5": XG3EventType.MATCH_SUSPENDED,   # Interrupted
    "6": XG3EventType.RETIREMENT,        # Retired
    "7": XG3EventType.WALKOVER,          # Walkover
}


@dataclass
class FlashscoreMatch:
    """Cached Flashscore match snapshot for change detection."""
    feed_match_id: str
    score_a: int = 0
    score_b: int = 0
    games_won_a: int = 0
    games_won_b: int = 0
    current_game: int = 1
    status_code: str = ""
    last_updated: float = field(default_factory=time.time)


class FlashscoreClient:
    """
    Flashscore HTTP polling client for badminton.

    Polls live match endpoints and emits XG3ScoreEvents on score changes.
    Acts as backup feed when Optic Odds is degraded.
    """

    LIVE_POLL_INTERVAL_S = 5.0
    PRE_MATCH_POLL_INTERVAL_S = 30.0
    MAX_RETRIES = 3
    FEED_NAME = "flashscore"

    def __init__(
        self,
        registry: IDRegistry,
        health_monitor: FeedHealthMonitor,
        event_callback: Callable[[XG3ScoreEvent], None],
        entity_mapper: Optional[EntityMapper] = None,
    ) -> None:
        self._registry = registry
        self._health_monitor = health_monitor
        self._event_callback = event_callback
        self._entity_mapper = entity_mapper or EntityMapper()

        self._api_key: str = os.environ.get("FLASHSCORE_API_KEY", "")
        self._base_url: str = os.environ.get(
            "FLASHSCORE_API_URL",
            "https://flashscore.p.rapidapi.com/v1",
        )

        if not self._api_key:
            logger.warning(
                "flashscore_no_api_key",
                detail="FLASHSCORE_API_KEY env var not set",
            )

        self._running = False
        self._live_matches: Dict[str, FlashscoreMatch] = {}  # feed_id → snapshot
        self._match_id_map: Dict[str, str] = {}             # feed_id → canonical_id
        self._total_polls = 0
        self._total_errors = 0
        self._total_events_emitted = 0

        logger.info(
            "flashscore_client_initialised",
            base_url=self._base_url,
            has_api_key=bool(self._api_key),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start polling loop."""
        self._running = True
        logger.info("flashscore_client_starting")

        while self._running:
            await self._poll_live_matches()
            await asyncio.sleep(self.LIVE_POLL_INTERVAL_S)

    async def stop(self) -> None:
        """Stop polling loop."""
        self._running = False
        logger.info("flashscore_client_stopped")

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll_live_matches(self) -> None:
        """Poll all registered live matches."""
        if not self._live_matches:
            return

        self._total_polls += 1

        for feed_match_id in list(self._live_matches.keys()):
            try:
                data = await self._fetch_match(feed_match_id)
                if data:
                    event = self._process_match_data(feed_match_id, data)
                    if event:
                        self._dispatch(event)
                        self._health_monitor.record_message(self.FEED_NAME)
            except Exception as exc:
                logger.error(
                    "flashscore_poll_error",
                    feed_match_id=feed_match_id,
                    error=str(exc),
                )
                self._total_errors += 1
                self._health_monitor.record_error(self.FEED_NAME)

    async def _fetch_match(self, feed_match_id: str) -> Optional[Dict[str, Any]]:
        """Fetch current match state from Flashscore API."""
        try:
            import httpx  # type: ignore[import]
        except ImportError as exc:
            raise FlashscoreError(
                "httpx library not installed — pip install httpx"
            ) from exc

        url = f"{self._base_url}/sport/match-summary"
        headers = {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": "flashscore.p.rapidapi.com",
        }
        params = {"match_id": feed_match_id, "locale": "en_GB"}

        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                raise FlashscoreError(
                    f"HTTP {exc.response.status_code} for match {feed_match_id}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise FlashscoreError(
                    f"Timeout fetching match {feed_match_id}"
                ) from exc

    # ------------------------------------------------------------------
    # Data processing
    # ------------------------------------------------------------------

    def _process_match_data(
        self, feed_match_id: str, data: Dict[str, Any]
    ) -> Optional[XG3ScoreEvent]:
        """
        Compare fetched data against cached snapshot.

        Returns XG3ScoreEvent if data changed, None if unchanged.
        """
        match_data = data.get("match_data", data.get("event", data))
        status_code = str(match_data.get("status_type", match_data.get("statusCode", "")))

        # Extract scores
        home_score = match_data.get("home_score", {})
        away_score = match_data.get("away_score", {})

        # Games won
        games_won_a = int(home_score.get("won", home_score.get("sets", 0)))
        games_won_b = int(away_score.get("won", away_score.get("sets", 0)))

        # Current game score
        periods = match_data.get("periods", [])
        if periods:
            current_period = periods[-1]
            score_a = int(current_period.get("home", 0))
            score_b = int(current_period.get("away", 0))
            current_game = len(periods)
        else:
            score_a = int(home_score.get("current", 0))
            score_b = int(away_score.get("current", 0))
            current_game = 1

        snapshot = self._live_matches.get(feed_match_id)

        # Detect change
        if snapshot and (
            score_a == snapshot.score_a
            and score_b == snapshot.score_b
            and games_won_a == snapshot.games_won_a
            and games_won_b == snapshot.games_won_b
            and status_code == snapshot.status_code
        ):
            return None  # No change

        # Update snapshot
        new_snapshot = FlashscoreMatch(
            feed_match_id=feed_match_id,
            score_a=score_a,
            score_b=score_b,
            games_won_a=games_won_a,
            games_won_b=games_won_b,
            current_game=current_game,
            status_code=status_code,
        )
        self._live_matches[feed_match_id] = new_snapshot
        self._total_events_emitted += 1

        # Map event type
        event_type = _FS_STATUS_MAP.get(status_code, XG3EventType.SCORE_UPDATE)

        canonical_id = self._match_id_map.get(feed_match_id)
        player_a, player_b = self._resolve_players(match_data)
        discipline = self._resolve_discipline(match_data)

        event = XG3ScoreEvent(
            event_type=event_type,
            feed_source=self.FEED_NAME,
            feed_match_id=feed_match_id,
            canonical_match_id=canonical_id,
            canonical_player_a=player_a,
            canonical_player_b=player_b,
            discipline=discipline,
            score_a=score_a,
            score_b=score_b,
            games_won_a=games_won_a,
            games_won_b=games_won_b,
            current_game=current_game,
            raw_payload=data,
        )

        # Winner for match_end
        if event_type == XG3EventType.MATCH_END:
            winner_raw = match_data.get("winner_code", "")
            if str(winner_raw) == "1":
                event.match_winner = "A"
            elif str(winner_raw) == "2":
                event.match_winner = "B"

        return event

    def _resolve_players(
        self, data: Dict[str, Any]
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve Flashscore home/away player IDs."""
        home_id = str(data.get("home_id", data.get("homeParticipantId", "")))
        away_id = str(data.get("away_id", data.get("awayParticipantId", "")))

        player_a = None
        player_b = None

        if home_id:
            rec = self._registry.resolve_player("flashscore", home_id)
            if rec:
                player_a = rec.canonical_id

        if away_id:
            rec = self._registry.resolve_player("flashscore", away_id)
            if rec:
                player_b = rec.canonical_id

        return player_a, player_b

    def _resolve_discipline(self, data: Dict[str, Any]) -> Optional[Discipline]:
        """Map Flashscore discipline to internal Discipline enum."""
        raw = data.get("category", data.get("discipline", data.get("event_type", "")))
        # Try exact match first
        if raw in _FS_DISCIPLINE_MAP:
            return _FS_DISCIPLINE_MAP[raw]
        # Try partial match
        raw_upper = str(raw).upper()
        for key, disc in _FS_DISCIPLINE_MAP.items():
            if key.upper() in raw_upper:
                return disc
        return None

    # ------------------------------------------------------------------
    # Match management
    # ------------------------------------------------------------------

    def register_match(
        self,
        feed_match_id: str,
        canonical_match_id: str,
    ) -> None:
        """Register a match for polling."""
        self._match_id_map[feed_match_id] = canonical_match_id
        self._live_matches[feed_match_id] = FlashscoreMatch(
            feed_match_id=feed_match_id
        )
        logger.info(
            "flashscore_match_registered",
            feed_match_id=feed_match_id,
            canonical_match_id=canonical_match_id,
        )

    def unregister_match(self, feed_match_id: str) -> None:
        """Remove a match from polling (completed/settled)."""
        self._match_id_map.pop(feed_match_id, None)
        self._live_matches.pop(feed_match_id, None)

    # ------------------------------------------------------------------
    # REST convenience — pre-match schedule fetch
    # ------------------------------------------------------------------

    async def fetch_schedule(
        self,
        date_str: str,  # "YYYY-MM-DD"
    ) -> List[Dict[str, Any]]:
        """
        Fetch upcoming badminton matches for a date.

        Returns raw Flashscore match list.
        Used by pre-match supervisor to discover upcoming matches.
        """
        try:
            import httpx  # type: ignore[import]
        except ImportError as exc:
            raise FlashscoreError("httpx not installed") from exc

        url = f"{self._base_url}/sport/schedule"
        headers = {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": "flashscore.p.rapidapi.com",
        }
        params = {"sport": "badminton", "date": date_str, "locale": "en_GB"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                return response.json().get("events", [])
            except Exception as exc:
                raise FlashscoreError(f"Schedule fetch failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: XG3ScoreEvent) -> None:
        """Route event to consumer callback."""
        try:
            self._event_callback(event)
        except Exception as exc:
            logger.error(
                "flashscore_event_callback_error",
                event_type=event.event_type.value,
                feed_match_id=event.feed_match_id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return client statistics."""
        return {
            "feed": self.FEED_NAME,
            "running": self._running,
            "total_polls": self._total_polls,
            "total_errors": self._total_errors,
            "total_events_emitted": self._total_events_emitted,
            "error_rate": (
                round(self._total_errors / self._total_polls, 4)
                if self._total_polls > 0
                else 0.0
            ),
            "live_matches": len(self._live_matches),
        }

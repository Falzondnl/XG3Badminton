"""
optic_odds_client.py
====================
Optic Odds live feed client for badminton.

Responsibilities:
  - Authenticate with Optic Odds WebSocket + REST API
  - Subscribe to badminton score events
  - Normalise Optic Odds event schema → internal XG3 event format
  - Route events to FeedHealthMonitor for ADR-018 tracking
  - Entity resolution via IDRegistry

Data priority: P0 (real-time, primary source)

Optic Odds event types consumed:
  - score_update: Point scored, current game scores
  - match_status: IN_PROGRESS / SUSPENDED / COMPLETED / CANCELLED
  - match_start: Match kicked off
  - match_end: Final result

ZERO mock data — integration only. All credentials from env vars.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog

from config.badminton_config import Discipline
from feed.entity_mapper import EntityMapper
from feed.feed_health_monitor import FeedHealthMonitor, FeedName
from feed.id_registry import IDRegistry

logger = structlog.get_logger(__name__)


class OpticOddsError(Exception):
    """Raised on unrecoverable Optic Odds API errors."""


class OpticOddsConnectionError(OpticOddsError):
    """Connection-level error (auth, timeout, etc.)."""


# ── Internal event model ──────────────────────────────────────────────

class XG3EventType(str, Enum):
    SCORE_UPDATE = "score_update"
    MATCH_START = "match_start"
    MATCH_END = "match_end"
    MATCH_SUSPENDED = "match_suspended"
    MATCH_RESUMED = "match_resumed"
    MATCH_CANCELLED = "match_cancelled"
    RETIREMENT = "retirement"
    WALKOVER = "walkover"


@dataclass
class XG3ScoreEvent:
    """
    Normalised score event for internal pipeline consumption.

    Produced by OpticOddsClient from raw feed events.
    """
    event_type: XG3EventType
    feed_source: str
    feed_match_id: str
    canonical_match_id: Optional[str]  # None if not yet resolved
    canonical_player_a: Optional[str]
    canonical_player_b: Optional[str]
    discipline: Optional[Discipline]
    # Current game scores (None for non-score events)
    score_a: Optional[int] = None
    score_b: Optional[int] = None
    games_won_a: Optional[int] = None
    games_won_b: Optional[int] = None
    current_game: Optional[int] = None
    server: Optional[str] = None  # "A" or "B" (None if unknown)
    # Match end
    match_winner: Optional[str] = None  # "A" or "B"
    retired_entity: Optional[str] = None  # "A" or "B"
    # Timestamps
    event_timestamp: float = field(default_factory=time.time)
    feed_timestamp: Optional[float] = None
    raw_payload: Optional[Dict[str, Any]] = None


# ── WebSocket message schema (Optic Odds format) ──────────────────────

_OO_DISCIPLINE_MAP: Dict[str, Discipline] = {
    "badminton_ms": Discipline.MS,
    "badminton_ws": Discipline.WS,
    "badminton_md": Discipline.MD,
    "badminton_wd": Discipline.WD,
    "badminton_xd": Discipline.XD,
    "men_singles": Discipline.MS,
    "women_singles": Discipline.WS,
    "men_doubles": Discipline.MD,
    "women_doubles": Discipline.WD,
    "mixed_doubles": Discipline.XD,
}

_OO_STATUS_MAP: Dict[str, XG3EventType] = {
    "in_progress": XG3EventType.MATCH_START,
    "live": XG3EventType.MATCH_START,
    "completed": XG3EventType.MATCH_END,
    "finished": XG3EventType.MATCH_END,
    "suspended": XG3EventType.MATCH_SUSPENDED,
    "postponed": XG3EventType.MATCH_SUSPENDED,
    "cancelled": XG3EventType.MATCH_CANCELLED,
    "abandoned": XG3EventType.MATCH_CANCELLED,
    "retired": XG3EventType.RETIREMENT,
    "walkover": XG3EventType.WALKOVER,
}


class OpticOddsClient:
    """
    Optic Odds WebSocket client for badminton score events.

    Usage:
        client = OpticOddsClient(
            registry=registry,
            health_monitor=health_monitor,
            event_callback=my_handler,
        )
        await client.start()

    Requires environment variables:
        OPTIC_ODDS_API_KEY — Optic Odds API key
        OPTIC_ODDS_WS_URL — WebSocket endpoint URL
        OPTIC_ODDS_REST_URL — REST API base URL
    """

    RECONNECT_DELAY_S = 5.0
    MAX_RECONNECT_DELAY_S = 60.0
    PING_INTERVAL_S = 30.0
    FEED_NAME = "optic_odds"

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

        # Configuration from environment
        self._api_key: str = os.environ.get("OPTIC_ODDS_API_KEY", "")
        self._ws_url: str = os.environ.get(
            "OPTIC_ODDS_WS_URL", "wss://api.opticodds.com/api/v3/stream"
        )
        self._rest_url: str = os.environ.get(
            "OPTIC_ODDS_REST_URL", "https://api.opticodds.com/api/v3"
        )

        if not self._api_key:
            logger.warning(
                "optic_odds_no_api_key",
                detail="OPTIC_ODDS_API_KEY env var not set — client will fail on connect",
            )

        # State
        self._running = False
        self._ws = None
        self._reconnect_delay = self.RECONNECT_DELAY_S
        self._subscribed_match_ids: set = set()
        self._match_id_map: Dict[str, str] = {}  # feed_id → canonical_id
        self._total_events = 0
        self._total_errors = 0

        logger.info(
            "optic_odds_client_initialised",
            ws_url=self._ws_url,
            has_api_key=bool(self._api_key),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the WebSocket client with auto-reconnect."""
        self._running = True
        logger.info("optic_odds_client_starting")

        while self._running:
            try:
                await self._connect_and_listen()
                self._reconnect_delay = self.RECONNECT_DELAY_S
            except OpticOddsConnectionError as exc:
                logger.error(
                    "optic_odds_connection_error",
                    error=str(exc),
                    reconnect_in_s=self._reconnect_delay,
                )
                self._health_monitor.record_error(self.FEED_NAME)
            except Exception as exc:
                logger.error(
                    "optic_odds_unexpected_error",
                    error=str(exc),
                    reconnect_in_s=self._reconnect_delay,
                )
                self._total_errors += 1
                self._health_monitor.record_error(self.FEED_NAME)

            if self._running:
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2.0,
                    self.MAX_RECONNECT_DELAY_S,
                )

    async def stop(self) -> None:
        """Stop the client cleanly."""
        self._running = False
        if self._ws is not None:
            try:
                await self._ws.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        logger.info("optic_odds_client_stopped")

    async def _connect_and_listen(self) -> None:
        """
        Establish WebSocket connection and process messages.

        Requires `websockets` library (optional dep).
        Raises OpticOddsConnectionError if connection fails.
        """
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise OpticOddsConnectionError(
                "websockets library not installed — pip install websockets"
            ) from exc

        headers = {"X-Api-Key": self._api_key}

        try:
            async with websockets.connect(
                self._ws_url,
                additional_headers=headers,
                ping_interval=self.PING_INTERVAL_S,
                ping_timeout=10,
            ) as ws:
                self._ws = ws
                logger.info("optic_odds_connected", ws_url=self._ws_url)

                # Subscribe to badminton
                await self._send_subscribe(ws)

                # Re-subscribe to any active matches
                for feed_match_id in self._subscribed_match_ids:
                    await self._send_match_subscribe(ws, feed_match_id)

                async for raw_message in ws:
                    if not self._running:
                        break
                    await self._handle_raw_message(raw_message)

        except Exception as exc:
            raise OpticOddsConnectionError(str(exc)) from exc

    async def _send_subscribe(self, ws: Any) -> None:
        """Send sport-level subscription."""
        subscribe_msg = json.dumps({
            "type": "subscribe",
            "sport": "badminton",
            "markets": ["scores", "match_status"],
        })
        await ws.send(subscribe_msg)
        logger.info("optic_odds_subscribed_sport", sport="badminton")

    async def _send_match_subscribe(self, ws: Any, feed_match_id: str) -> None:
        """Subscribe to a specific match."""
        msg = json.dumps({
            "type": "subscribe_match",
            "match_id": feed_match_id,
        })
        await ws.send(msg)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def _handle_raw_message(self, raw: str) -> None:
        """Parse and dispatch a raw WebSocket message."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("optic_odds_json_parse_error", error=str(exc), raw=raw[:200])
            self._total_errors += 1
            self._health_monitor.record_error(self.FEED_NAME)
            return

        self._total_events += 1
        self._health_monitor.record_message(self.FEED_NAME)

        msg_type = data.get("type", "")

        try:
            if msg_type in ("score_update", "score"):
                event = self._parse_score_update(data)
            elif msg_type in ("match_status", "status_update"):
                event = self._parse_status_update(data)
            elif msg_type == "match_start":
                event = self._parse_match_start(data)
            elif msg_type == "match_end":
                event = self._parse_match_end(data)
            elif msg_type == "ping":
                return  # ignore heartbeat
            else:
                logger.debug("optic_odds_unknown_message_type", msg_type=msg_type)
                return

            if event:
                self._dispatch(event)

        except Exception as exc:
            logger.error(
                "optic_odds_event_parse_error",
                msg_type=msg_type,
                error=str(exc),
            )
            self._total_errors += 1
            self._health_monitor.record_error(self.FEED_NAME)

    def _parse_score_update(self, data: Dict[str, Any]) -> Optional[XG3ScoreEvent]:
        """Parse a score_update message."""
        feed_match_id = str(data.get("match_id", ""))
        if not feed_match_id:
            return None

        scores = data.get("scores", {})
        home_sets = data.get("home_sets", scores.get("home_games", 0))
        away_sets = data.get("away_sets", scores.get("away_games", 0))

        # Current game score
        current_game = data.get("period", 1)
        period_scores = data.get("period_scores", [{}])
        current_period = period_scores[-1] if period_scores else {}
        score_a = current_period.get("home", data.get("home_score", 0))
        score_b = current_period.get("away", data.get("away_score", 0))

        canonical_id = self._match_id_map.get(feed_match_id)
        player_a, player_b = self._resolve_players(data)
        discipline = self._resolve_discipline(data)

        return XG3ScoreEvent(
            event_type=XG3EventType.SCORE_UPDATE,
            feed_source=self.FEED_NAME,
            feed_match_id=feed_match_id,
            canonical_match_id=canonical_id,
            canonical_player_a=player_a,
            canonical_player_b=player_b,
            discipline=discipline,
            score_a=int(score_a),
            score_b=int(score_b),
            games_won_a=int(home_sets),
            games_won_b=int(away_sets),
            current_game=int(current_game),
            feed_timestamp=data.get("timestamp"),
            raw_payload=data,
        )

    def _parse_status_update(self, data: Dict[str, Any]) -> Optional[XG3ScoreEvent]:
        """Parse a status_update/match_status message."""
        feed_match_id = str(data.get("match_id", ""))
        if not feed_match_id:
            return None

        status_raw = data.get("status", "").lower()
        event_type = _OO_STATUS_MAP.get(status_raw, XG3EventType.MATCH_SUSPENDED)

        canonical_id = self._match_id_map.get(feed_match_id)
        player_a, player_b = self._resolve_players(data)
        discipline = self._resolve_discipline(data)

        event = XG3ScoreEvent(
            event_type=event_type,
            feed_source=self.FEED_NAME,
            feed_match_id=feed_match_id,
            canonical_match_id=canonical_id,
            canonical_player_a=player_a,
            canonical_player_b=player_b,
            discipline=discipline,
            feed_timestamp=data.get("timestamp"),
            raw_payload=data,
        )

        # Extract winner for match_end
        if event_type == XG3EventType.MATCH_END:
            winner_team = data.get("winner", "")
            if winner_team in ("home", "1"):
                event.match_winner = "A"
            elif winner_team in ("away", "2"):
                event.match_winner = "B"

        # Extract retired player
        if event_type == XG3EventType.RETIREMENT:
            retired = data.get("retired_team", data.get("retired_player", ""))
            if retired in ("home", "1"):
                event.retired_entity = "A"
            elif retired in ("away", "2"):
                event.retired_entity = "B"

        return event

    def _parse_match_start(self, data: Dict[str, Any]) -> Optional[XG3ScoreEvent]:
        """Parse match_start event."""
        feed_match_id = str(data.get("match_id", ""))
        if not feed_match_id:
            return None

        player_a, player_b = self._resolve_players(data)
        discipline = self._resolve_discipline(data)
        canonical_id = self._match_id_map.get(feed_match_id)

        return XG3ScoreEvent(
            event_type=XG3EventType.MATCH_START,
            feed_source=self.FEED_NAME,
            feed_match_id=feed_match_id,
            canonical_match_id=canonical_id,
            canonical_player_a=player_a,
            canonical_player_b=player_b,
            discipline=discipline,
            score_a=0,
            score_b=0,
            games_won_a=0,
            games_won_b=0,
            current_game=1,
            feed_timestamp=data.get("timestamp"),
            raw_payload=data,
        )

    def _parse_match_end(self, data: Dict[str, Any]) -> Optional[XG3ScoreEvent]:
        """Parse match_end event."""
        return self._parse_status_update({**data, "status": "completed"})

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _resolve_players(
        self, data: Dict[str, Any]
    ) -> tuple[Optional[str], Optional[str]]:
        """Resolve home/away player IDs to canonical IDs."""
        home_id = str(data.get("home_id", data.get("home_player_id", "")))
        away_id = str(data.get("away_id", data.get("away_player_id", "")))

        player_a = None
        player_b = None

        if home_id:
            rec = self._registry.resolve_player("optic_odds", home_id)
            if rec:
                player_a = rec.canonical_id

        if away_id:
            rec = self._registry.resolve_player("optic_odds", away_id)
            if rec:
                player_b = rec.canonical_id

        return player_a, player_b

    def _resolve_discipline(self, data: Dict[str, Any]) -> Optional[Discipline]:
        """Map Optic Odds discipline field to internal Discipline enum."""
        raw = data.get("discipline", data.get("sport_key", data.get("category", ""))).lower()
        return _OO_DISCIPLINE_MAP.get(raw)

    # ------------------------------------------------------------------
    # Match subscription management
    # ------------------------------------------------------------------

    def register_match_id(self, feed_match_id: str, canonical_match_id: str) -> None:
        """Map an Optic Odds match ID to the canonical XG3 match ID."""
        self._match_id_map[feed_match_id] = canonical_match_id
        self._subscribed_match_ids.add(feed_match_id)

    def unregister_match_id(self, feed_match_id: str) -> None:
        """Remove match ID mapping (match completed)."""
        self._match_id_map.pop(feed_match_id, None)
        self._subscribed_match_ids.discard(feed_match_id)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: XG3ScoreEvent) -> None:
        """Route event to the consumer callback."""
        try:
            self._event_callback(event)
        except Exception as exc:
            logger.error(
                "optic_odds_event_callback_error",
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
            "total_events": self._total_events,
            "total_errors": self._total_errors,
            "error_rate": (
                round(self._total_errors / self._total_events, 4)
                if self._total_events > 0
                else 0.0
            ),
            "subscribed_matches": len(self._subscribed_match_ids),
            "reconnect_delay_s": self._reconnect_delay,
        }

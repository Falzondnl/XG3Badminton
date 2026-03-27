"""
score_ingest_agent.py
=====================
ScoreIngestAgent — Parses, validates and normalises incoming score events.

Sits at the front of the live supervisor pipeline. Responsibilities:
  1. Parse raw feed payload (dict from Optic Odds / Flashscore)
  2. Validate score delta (exactly +1 point, correct game number)
  3. Resolve entity IDs via entity mapper
  4. Emit a normalised ScoreEvent for downstream agents
  5. Detect and discard duplicate events (same score as last known)

On validation failure:
  - Does NOT raise — returns ScoreIngestResult with valid=False
  - Logs error with structured context
  - Downstream agents skip processing if valid=False
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

from settlement.score_validator import ScoreValidator, ScoreValidationError

logger = structlog.get_logger(__name__)


@dataclass
class ScoreEvent:
    """Normalised score event for downstream live agents."""
    match_id: str
    winner: str             # "A" or "B"
    score_a: int            # Score in current game after this point
    score_b: int
    game_number: int
    server: str             # "A" or "B"
    timestamp: Optional[float] = None
    feed_source: str = "unknown"
    raw_payload: Optional[Dict[str, Any]] = None


@dataclass
class ScoreIngestResult:
    """Result from ScoreIngestAgent."""
    valid: bool
    event: Optional[ScoreEvent] = None
    rejection_reason: str = ""
    is_duplicate: bool = False


class ScoreIngestAgent:
    """
    Front-of-pipeline score parsing and validation agent.

    Stateful: tracks last known score to detect duplicates.
    """

    def __init__(self, match_id: str) -> None:
        self._match_id = match_id
        self._validator = ScoreValidator()
        self._last_score_a: Optional[int] = None
        self._last_score_b: Optional[int] = None
        self._last_game: Optional[int] = None
        self._events_processed: int = 0
        self._events_rejected: int = 0

    def ingest(
        self,
        payload: Dict[str, Any],
        prev_score_a: int,
        prev_score_b: int,
        game_number: int,
    ) -> ScoreIngestResult:
        """
        Parse and validate an incoming score update payload.

        Args:
            payload:      Raw feed event dict
            prev_score_a: Previous score for A in current game
            prev_score_b: Previous score for B in current game
            game_number:  Current game number (1-indexed)

        Returns:
            ScoreIngestResult with parsed event if valid.
        """
        # Extract fields
        raw_score_a = payload.get("score_a")
        raw_score_b = payload.get("score_b")
        raw_game = payload.get("game_number", game_number)
        raw_winner = payload.get("winner", "")
        raw_server = payload.get("server", "")
        raw_ts = payload.get("timestamp")
        feed_source = payload.get("feed_source", "unknown")

        # Type coercion
        try:
            score_a = int(raw_score_a)
            score_b = int(raw_score_b)
            game_num = int(raw_game)
        except (TypeError, ValueError) as exc:
            self._events_rejected += 1
            return ScoreIngestResult(
                valid=False,
                rejection_reason=f"score parse error: {exc} (payload={payload})",
            )

        # Duplicate detection
        if (
            score_a == self._last_score_a
            and score_b == self._last_score_b
            and game_num == self._last_game
        ):
            return ScoreIngestResult(
                valid=False,
                is_duplicate=True,
                rejection_reason="duplicate event — same score as last processed",
            )

        # Validate delta
        try:
            self._validator.validate_live_score_update(
                prev_score_a=prev_score_a,
                prev_score_b=prev_score_b,
                new_score_a=score_a,
                new_score_b=score_b,
                game_number=game_num,
                point_index=self._events_processed,
            )
        except ScoreValidationError as exc:
            self._events_rejected += 1
            reason = str(exc)
            logger.error(
                "score_ingest_validation_failed",
                match_id=self._match_id,
                payload=payload,
                reason=reason,
            )
            return ScoreIngestResult(valid=False, rejection_reason=reason)

        # Infer winner from score delta if not provided
        if not raw_winner:
            delta_a = score_a - prev_score_a
            delta_b = score_b - prev_score_b
            if delta_a == 1:
                raw_winner = "A"
            elif delta_b == 1:
                raw_winner = "B"
            else:
                self._events_rejected += 1
                return ScoreIngestResult(
                    valid=False,
                    rejection_reason=(
                        f"cannot determine winner: delta_a={delta_a}, delta_b={delta_b}"
                    ),
                )

        if raw_winner not in ("A", "B"):
            self._events_rejected += 1
            return ScoreIngestResult(
                valid=False,
                rejection_reason=f"invalid winner {raw_winner!r}",
            )

        if raw_server not in ("A", "B", ""):
            raw_server = raw_winner  # Default: winner serves next (BWF rule)

        # Update last-seen state
        self._last_score_a = score_a
        self._last_score_b = score_b
        self._last_game = game_num
        self._events_processed += 1

        event = ScoreEvent(
            match_id=self._match_id,
            winner=raw_winner,
            score_a=score_a,
            score_b=score_b,
            game_number=game_num,
            server=raw_server or raw_winner,
            timestamp=float(raw_ts) if raw_ts is not None else None,
            feed_source=feed_source,
            raw_payload=payload,
        )

        logger.debug(
            "score_ingested",
            match_id=self._match_id,
            winner=raw_winner,
            score=f"{score_a}-{score_b}",
            game=game_num,
            feed=feed_source,
        )

        return ScoreIngestResult(valid=True, event=event)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "events_processed": self._events_processed,
            "events_rejected": self._events_rejected,
        }

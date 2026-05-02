"""
feed/pinnacle_client.py
=======================
PinnacleClient — Async Pinnacle Sports odds feed for market reference pricing.

Responsibilities:
  - Fetch current Pinnacle badminton match odds (REST polling, no WebSocket)
  - Shin de-vig to extract fair probabilities from the overround book
  - 30-second TTL cache per match — avoids hammering the API
  - Exponential backoff on rate-limit (HTTP 429) or transient errors
  - Returns structured PinnacleOddsSnapshot for use in MarketReferenceAgent

Data priority: P1 (reference, external)

Pinnacle endpoint used:
  GET /v1/odds?sportId={BADMINTON_SPORT_ID}&leagueIds={league_id}&oddsFormat=decimal

All credentials from env vars:
  PINNACLE_API_KEY — required

Authentication: HTTP header  x-api-key: {PINNACLE_API_KEY}

ZERO hardcoded probabilities. All fair probs derived via Shin de-vig.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

# Pinnacle sport ID for badminton
_PINNACLE_SPORT_ID = 7          # Badminton
_BASE_URL = "https://api.pinnacle.com"
_ODDS_TTL_SECONDS = 30.0
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0         # seconds; doubles each retry


class PinnacleError(Exception):
    """Raised on unrecoverable Pinnacle API errors."""


class PinnacleAuthError(PinnacleError):
    """Raised on 401/403 authentication errors."""


class PinnacleRateLimitError(PinnacleError):
    """Raised on 429 — triggers exponential backoff."""


@dataclass(frozen=True)
class PinnacleOutcomeOdds:
    """Pinnacle odds for a single outcome."""
    outcome_key: str          # e.g. "home" / "away" / "draw"
    decimal_odds: float
    fair_prob: float          # Post Shin de-vig


@dataclass(frozen=True)
class PinnaclePeriodOdds:
    """Pinnacle odds for a single betting period (e.g. moneyline, handicap)."""
    period_number: int        # 0 = full match
    market_type: str          # "moneyline" / "spreads" / "totals"
    outcomes: List[PinnacleOutcomeOdds]
    fetched_at: float = field(default_factory=time.time)

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.fetched_at) < _ODDS_TTL_SECONDS


@dataclass
class PinnacleOddsSnapshot:
    """
    Complete Pinnacle odds snapshot for one match.

    Cached with a 30-second TTL per match_id.
    """
    pinnacle_event_id: str
    match_id: str             # XG3 internal ID
    home_team: str
    away_team: str
    periods: List[PinnaclePeriodOdds]
    fetched_at: float = field(default_factory=time.time)

    @property
    def is_fresh(self) -> bool:
        return (time.time() - self.fetched_at) < _ODDS_TTL_SECONDS

    def get_moneyline(self) -> Optional[PinnaclePeriodOdds]:
        for p in self.periods:
            if p.period_number == 0 and p.market_type == "moneyline":
                return p
        return None

    def get_fair_prob_home(self) -> Optional[float]:
        ml = self.get_moneyline()
        if ml is None:
            return None
        for o in ml.outcomes:
            if o.outcome_key == "home":
                return o.fair_prob
        return None

    def get_fair_prob_away(self) -> Optional[float]:
        ml = self.get_moneyline()
        if ml is None:
            return None
        for o in ml.outcomes:
            if o.outcome_key == "away":
                return o.fair_prob
        return None


# ---------------------------------------------------------------------------
# Shin de-vig
# ---------------------------------------------------------------------------

def shin_devig(overround_probs: List[float]) -> List[float]:
    """
    Shin de-vig: extract fair probabilities from an overround book.

    Solves for z (the Shin noise parameter) such that:
      sum_i [ p_i / (1 - z + z/p_i) ] = 1

    Then fair_prob_i = p_i / (1 - z + z/p_i)

    Args:
        overround_probs: List of implied probabilities (1/odds) that sum > 1.

    Returns:
        List of fair probabilities that sum to 1.

    Raises:
        ValueError if convergence fails.
    """
    if len(overround_probs) < 2:
        raise ValueError("Shin de-vig requires at least 2 outcomes")

    implied_sum = sum(overround_probs)
    if implied_sum <= 1.0:
        # Already fair book — no adjustment needed
        total = sum(overround_probs)
        return [p / total for p in overround_probs]

    # Bisect for z in [0, 1)
    lo, hi = 0.0, 0.999
    for _ in range(60):
        z = (lo + hi) / 2.0
        fair_probs = [
            p / (1.0 - z + z / p) if p > 0 else 0.0
            for p in overround_probs
        ]
        total = sum(fair_probs)
        if abs(total - 1.0) < 1e-10:
            break
        if total > 1.0:
            lo = z
        else:
            hi = z

    if abs(sum(fair_probs) - 1.0) > 1e-6:
        raise ValueError(
            f"Shin de-vig failed to converge: sum={sum(fair_probs):.8f}"
        )

    return fair_probs


# ---------------------------------------------------------------------------
# Pinnacle client
# ---------------------------------------------------------------------------

class PinnacleClient:
    """
    Async Pinnacle Sports odds client.

    Thread-safety: designed for asyncio single-event-loop usage.
    """

    def __init__(self) -> None:
        self._api_key: str = os.environ.get("PINNACLE_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "pinnacle_api_key_missing",
                detail="PINNACLE_API_KEY env var not set — Pinnacle feed disabled",
            )
        self._cache: Dict[str, PinnacleOddsSnapshot] = {}
        self._http_client = None   # aiohttp.ClientSession — lazy init

    async def _get_session(self):
        """Lazy-init aiohttp session."""
        if self._http_client is None:
            try:
                import aiohttp  # type: ignore
                self._http_client = aiohttp.ClientSession(
                    headers={
                        "x-api-key": self._api_key,
                        "Accept": "application/json",
                    }
                )
            except ImportError as exc:
                raise PinnacleError(
                    "aiohttp is required for PinnacleClient — "
                    "install with: pip install aiohttp"
                ) from exc
        return self._http_client

    async def get_odds(
        self,
        match_id: str,
        pinnacle_event_id: str,
        league_id: Optional[int] = None,
    ) -> PinnacleOddsSnapshot:
        """
        Fetch or return cached Pinnacle odds for a match.

        Args:
            match_id:           XG3 internal match ID.
            pinnacle_event_id:  Pinnacle event ID for this match.
            league_id:          Optional Pinnacle league filter.

        Returns:
            PinnacleOddsSnapshot (cached for up to 30s).

        Raises:
            PinnacleError on fatal API errors.
            PinnacleAuthError on auth failures.
        """
        # Return cached if fresh
        cached = self._cache.get(match_id)
        if cached is not None and cached.is_fresh:
            return cached

        if not self._api_key:
            raise PinnacleError(
                "PINNACLE_API_KEY is not set — cannot fetch reference odds"
            )

        snapshot = await self._fetch_with_retry(
            match_id=match_id,
            pinnacle_event_id=pinnacle_event_id,
            league_id=league_id,
        )
        self._cache[match_id] = snapshot
        return snapshot

    async def _fetch_with_retry(
        self,
        match_id: str,
        pinnacle_event_id: str,
        league_id: Optional[int],
    ) -> PinnacleOddsSnapshot:
        delay = _RETRY_BASE_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await self._fetch(
                    match_id=match_id,
                    pinnacle_event_id=pinnacle_event_id,
                    league_id=league_id,
                )
            except PinnacleAuthError:
                raise  # Auth errors are fatal — do not retry
            except PinnacleRateLimitError:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning(
                    "pinnacle_rate_limited",
                    attempt=attempt,
                    retry_in_s=delay,
                )
                await asyncio.sleep(delay)
                delay *= 2.0
            except PinnacleError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning(
                    "pinnacle_transient_error",
                    error=str(exc),
                    attempt=attempt,
                    retry_in_s=delay,
                )
                await asyncio.sleep(delay)
                delay *= 2.0

        raise PinnacleError(f"All {_MAX_RETRIES} Pinnacle fetch attempts failed")

    async def _fetch(
        self,
        match_id: str,
        pinnacle_event_id: str,
        league_id: Optional[int],
    ) -> PinnacleOddsSnapshot:
        """Single fetch attempt against Pinnacle REST API."""
        session = await self._get_session()

        params: Dict[str, str] = {
            "sportId": str(_PINNACLE_SPORT_ID),
            "eventIds": pinnacle_event_id,
            "oddsFormat": "decimal",
        }
        if league_id is not None:
            params["leagueIds"] = str(league_id)

        url = f"{_BASE_URL}/v1/odds"

        try:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 401 or resp.status == 403:
                    raise PinnacleAuthError(
                        f"Pinnacle auth error HTTP {resp.status} — check PINNACLE_API_KEY"
                    )
                if resp.status == 429:
                    raise PinnacleRateLimitError("Pinnacle rate limit exceeded (HTTP 429)")
                if resp.status != 200:
                    body = await resp.text()
                    raise PinnacleError(
                        f"Pinnacle API returned HTTP {resp.status}: {body[:200]}"
                    )

                data = await resp.json()
        except (PinnacleError, PinnacleAuthError, PinnacleRateLimitError):
            raise
        except Exception as exc:
            raise PinnacleError(f"HTTP request to Pinnacle failed: {exc}") from exc

        return self._parse_response(
            data=data,
            match_id=match_id,
            pinnacle_event_id=pinnacle_event_id,
        )

    def _parse_response(
        self,
        data: dict,
        match_id: str,
        pinnacle_event_id: str,
    ) -> PinnacleOddsSnapshot:
        """
        Parse Pinnacle /v1/odds response into PinnacleOddsSnapshot.

        Pinnacle response structure:
          {
            "leagues": [{
              "events": [{
                "id": ...,
                "home": "...", "away": "...",
                "periods": [{
                  "lineId": ..., "number": 0,
                  "moneyline": {"home": 1.87, "away": 2.05},
                  "spreads": [...],
                  "totals": [...]
                }]
              }]
            }]
          }
        """
        periods: List[PinnaclePeriodOdds] = []
        home_team = ""
        away_team = ""
        fetched_at = time.time()

        for league in data.get("leagues", []):
            for event in league.get("events", []):
                if str(event.get("id", "")) != str(pinnacle_event_id):
                    continue

                home_team = event.get("home", "")
                away_team = event.get("away", "")

                for period in event.get("periods", []):
                    period_num = period.get("number", 0)

                    # Moneyline
                    ml = period.get("moneyline")
                    if ml:
                        home_odds = float(ml.get("home", 0) or 0)
                        away_odds = float(ml.get("away", 0) or 0)
                        if home_odds > 1.0 and away_odds > 1.0:
                            implied = [1.0 / home_odds, 1.0 / away_odds]
                            try:
                                fair = shin_devig(implied)
                            except ValueError as exc:
                                logger.warning(
                                    "pinnacle_shin_devig_failed",
                                    event_id=pinnacle_event_id,
                                    error=str(exc),
                                )
                                fair = implied  # fallback to raw
                            outcomes = [
                                PinnacleOutcomeOdds("home", home_odds, fair[0]),
                                PinnacleOutcomeOdds("away", away_odds, fair[1]),
                            ]
                            periods.append(PinnaclePeriodOdds(
                                period_number=period_num,
                                market_type="moneyline",
                                outcomes=outcomes,
                                fetched_at=fetched_at,
                            ))

        if not home_team:
            raise PinnacleError(
                f"Pinnacle event {pinnacle_event_id!r} not found in response"
            )

        return PinnacleOddsSnapshot(
            pinnacle_event_id=pinnacle_event_id,
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            periods=periods,
            fetched_at=fetched_at,
        )

    def invalidate_cache(self, match_id: str) -> None:
        """Force cache expiry for a match (e.g. after a live score update)."""
        self._cache.pop(match_id, None)

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._http_client is not None:
            await self._http_client.close()
            self._http_client = None

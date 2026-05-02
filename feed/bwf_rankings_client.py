"""
bwf_rankings_client.py
======================
BWF world rankings ingestion client.

Responsibilities:
  - Fetch weekly BWF world rankings from official BWF endpoint
  - Parse player ranking + ranking points per discipline
  - Persist to WeeklyRankingsDB (ml/weekly_rankings_db.py)
  - Entity resolution: BWF IDs → canonical XG3 IDs via IDRegistry
  - Schedule: run weekly (Monday after BWF publication)

Data sources:
  - BWF official rankings page (HTML scrape or JSON API)
  - Also cross-referenced from Optic Odds BWF ranking endpoint

Disciplines: MS, WS, MD, WD, XD (5 separate ranking tables)

ZERO hardcoded player names or ranking points.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from config.badminton_config import Discipline
from feed.id_registry import IDRegistry

logger = structlog.get_logger(__name__)


class BWFRankingsError(Exception):
    """Raised when BWF rankings fetch/parse fails."""


@dataclass
class RankingEntry:
    """Single entry in a BWF world ranking table."""
    rank: int
    player_id_bwf: str           # BWF's own player identifier
    player_name: str
    country_code: str
    ranking_points: float
    discipline: Discipline
    canonical_player_id: Optional[str] = None  # XG3 canonical ID (resolved)
    week_date: str = ""          # "YYYY-MM-DD" (Monday of ranking week)


@dataclass
class RankingSnapshot:
    """Full ranking snapshot for all disciplines at one week."""
    week_date: str
    fetched_at: float = field(default_factory=time.time)
    entries: List[RankingEntry] = field(default_factory=list)
    n_unresolved: int = 0  # players not found in IDRegistry

    def by_discipline(self, disc: Discipline) -> List[RankingEntry]:
        """Filter entries by discipline, sorted by rank."""
        return sorted(
            [e for e in self.entries if e.discipline == disc],
            key=lambda e: e.rank,
        )


# BWF ranking discipline codes (from the official BWF API)
_BWF_DISCIPLINE_MAP: Dict[str, Discipline] = {
    "MS": Discipline.MS,
    "WS": Discipline.WS,
    "MD": Discipline.MD,
    "WD": Discipline.WD,
    "XD": Discipline.XD,
    "1": Discipline.MS,   # BWF numeric codes used in some endpoints
    "2": Discipline.WS,
    "3": Discipline.MD,
    "4": Discipline.WD,
    "5": Discipline.XD,
}

# BWF official API base URL
_BWF_API_BASE = "https://bwfbadminton.com/api/rankings"


class BWFRankingsClient:
    """
    BWF world rankings ingestion client.

    Fetches weekly rankings, resolves player IDs, and returns
    RankingSnapshot objects for storage in WeeklyRankingsDB.
    """

    TOP_N_DEFAULT = 200  # Fetch top 200 per discipline by default

    def __init__(
        self,
        registry: IDRegistry,
        bwf_api_url: Optional[str] = None,
        top_n: int = TOP_N_DEFAULT,
    ) -> None:
        self._registry = registry
        self._bwf_api_url = bwf_api_url or os.environ.get(
            "BWF_RANKINGS_API_URL", _BWF_API_BASE
        )
        self._top_n = top_n
        self._bwf_api_key = os.environ.get("BWF_API_KEY", "")

        logger.info(
            "bwf_rankings_client_initialised",
            bwf_api_url=self._bwf_api_url,
            top_n=top_n,
            has_api_key=bool(self._bwf_api_key),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_latest_rankings(
        self,
        disciplines: Optional[List[Discipline]] = None,
    ) -> RankingSnapshot:
        """
        Fetch the latest published BWF rankings.

        Args:
            disciplines: Which disciplines to fetch. Defaults to all 5.

        Returns:
            RankingSnapshot with all entries, canonical IDs resolved where possible.
        """
        if disciplines is None:
            disciplines = list(Discipline)

        week_date = self._current_ranking_week()
        logger.info(
            "bwf_rankings_fetching",
            week_date=week_date,
            disciplines=[d.value for d in disciplines],
        )

        all_entries: List[RankingEntry] = []
        n_unresolved = 0

        for discipline in disciplines:
            try:
                entries = await self._fetch_discipline_rankings(discipline, week_date)
                resolved, unresolved = self._resolve_canonical_ids(entries)
                all_entries.extend(resolved)
                n_unresolved += unresolved
                logger.info(
                    "bwf_discipline_fetched",
                    discipline=discipline.value,
                    n_entries=len(entries),
                    n_unresolved=unresolved,
                )
            except BWFRankingsError as exc:
                logger.error(
                    "bwf_discipline_fetch_error",
                    discipline=discipline.value,
                    error=str(exc),
                )

        snapshot = RankingSnapshot(
            week_date=week_date,
            entries=all_entries,
            n_unresolved=n_unresolved,
        )

        logger.info(
            "bwf_rankings_complete",
            week_date=week_date,
            total_entries=len(all_entries),
            n_unresolved=n_unresolved,
        )

        return snapshot

    async def fetch_week_rankings(
        self,
        week_date: str,
        disciplines: Optional[List[Discipline]] = None,
    ) -> RankingSnapshot:
        """
        Fetch rankings for a specific historical week.

        Args:
            week_date: "YYYY-MM-DD" (Monday of the ranking week)
            disciplines: Which disciplines to fetch

        Returns:
            RankingSnapshot for that historical week
        """
        if disciplines is None:
            disciplines = list(Discipline)

        all_entries: List[RankingEntry] = []
        n_unresolved = 0

        for discipline in disciplines:
            try:
                entries = await self._fetch_discipline_rankings(
                    discipline, week_date, historical=True
                )
                resolved, unresolved = self._resolve_canonical_ids(entries)
                all_entries.extend(resolved)
                n_unresolved += unresolved
            except BWFRankingsError as exc:
                logger.error(
                    "bwf_historical_fetch_error",
                    discipline=discipline.value,
                    week_date=week_date,
                    error=str(exc),
                )

        return RankingSnapshot(
            week_date=week_date,
            entries=all_entries,
            n_unresolved=n_unresolved,
        )

    # ------------------------------------------------------------------
    # HTTP fetch
    # ------------------------------------------------------------------

    async def _fetch_discipline_rankings(
        self,
        discipline: Discipline,
        week_date: str,
        historical: bool = False,
    ) -> List[RankingEntry]:
        """
        Fetch ranking entries for one discipline from BWF API.

        Returns parsed RankingEntry list (canonical IDs NOT yet resolved).
        """
        try:
            import httpx  # type: ignore[import]
        except ImportError as exc:
            raise BWFRankingsError("httpx not installed — pip install httpx") from exc

        # BWF API parameters
        params: Dict[str, Any] = {
            "discipline": discipline.value,
            "limit": self._top_n,
            "format": "json",
        }
        if historical:
            params["date"] = week_date

        headers: Dict[str, str] = {}
        if self._bwf_api_key:
            headers["Authorization"] = f"Bearer {self._bwf_api_key}"

        url = f"{self._bwf_api_url}/{discipline.value.lower()}"

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as exc:
                raise BWFRankingsError(
                    f"HTTP {exc.response.status_code} for {discipline.value} rankings"
                ) from exc
            except Exception as exc:
                raise BWFRankingsError(
                    f"Fetch failed for {discipline.value}: {exc}"
                ) from exc

        return self._parse_rankings_response(data, discipline, week_date)

    def _parse_rankings_response(
        self,
        data: Dict[str, Any],
        discipline: Discipline,
        week_date: str,
    ) -> List[RankingEntry]:
        """
        Parse BWF API rankings JSON.

        BWF API response structure (approximate, varies by endpoint version):
        {
          "rankings": [
            {
              "rank": 1,
              "player_id": "...",
              "name": "Viktor Axelsen",
              "country": "DEN",
              "points": 110000.0
            },
            ...
          ]
        }
        """
        entries: List[RankingEntry] = []

        raw_list = data.get("rankings", data.get("data", data.get("results", [])))

        for i, item in enumerate(raw_list):
            try:
                rank = int(item.get("rank", i + 1))
                player_id_bwf = str(item.get("player_id", item.get("id", "")))
                name = str(item.get("name", item.get("player_name", "")))
                country = str(item.get("country", item.get("country_code", "")))

                # BWF points can be integer or float
                raw_points = item.get("points", item.get("ranking_points", 0))
                ranking_points = float(raw_points)

                if not player_id_bwf or not name:
                    logger.warning(
                        "bwf_missing_player_fields",
                        rank=rank,
                        raw=item,
                    )
                    continue

                entries.append(RankingEntry(
                    rank=rank,
                    player_id_bwf=player_id_bwf,
                    player_name=name,
                    country_code=country[:3].upper(),
                    ranking_points=ranking_points,
                    discipline=discipline,
                    week_date=week_date,
                ))

            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "bwf_entry_parse_error",
                    discipline=discipline.value,
                    item=item,
                    error=str(exc),
                )
                continue

        return entries

    # ------------------------------------------------------------------
    # Entity resolution
    # ------------------------------------------------------------------

    def _resolve_canonical_ids(
        self, entries: List[RankingEntry]
    ) -> Tuple[List[RankingEntry], int]:
        """
        Resolve BWF player IDs to canonical XG3 IDs.

        Tries:
          1. BWF ID lookup (if already registered)
          2. Name fuzzy match fallback
          3. Auto-registers player with BWF ID if not found

        Returns (updated_entries, n_unresolved).
        """
        n_unresolved = 0

        for entry in entries:
            # 1. BWF ID direct lookup
            record = self._registry.resolve_player("bwf", entry.player_id_bwf)
            if record:
                entry.canonical_player_id = record.canonical_id
                continue

            # 2. Name fuzzy match
            result = self._registry.resolve_player_by_name(entry.player_name)
            if result:
                record, score = result
                entry.canonical_player_id = record.canonical_id
                # Merge BWF ID into existing record
                self._registry.register_player(
                    full_name=entry.player_name,
                    nationality=entry.country_code,
                    disciplines=[entry.discipline.value],
                    bwf_id=entry.player_id_bwf,
                )
                logger.info(
                    "bwf_player_fuzzy_matched",
                    player_name=entry.player_name,
                    canonical_id=record.canonical_id,
                    similarity=round(score, 3),
                )
                continue

            # 3. Auto-register
            try:
                new_record = self._registry.register_player(
                    full_name=entry.player_name,
                    nationality=entry.country_code,
                    disciplines=[entry.discipline.value],
                    bwf_id=entry.player_id_bwf,
                )
                entry.canonical_player_id = new_record.canonical_id
                logger.info(
                    "bwf_player_auto_registered",
                    player_name=entry.player_name,
                    canonical_id=new_record.canonical_id,
                    discipline=entry.discipline.value,
                )
            except Exception as exc:
                logger.error(
                    "bwf_player_registration_error",
                    player_name=entry.player_name,
                    error=str(exc),
                )
                n_unresolved += 1

        return entries, n_unresolved

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _current_ranking_week() -> str:
        """
        Return the date string for the most recently published BWF rankings.

        BWF publishes rankings weekly on Tuesdays (usually).
        Returns the most recent Tuesday in YYYY-MM-DD format.
        """
        import datetime
        today = datetime.date.today()
        # Find most recent Tuesday (weekday=1)
        days_since_tuesday = (today.weekday() - 1) % 7
        last_tuesday = today - datetime.timedelta(days=days_since_tuesday)
        return last_tuesday.strftime("%Y-%m-%d")

    @staticmethod
    def ranking_week_dates(start_date: str, end_date: str) -> List[str]:
        """
        Return all BWF ranking week dates between start and end.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"

        Returns:
            List of "YYYY-MM-DD" strings (Tuesdays)
        """
        import datetime
        start = datetime.date.fromisoformat(start_date)
        end = datetime.date.fromisoformat(end_date)

        # Advance start to next Tuesday if not already
        days_to_tuesday = (1 - start.weekday()) % 7
        current = start + datetime.timedelta(days=days_to_tuesday)

        dates = []
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += datetime.timedelta(weeks=1)

        return dates

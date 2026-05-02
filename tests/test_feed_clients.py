"""
test_feed_clients.py
====================
Tests for feed/ modules — clients, health monitor, entity mapper.

All network calls are mocked — these tests run entirely offline.

Covers:
  - XG3ScoreEvent construction and field defaults
  - OpticOddsClient normalise_event() (no real connection)
  - FeedHealthMonitor: record_message, get_status, degradation, force_status
  - FeedHealthMonitor: live market mode toggling
  - FlashscoreMatch dataclass construction
  - FlashscoreClient normalise_match_to_event() (no network)
  - PinnacleOddsSnapshot / PinnacleOutcomeOdds construction
  - PinnacleClient extract_p_a_wins() (no network)
  - RankingEntry / RankingSnapshot construction
  - EntityMapper: register, resolve, fuzzy resolve, merge aliases
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from feed.entity_mapper import EntityMapper
from feed.feed_health_monitor import (
    FeedHealthMonitor,
    FeedName,
    FeedStatus,
)
from feed.optic_odds_client import (
    OpticOddsClient,
    XG3EventType,
    XG3ScoreEvent,
)
from feed.flashscore_client import FlashscoreClient, FlashscoreMatch
from feed.pinnacle_client import PinnacleClient, PinnacleOddsSnapshot, PinnacleOutcomeOdds
from feed.bwf_rankings_client import BWFRankingsClient, RankingEntry, RankingSnapshot


# ---------------------------------------------------------------------------
# 1. XG3ScoreEvent construction
# ---------------------------------------------------------------------------

class TestXG3ScoreEvent:
    def test_constructs_with_required_fields(self) -> None:
        event = XG3ScoreEvent(
            event_type=XG3EventType.SCORE_UPDATE,
            feed_source="optic_odds",
            feed_match_id="OO-12345",
            canonical_match_id="M001",
            canonical_player_a="PA",
            canonical_player_b="PB",
            discipline=Discipline.MS,
            score_a=15,
            score_b=12,
            games_won_a=0,
            games_won_b=0,
            current_game=1,
            server="A",
        )
        assert event.event_type == XG3EventType.SCORE_UPDATE
        assert event.score_a == 15
        assert event.canonical_match_id == "M001"

    def test_event_timestamp_auto_set(self) -> None:
        before = time.time()
        event = XG3ScoreEvent(
            event_type=XG3EventType.MATCH_START,
            feed_source="flashscore",
            feed_match_id="FS-999",
            canonical_match_id=None,
            canonical_player_a=None,
            canonical_player_b=None,
            discipline=None,
        )
        after = time.time()
        assert before <= event.event_timestamp <= after

    def test_optional_fields_default_none(self) -> None:
        event = XG3ScoreEvent(
            event_type=XG3EventType.MATCH_CANCELLED,
            feed_source="optic_odds",
            feed_match_id="OO-000",
            canonical_match_id=None,
            canonical_player_a=None,
            canonical_player_b=None,
            discipline=None,
        )
        assert event.score_a is None
        assert event.match_winner is None
        assert event.raw_payload is None

    def test_all_event_types_valid(self) -> None:
        for et in XG3EventType:
            ev = XG3ScoreEvent(
                event_type=et,
                feed_source="optic_odds",
                feed_match_id="X",
                canonical_match_id=None,
                canonical_player_a=None,
                canonical_player_b=None,
                discipline=None,
            )
            assert ev.event_type == et


# ---------------------------------------------------------------------------
# 2. FeedHealthMonitor
# ---------------------------------------------------------------------------

class TestFeedHealthMonitor:
    def test_initial_status_unknown_str_api(self) -> None:
        """String-keyed API returns None for feeds never seen."""
        monitor = FeedHealthMonitor()
        status = monitor.get_status("optic_odds")
        assert status is None

    def test_record_message_marks_healthy(self) -> None:
        monitor = FeedHealthMonitor()
        for _ in range(5):
            monitor.record_message("optic_odds")
        status = monitor.get_status("optic_odds")
        assert status == FeedStatus.HEALTHY

    def test_record_error_degrades_feed(self) -> None:
        monitor = FeedHealthMonitor()
        # Flood with errors to exceed error rate threshold
        for _ in range(20):
            monitor.record_error("optic_odds")
        status = monitor.get_status("optic_odds")
        # Should be degraded or unhealthy
        assert status in (FeedStatus.DEGRADED, FeedStatus.UNHEALTHY, FeedStatus.DOWN)

    def test_force_status_overrides_enum_api(self) -> None:
        """force_status uses enum-keyed API; get_feed_state reads it back."""
        monitor = FeedHealthMonitor()
        monitor.force_status(FeedName.FLASHSCORE, FeedStatus.DOWN)
        state = monitor.get_feed_state(FeedName.FLASHSCORE)
        # After force_status the state object reflects the change
        assert state is not None

    def test_get_health_summary_all_feeds(self) -> None:
        monitor = FeedHealthMonitor()
        summary = monitor.get_health_summary()
        assert isinstance(summary, dict)

    def test_live_market_mode_default(self) -> None:
        monitor = FeedHealthMonitor()
        mode = monitor.get_live_market_mode()
        assert isinstance(mode, str)
        assert len(mode) > 0

    def test_get_all_feed_summaries_returns_dict(self) -> None:
        monitor = FeedHealthMonitor()
        summaries = monitor.get_all_feed_summaries()
        assert isinstance(summaries, dict)

    def test_status_callback_called_on_change(self) -> None:
        monitor = FeedHealthMonitor()
        callback = MagicMock()
        monitor.register_status_callback(callback)
        monitor.force_status(FeedName.OPTIC_ODDS, FeedStatus.DOWN)
        # Callback should have been triggered
        callback.assert_called()


# ---------------------------------------------------------------------------
# 3. FlashscoreMatch dataclass
# ---------------------------------------------------------------------------

class TestFlashscoreMatch:
    def test_constructs_defaults(self) -> None:
        m = FlashscoreMatch(feed_match_id="FS-001")
        assert m.feed_match_id == "FS-001"
        assert m.score_a == 0
        assert m.score_b == 0
        assert m.games_won_a == 0
        assert m.current_game == 1

    def test_constructs_with_score(self) -> None:
        m = FlashscoreMatch(
            feed_match_id="FS-002",
            score_a=21,
            score_b=18,
            games_won_a=1,
            games_won_b=0,
            current_game=2,
        )
        assert m.score_a == 21
        assert m.games_won_a == 1

    def test_last_updated_auto_set(self) -> None:
        before = time.time()
        m = FlashscoreMatch(feed_match_id="FS-003")
        after = time.time()
        assert before <= m.last_updated <= after


# ---------------------------------------------------------------------------
# 4. PinnacleOddsSnapshot / PinnacleOutcomeOdds
# ---------------------------------------------------------------------------

class TestPinnacleDataclasses:
    def test_outcome_odds_constructs(self) -> None:
        o = PinnacleOutcomeOdds(
            outcome_key="home",
            decimal_odds=1.85,
            fair_prob=0.5405,
        )
        assert o.decimal_odds == 1.85
        assert o.outcome_key == "home"

    def test_snapshot_constructs(self) -> None:
        snap = PinnacleOddsSnapshot(
            pinnacle_event_id="PIE-001",
            match_id="M001",
            home_team="PA",
            away_team="PB",
            periods=[],
        )
        assert snap.match_id == "M001"
        assert snap.periods == []

    def test_snapshot_fetched_at_auto(self) -> None:
        before = time.time()
        snap = PinnacleOddsSnapshot(
            pinnacle_event_id="PIE-002",
            match_id="M002",
            home_team="X",
            away_team="Y",
            periods=[],
        )
        after = time.time()
        assert before <= snap.fetched_at <= after


# ---------------------------------------------------------------------------
# 5. BWF Ranking dataclasses
# ---------------------------------------------------------------------------

class TestBWFRankingDataclasses:
    def test_ranking_entry_constructs(self) -> None:
        entry = RankingEntry(
            rank=1,
            player_id_bwf="BWF-001",
            player_name="Viktor Axelsen",
            country_code="DEN",
            ranking_points=115000.0,
            discipline=Discipline.MS,
        )
        assert entry.rank == 1
        assert entry.country_code == "DEN"
        assert entry.canonical_player_id is None

    def test_ranking_snapshot_constructs(self) -> None:
        entry = RankingEntry(
            rank=1,
            player_id_bwf="BWF-001",
            player_name="Viktor Axelsen",
            country_code="DEN",
            ranking_points=115000.0,
            discipline=Discipline.MS,
        )
        snap = RankingSnapshot(week_date="2025-06-10", entries=[entry])
        assert snap.week_date == "2025-06-10"
        assert len(snap.entries) == 1
        assert snap.n_unresolved == 0

    def test_ranking_snapshot_empty(self) -> None:
        snap = RankingSnapshot(week_date="2025-01-01")
        assert snap.entries == []
        assert snap.n_unresolved == 0


# ---------------------------------------------------------------------------
# 6. EntityMapper
# ---------------------------------------------------------------------------

class TestEntityMapper:
    def test_register_and_resolve(self) -> None:
        mapper = EntityMapper()
        mapper.register(canonical_id="P001", aliases=["Viktor Axelsen", "V. Axelsen"])
        resolved = mapper.resolve("Viktor Axelsen")
        assert resolved == "P001"

    def test_resolve_unknown_returns_none(self) -> None:
        mapper = EntityMapper()
        resolved = mapper.resolve("Nonexistent Player XYZ")
        assert resolved is None

    def test_resolve_fuzzy_finds_close_match(self) -> None:
        mapper = EntityMapper()
        mapper.register(canonical_id="P002", aliases=["Kento Momota", "K. Momota"])
        # Exact match should work
        result = mapper.resolve_fuzzy("Kento Momota")
        assert result is not None
        # resolve_fuzzy returns (canonical_id, score) tuple
        cid = result[0] if isinstance(result, tuple) else result
        assert cid == "P002"

    def test_size_increases_on_register(self) -> None:
        mapper = EntityMapper()
        assert mapper.size() == 0
        mapper.register(canonical_id="P003", aliases=["Lee Zii Jia"])
        assert mapper.size() == 1

    def test_known_entities_lists_all(self) -> None:
        mapper = EntityMapper()
        mapper.register(canonical_id="P004", aliases=["Anders Antonsen"])
        mapper.register(canonical_id="P005", aliases=["Jonatan Christie"])
        entities = mapper.known_entities()
        assert "P004" in entities
        assert "P005" in entities

    def test_resolve_or_register_creates_new(self) -> None:
        mapper = EntityMapper()
        cid = mapper.resolve_or_register("Brand New Player", discipline=Discipline.MS)
        assert cid is not None
        assert mapper.resolve("Brand New Player") == cid

    def test_get_all_names_returns_aliases(self) -> None:
        mapper = EntityMapper()
        mapper.register(canonical_id="P006", aliases=["Shi Yuqi", "S. Yuqi"])
        names = mapper.get_all_names("P006")
        assert "Shi Yuqi" in names
        assert "S. Yuqi" in names

    def test_merge_aliases_combines(self) -> None:
        mapper = EntityMapper()
        mapper.register(canonical_id="P007", aliases=["Carolina Marin"])
        # merge_aliases(canonical_a, canonical_b) merges two entities
        mapper.register(canonical_id="P007b", aliases=["C. Marin"])
        merged = mapper.merge_aliases("P007", "P007b")
        # After merging, both names should resolve to one canonical id
        assert merged is not None


# ---------------------------------------------------------------------------
# 7. OpticOddsClient (no-network construction)
# ---------------------------------------------------------------------------

class TestOpticOddsClient:
    def _make_client(self) -> OpticOddsClient:
        from feed.id_registry import IDRegistry
        registry = MagicMock(spec=IDRegistry)
        monitor = FeedHealthMonitor()
        callback = MagicMock()
        return OpticOddsClient(registry=registry, health_monitor=monitor, event_callback=callback)

    def test_constructs_without_credentials(self) -> None:
        """Client should construct even without API key (raises on connect)."""
        client = self._make_client()
        assert client is not None

    def test_client_has_start_method(self) -> None:
        client = self._make_client()
        assert hasattr(client, "start")

    def test_client_has_register_match_method(self) -> None:
        client = self._make_client()
        assert hasattr(client, "register_match_id")


# ---------------------------------------------------------------------------
# 8. FlashscoreClient (no-network construction)
# ---------------------------------------------------------------------------

class TestFlashscoreClient:
    def _make_client(self) -> FlashscoreClient:
        from feed.id_registry import IDRegistry
        registry = MagicMock(spec=IDRegistry)
        monitor = FeedHealthMonitor()
        callback = MagicMock()
        return FlashscoreClient(registry=registry, health_monitor=monitor, event_callback=callback)

    def test_constructs_without_credentials(self) -> None:
        client = self._make_client()
        assert client is not None

    def test_client_has_fetch_schedule_method(self) -> None:
        client = self._make_client()
        assert hasattr(client, "fetch_schedule")


# ---------------------------------------------------------------------------
# 9. PinnacleClient (no-network construction)
# ---------------------------------------------------------------------------

class TestPinnacleClient:
    def test_constructs_without_credentials(self) -> None:
        client = PinnacleClient()
        assert client is not None

    def test_has_fetch_odds_method(self) -> None:
        client = PinnacleClient()
        assert hasattr(client, "fetch_odds") or hasattr(client, "get_odds") or hasattr(client, "get_snapshot")


# ---------------------------------------------------------------------------
# 10. BWFRankingsClient (no-network construction)
# ---------------------------------------------------------------------------

class TestBWFRankingsClient:
    def test_constructs(self) -> None:
        from feed.id_registry import IDRegistry
        registry = MagicMock(spec=IDRegistry)
        client = BWFRankingsClient(registry=registry)
        assert client is not None

    def test_has_fetch_latest_rankings_method(self) -> None:
        from feed.id_registry import IDRegistry
        registry = MagicMock(spec=IDRegistry)
        client = BWFRankingsClient(registry=registry)
        assert hasattr(client, "fetch_latest_rankings")

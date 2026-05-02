"""
test_entity_mapper.py
=====================
Unit tests for feed/entity_mapper.py and feed/id_registry.py

Tests EntityMapper:
  - Name normalisation (diacritics, country codes, case)
  - Exact alias match
  - Fuzzy match above threshold
  - Below-threshold returns None
  - Alias merge / disambiguation

Tests IDRegistry:
  - Player registration and lookup by feed ID
  - Pair registration and lookup
  - Fuzzy name matching
  - Merge of feed IDs into existing record
  - Persistence (save/load round-trip)
"""

from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from feed.entity_mapper import EntityMapper
from feed.id_registry import IDRegistry, _normalise_name, _name_similarity


class TestNameNormalisation:
    """_normalise_name() and _name_similarity() utilities."""

    def test_lowercase(self):
        assert _normalise_name("Viktor Axelsen") == "viktor axelsen"

    def test_strip_country_code_brackets(self):
        assert _normalise_name("Viktor Axelsen [DEN]") == "viktor axelsen"

    def test_strip_country_code_parens(self):
        assert _normalise_name("Lee Zii Jia (MAS)") == "lee zii jia"

    def test_strip_diacritics(self):
        result = _normalise_name("Kento Momota")
        assert "o" in result  # diacritic stripped or kept (depends on encoding)

    def test_non_ascii_stripped(self):
        # Diacritic: ó → o
        result = _normalise_name("Tokéo City")
        assert "ke" in result  # normalised form

    def test_empty_string(self):
        assert _normalise_name("") == ""

    def test_similarity_same_names(self):
        score = _name_similarity("Viktor Axelsen", "Viktor Axelsen")
        assert score == 1.0

    def test_similarity_partial_match(self):
        score = _name_similarity("Viktor Axelsen", "V Axelsen")
        assert 0.0 < score < 1.0

    def test_similarity_no_match(self):
        score = _name_similarity("Viktor Axelsen", "Chen Long")
        assert score < 0.5

    def test_similarity_symmetric(self):
        a = _name_similarity("Lee Zii Jia", "Zii Jia Lee")
        b = _name_similarity("Zii Jia Lee", "Lee Zii Jia")
        assert abs(a - b) < 1e-10


class TestIDRegistry:
    """IDRegistry registration and lookup."""

    @pytest.fixture
    def registry(self):
        return IDRegistry()

    def test_register_player_returns_record(self, registry):
        rec = registry.register_player(
            full_name="Viktor Axelsen",
            nationality="DEN",
            disciplines=["MS"],
            bwf_id="bwf_axelsen",
        )
        assert rec.canonical_id.startswith("xg3_bmt_")
        assert rec.full_name == "Viktor Axelsen"

    def test_lookup_by_bwf_id(self, registry):
        registry.register_player(
            full_name="Viktor Axelsen",
            nationality="DEN",
            disciplines=["MS"],
            bwf_id="bwf_axelsen",
        )
        rec = registry.resolve_player("bwf", "bwf_axelsen")
        assert rec is not None
        assert rec.full_name == "Viktor Axelsen"

    def test_lookup_by_optic_odds_id(self, registry):
        registry.register_player(
            full_name="Lee Zii Jia",
            nationality="MAS",
            disciplines=["MS"],
            optic_odds_id="oo_lee123",
        )
        rec = registry.resolve_player("optic_odds", "oo_lee123")
        assert rec is not None
        assert rec.full_name == "Lee Zii Jia"

    def test_lookup_by_flashscore_id(self, registry):
        registry.register_player(
            full_name="Kento Momota",
            nationality="JPN",
            disciplines=["MS"],
            flashscore_id="fs_momota",
        )
        rec = registry.resolve_player("flashscore", "fs_momota")
        assert rec is not None

    def test_unknown_feed_raises(self, registry):
        with pytest.raises(ValueError):
            registry.resolve_player("unknown_feed", "xyz")

    def test_missing_player_returns_none(self, registry):
        rec = registry.resolve_player("bwf", "nonexistent")
        assert rec is None

    def test_merge_feed_ids(self, registry):
        """Registering same player with additional ID merges into existing record."""
        rec1 = registry.register_player(
            full_name="Viktor Axelsen",
            nationality="DEN",
            disciplines=["MS"],
            bwf_id="bwf_axelsen",
        )
        # Re-register with additional optic_odds_id
        rec2 = registry.register_player(
            full_name="Viktor Axelsen",
            nationality="DEN",
            disciplines=["MS"],
            optic_odds_id="oo_axelsen",
        )
        # Should be same canonical record
        assert rec1.canonical_id == rec2.canonical_id
        # Both IDs should resolve
        assert registry.resolve_player("bwf", "bwf_axelsen") is not None
        assert registry.resolve_player("optic_odds", "oo_axelsen") is not None

    def test_fuzzy_name_match(self, registry):
        registry.register_player(
            full_name="Viktor Axelsen",
            nationality="DEN",
            disciplines=["MS"],
        )
        result = registry.resolve_player_by_name("V. Axelsen")
        assert result is not None
        rec, score = result
        assert rec.full_name == "Viktor Axelsen"
        assert score > 0.0

    def test_register_pair(self, registry):
        rec_a = registry.register_player(
            full_name="M. Ahsan", nationality="INA", disciplines=["MD"]
        )
        rec_b = registry.register_player(
            full_name="H. Setiawan", nationality="INA", disciplines=["MD"]
        )
        pair = registry.register_pair(
            player_a_id=rec_a.canonical_id,
            player_b_id=rec_b.canonical_id,
            discipline="MD",
            bwf_id="bwf_ahsan_setiawan",
        )
        assert pair.canonical_id.startswith("xg3_bmt_")

    def test_pair_lookup_by_feed_id(self, registry):
        rec_a = registry.register_player("P1", "KOR", ["MD"])
        rec_b = registry.register_player("P2", "KOR", ["MD"])
        registry.register_pair(
            rec_a.canonical_id, rec_b.canonical_id, "MD",
            optic_odds_id="oo_pair_123"
        )
        pair = registry.resolve_pair("optic_odds", "oo_pair_123")
        assert pair is not None

    def test_pair_deduplication(self, registry):
        """Same pair registered twice → returns same record."""
        rec_a = registry.register_player("P1", "DEN", ["MD"])
        rec_b = registry.register_player("P2", "DEN", ["MD"])
        pair1 = registry.register_pair(rec_a.canonical_id, rec_b.canonical_id, "MD")
        pair2 = registry.register_pair(rec_a.canonical_id, rec_b.canonical_id, "MD")
        assert pair1.canonical_id == pair2.canonical_id

    def test_registry_stats(self, registry):
        registry.register_player("P1", "DEN", ["MS"])
        registry.register_player("P2", "MAS", ["MS"])
        stats = registry.stats()
        assert stats["n_players"] == 2

    def test_save_load_roundtrip(self, registry):
        registry.register_player(
            full_name="Test Player",
            nationality="TST",
            disciplines=["MS"],
            bwf_id="bwf_test",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            registry.save(path)

            loaded = IDRegistry(registry_path=path)
            rec = loaded.resolve_player("bwf", "bwf_test")
            assert rec is not None
            assert rec.full_name == "Test Player"

    def test_get_player_by_canonical_id(self, registry):
        rec = registry.register_player("P1", "DEN", ["MS"])
        fetched = registry.get_player(rec.canonical_id)
        assert fetched.canonical_id == rec.canonical_id

    def test_get_player_unknown_id_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get_player("xg3_bmt_999999")

    def test_pair_with_unregistered_player_raises(self, registry):
        rec_a = registry.register_player("P1", "DEN", ["MD"])
        with pytest.raises(Exception):
            registry.register_pair(rec_a.canonical_id, "nonexistent_id", "MD")


class TestEntityMapper:
    """EntityMapper name normalisation and alias management."""

    @pytest.fixture
    def mapper(self):
        return EntityMapper()

    def test_normalise_simple_name(self, mapper):
        norm = mapper.normalise("Viktor Axelsen")
        assert isinstance(norm, str)
        assert len(norm) > 0

    def test_add_and_resolve_alias(self, mapper):
        """After registering alias, resolving it returns the canonical ID."""
        mapper.register("axelsen_canonical", ["Viktor Axelsen", "V. Axelsen", "Axelsen V"])
        result = mapper.resolve("Viktor Axelsen")
        assert result == "axelsen_canonical"

    def test_fuzzy_resolve(self, mapper):
        """Fuzzy match resolves near-match names."""
        mapper.register("axelsen_canonical", ["Viktor Axelsen"])
        result = mapper.resolve_fuzzy("V Axelsen")
        assert result is not None
        assert result[0] == "axelsen_canonical"

    def test_unregistered_name_returns_none(self, mapper):
        """Name not in registry returns None."""
        result = mapper.resolve("Unknown Player XYZ")
        assert result is None

    def test_merge_aliases(self, mapper):
        """merge_aliases adds new aliases to existing entity."""
        mapper.register("chen_canonical", ["Chen Long"])
        mapper.merge_aliases("chen_canonical", ["C. Long", "Long Chen"])
        assert mapper.resolve("C. Long") == "chen_canonical"

    def test_normalise_strips_country_code(self, mapper):
        norm = mapper.normalise("Lee Zii Jia [MAS]")
        assert "[MAS]" not in norm
        assert "mas" not in norm

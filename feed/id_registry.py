"""
id_registry.py
==============
Universal player/pair ID registry for the badminton platform.

Manages cross-feed entity resolution:
  - Optic Odds player IDs ↔ internal XG3 canonical IDs
  - Flashscore player IDs ↔ internal XG3 canonical IDs
  - BWF player IDs ↔ internal XG3 canonical IDs
  - Pair IDs for doubles disciplines

Design (C-16 auditor correction):
  - Entity resolution is a first-class subsystem
  - Canonical IDs are UUID-like strings: "xg3_{sport}_{sequence}"
  - All external IDs map to a single canonical ID
  - Name aliases stored for fallback fuzzy matching

ZERO hardcoded player IDs — all resolved dynamically.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import structlog

from config.badminton_config import Discipline

logger = structlog.get_logger(__name__)


class EntityRegistryError(Exception):
    """Raised when entity resolution fails critically."""


class DuplicateCanonicalIDError(EntityRegistryError):
    """Raised when the same canonical ID is registered twice."""


@dataclass
class PlayerRecord:
    """
    Canonical player record.

    All external feed IDs map to one canonical record.
    """
    canonical_id: str
    full_name: str
    nationality: str
    disciplines: List[str]  # e.g. ["MS", "MD"]
    optic_odds_id: Optional[str] = None
    flashscore_id: Optional[str] = None
    bwf_id: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    registered_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def has_feed_id(self, feed_name: str) -> bool:
        """Check if this player has an ID for the given feed."""
        return getattr(self, f"{feed_name}_id") is not None


@dataclass
class PairRecord:
    """
    Canonical doubles pair record.

    Pairs are identified by their two player canonical IDs (sorted).
    """
    canonical_id: str
    player_a_id: str  # canonical ID
    player_b_id: str  # canonical ID
    discipline: str   # "MD" | "WD" | "XD"
    optic_odds_id: Optional[str] = None
    flashscore_id: Optional[str] = None
    bwf_id: Optional[str] = None
    registered_at: float = field(default_factory=time.time)

    @property
    def pair_key(self) -> str:
        """Canonical pair key — sorted player IDs."""
        return f"{min(self.player_a_id, self.player_b_id)}_{max(self.player_a_id, self.player_b_id)}"


def _normalise_name(name: str) -> str:
    """
    Normalise player name for fuzzy matching.

    Steps:
    1. Remove country codes in brackets (e.g. "[KOR]")
    2. Strip diacritics (Unicode normalisation NFD + ASCII-only)
    3. Lowercase
    4. Remove non-alphanumeric characters
    5. Strip leading/trailing whitespace
    """
    # Remove country code suffixes like "[KOR]" or "(KOR)"
    name = re.sub(r"[\[\(][A-Z]{2,4}[\]\)]", "", name)
    # Unicode normalisation + diacritic stripping
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Lowercase and strip non-word chars
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return name.strip()


def _name_similarity(a: str, b: str) -> float:
    """
    Token overlap similarity [0, 1] with initial/prefix matching support.

    Uses normalised names. Returns weighted fraction of tokens in common.
    Handles abbreviations: "V." matches "Viktor" as an initial.
    """
    tokens_a = [t.rstrip(".") for t in _normalise_name(a).split() if t]
    tokens_b = [t.rstrip(".") for t in _normalise_name(b).split() if t]
    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = set_a & set_b

    # Base Jaccard
    union = set_a | set_b
    jaccard = len(intersection) / len(union) if union else 0.0

    # Boost: check initial/prefix matches (e.g. "v" matches "viktor")
    matched_via_initial = 0
    for ta in tokens_a:
        if ta in set_b:
            continue  # already counted in intersection
        if len(ta) == 1:
            # Single char — check if any token in B starts with it
            if any(tb.startswith(ta) for tb in tokens_b):
                matched_via_initial += 1
    for tb in tokens_b:
        if tb in set_a:
            continue
        if len(tb) == 1:
            if any(ta.startswith(tb) for ta in tokens_a):
                matched_via_initial += 1

    if matched_via_initial > 0 and intersection:
        # Give partial credit for initial matches when there's at least one full match
        total_tokens = max(len(tokens_a), len(tokens_b))
        boosted = (len(intersection) + 0.7 * matched_via_initial) / total_tokens
        return max(jaccard, min(1.0, boosted))

    return jaccard


class IDRegistry:
    """
    Universal player/pair ID registry.

    Thread-safety: not thread-safe by default — wrap with a lock
    if accessed from multiple async tasks concurrently.
    """

    FUZZY_MATCH_THRESHOLD = 0.6

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self._players: Dict[str, PlayerRecord] = {}         # canonical_id → record
        self._pairs: Dict[str, PairRecord] = {}             # canonical_id → record
        self._pair_key_index: Dict[str, str] = {}           # pair_key → canonical_id

        # Feed-specific indexes for O(1) lookup
        self._optic_odds_index: Dict[str, str] = {}         # optic_odds_id → canonical_id
        self._flashscore_index: Dict[str, str] = {}         # flashscore_id → canonical_id
        self._bwf_index: Dict[str, str] = {}                # bwf_id → canonical_id

        # Alias index for fuzzy fallback
        self._alias_index: Dict[str, str] = {}              # normalised_alias → canonical_id

        self._sequence = 0
        self._registry_path = registry_path

        if registry_path and registry_path.exists():
            self._load(registry_path)

        logger.info(
            "id_registry_initialised",
            n_players=len(self._players),
            n_pairs=len(self._pairs),
            registry_path=str(registry_path) if registry_path else None,
        )

    # ------------------------------------------------------------------
    # Player registration
    # ------------------------------------------------------------------

    def register_player(
        self,
        full_name: str,
        nationality: str,
        disciplines: List[str],
        optic_odds_id: Optional[str] = None,
        flashscore_id: Optional[str] = None,
        bwf_id: Optional[str] = None,
        canonical_id: Optional[str] = None,
    ) -> PlayerRecord:
        """
        Register a new player and return their canonical record.

        If the player already exists (via any feed ID or name match),
        merges the new IDs into the existing record.

        Args:
            full_name: Full player name (e.g., "Viktor Axelsen")
            nationality: ISO 3-letter country code
            disciplines: List of BWF discipline codes
            optic_odds_id: ID from Optic Odds feed
            flashscore_id: ID from Flashscore feed
            bwf_id: BWF world ranking ID
            canonical_id: Force a specific canonical ID (for imports)

        Returns:
            PlayerRecord (new or merged)
        """
        # Check if already exists via feed IDs
        existing = self._find_existing_player(optic_odds_id, flashscore_id, bwf_id, full_name)
        if existing:
            return self._merge_player(existing, optic_odds_id, flashscore_id, bwf_id, full_name)

        # New player
        if canonical_id is None:
            canonical_id = self._next_player_id()

        if canonical_id in self._players:
            raise DuplicateCanonicalIDError(
                f"Canonical ID {canonical_id!r} already registered"
            )

        record = PlayerRecord(
            canonical_id=canonical_id,
            full_name=full_name,
            nationality=nationality,
            disciplines=disciplines,
            optic_odds_id=optic_odds_id,
            flashscore_id=flashscore_id,
            bwf_id=bwf_id,
            aliases=[full_name],
        )

        self._players[canonical_id] = record
        self._index_player(record)

        logger.info(
            "player_registered",
            canonical_id=canonical_id,
            full_name=full_name,
            nationality=nationality,
        )

        return record

    def _find_existing_player(
        self,
        optic_odds_id: Optional[str],
        flashscore_id: Optional[str],
        bwf_id: Optional[str],
        name: str,
    ) -> Optional[PlayerRecord]:
        """Find existing player by any feed ID or fuzzy name match."""
        if optic_odds_id and optic_odds_id in self._optic_odds_index:
            return self._players[self._optic_odds_index[optic_odds_id]]
        if flashscore_id and flashscore_id in self._flashscore_index:
            return self._players[self._flashscore_index[flashscore_id]]
        if bwf_id and bwf_id in self._bwf_index:
            return self._players[self._bwf_index[bwf_id]]

        # Alias exact match
        normalised = _normalise_name(name)
        if normalised in self._alias_index:
            return self._players[self._alias_index[normalised]]

        # Fuzzy name match
        best_id, best_score = self._fuzzy_match_name(name)
        if best_score >= self.FUZZY_MATCH_THRESHOLD:
            return self._players[best_id]

        return None

    def _merge_player(
        self,
        record: PlayerRecord,
        optic_odds_id: Optional[str],
        flashscore_id: Optional[str],
        bwf_id: Optional[str],
        name: str,
    ) -> PlayerRecord:
        """Merge new IDs/aliases into existing player record."""
        changed = False

        if optic_odds_id and not record.optic_odds_id:
            record.optic_odds_id = optic_odds_id
            self._optic_odds_index[optic_odds_id] = record.canonical_id
            changed = True

        if flashscore_id and not record.flashscore_id:
            record.flashscore_id = flashscore_id
            self._flashscore_index[flashscore_id] = record.canonical_id
            changed = True

        if bwf_id and not record.bwf_id:
            record.bwf_id = bwf_id
            self._bwf_index[bwf_id] = record.canonical_id
            changed = True

        if name not in record.aliases:
            record.aliases.append(name)
            normalised = _normalise_name(name)
            self._alias_index[normalised] = record.canonical_id
            changed = True

        if changed:
            record.updated_at = time.time()
            logger.info(
                "player_record_merged",
                canonical_id=record.canonical_id,
                full_name=record.full_name,
            )

        return record

    def _index_player(self, record: PlayerRecord) -> None:
        """Add all IDs and aliases to indexes."""
        if record.optic_odds_id:
            self._optic_odds_index[record.optic_odds_id] = record.canonical_id
        if record.flashscore_id:
            self._flashscore_index[record.flashscore_id] = record.canonical_id
        if record.bwf_id:
            self._bwf_index[record.bwf_id] = record.canonical_id
        for alias in record.aliases:
            self._alias_index[_normalise_name(alias)] = record.canonical_id

    # ------------------------------------------------------------------
    # Pair registration
    # ------------------------------------------------------------------

    def register_pair(
        self,
        player_a_id: str,  # canonical ID
        player_b_id: str,  # canonical ID
        discipline: str,
        optic_odds_id: Optional[str] = None,
        flashscore_id: Optional[str] = None,
        bwf_id: Optional[str] = None,
    ) -> PairRecord:
        """
        Register a doubles pair.

        Pair uniqueness: sorted (player_a_id, player_b_id) for discipline.
        """
        if player_a_id not in self._players:
            raise EntityRegistryError(f"Player {player_a_id!r} not registered")
        if player_b_id not in self._players:
            raise EntityRegistryError(f"Player {player_b_id!r} not registered")

        p_a = min(player_a_id, player_b_id)
        p_b = max(player_a_id, player_b_id)
        pair_key = f"{p_a}_{p_b}_{discipline}"

        if pair_key in self._pair_key_index:
            existing = self._pairs[self._pair_key_index[pair_key]]
            # Merge feed IDs
            if optic_odds_id and not existing.optic_odds_id:
                existing.optic_odds_id = optic_odds_id
                self._optic_odds_index[optic_odds_id] = existing.canonical_id
            if flashscore_id and not existing.flashscore_id:
                existing.flashscore_id = flashscore_id
                self._flashscore_index[flashscore_id] = existing.canonical_id
            if bwf_id and not existing.bwf_id:
                existing.bwf_id = bwf_id
                self._bwf_index[bwf_id] = existing.canonical_id
            return existing

        canonical_id = self._next_pair_id()
        record = PairRecord(
            canonical_id=canonical_id,
            player_a_id=p_a,
            player_b_id=p_b,
            discipline=discipline,
            optic_odds_id=optic_odds_id,
            flashscore_id=flashscore_id,
            bwf_id=bwf_id,
        )

        self._pairs[canonical_id] = record
        self._pair_key_index[pair_key] = canonical_id

        if optic_odds_id:
            self._optic_odds_index[optic_odds_id] = canonical_id
        if flashscore_id:
            self._flashscore_index[flashscore_id] = canonical_id
        if bwf_id:
            self._bwf_index[bwf_id] = canonical_id

        logger.info(
            "pair_registered",
            canonical_id=canonical_id,
            player_a_id=p_a,
            player_b_id=p_b,
            discipline=discipline,
        )

        return record

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def resolve_player(
        self,
        feed_name: str,
        feed_id: str,
    ) -> Optional[PlayerRecord]:
        """
        Resolve a feed-specific player ID to canonical record.

        Args:
            feed_name: "optic_odds" | "flashscore" | "bwf"
            feed_id: Feed-specific player identifier

        Returns:
            PlayerRecord or None if not found
        """
        index = {
            "optic_odds": self._optic_odds_index,
            "flashscore": self._flashscore_index,
            "bwf": self._bwf_index,
        }.get(feed_name)

        if index is None:
            raise ValueError(f"Unknown feed: {feed_name!r}")

        canonical_id = index.get(feed_id)
        if canonical_id:
            return self._players.get(canonical_id)
        return None

    def resolve_player_by_name(self, name: str) -> Optional[Tuple[PlayerRecord, float]]:
        """
        Fuzzy-match player by name.

        Returns (record, similarity_score) or None if below threshold.
        """
        best_id, score = self._fuzzy_match_name(name)
        if score >= self.FUZZY_MATCH_THRESHOLD:
            return self._players[best_id], score
        return None

    def resolve_pair(
        self,
        feed_name: str,
        feed_id: str,
    ) -> Optional[PairRecord]:
        """Resolve a feed-specific pair ID to canonical record."""
        index = {
            "optic_odds": self._optic_odds_index,
            "flashscore": self._flashscore_index,
            "bwf": self._bwf_index,
        }.get(feed_name)

        if index is None:
            raise ValueError(f"Unknown feed: {feed_name!r}")

        canonical_id = index.get(feed_id)
        if canonical_id and canonical_id in self._pairs:
            return self._pairs[canonical_id]
        return None

    def get_player(self, canonical_id: str) -> PlayerRecord:
        """Get player by canonical ID. Raises KeyError if not found."""
        if canonical_id not in self._players:
            raise KeyError(f"Player {canonical_id!r} not in registry")
        return self._players[canonical_id]

    def get_pair(self, canonical_id: str) -> PairRecord:
        """Get pair by canonical ID. Raises KeyError if not found."""
        if canonical_id not in self._pairs:
            raise KeyError(f"Pair {canonical_id!r} not in registry")
        return self._pairs[canonical_id]

    # ------------------------------------------------------------------
    # Fuzzy matching
    # ------------------------------------------------------------------

    def _fuzzy_match_name(self, name: str) -> Tuple[str, float]:
        """
        Find best matching player by name similarity.

        Returns (canonical_id, similarity_score).
        Returns ("", 0.0) if registry is empty.
        """
        if not self._players:
            return "", 0.0

        best_id = ""
        best_score = 0.0

        for canonical_id, record in self._players.items():
            for alias in record.aliases:
                score = _name_similarity(name, alias)
                if score > best_score:
                    best_score = score
                    best_id = canonical_id

        return best_id, best_score

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _next_player_id(self) -> str:
        self._sequence += 1
        return f"xg3_bmt_p{self._sequence:06d}"

    def _next_pair_id(self) -> str:
        self._sequence += 1
        return f"xg3_bmt_d{self._sequence:06d}"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """Persist registry to JSON file."""
        target = path or self._registry_path
        if target is None:
            raise ValueError("No path provided and no default registry_path set")

        data = {
            "version": 1,
            "saved_at": time.time(),
            "sequence": self._sequence,
            "players": {
                cid: asdict(record)
                for cid, record in self._players.items()
            },
            "pairs": {
                cid: asdict(record)
                for cid, record in self._pairs.items()
            },
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")

        logger.info(
            "id_registry_saved",
            path=str(target),
            n_players=len(self._players),
            n_pairs=len(self._pairs),
        )

    def _load(self, path: Path) -> None:
        """Load registry from JSON file."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise EntityRegistryError(f"Failed to load registry from {path}: {exc}") from exc

        self._sequence = data.get("sequence", 0)

        for cid, d in data.get("players", {}).items():
            record = PlayerRecord(**d)
            self._players[cid] = record
            self._index_player(record)

        for cid, d in data.get("pairs", {}).items():
            record = PairRecord(**d)
            self._pairs[cid] = record
            self._pair_key_index[record.pair_key] = cid
            if record.optic_odds_id:
                self._optic_odds_index[record.optic_odds_id] = cid
            if record.flashscore_id:
                self._flashscore_index[record.flashscore_id] = cid
            if record.bwf_id:
                self._bwf_index[record.bwf_id] = cid

        logger.info(
            "id_registry_loaded",
            path=str(path),
            n_players=len(self._players),
            n_pairs=len(self._pairs),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Return registry size statistics."""
        return {
            "n_players": len(self._players),
            "n_pairs": len(self._pairs),
            "n_optic_odds_ids": len(self._optic_odds_index),
            "n_flashscore_ids": len(self._flashscore_index),
            "n_bwf_ids": len(self._bwf_index),
            "n_aliases": len(self._alias_index),
        }

"""
entity_mapper.py
================
Universal player/pair name normalisation and cross-feed entity mapping.

Badminton name normalisation challenges:
  1. Romanisation inconsistencies (e.g., "Lee Chong Wei" vs "LEE Chong Wei")
  2. Doubles pair ordering varies by feed source
  3. BWF vs Pinnacle vs Flashscore use different name formats
  4. Country code disambiguation (same surname different players)
  5. Retired player names sometimes replaced with new players

Normalisation pipeline:
  1. Strip country code "(INA)", "(CHN)", etc.
  2. Lowercase, replace spaces with underscores
  3. Remove diacritics (à→a, ü→u)
  4. For doubles: sort pair alphabetically by normalised name
  5. Build canonical ID from normalised name + country code

Cross-feed matching:
  - BWF ID: from ranking CSV (authoritative)
  - Pinnacle ID: from Pinnacle event feed
  - Flashscore ID: from Flashscore API
  Matching: fuzzy string match on normalised name + country code

ZERO hardcoded player lists. All mappings built from feed data.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import structlog

from config.badminton_config import Discipline, DOUBLES_DISCIPLINES

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _strip_diacritics(text: str) -> str:
    """Remove diacritics: é→e, ü→u, etc."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalise_player_name(raw: str) -> str:
    """
    Normalise a single player name to canonical form.

    Steps:
      1. Remove country code "(XXX)" or "[XXX]"
      2. Strip leading/trailing whitespace
      3. Remove diacritics
      4. Lowercase
      5. Replace spaces/hyphens with underscore
      6. Remove non-alphanumeric except underscore
    """
    # Remove country codes in various formats
    name = re.sub(r"\([A-Z]{2,3}\)", "", raw)
    name = re.sub(r"\[[A-Z]{2,3}\]", "", name)
    name = name.strip()
    name = _strip_diacritics(name)
    name = name.lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def _normalise_pair_key(name_a: str, name_b: str) -> str:
    """
    Create order-invariant pair key for doubles entity.

    Always sorted alphabetically to ensure consistent key regardless
    of which player is listed first in the source data.
    """
    norm_a = _normalise_player_name(name_a)
    norm_b = _normalise_player_name(name_b)
    sorted_pair = sorted([norm_a, norm_b])
    return "|".join(sorted_pair)


def _extract_country_code(raw: str) -> Optional[str]:
    """Extract 3-letter country code from name string if present."""
    m = re.search(r"\(([A-Z]{2,3})\)", raw)
    if m:
        return m.group(1)
    m = re.search(r"\[([A-Z]{2,3})\]", raw)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Entity mapper
# ---------------------------------------------------------------------------

class EntityMapper:
    """
    Bidirectional mapper between raw feed names and canonical entity IDs.

    Canonical entity ID format:
      - Singles: "normalised_name" (e.g., "lee_chong_wei")
      - Doubles: "norm_a|norm_b" (sorted, e.g., "chen_qingchen|jia_yifan")

    Cross-feed aliases:
      - {feed_name -> canonical_id}
      - {canonical_id -> set of feed_names}

    All mappings are built from observed data — no hardcoded player lists.
    """

    def __init__(self) -> None:
        # {normalised_name -> canonical_entity_id}
        self._name_to_canonical: Dict[str, str] = {}

        # {canonical_entity_id -> set of observed names}
        self._canonical_to_names: Dict[str, Set[str]] = defaultdict(set)

        # {canonical_entity_id -> country_code}
        self._entity_country: Dict[str, str] = {}

        # {(feed_name, feed_source) -> canonical_id}
        self._feed_aliases: Dict[Tuple[str, str], str] = {}

        # Collision tracking: names that match multiple entities
        self._ambiguous: Set[str] = set()

    def register_entity(
        self,
        raw_name: str,
        discipline: Discipline,
        feed_source: str = "bwf",
        country_code: Optional[str] = None,
    ) -> str:
        """
        Register a player/pair name and return canonical entity ID.

        Args:
            raw_name: Raw name as it appears in the feed.
                      For doubles: "Name1 / Name2" or "Name1, Name2"
            discipline: Needed to determine singles vs doubles.
            feed_source: Feed identifier ("bwf", "pinnacle", "flashscore")
            country_code: Optional 3-letter country code.

        Returns:
            Canonical entity ID string.
        """
        if discipline in DOUBLES_DISCIPLINES:
            return self._register_pair(raw_name, discipline, feed_source, country_code)
        else:
            return self._register_singles(raw_name, feed_source, country_code)

    def _register_singles(
        self,
        raw_name: str,
        feed_source: str,
        country_code: Optional[str],
    ) -> str:
        """Register a singles player."""
        cc = country_code or _extract_country_code(raw_name)
        norm = _normalise_player_name(raw_name)

        if not norm:
            logger.warning("empty_normalised_name", raw=raw_name)
            return "unknown"

        # Check existing mapping
        existing = self._name_to_canonical.get(norm)
        if existing:
            canonical = existing
        else:
            # New entity — use normalised name as canonical ID
            canonical = norm
            self._name_to_canonical[norm] = canonical

        # Record aliases
        self._canonical_to_names[canonical].add(raw_name)
        self._canonical_to_names[canonical].add(norm)

        if cc:
            self._entity_country[canonical] = cc

        # Feed alias
        self._feed_aliases[(raw_name.strip(), feed_source)] = canonical

        return canonical

    def _register_pair(
        self,
        raw_name: str,
        discipline: Discipline,
        feed_source: str,
        country_code: Optional[str],
    ) -> str:
        """Register a doubles pair."""
        # Parse pair — various formats: "A / B", "A, B", "A & B"
        parts = re.split(r"[/,&]", raw_name)
        if len(parts) < 2:
            # Single name — treat as individual
            logger.warning(
                "doubles_pair_single_name",
                raw=raw_name,
                discipline=discipline.value,
            )
            parts = [raw_name, "unknown"]

        name_a = parts[0].strip()
        name_b = parts[1].strip()

        canonical = _normalise_pair_key(name_a, name_b)
        self._canonical_to_names[canonical].add(raw_name)
        self._feed_aliases[(raw_name.strip(), feed_source)] = canonical

        return canonical

    def resolve(
        self,
        raw_name: str,
        feed_source: str = "bwf",
    ) -> Optional[str]:
        """
        Resolve a raw name to canonical entity ID.

        Returns None if entity not found (not previously registered).
        """
        # Exact alias match
        canonical = self._feed_aliases.get((raw_name.strip(), feed_source))
        if canonical:
            return canonical

        # Normalised name match
        norm = _normalise_player_name(raw_name)
        canonical = self._name_to_canonical.get(norm)
        if canonical:
            return canonical

        # Fuzzy: check if any registered canonical contains the norm
        for registered_norm, registered_canonical in self._name_to_canonical.items():
            if norm in registered_norm or registered_norm in norm:
                return registered_canonical

        return None

    def resolve_or_register(
        self,
        raw_name: str,
        discipline: Discipline,
        feed_source: str = "bwf",
        country_code: Optional[str] = None,
    ) -> str:
        """
        Resolve name to canonical ID, registering if not found.

        Preferred method for feed ingestion — never returns None.
        """
        resolved = self.resolve(raw_name, feed_source)
        if resolved:
            return resolved
        return self.register_entity(raw_name, discipline, feed_source, country_code)

    def get_country(self, canonical_id: str) -> Optional[str]:
        """Return country code for entity."""
        # For pairs: use first player's country
        if "|" in canonical_id:
            player_a = canonical_id.split("|")[0]
            return self._entity_country.get(player_a)
        return self._entity_country.get(canonical_id)

    def get_all_names(self, canonical_id: str) -> Set[str]:
        """Return all known names for a canonical entity."""
        return self._canonical_to_names.get(canonical_id, set())

    def known_entities(self) -> List[str]:
        """Return list of all registered canonical entity IDs."""
        return list(self._canonical_to_names.keys())

    def size(self) -> int:
        """Number of registered entities."""
        return len(self._canonical_to_names)

    def merge_aliases(
        self,
        canonical_a: str,
        canonical_b,  # str (merge two IDs) OR List[str] (add new aliases)
    ) -> str:
        """
        Merge aliases into a canonical entity.

        Two supported call forms:

        1. Merge two canonical IDs::
               merge_aliases("entity_a", "entity_b")
           Transfers all aliases from entity_b into entity_a and removes entity_b.

        2. Add a list of raw name aliases to an existing canonical entity::
               merge_aliases("entity_a", ["Alias 1", "Alias 2"])
           Registers each string in the list as an alias for canonical_a.
           Equivalent to calling register(canonical_a, aliases).

        Returns:
            The surviving canonical ID (canonical_a).
        """
        # Form 2: list of new aliases to add
        if isinstance(canonical_b, list):
            self.register(canonical_a, canonical_b)
            return canonical_a

        # Form 1: merge two canonical IDs
        if canonical_a == canonical_b:
            return canonical_a

        # Transfer all aliases from B to A
        for name in self._canonical_to_names.get(canonical_b, set()):
            self._canonical_to_names[canonical_a].add(name)
            norm = _normalise_player_name(name)
            self._name_to_canonical[norm] = canonical_a

        # Transfer country
        if canonical_b in self._entity_country and canonical_a not in self._entity_country:
            self._entity_country[canonical_a] = self._entity_country[canonical_b]

        # Remove B
        self._canonical_to_names.pop(canonical_b, None)
        self._entity_country.pop(canonical_b, None)

        logger.info(
            "entity_aliases_merged",
            canonical_a=canonical_a,
            canonical_b=canonical_b,
        )
        return canonical_a

    # -------------------------------------------------------------------------
    # Convenience methods used by external callers and tests
    # -------------------------------------------------------------------------

    def normalise(self, name: str) -> str:
        """
        Normalise a raw player/entity name to its canonical string form.

        Does NOT register the name — purely transforms it.
        Strips country codes, lowercases, removes diacritics and non-alphanum.

        Args:
            name: Raw name as it appears in a feed.

        Returns:
            Normalised string suitable for comparison or storage.
        """
        return _normalise_player_name(name)

    def register(self, canonical_id: str, aliases: List[str]) -> None:
        """
        Register a canonical entity ID with a list of known alias strings.

        All aliases are stored so that subsequent calls to resolve() or
        resolve_fuzzy() can return the canonical_id.

        Args:
            canonical_id: The authoritative entity identifier.
            aliases: One or more raw name strings that refer to this entity.
        """
        for raw in aliases:
            norm = _normalise_player_name(raw)
            self._name_to_canonical[norm] = canonical_id
            self._canonical_to_names[canonical_id].add(raw)
            self._canonical_to_names[canonical_id].add(norm)
            # Register under a synthetic "direct" feed source for fast lookup
            self._feed_aliases[(raw.strip(), "direct")] = canonical_id
            self._feed_aliases[(norm, "direct")] = canonical_id

        logger.debug(
            "entity_registered",
            canonical_id=canonical_id,
            n_aliases=len(aliases),
        )

    def resolve_fuzzy(
        self, name: str, threshold: float = 0.6
    ) -> Optional[Tuple[str, float]]:
        """
        Attempt to fuzzy-match a name to a registered canonical entity.

        Uses token overlap + substring + initial-match heuristics; no external
        dependency on fuzzywuzzy or rapidfuzz.

        Matching strategy (in priority order):
          1. Exact normalised match → score 1.0
          2. Substring containment → score 0.85
          3. Jaccard token overlap (surname/given name tokens) → overlap ratio
          4. One token is a prefix initial of the other (e.g. "V" → "Viktor")
             and all other tokens match → boosted score

        Args:
            name: Raw name to match.
            threshold: Minimum similarity ratio (0-1) to consider a match.

        Returns:
            (canonical_id, score) tuple if a match found, else None.
        """
        norm = _normalise_player_name(name)
        if not norm:
            return None

        # Exact match shortcut
        exact = self._name_to_canonical.get(norm)
        if exact:
            return (exact, 1.0)

        best_canonical: Optional[str] = None
        best_score: float = 0.0
        tokens_query = set(norm.split("_"))

        for registered_norm, registered_canonical in self._name_to_canonical.items():
            tokens_reg = set(registered_norm.split("_"))
            if not tokens_query or not tokens_reg:
                continue

            score = 0.0

            # Substring containment
            if norm in registered_norm or registered_norm in norm:
                score = max(score, 0.85)

            # Jaccard token overlap
            intersection = tokens_query & tokens_reg
            union = tokens_query | tokens_reg
            if union:
                jaccard = len(intersection) / len(union)
                score = max(score, jaccard)

            # Initial/prefix matching: "v_axelsen" vs "viktor_axelsen"
            # Check if each query token is either in reg tokens or is a single-char
            # prefix (initial) of a reg token.
            if score < threshold:
                matched_tokens = 0
                total_tokens = len(tokens_query)
                for qt in tokens_query:
                    if qt in tokens_reg:
                        matched_tokens += 1
                    else:
                        # Check if qt is an initial of any registered token
                        for rt in tokens_reg:
                            if len(qt) == 1 and rt.startswith(qt):
                                matched_tokens += 1
                                break
                if total_tokens > 0:
                    initial_score = matched_tokens / total_tokens
                    # Require that at least one surname token fully matches
                    # (to avoid spurious matches on common initials)
                    full_match_exists = bool(intersection)
                    if full_match_exists and initial_score > 0.5:
                        score = max(score, initial_score * 0.9)

            if score > best_score:
                best_score = score
                best_canonical = registered_canonical

        if best_canonical is not None and best_score >= threshold:
            return (best_canonical, best_score)
        return None

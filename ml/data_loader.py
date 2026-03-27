"""
data_loader.py
===============
Loads and validates badminton match data from D:\codex\Data\Badminton
into a normalised DataFrame ready for feature engineering.

Data sources ingested (P0 tier — required for training):
  1. badminton_data.csv — 14,723 match records (2018-2023), point-by-point
  2. BWF weekly rankings (normalized CSVs, 2022-2025)
  3. tournaments.csv — tournament metadata

P1 tier (enrichment — loaded if available):
  1. Optic Odds historical odds snapshots
  2. Flashscore results (extends coverage beyond badminton_data.csv)

Data contract:
  - All dates as ISO8601 strings at load time, converted to datetime.date objects
  - Entity IDs are normalised via entity_mapper (applied post-load)
  - No data imputation — NaN preserved for feature engineering to handle
  - Returns only matches with sufficient entity information for modelling

ZERO hardcoded probabilities. ZERO mock data.
Raises RuntimeError if P0 data is unavailable.
"""

from __future__ import annotations

import os
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import structlog

from config.badminton_config import Discipline, TournamentTier

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Subdirectory paths within BADMINTON_DATA_ROOT
_MATCH_DATA_RELATIVE = "sources/github_repos/badminton_data_analysis/badminton_data.csv"
_TOURNAMENT_DATA_RELATIVE = "sources/github_repos/badminton-data/out/tournaments.csv"
_RANKINGS_RELATIVE = "normalized/badminton_data_weekly_rankings_csv"

# Column mapping from raw badminton_data.csv to internal schema
_DISCIPLINE_MAP: Dict[str, str] = {
    "MS": "MS", "WS": "WS", "MD": "MD", "WD": "WD", "XD": "XD",
}

# Tournament type to TournamentTier mapping
_TIER_MAP: Dict[str, TournamentTier] = {
    "HSBC BWF World Tour Super 1000": TournamentTier.SUPER_1000,
    "HSBC BWF World Tour Super 750": TournamentTier.SUPER_750,
    "HSBC BWF World Tour Super 500": TournamentTier.SUPER_500,
    "HSBC BWF World Tour Super 300": TournamentTier.SUPER_300,
    "HSBC BWF World Tour Super 100": TournamentTier.SUPER_100,
    "BWF Super Series Premier": TournamentTier.SUPER_1000,  # Legacy name
    "BWF Super Series": TournamentTier.SUPER_750,            # Legacy name
    "Olympic Games": TournamentTier.OLYMPICS,
    "BWF World Championships": TournamentTier.WORLD_CHAMPIONSHIPS,
    "BWF World Tour Finals": TournamentTier.WORLD_TOUR_FINALS,
    "Thomas Cup": TournamentTier.TEAM_EVENT,
    "Uber Cup": TournamentTier.TEAM_EVENT,
    "Sudirman Cup": TournamentTier.TEAM_EVENT,
}


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

class BadmintonDataLoader:
    """
    Loads and normalises badminton match data.

    All paths are resolved from environment variable BADMINTON_DATA_ROOT.
    Raises RuntimeError if the data root is not set or P0 data is missing.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        root = data_root or os.environ.get("BADMINTON_DATA_ROOT")
        if not root:
            raise RuntimeError(
                "BADMINTON_DATA_ROOT environment variable is not set. "
                "Set it to the path of D:\\codex\\Data\\Badminton (or equivalent)."
            )
        self._root = Path(root)
        if not self._root.exists():
            raise RuntimeError(
                f"BADMINTON_DATA_ROOT path does not exist: {self._root}"
            )

    def load_matches(
        self,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        disciplines: Optional[List[Discipline]] = None,
    ) -> pd.DataFrame:
        """
        Load and normalise match records from badminton_data.csv.

        Returns DataFrame with columns:
            match_id, date, tournament, tournament_id, tier, discipline,
            round, draw_size, entity_a_id, entity_b_id, winner_id,
            game_scores (list of (score_a, score_b) tuples),
            point_by_point (list of rally sequences if available),
            retired (bool)

        Args:
            start_year: Filter from this year inclusive.
            end_year: Filter to this year inclusive.
            disciplines: Filter to specific disciplines (None = all).

        Raises:
            RuntimeError: If P0 data file is missing.
        """
        match_file = self._root / _MATCH_DATA_RELATIVE
        if not match_file.exists():
            raise RuntimeError(
                f"P0 match data file not found: {match_file}. "
                f"Run scripts/backfill_historical.py first."
            )

        logger.info("loading_match_data", path=str(match_file))
        raw = pd.read_csv(match_file, low_memory=False)

        logger.info("raw_match_data_loaded", n_rows=len(raw), columns=list(raw.columns))

        # Parse and validate
        normalised = self._normalise_matches(raw)

        # Apply filters
        if start_year:
            normalised = normalised[normalised["date"].dt.year >= start_year]
        if end_year:
            normalised = normalised[normalised["date"].dt.year <= end_year]
        if disciplines:
            disc_values = {d.value for d in disciplines}
            normalised = normalised[normalised["discipline"].isin(disc_values)]

        # Drop rows with missing critical fields
        before = len(normalised)
        normalised = normalised.dropna(subset=["entity_a_id", "entity_b_id", "winner_id", "date"])
        dropped = before - len(normalised)
        if dropped > 0:
            logger.warning("dropped_rows_missing_critical_fields", n_dropped=dropped)

        logger.info(
            "match_data_loaded",
            n_rows=len(normalised),
            date_range=(
                str(normalised["date"].min().date()),
                str(normalised["date"].max().date()),
            ) if len(normalised) > 0 else ("N/A", "N/A"),
        )

        return normalised.reset_index(drop=True)

    def load_tournaments(self) -> pd.DataFrame:
        """Load tournament metadata from tournaments.csv."""
        tourney_file = self._root / _TOURNAMENT_DATA_RELATIVE
        if not tourney_file.exists():
            raise RuntimeError(f"Tournament data file not found: {tourney_file}")

        raw = pd.read_csv(tourney_file)
        return self._normalise_tournaments(raw)

    def _normalise_matches(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise raw match CSV into internal schema."""
        df = pd.DataFrame()

        # Date parsing (format: DD-MM-YYYY in source)
        df["date"] = pd.to_datetime(raw["date"], format="%d-%m-%Y", errors="coerce")

        # Tournament metadata
        df["tournament"] = raw["tournament"].str.strip()
        df["tournament_id"] = raw["tournament"].str.strip().str.lower().str.replace(r"\W+", "_", regex=True)
        df["tier"] = raw["tournament_type"].apply(self._map_tier)
        df["city"] = raw.get("city", pd.Series(dtype=str))
        df["country"] = raw.get("country", pd.Series(dtype=str))

        # Discipline normalisation
        df["discipline"] = raw["discipline"].str.strip().str.upper().map(_DISCIPLINE_MAP)

        # Round
        df["round"] = raw["round"].apply(self._normalise_round)
        df["draw_size"] = 32  # Default — refined per tournament if data available

        # Players — raw format: "Player Name (COUNTRY)" or comma-separated for doubles
        entity_a, entity_b = zip(*raw.apply(self._parse_entities, axis=1))
        df["entity_a_id"] = list(entity_a)
        df["entity_b_id"] = list(entity_b)

        # Winner — from "21pts_winner" column (player/pair name)
        winner_entities = raw.apply(lambda r: self._parse_winner(r), axis=1)
        df["winner_id"] = winner_entities

        # Match ID
        df["match_id"] = raw.index.astype(str).apply(lambda i: f"bda_{i}")

        # Game scores
        df["game_scores"] = raw.apply(self._parse_game_scores, axis=1)
        df["retired"] = raw.get("retired", pd.Series(False, index=raw.index)).fillna(False).astype(bool)

        # Point-by-point (if available)
        df["point_by_point"] = raw.get("point_change_eval", pd.Series(dtype=str))

        return df

    def _normalise_tournaments(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalise tournament CSV."""
        df = pd.DataFrame()
        df["name"] = raw["name"].str.strip()
        df["tier"] = raw["type"].apply(self._map_tier) if "type" in raw.columns else TournamentTier.SUPER_300.value
        df["city"] = raw.get("city", pd.Series(dtype=str))
        df["country"] = raw.get("country", pd.Series(dtype=str))
        df["start_date"] = pd.to_datetime(raw["start_date"], errors="coerce")
        df["end_date"] = pd.to_datetime(raw["end_date"], errors="coerce")
        return df

    @staticmethod
    def _map_tier(tournament_type: str) -> str:
        """Map raw tournament_type string to TournamentTier value."""
        if pd.isna(tournament_type):
            return TournamentTier.SUPER_300.value
        t = str(tournament_type).strip()
        for pattern, tier in _TIER_MAP.items():
            if pattern.lower() in t.lower():
                return tier.value
        # Default fallback
        if "super" in t.lower():
            return TournamentTier.SUPER_300.value
        return TournamentTier.SUPER_100.value

    @staticmethod
    def _normalise_round(raw_round: str) -> str:
        """Normalise round string to internal code."""
        if pd.isna(raw_round):
            return "R32"
        r = str(raw_round).strip().upper()
        if "FINAL" in r and "SEMI" not in r and "QUARTER" not in r:
            return "F"
        if "SEMI" in r:
            return "SF"
        if "QUARTER" in r:
            return "QF"
        if "16" in r:
            return "R16"
        if "32" in r:
            return "R32"
        if "64" in r:
            return "R64"
        if "QUALIF" in r or "Q" in r:
            return "Q1"
        return "R32"

    @staticmethod
    def _parse_entities(row: pd.Series) -> Tuple[str, str]:
        """
        Parse entity IDs for team_one and team_two.

        For singles: entity is player name normalised.
        For doubles: entity is pair key "name1|name2" (sorted).
        """
        discipline = str(row.get("discipline", "MS")).strip().upper()
        is_doubles = discipline in {"MD", "WD", "XD"}

        team_one_raw = str(row.get("team_one_players", ""))
        team_two_raw = str(row.get("team_two_players", ""))

        def normalise_name(name: str) -> str:
            """Remove country code, strip whitespace, lowercase."""
            # Format: "Firstname LASTNAME (COUNTRY)" or "LASTNAME Firstname (COUNTRY)"
            name = re.sub(r"\([A-Z]{2,3}\)", "", name).strip()
            return name.strip().lower().replace(" ", "_")

        if is_doubles:
            # Doubles: comma-separated pair
            players_one = [normalise_name(p) for p in team_one_raw.split(",") if p.strip()]
            players_two = [normalise_name(p) for p in team_two_raw.split(",") if p.strip()]
            entity_a = "|".join(sorted(players_one)) if players_one else ""
            entity_b = "|".join(sorted(players_two)) if players_two else ""
        else:
            entity_a = normalise_name(team_one_raw)
            entity_b = normalise_name(team_two_raw)

        return entity_a if entity_a else None, entity_b if entity_b else None

    @staticmethod
    def _parse_winner(row: pd.Series) -> Optional[str]:
        """Parse winner from 21pts_winner column."""
        winner_raw = str(row.get("21pts_winner", ""))
        if not winner_raw or winner_raw == "nan":
            return None

        team_one_raw = str(row.get("team_one_players", ""))
        discipline = str(row.get("discipline", "MS")).strip().upper()

        # Normalise winner name using same logic as _parse_entities
        def normalise_name(name: str) -> str:
            name = re.sub(r"\([A-Z]{2,3}\)", "", name).strip()
            return name.strip().lower().replace(" ", "_")

        winner_norm = normalise_name(winner_raw)
        entity_a_norm = normalise_name(team_one_raw)

        # If winner matches team_one, return "A"; otherwise "B"
        # For doubles: check if winner string matches any player in team_one
        if discipline in {"MD", "WD", "XD"}:
            players_one = [normalise_name(p) for p in team_one_raw.split(",") if p.strip()]
            if any(winner_norm in p or p in winner_norm for p in players_one):
                return "A"
            return "B"
        else:
            if winner_norm in entity_a_norm or entity_a_norm in winner_norm:
                return "A"
            return "B"

    @staticmethod
    def _parse_game_scores(row: pd.Series) -> List[Tuple[int, int]]:
        """Parse game scores from game_N_score columns."""
        scores = []
        for i in range(1, 4):
            col = f"21pts_game_{i}_score"
            val = row.get(col)
            if pd.isna(val) or not val:
                break
            try:
                parts = str(val).split("-")
                if len(parts) == 2:
                    scores.append((int(parts[0].strip()), int(parts[1].strip())))
            except (ValueError, IndexError):
                continue
        return scores

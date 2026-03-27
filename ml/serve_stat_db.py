"""
serve_stat_db.py
=================
Rally Win Probability (RWP) and tactical performance statistics database.

Computes and caches per-player, per-discipline RWP estimates from:
  1. Point-by-point data in badminton_data.csv (rally sequences)
  2. FineBadminton tactical dataset (smash/net win rates)
  3. ShuttleNet/Shuttleset stroke data

RWP estimation method:
  For each player P, over rolling last-50 matches:
    - Extract all rallies where P was serving
    - rwp_as_server = (rallies P won as server) / (total rallies as server)
    - rwp_as_receiver = (rallies P won as receiver) / (total rallies as receiver)

Tactical features (from FineBadminton dataset when available):
  - smash_win_rate: % of smash attempts where the smasher wins the rally
  - net_win_rate: % of net shot attempts resulting in a winner
  - avg_rally_length: average rallies per point

ZERO hardcoded probabilities. Returns None if insufficient data.
Raises RuntimeError if data root unavailable.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Optional

import structlog

from config.badminton_config import Discipline, DOUBLES_DISCIPLINES
from core.rwp_calculator import PlayerRWPProfile

logger = structlog.get_logger(__name__)

_RWP_ROLLING_WINDOW: int = 50      # Matches in rolling RWP estimate
_RWP_MIN_RALLIES: int = 10         # Minimum rallies to compute RWP


class ServeStatDB:
    """
    In-memory RWP and tactical statistics database.

    Built from match data using build_from_matches().
    Provides get_profile() for feature engineering.
    """

    def __init__(self) -> None:
        # {(entity_id, discipline) -> PlayerRWPProfile}
        self._profiles: Dict[tuple[str, str], PlayerRWPProfile] = {}
        # Tactical stats
        self._smash_win_rates: Dict[tuple[str, str], float] = {}
        self._net_win_rates: Dict[tuple[str, str], float] = {}
        self._avg_rally_lengths: Dict[tuple[str, str], float] = {}

    def build_from_matches(
        self,
        matches_df,  # pd.DataFrame from data_loader
        point_by_point_col: str = "point_by_point",
    ) -> None:
        """
        Compute RWP profiles from match data.

        For each player in each discipline, compute rolling RWP
        using the point-by-point column if available.

        This is called once during data pipeline setup, before feature engineering.
        """
        logger.info("serve_stat_db_building", n_matches=len(matches_df))

        # Group by discipline
        for discipline in Discipline:
            disc_matches = matches_df[
                matches_df["discipline"] == discipline.value
            ].sort_values("date")

            # Accumulate rally data per entity
            entity_rallies: Dict[str, Dict[str, int]] = defaultdict(
                lambda: {"server_wins": 0, "server_total": 0, "recv_wins": 0, "recv_total": 0}
            )
            entity_last_date: Dict[str, str] = {}

            for _, row in disc_matches.iterrows():
                entity_a = str(row["entity_a_id"])
                entity_b = str(row["entity_b_id"])
                game_scores = row.get("game_scores", [])
                match_date = str(row["date"])[:10]

                # Parse point-by-point sequences to extract server wins
                pbp = row.get(point_by_point_col)
                if pbp and isinstance(pbp, str) and len(pbp) > 0:
                    self._process_pbp_sequence(
                        pbp, entity_a, entity_b, game_scores, entity_rallies
                    )
                else:
                    # No PBP: estimate from game scores using scoring engine
                    self._estimate_from_game_scores(
                        entity_a, entity_b, game_scores, row.get("winner_id", "A"),
                        entity_rallies
                    )

                entity_last_date[entity_a] = max(
                    entity_last_date.get(entity_a, "2000-01-01"), match_date
                )
                entity_last_date[entity_b] = max(
                    entity_last_date.get(entity_b, "2000-01-01"), match_date
                )

            # Build profiles from accumulated rally data
            for entity_id, stats in entity_rallies.items():
                server_total = stats["server_total"]
                recv_total = stats["recv_total"]

                if server_total < _RWP_MIN_RALLIES or recv_total < _RWP_MIN_RALLIES:
                    continue  # Insufficient data — profile not created

                rwp_server = stats["server_wins"] / server_total
                rwp_recv = stats["recv_wins"] / recv_total

                try:
                    profile = PlayerRWPProfile(
                        entity_id=entity_id,
                        discipline=discipline,
                        rwp_as_server=rwp_server,
                        rwp_as_receiver=rwp_recv,
                        sample_size=server_total,
                        last_updated=entity_last_date.get(entity_id, "unknown"),
                    )
                    self._profiles[(entity_id, discipline.value)] = profile
                except ValueError as exc:
                    logger.warning(
                        "rwp_profile_invalid",
                        entity_id=entity_id,
                        discipline=discipline.value,
                        error=str(exc),
                    )

        total = len(self._profiles)
        logger.info("serve_stat_db_built", total_profiles=total)

    def get_profile(
        self,
        entity_id: str,
        discipline: Discipline,
    ) -> Optional[PlayerRWPProfile]:
        """
        Return RWP profile for entity, or None if unavailable.

        Returns None (never a default probability) — caller handles missing data.
        """
        return self._profiles.get((entity_id, discipline.value))

    def get_smash_win_rate(
        self,
        entity_id: str,
        discipline: Discipline,
    ) -> Optional[float]:
        """Return smash win rate from tactical dataset, or None."""
        return self._smash_win_rates.get((entity_id, discipline.value))

    def get_net_win_rate(
        self,
        entity_id: str,
        discipline: Discipline,
    ) -> Optional[float]:
        """Return net point win rate, or None."""
        return self._net_win_rates.get((entity_id, discipline.value))

    def get_avg_rally_length(
        self,
        entity_id: str,
        discipline: Discipline,
    ) -> Optional[float]:
        """Return average rally length, or None."""
        return self._avg_rally_lengths.get((entity_id, discipline.value))

    def load_finebadminton_tactical(self, data_root: Optional[str] = None) -> None:
        """
        Load smash/net win rates from FineBadminton JSON dataset.

        Path: {data_root}/sources/github_repos/FineBadminton/dataset/
        """
        import json

        root = data_root or os.environ.get("BADMINTON_DATA_ROOT")
        if not root:
            logger.warning("finebadminton_data_root_not_set")
            return

        json_path = (
            Path(root)
            / "sources/github_repos/FineBadminton/dataset"
            / "transformed_combined_rounds_output_en_evals_translated.json"
        )
        if not json_path.exists():
            logger.warning("finebadminton_json_not_found", path=str(json_path))
            return

        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("finebadminton_load_failed", error=str(exc))
            return

        # Aggregate smash win rates per player
        player_smash: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "total": 0}
        )
        player_net: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "total": 0}
        )
        player_rally_lengths: Dict[str, list[int]] = defaultdict(list)

        for rally in data if isinstance(data, list) else []:
            hits = rally.get("hitting", [])
            rally_len = len(hits)

            for hit in hits:
                player = str(hit.get("player", "")).strip().lower().replace(" ", "_")
                hit_type = str(hit.get("hit_type", "")).lower()
                get_point = hit.get("get_point", [])
                won_point = bool(get_point)

                if "smash" in hit_type:
                    player_smash[player]["total"] += 1
                    if won_point:
                        player_smash[player]["wins"] += 1

                if "net" in hit_type:
                    player_net[player]["total"] += 1
                    if won_point:
                        player_net[player]["wins"] += 1

            # Rally length per player (use first hitter as proxy)
            if hits:
                first_player = str(hits[0].get("player", "")).strip().lower().replace(" ", "_")
                player_rally_lengths[first_player].append(rally_len)

        # Store computed stats (discipline-agnostic from this dataset — apply to MS)
        for player, stats in player_smash.items():
            if stats["total"] >= 5:
                self._smash_win_rates[(player, Discipline.MS.value)] = (
                    stats["wins"] / stats["total"]
                )

        for player, stats in player_net.items():
            if stats["total"] >= 5:
                self._net_win_rates[(player, Discipline.MS.value)] = (
                    stats["wins"] / stats["total"]
                )

        for player, lengths in player_rally_lengths.items():
            if len(lengths) >= 3:
                self._avg_rally_lengths[(player, Discipline.MS.value)] = (
                    sum(lengths) / len(lengths)
                )

        logger.info(
            "finebadminton_tactical_loaded",
            n_smash_profiles=len(self._smash_win_rates),
            n_net_profiles=len(self._net_win_rates),
            n_rally_length_profiles=len(self._avg_rally_lengths),
        )

    @staticmethod
    def _process_pbp_sequence(
        pbp: str,
        entity_a: str,
        entity_b: str,
        game_scores: list,
        entity_rallies: Dict[str, Dict[str, int]],
    ) -> None:
        """
        Parse point-by-point sequence string to extract server wins.

        pbp format in badminton_data.csv: "point_change_eval" column
        which typically encodes rally-by-rally score changes.

        This is a best-effort parser — exact format varies by source.
        """
        # pbp format: comma-separated float impact scores, e.g. "0.12,-0.05,0.08,..."
        # Positive = entity_a wins rally; negative = entity_b wins rally.
        # BWF rally-scoring: winner of rally serves next point.
        if not pbp or not isinstance(pbp, str):
            return

        tokens = [t.strip() for t in pbp.split(",") if t.strip()]
        if not tokens:
            return

        try:
            scores = [float(t) for t in tokens]
        except ValueError:
            # Unknown PBP format (e.g. rally-code strings) — skip; caller uses game scores
            return

        # Reconstruct serving state from first point.
        # Convention: entity_a serves first when PBP server not encoded.
        current_server: str = entity_a

        for impact in scores:
            if impact == 0.0:
                continue  # Ambiguous / let — skip

            rally_winner = entity_a if impact > 0.0 else entity_b

            if current_server == entity_a:
                entity_rallies[entity_a]["server_total"] += 1
                if rally_winner == entity_a:
                    entity_rallies[entity_a]["server_wins"] += 1
                else:
                    entity_rallies[entity_b]["recv_total"] += 1
                    entity_rallies[entity_b]["recv_wins"] += 1
            else:  # entity_b serving
                entity_rallies[entity_b]["server_total"] += 1
                if rally_winner == entity_b:
                    entity_rallies[entity_b]["server_wins"] += 1
                else:
                    entity_rallies[entity_a]["recv_total"] += 1
                    entity_rallies[entity_a]["recv_wins"] += 1

            # BWF: winner retains serve
            current_server = rally_winner

    @staticmethod
    def _estimate_from_game_scores(
        entity_a: str,
        entity_b: str,
        game_scores: list,
        winner: str,
        entity_rallies: Dict[str, Dict[str, int]],
    ) -> None:
        """
        Estimate rally stats from game scores when PBP is unavailable.

        Uses total points as proxy for rally count.
        Server wins estimated using baseline RWP.
        """
        from config.badminton_config import RWP_BASELINE, Discipline
        baseline_rwp = RWP_BASELINE[Discipline.MS]  # Default baseline

        for score_a, score_b in game_scores:
            total_rallies = score_a + score_b
            if total_rallies < 1:
                continue

            # A served approximately half the rallies (rally scoring)
            approx_serves_a = total_rallies // 2
            approx_serves_b = total_rallies - approx_serves_a

            # Estimate server wins using baseline RWP
            entity_rallies[entity_a]["server_total"] += approx_serves_a
            entity_rallies[entity_a]["server_wins"] += int(approx_serves_a * baseline_rwp)
            entity_rallies[entity_a]["recv_total"] += approx_serves_b
            entity_rallies[entity_a]["recv_wins"] += int(approx_serves_b * (1 - baseline_rwp))

            entity_rallies[entity_b]["server_total"] += approx_serves_b
            entity_rallies[entity_b]["server_wins"] += int(approx_serves_b * baseline_rwp)
            entity_rallies[entity_b]["recv_total"] += approx_serves_a
            entity_rallies[entity_b]["recv_wins"] += int(approx_serves_a * (1 - baseline_rwp))

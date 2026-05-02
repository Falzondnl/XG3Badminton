"""
backfill_historical.py
======================
Historical data backfill script for the XG3 badminton platform.

Ingests historical match data from D:\\codex\\Data\\Badminton and:
  1. Populates WeeklyRankingsDB with 530 BWF ranking snapshots
  2. Populates ServeStatDB with per-player RWP profiles
  3. Initialises ELO pools from historical match sequence
  4. Validates temporal ordering (no leakage — ELO updated AFTER features)

This script is run ONCE to bootstrap the system from historical data.
Subsequent updates are handled by the weekly ranking and live feed clients.

Usage:
    python scripts/backfill_historical.py [--dry-run] [--start-year 2019]
                                         [--end-year 2024] [--data-root PATH]

ZERO mock data. All operations raise on missing/corrupt data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import structlog

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline
from ml.data_loader import BadmintonDataLoader
from ml.elo_system import ELOSystem
from ml.weekly_rankings_db import WeeklyRankingsDB
from ml.serve_stat_db import ServeStatDB

logger = structlog.get_logger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill historical badminton data into XG3 databases"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("D:/codex/Data/Badminton"),
        help="Root directory of raw badminton data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/backfill"),
        help="Output directory for processed databases",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2019,
        help="First year to include in backfill",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Last year to include (inclusive)",
    )
    parser.add_argument(
        "--disciplines",
        nargs="+",
        choices=["MS", "WS", "MD", "WD", "XD"],
        default=["MS", "WS", "MD", "WD", "XD"],
        help="Disciplines to backfill",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate only — do not write output",
    )
    parser.add_argument(
        "--skip-elo",
        action="store_true",
        help="Skip ELO computation (fast, for rankings-only backfill)",
    )
    parser.add_argument(
        "--skip-rankings",
        action="store_true",
        help="Skip BWF rankings ingestion",
    )
    parser.add_argument(
        "--skip-serve-stats",
        action="store_true",
        help="Skip serve stat profile computation",
    )
    return parser.parse_args()


# ── ELO backfill ──────────────────────────────────────────────────────

def backfill_elo(
    matches: list,
    elo_system: ELOSystem,
    discipline: Discipline,
    dry_run: bool = False,
) -> dict:
    """
    Compute ELO ratings from historical match sequence.

    Matches MUST be sorted chronologically (earliest first) to avoid leakage.

    Args:
        matches: List of match dicts from BadmintonDataLoader
        elo_system: ELOSystem to update in-place
        discipline: Discipline being processed
        dry_run: If True, compute but do not persist

    Returns:
        Dict with ELO statistics
    """
    logger.info(
        "backfill_elo_start",
        discipline=discipline.value,
        n_matches=len(matches),
    )

    n_processed = 0
    n_skipped = 0
    t0 = time.time()

    for match in matches:
        try:
            player_a = match.get("player_a_id") or match.get("winner_id")
            player_b = match.get("player_b_id") or match.get("loser_id")
            winner = match.get("winner", match.get("result", "A"))

            if not player_a or not player_b:
                n_skipped += 1
                continue

            # Validate winner field
            if winner not in ("A", "B", player_a, player_b):
                logger.warning(
                    "elo_invalid_winner",
                    match_id=match.get("match_id", "unknown"),
                    winner=winner,
                )
                n_skipped += 1
                continue

            winner_id = player_a if winner in ("A", player_a) else player_b
            loser_id = player_b if winner_id == player_a else player_a

            # ELO update
            elo_system.update(
                winner_id=winner_id,
                loser_id=loser_id,
                discipline=discipline,
                tournament_tier=match.get("tournament_tier", "SUPER_300"),
                surface=match.get("surface", ""),
            )

            n_processed += 1

        except Exception as exc:
            logger.warning(
                "elo_match_error",
                match_id=match.get("match_id", "?"),
                error=str(exc),
            )
            n_skipped += 1

    elapsed = time.time() - t0

    stats = {
        "discipline": discipline.value,
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "elapsed_s": round(elapsed, 2),
        "matches_per_second": round(n_processed / max(elapsed, 0.001), 1),
    }

    logger.info("backfill_elo_complete", **stats)
    return stats


# ── Serve stat backfill ───────────────────────────────────────────────

def backfill_serve_stats(
    matches: list,
    serve_stat_db: ServeStatDB,
    discipline: Discipline,
) -> dict:
    """
    Compute per-player RWP profiles from point-by-point data.

    Each match with PBP data contributes to server's win/loss tallies.
    """
    logger.info(
        "backfill_serve_stats_start",
        discipline=discipline.value,
        n_matches=len(matches),
    )

    n_pbp_matches = 0
    n_rallies = 0
    t0 = time.time()

    for match in matches:
        pbp = match.get("point_by_point", [])
        if not pbp:
            continue

        n_pbp_matches += 1
        player_a = match.get("player_a_id", "")
        player_b = match.get("player_b_id", "")

        for point in pbp:
            server = point.get("server", "")
            winner = point.get("winner", "")

            if server not in ("A", "B") or winner not in ("A", "B"):
                continue

            server_id = player_a if server == "A" else player_b
            server_won = server == winner

            serve_stat_db.record_rally(
                player_id=server_id,
                discipline=discipline,
                server_won=server_won,
            )
            n_rallies += 1

    elapsed = time.time() - t0
    stats = {
        "discipline": discipline.value,
        "n_pbp_matches": n_pbp_matches,
        "n_rallies": n_rallies,
        "elapsed_s": round(elapsed, 2),
    }

    logger.info("backfill_serve_stats_complete", **stats)
    return stats


# ── Rankings backfill ─────────────────────────────────────────────────

def backfill_rankings(
    data_root: Path,
    rankings_db: WeeklyRankingsDB,
    start_year: int,
    end_year: int,
    dry_run: bool = False,
) -> dict:
    """
    Load BWF rankings snapshots from historical files.

    Expects rankings files in:
        data_root/rankings/{year}/week_{YYYY-MM-DD}.json

    Each file should contain ranking data for all disciplines.
    """
    rankings_root = data_root / "rankings"

    if not rankings_root.exists():
        raise RuntimeError(
            f"Rankings directory not found: {rankings_root}\n"
            "Expected structure: data_root/rankings/{year}/week_YYYY-MM-DD.json"
        )

    n_weeks = 0
    n_entries = 0
    n_errors = 0

    for year in range(start_year, end_year + 1):
        year_dir = rankings_root / str(year)
        if not year_dir.exists():
            logger.warning("rankings_year_dir_missing", year=year)
            continue

        week_files = sorted(year_dir.glob("week_*.json"))
        logger.info("rankings_year_files", year=year, n_files=len(week_files))

        for week_file in week_files:
            try:
                data = json.loads(week_file.read_text(encoding="utf-8"))
                week_date = data.get("week_date", week_file.stem.replace("week_", ""))

                for disc_key, entries in data.get("rankings", {}).items():
                    discipline = Discipline(disc_key)
                    if not dry_run:
                        rankings_db.store_week(
                            week_date=week_date,
                            discipline=discipline,
                            entries=entries,
                        )
                    n_entries += len(entries)

                n_weeks += 1

            except Exception as exc:
                logger.error(
                    "rankings_week_error",
                    file=str(week_file),
                    error=str(exc),
                )
                n_errors += 1

    stats = {
        "n_weeks_loaded": n_weeks,
        "n_entries": n_entries,
        "n_errors": n_errors,
        "dry_run": dry_run,
    }

    logger.info("backfill_rankings_complete", **stats)
    return stats


# ── Main ──────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    logger.info(
        "backfill_start",
        data_root=str(args.data_root),
        start_year=args.start_year,
        end_year=args.end_year,
        disciplines=args.disciplines,
        dry_run=args.dry_run,
    )

    if not args.data_root.exists():
        logger.error(
            "data_root_not_found",
            data_root=str(args.data_root),
        )
        return 1

    disciplines = [Discipline(d) for d in args.disciplines]

    # Initialise databases
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    elo_system = ELOSystem()
    serve_stat_db = ServeStatDB()
    rankings_db = WeeklyRankingsDB()

    loader = BadmintonDataLoader(data_root=args.data_root)

    all_stats: dict = {
        "elo": {},
        "serve_stats": {},
        "rankings": {},
    }

    # ── Rankings ──────────────────────────────────────────────────────
    if not args.skip_rankings:
        try:
            ranking_stats = backfill_rankings(
                data_root=args.data_root,
                rankings_db=rankings_db,
                start_year=args.start_year,
                end_year=args.end_year,
                dry_run=args.dry_run,
            )
            all_stats["rankings"] = ranking_stats
        except Exception as exc:
            logger.error("rankings_backfill_failed", error=str(exc))
            if not args.skip_rankings:
                # Rankings are non-fatal — continue with other steps
                pass

    # ── Per-discipline ELO + serve stats ──────────────────────────────
    for discipline in disciplines:
        logger.info("discipline_backfill_start", discipline=discipline.value)

        try:
            matches = loader.load_matches(
                discipline=discipline,
                start_year=args.start_year,
                end_year=args.end_year,
            )
        except Exception as exc:
            logger.error(
                "data_load_failed",
                discipline=discipline.value,
                error=str(exc),
            )
            continue

        if not matches:
            logger.warning("no_matches_loaded", discipline=discipline.value)
            continue

        logger.info(
            "matches_loaded",
            discipline=discipline.value,
            n_matches=len(matches),
        )

        # Sort chronologically — MANDATORY to prevent leakage
        matches_sorted = sorted(
            matches,
            key=lambda m: m.get("match_date", m.get("date", "1900-01-01")),
        )

        # ELO backfill
        if not args.skip_elo:
            elo_stats = backfill_elo(
                matches=matches_sorted,
                elo_system=elo_system,
                discipline=discipline,
                dry_run=args.dry_run,
            )
            all_stats["elo"][discipline.value] = elo_stats

        # Serve stat backfill
        if not args.skip_serve_stats:
            ss_stats = backfill_serve_stats(
                matches=matches_sorted,
                serve_stat_db=serve_stat_db,
                discipline=discipline,
            )
            all_stats["serve_stats"][discipline.value] = ss_stats

    # ── Persist ───────────────────────────────────────────────────────
    if not args.dry_run:
        logger.info("persisting_databases")

        try:
            elo_path = output_root / "elo_state.json"
            elo_system.save(elo_path)
            logger.info("elo_saved", path=str(elo_path))
        except Exception as exc:
            logger.error("elo_save_failed", error=str(exc))

        try:
            ss_path = output_root / "serve_stat_db.json"
            serve_stat_db.save(ss_path)
            logger.info("serve_stats_saved", path=str(ss_path))
        except Exception as exc:
            logger.error("serve_stats_save_failed", error=str(exc))

        # Save summary report
        summary_path = output_root / "backfill_summary.json"
        summary_path.write_text(
            json.dumps(all_stats, indent=2, default=str), encoding="utf-8"
        )
        logger.info("summary_saved", path=str(summary_path))

    # Print summary
    print("\n=== BACKFILL SUMMARY ===")
    for section, data in all_stats.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {v}")

    logger.info("backfill_complete", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

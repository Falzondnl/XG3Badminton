"""
config/__init__.py — re-export all constants from the flat config module.

The config/ package exists for the advanced domain config (badminton_config.py).
This __init__ re-exports the flat constants needed by ml/ and pricing/ modules
so that `from config import MS_CSV` resolves correctly regardless of import path.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Raw data paths
# --------------------------------------------------------------------------- #
BWF_DATA_DIR = "D:/codex/Data/Badminton/raw/kaggle/sanderp_bwf_world_tour/"
MS_CSV = BWF_DATA_DIR + "ms.csv"
WS_CSV = BWF_DATA_DIR + "ws.csv"
MD_CSV = BWF_DATA_DIR + "md.csv"
XD_CSV = BWF_DATA_DIR + "xd.csv"
WD_CSV = BWF_DATA_DIR + "wd.csv"

# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
PORT = 8034

# --------------------------------------------------------------------------- #
# Model storage directories
# --------------------------------------------------------------------------- #
R0_DIR = "models/r0"   # Men's Singles
R1_DIR = "models/r1"   # Women's Singles
R2_DIR = "models/r2"   # Doubles (MD + XD + WD combined)

# --------------------------------------------------------------------------- #
# ELO settings
# --------------------------------------------------------------------------- #
ELO_K = 32
ELO_DEFAULT = 1500.0

# --------------------------------------------------------------------------- #
# Disciplines (plain strings for ML — not enums, for serialisation simplicity)
# --------------------------------------------------------------------------- #
SINGLES_DISCIPLINES = ("MS", "WS")
DOUBLES_DISCIPLINES = ("MD", "XD", "WD")
ALL_DISCIPLINES = SINGLES_DISCIPLINES + DOUBLES_DISCIPLINES

# --------------------------------------------------------------------------- #
# Tournament tier encoding
# --------------------------------------------------------------------------- #
TOURNAMENT_TIER: dict[str, int] = {
    "BWF Tour Super 100": 1,
    "HSBC BWF World Tour Super 300": 2,
    "HSBC BWF World Tour Super 500": 3,
    "HSBC BWF World Tour Super 750": 4,
    "HSBC BWF World Tour Super 1000": 5,
    "HSBC BWF World Tour Finals": 6,
}

# --------------------------------------------------------------------------- #
# Round encoding
# --------------------------------------------------------------------------- #
ROUND_ENCODING: dict[str, int] = {
    "Qualification round of 32": 1,
    "Qualification round of 16": 2,
    "Qualification quarter final": 3,
    "Round of 64": 4,
    "Round of 32": 5,
    "Round 1": 5,
    "Round of 16": 6,
    "Round 2": 6,
    "Quarter final": 7,
    "Round 3": 7,
    "Semi final": 8,
    "Final": 9,
}

# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
MATCH_WINNER_MARGIN = 0.05   # 5% — tight Asian badminton market

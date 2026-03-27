"""
BadmintonFeatureExtractor — builds per-match feature vectors.

Design:
- Separate ELO pools per discipline (MS, WS, MD, XD, WD)
- Rolling win-rate (last 5 / last 10 matches) per player/pair
- Head-to-head win rate + match count
- Tournament tier encoding  (1–6)
- Round encoding            (1–9)
- Nationality advantage flag (home country bonus)
- Career matches played + career win rate
- Doubles partnership win rate (MD/XD/WD only)

Training-time: 50% random P1/P2 swap to prevent positional bias.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import (
    ALL_DISCIPLINES,
    DOUBLES_DISCIPLINES,
    ELO_DEFAULT,
    ELO_K,
    ROUND_ENCODING,
    TOURNAMENT_TIER,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Picklable default factories (lambdas cannot be pickled)
# --------------------------------------------------------------------------- #

def _elo_default() -> float:
    return ELO_DEFAULT


def _career_default() -> list:
    return [0, 0]


def _partnership_default() -> list:
    return [0, 0]


# --------------------------------------------------------------------------- #
# ELO helper
# --------------------------------------------------------------------------- #

def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def _update_elo(ra: float, rb: float, score: float) -> tuple[float, float]:
    """Return updated (ra, rb) after a match where score=1 means A wins."""
    ea = _expected(ra, rb)
    return ra + ELO_K * (score - ea), rb + ELO_K * ((1.0 - score) - (1.0 - ea))


# --------------------------------------------------------------------------- #
# Feature names
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "elo_p1",
    "elo_p2",
    "elo_diff",            # p1_elo - p2_elo
    "win_rate_5_p1",
    "win_rate_5_p2",
    "win_rate_10_p1",
    "win_rate_10_p2",
    "h2h_win_rate_p1",
    "h2h_matches",
    "tournament_tier",
    "round_enc",
    "home_advantage_p1",   # 1 if p1 same nationality as host country
    "home_advantage_p2",   # 1 if p2 same nationality as host country
    "career_matches_p1",
    "career_matches_p2",
    "career_win_rate_p1",
    "career_win_rate_p2",
    "partnership_win_rate_p1",  # doubles only (0 for singles)
    "partnership_matches_p1",   # doubles only (0 for singles)
    "discipline_enc",           # 0=MS,1=WS,2=MD,3=XD,4=WD
]

DISCIPLINE_ENC = {"MS": 0, "WS": 1, "MD": 2, "XD": 3, "WD": 4}


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class BadmintonFeatureExtractor:
    """
    Stateful extractor that iterates matches chronologically,
    updating ELO/rolling stats after each match.
    """

    def __init__(self) -> None:
        # ELO per discipline per player-key (use named default factory — lambdas not picklable)
        self._elo: dict[str, dict[str, float]] = {
            d: defaultdict(_elo_default) for d in ALL_DISCIPLINES
        }
        # rolling history  key → list[(date, outcome)]  outcome=1=win
        self._history: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
        # h2h: frozenset(key1, key2) → [outcomes from p1 perspective where p1 < p2 lexically]
        self._h2h: dict[frozenset, list[int]] = defaultdict(list)
        # career totals: key → [total_matches, total_wins]
        self._career: dict[str, list[int]] = defaultdict(_career_default)
        # partnership stats: pair_key → [total, wins]
        self._partnership: dict[str, list[int]] = defaultdict(_partnership_default)

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def extract_training_dataset(
        self,
        df: pd.DataFrame,
        discipline: str,
        apply_swap: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Process df sorted by date; return (X, y).
        apply_swap: randomly swap P1/P2 50% of time to prevent positional bias.
        """
        # Reset state for clean training run per discipline
        self._reset_discipline(discipline)

        rows: list[np.ndarray] = []
        targets: list[int] = []

        df = df.copy()
        df["_date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df = df.dropna(subset=["_date"])
        df = df.sort_values("_date").reset_index(drop=True)

        rng = np.random.default_rng(seed=42)

        for _, row in df.iterrows():
            winner_raw = int(row["winner"])
            # winner=0 → void/no result → skip
            if winner_raw == 0:
                continue
            # winner=1 → team_one wins; winner=2 → team_two wins
            team_one_wins = winner_raw == 1

            p1_key, p2_key = self._extract_player_keys(row, discipline)
            country = str(row.get("country", ""))
            nat_p1 = self._get_nationality(row, discipline, team="one")
            nat_p2 = self._get_nationality(row, discipline, team="two")
            tournament_type = str(row.get("tournament_type", ""))
            round_str = str(row.get("round", ""))

            feat = self._build_feature_vector(
                p1_key=p1_key,
                p2_key=p2_key,
                discipline=discipline,
                tournament_type=tournament_type,
                round_str=round_str,
                country=country,
                nat_p1=nat_p1,
                nat_p2=nat_p2,
            )

            target = 1 if team_one_wins else 0

            if apply_swap and rng.random() < 0.5:
                feat = self._swap_features(feat)
                target = 1 - target

            rows.append(feat)
            targets.append(target)

            # Update state AFTER feature extraction (no leakage)
            score = 1.0 if team_one_wins else 0.0
            self._update_state(p1_key, p2_key, discipline, score)

        if not rows:
            raise ValueError(f"No valid rows extracted for discipline={discipline}")

        X = np.stack(rows)
        y = np.array(targets, dtype=np.int32)
        logger.info(
            "extracted discipline=%s n=%d class_balance=%.3f",
            discipline, len(y), y.mean(),
        )
        return X, y

    def predict_features(
        self,
        player1: str,
        player2: str,
        partner1: str | None,
        partner2: str | None,
        discipline: str,
        nationality1: str,
        nationality2: str,
        tournament_type: str,
        round_str: str,
        country: str,
    ) -> np.ndarray:
        """Build a single feature vector for live prediction (no state update)."""
        is_doubles = discipline in DOUBLES_DISCIPLINES
        if is_doubles and partner1 and partner2:
            p1_key = _pair_key(player1, partner1)
            p2_key = _pair_key(player2, partner2)
        else:
            p1_key = player1.strip()
            p2_key = player2.strip()

        return self._build_feature_vector(
            p1_key=p1_key,
            p2_key=p2_key,
            discipline=discipline,
            tournament_type=tournament_type,
            round_str=round_str,
            country=country,
            nat_p1=nationality1,
            nat_p2=nationality2,
        )

    def get_elo(self, player_key: str, discipline: str) -> float:
        return self._elo[discipline].get(player_key, ELO_DEFAULT)

    def get_all_elos(self, discipline: str) -> dict[str, float]:
        return dict(self._elo[discipline])

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _reset_discipline(self, discipline: str) -> None:
        self._elo[discipline] = defaultdict(_elo_default)
        self._history.clear()
        self._h2h.clear()
        self._career.clear()
        self._partnership.clear()

    def _extract_player_keys(self, row: Any, discipline: str) -> tuple[str, str]:
        if discipline in DOUBLES_DISCIPLINES:
            p1a = str(row.get("team_one_player_one", "")).strip()
            p1b = str(row.get("team_one_player_two", "")).strip()
            p2a = str(row.get("team_two_player_one", "")).strip()
            p2b = str(row.get("team_two_player_two", "")).strip()
            return _pair_key(p1a, p1b), _pair_key(p2a, p2b)
        else:
            p1 = str(row.get("team_one_players", "")).strip()
            p2 = str(row.get("team_two_players", "")).strip()
            return p1, p2

    def _get_nationality(self, row: Any, discipline: str, team: str) -> str:
        if discipline in DOUBLES_DISCIPLINES:
            nat = str(row.get(f"team_{team}_player_one_nationality", "")).strip()
        else:
            nat = str(row.get(f"team_{team}_nationalities", "")).strip()
        return nat

    def _build_feature_vector(
        self,
        p1_key: str,
        p2_key: str,
        discipline: str,
        tournament_type: str,
        round_str: str,
        country: str,
        nat_p1: str,
        nat_p2: str,
    ) -> np.ndarray:
        elo_p1 = self._elo[discipline].get(p1_key, ELO_DEFAULT)
        elo_p2 = self._elo[discipline].get(p2_key, ELO_DEFAULT)
        elo_diff = elo_p1 - elo_p2

        wr5_p1, wr10_p1 = self._rolling_win_rate(p1_key)
        wr5_p2, wr10_p2 = self._rolling_win_rate(p2_key)

        h2h_wr, h2h_cnt = self._h2h_stats(p1_key, p2_key)

        tier = TOURNAMENT_TIER.get(tournament_type, 0)
        round_enc = ROUND_ENCODING.get(round_str, 0)

        home_p1 = 1 if nat_p1 and country and nat_p1[:3].upper() == country[:3].upper() else 0
        home_p2 = 1 if nat_p2 and country and nat_p2[:3].upper() == country[:3].upper() else 0

        c_p1 = self._career[p1_key]
        c_p2 = self._career[p2_key]
        career_m_p1 = c_p1[0]
        career_m_p2 = c_p2[0]
        career_wr_p1 = c_p1[1] / c_p1[0] if c_p1[0] > 0 else 0.5
        career_wr_p2 = c_p2[1] / c_p2[0] if c_p2[0] > 0 else 0.5

        # Doubles partnership stats
        if discipline in DOUBLES_DISCIPLINES:
            # For doubles, p1_key is already a pair key
            pk1 = frozenset([p1_key, "__SELF__"])  # partnerships tracked within pair key
            # Track partnerships under a different key structure
            pship = self._partnership.get(p1_key, [0, 0])
            pship_wr = pship[1] / pship[0] if pship[0] > 0 else 0.5
            pship_m = pship[0]
        else:
            pship_wr = 0.0
            pship_m = 0

        disc_enc = DISCIPLINE_ENC.get(discipline, 0)

        return np.array([
            elo_p1, elo_p2, elo_diff,
            wr5_p1, wr5_p2,
            wr10_p1, wr10_p2,
            h2h_wr, h2h_cnt,
            tier, round_enc,
            home_p1, home_p2,
            career_m_p1, career_m_p2,
            career_wr_p1, career_wr_p2,
            pship_wr, pship_m,
            disc_enc,
        ], dtype=np.float32)

    def _update_state(
        self, p1_key: str, p2_key: str, discipline: str, score: float
    ) -> None:
        """Update ELO, rolling history, H2H, career, partnership. score=1 → P1 won."""
        # ELO
        ra = self._elo[discipline].get(p1_key, ELO_DEFAULT)
        rb = self._elo[discipline].get(p2_key, ELO_DEFAULT)
        ra_new, rb_new = _update_elo(ra, rb, score)
        self._elo[discipline][p1_key] = ra_new
        self._elo[discipline][p2_key] = rb_new

        # Rolling history
        now = datetime.utcnow()
        self._history[p1_key].append((now, int(score == 1.0)))
        self._history[p2_key].append((now, int(score == 0.0)))

        # H2H
        canonical = tuple(sorted([p1_key, p2_key]))
        h2h_key = frozenset(canonical)
        if p1_key == canonical[0]:
            self._h2h[h2h_key].append(int(score == 1.0))
        else:
            self._h2h[h2h_key].append(int(score == 0.0))

        # Career
        self._career[p1_key][0] += 1
        self._career[p1_key][1] += int(score == 1.0)
        self._career[p2_key][0] += 1
        self._career[p2_key][1] += int(score == 0.0)

        # Partnership stats (for doubles, p1_key IS the pair key)
        if discipline in DOUBLES_DISCIPLINES:
            self._partnership.setdefault(p1_key, [0, 0])
            self._partnership[p1_key][0] += 1
            self._partnership[p1_key][1] += int(score == 1.0)
            self._partnership.setdefault(p2_key, [0, 0])
            self._partnership[p2_key][0] += 1
            self._partnership[p2_key][1] += int(score == 0.0)

    def _rolling_win_rate(self, key: str) -> tuple[float, float]:
        history = self._history.get(key, [])
        if not history:
            return 0.5, 0.5
        last5 = [o for _, o in history[-5:]]
        last10 = [o for _, o in history[-10:]]
        wr5 = sum(last5) / len(last5) if last5 else 0.5
        wr10 = sum(last10) / len(last10) if last10 else 0.5
        return wr5, wr10

    def _h2h_stats(self, p1_key: str, p2_key: str) -> tuple[float, int]:
        canonical = tuple(sorted([p1_key, p2_key]))
        h2h_key = frozenset(canonical)
        results = self._h2h.get(h2h_key, [])
        if not results:
            return 0.5, 0
        # results from canonical[0]'s perspective
        if p1_key == canonical[0]:
            wr = sum(results) / len(results)
        else:
            wr = 1.0 - sum(results) / len(results)
        return wr, len(results)

    def _swap_features(self, feat: np.ndarray) -> np.ndarray:
        """Swap P1↔P2 features to augment training set."""
        f = feat.copy()
        # elo_p1 <-> elo_p2 (indices 0,1)
        f[0], f[1] = feat[1], feat[0]
        # elo_diff negated (index 2)
        f[2] = -feat[2]
        # win_rate_5 (3,4)
        f[3], f[4] = feat[4], feat[3]
        # win_rate_10 (5,6)
        f[5], f[6] = feat[6], feat[5]
        # h2h_win_rate_p1 → 1 - h2h (index 7), h2h_matches unchanged (8)
        f[7] = 1.0 - feat[7]
        # home_advantage (11,12)
        f[11], f[12] = feat[12], feat[11]
        # career_matches (13,14)
        f[13], f[14] = feat[14], feat[13]
        # career_win_rate (15,16)
        f[15], f[16] = feat[16], feat[15]
        # partnership stats (17,18) — swap P1↔P2 partnership
        f[17], f[18] = feat[17], feat[18]  # keeps same (partnership of each side)
        return f


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _pair_key(p1: str, p2: str) -> str:
    """Canonical pair key — sorted so (A,B)==(B,A) for ELO/career tracking."""
    parts = sorted([p1.strip(), p2.strip()])
    return f"{parts[0]}|{parts[1]}"

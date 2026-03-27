"""
BadmintonPricer — builds sportsbook markets for badminton matches.

Markets produced:
1. Match Winner        — 2-way, 5% margin (tight Asian market)
2. Set Handicap        — P1 -1.5 / +1.5 sets (3-set format)
3. Total Games O/U    — 1.5 / 2.5 sets (badminton plays best-of-3)
4. Correct Score      — 2-0, 2-1, 0-2, 1-2

Formulas:
- Match Winner uses calibrated ML probability.
- Set Handicap approximated via P(win 2-0) vs P(need 3rd set) breakdown.
- Correct Score uses Markov-style decomposition of game-win probability.
- O/U derived from correct score probabilities.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MATCH_WINNER_MARGIN = 0.05   # 5% overround


def _apply_margin(p1: float, p2: float, margin: float) -> tuple[float, float]:
    """Apply margin to a 2-way market; return (odds_p1, odds_p2)."""
    total = p1 + p2
    if total <= 0:
        return 2.0, 2.0
    p1_adj = p1 / total * (1.0 + margin / 2.0)
    p2_adj = p2 / total * (1.0 + margin / 2.0)
    odds_p1 = 1.0 / p1_adj if p1_adj > 0 else 99.0
    odds_p2 = 1.0 / p2_adj if p2_adj > 0 else 99.0
    return round(odds_p1, 3), round(odds_p2, 3)


def _three_way_margin(ps: list[float], margin: float) -> list[float]:
    """Apply margin to a 3+ way market; returns fair odds list."""
    total = sum(ps)
    if total <= 0:
        return [2.0] * len(ps)
    inflated = [p / total * (1.0 + margin) for p in ps]
    return [round(1.0 / p, 3) if p > 0 else 999.0 for p in inflated]


def _game_win_probability(p_match_win: float) -> float:
    """
    Estimate per-game win probability from match win probability.
    Uses the analytical relationship for best-of-3 with equal game probs.
    Solves: P(win >= 2 of 3 games) = p_match_win for p_game.
    Binary search.
    """
    target = float(np.clip(p_match_win, 0.01, 0.99))

    def match_win_from_game(pg: float) -> float:
        pg = float(np.clip(pg, 0.001, 0.999))
        # P(win match) = P(win 2-0) + P(win 2-1)
        p_20 = pg * pg
        p_21 = 2.0 * pg * pg * (1.0 - pg)
        return p_20 + p_21

    lo, hi = 0.001, 0.999
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if match_win_from_game(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _correct_score_probs(p_match_win: float) -> dict[str, float]:
    """
    Returns probabilities for each correct score outcome.
    Scores: 2-0, 2-1, 0-2, 1-2
    """
    pg = _game_win_probability(p_match_win)
    qa = 1.0 - pg

    p_20 = pg * pg
    p_21 = 2.0 * pg * qa * pg     # win g1 lose g2 win g3 + lose g1 win g2 win g3
    p_02 = qa * qa
    p_12 = 2.0 * pg * qa * qa     # win g1 lose g2 lose g3 + lose g1 win g2 lose g3

    total = p_20 + p_21 + p_02 + p_12
    if total <= 0:
        total = 1.0

    return {
        "2-0": p_20 / total,
        "2-1": p_21 / total,
        "0-2": p_02 / total,
        "1-2": p_12 / total,
    }


class BadmintonPricer:
    """Builds all sportsbook markets for a badminton match."""

    def price_match(
        self,
        player1: str,
        player2: str,
        p1_win_prob: float,
        p2_win_prob: float,
    ) -> dict[str, Any]:
        """
        Build all markets and return structured response.
        """
        p1_win_prob = float(np.clip(p1_win_prob, 0.02, 0.98))
        p2_win_prob = float(np.clip(p2_win_prob, 0.02, 0.98))

        # Re-normalise to sum=1 (calibrator guarantees this, but be safe)
        total = p1_win_prob + p2_win_prob
        if abs(total - 1.0) > 0.001:
            p1_win_prob /= total
            p2_win_prob /= total

        markets: dict[str, Any] = {}

        # 1. Match Winner
        markets["match_winner"] = self._market_match_winner(
            player1, player2, p1_win_prob, p2_win_prob
        )

        # 2. Set Handicap
        markets["set_handicap"] = self._market_set_handicap(
            player1, player2, p1_win_prob
        )

        # 3. Total Games (O/U 1.5, 2.5)
        markets["total_games"] = self._market_total_games(p1_win_prob)

        # 4. Correct Score
        markets["correct_score"] = self._market_correct_score(
            player1, player2, p1_win_prob
        )

        return {
            "player1": player1,
            "player2": player2,
            "p1_win_prob": round(p1_win_prob, 4),
            "p2_win_prob": round(p2_win_prob, 4),
            "markets": markets,
        }

    # ---------------------------------------------------------------------- #
    # Individual market builders
    # ---------------------------------------------------------------------- #

    def _market_match_winner(
        self,
        player1: str,
        player2: str,
        p1: float,
        p2: float,
    ) -> dict[str, Any]:
        odds_p1, odds_p2 = _apply_margin(p1, p2, MATCH_WINNER_MARGIN)
        return {
            "market_type": "match_winner",
            "margin": MATCH_WINNER_MARGIN,
            "selections": [
                {"name": player1, "probability": round(p1, 4), "odds": odds_p1},
                {"name": player2, "probability": round(p2, 4), "odds": odds_p2},
            ],
        }

    def _market_set_handicap(
        self,
        player1: str,
        player2: str,
        p1_win: float,
    ) -> dict[str, Any]:
        """
        Set Handicap: P1 -1.5 sets means P1 wins 2-0 (only 2 games played).
        P1 +1.5 sets means team_two wins 2-0.
        """
        cs = _correct_score_probs(p1_win)
        # P1 -1.5: P1 wins 2-0
        p_p1_minus = cs["2-0"]
        # P2 -1.5: P2 wins 2-0
        p_p2_minus = cs["0-2"]
        # Each side needs to cover the full handicap
        # P1 +1.5: P1 wins any OR P1 loses 2-1 (covers handicap, loses by less)
        p_p1_plus = cs["2-0"] + cs["2-1"] + cs["1-2"]  # anything except P2 wins 2-0
        p_p2_plus = cs["0-2"] + cs["1-2"] + cs["2-1"]  # anything except P1 wins 2-0

        odds_p1_minus, odds_p2_minus = _apply_margin(p_p1_minus, p_p2_minus, 0.065)

        return {
            "market_type": "set_handicap",
            "margin": 0.065,
            "selections": [
                {
                    "name": f"{player1} -1.5 sets",
                    "handicap": -1.5,
                    "probability": round(p_p1_minus, 4),
                    "odds": odds_p1_minus,
                },
                {
                    "name": f"{player2} -1.5 sets",
                    "handicap": -1.5,
                    "probability": round(p_p2_minus, 4),
                    "odds": odds_p2_minus,
                },
            ],
        }

    def _market_total_games(self, p1_win: float) -> dict[str, Any]:
        """
        Total Games:
          Over/Under 1.5 games (Under 1.5 impossible in BWF — min 2 games played)
          Over/Under 2.5 games (Under = 2 games / Over = 3 games)
        """
        cs = _correct_score_probs(p1_win)
        # 2 games played: 2-0 or 0-2
        p_2games = cs["2-0"] + cs["0-2"]
        # 3 games played: 2-1 or 1-2
        p_3games = cs["2-1"] + cs["1-2"]

        odds_under25, odds_over25 = _apply_margin(p_2games, p_3games, 0.065)

        return {
            "market_type": "total_games",
            "margin": 0.065,
            "lines": [
                {
                    "line": 2.5,
                    "over": {
                        "label": "Over 2.5",
                        "probability": round(p_3games, 4),
                        "odds": odds_over25,
                    },
                    "under": {
                        "label": "Under 2.5",
                        "probability": round(p_2games, 4),
                        "odds": odds_under25,
                    },
                }
            ],
        }

    def _market_correct_score(
        self,
        player1: str,
        player2: str,
        p1_win: float,
    ) -> dict[str, Any]:
        cs = _correct_score_probs(p1_win)
        probs = [cs["2-0"], cs["2-1"], cs["0-2"], cs["1-2"]]
        odds = _three_way_margin(probs, 0.08)
        names = [
            f"{player1} 2-0",
            f"{player1} 2-1",
            f"{player2} 2-0 ({player1} 0-2)",
            f"{player2} 2-1 ({player1} 1-2)",
        ]
        score_labels = ["2-0", "2-1", "0-2", "1-2"]
        selections = []
        for i, (name, prob, od, score) in enumerate(
            zip(names, probs, odds, score_labels)
        ):
            selections.append({
                "name": name,
                "score": score,
                "probability": round(prob, 4),
                "odds": od,
            })
        return {
            "market_type": "correct_score",
            "margin": 0.08,
            "selections": selections,
        }

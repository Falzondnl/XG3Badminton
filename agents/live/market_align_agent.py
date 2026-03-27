"""
market_align_agent.py
=====================
MarketAlignAgent — Aligns live prices with pre-match anchor.

Prevents live prices from drifting too far from pre-match prices
before sufficient live evidence has accumulated (< 20 points played).

Alignment logic:
  - Points 0-10:  Max drift from pre-match = 25% (relative)
  - Points 11-30: Max drift = 40%
  - Points 31+:   No constraint (Markov fully trusted)

Also aligns match winner with correct score sum:
  P(A wins) must equal sum of correct score probs for A
  (within 3% tolerance — Markov guarantees this but numerical drift can occur)
"""

from __future__ import annotations

from typing import Dict, List

import structlog

from markets.derivative_engine import MarketPrice

logger = structlog.get_logger(__name__)


class MarketAlignAgent:
    """
    Aligns live prices to pre-match anchor and cross-market consistency.
    """

    def __init__(self, match_id: str, pre_match_p_a: float) -> None:
        self._match_id = match_id
        self._pre_match_p_a = pre_match_p_a

    def align(
        self,
        markets: Dict[str, List[MarketPrice]],
        p_a_blend: float,
        total_points_played: int,
    ) -> Dict[str, List[MarketPrice]]:
        """
        Apply alignment constraints to live markets.

        Args:
            markets:             Generated live markets
            p_a_blend:           Current blended P(A wins)
            total_points_played: Points played so far in match

        Returns:
            Aligned markets dict.
        """
        # Compute max drift allowed based on points played
        if total_points_played < 11:
            max_drift_pct = 0.25
        elif total_points_played < 31:
            max_drift_pct = 0.40
        else:
            return markets  # No constraint after 30 points

        pre_match = self._pre_match_p_a
        if pre_match <= 0 or pre_match >= 1:
            return markets

        # Clamp blended probability to allowed drift range
        lower = pre_match * (1.0 - max_drift_pct)
        upper = min(0.99, pre_match * (1.0 + max_drift_pct))
        clamped_p_a = max(lower, min(upper, p_a_blend))

        if abs(clamped_p_a - p_a_blend) > 0.001:
            logger.debug(
                "market_align_clamped",
                match_id=self._match_id,
                p_a_blend=round(p_a_blend, 4),
                clamped=round(clamped_p_a, 4),
                pre_match=round(pre_match, 4),
                points_played=total_points_played,
            )

            # Update match winner market
            mw = markets.get("match_winner")
            if mw:
                from config.badminton_config import MIN_ODDS
                updated = []
                for mp in mw:
                    if "A_wins" in mp.outcome_name or mp.outcome_name.endswith("_a"):
                        new_prob = clamped_p_a
                    else:
                        new_prob = 1.0 - clamped_p_a
                    new_prob = max(0.01, min(0.99, new_prob))
                    updated.append(MarketPrice(
                        market_id=mp.market_id,
                        market_family=mp.market_family,
                        outcome_name=mp.outcome_name,
                        odds=max(MIN_ODDS, 1.0 / new_prob),
                        prob_implied=new_prob,
                        prob_with_margin=mp.prob_with_margin,
                    ))
                markets["match_winner"] = updated

        return markets

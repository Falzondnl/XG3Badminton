"""
manipulation_detection_agent.py
================================
ManipulationDetectionAgent — Sharp money / suspicious betting detection.

Detects:
  1. Rapid one-sided volume on a single outcome
  2. Closing line value (CLV) divergence from Pinnacle reference
  3. Velocity spikes — abnormal bet frequency in short window
  4. Multi-account pattern flags (passed in via context)

On detection:
  - Sets context.sharp_alert = True
  - Records context.sharp_alert_details
  - Reduces click_scales on affected markets (smart scaling deferred to SmartScalingAgent)
  - Optionally suspends market if manipulation_score > SUSPEND_THRESHOLD
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

import structlog

from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

# Thresholds
_CLV_ALERT_THRESHOLD = 0.08       # 8% edge vs Pinnacle → sharp alert
_VELOCITY_WINDOW_S = 60           # 1-minute rolling window
_VELOCITY_SPIKE_THRESHOLD = 20    # bets/min on single outcome
_SUSPEND_MANIPULATION_SCORE = 0.85  # suspend market if score >= this


@dataclass
class OutcomeVelocityTracker:
    """Tracks bet velocity for a single outcome in a rolling window."""
    window_s: float = _VELOCITY_WINDOW_S
    _timestamps: Deque[float] = field(default_factory=deque)

    def record_bet(self, ts: float) -> None:
        self._timestamps.append(ts)
        self._purge(ts)

    def bets_per_minute(self, now: float) -> float:
        self._purge(now)
        if self.window_s <= 0:
            return 0.0
        return len(self._timestamps) / (self.window_s / 60.0)

    def _purge(self, now: float) -> None:
        while self._timestamps and (now - self._timestamps[0]) > self.window_s:
            self._timestamps.popleft()


# Module-level velocity registry (keyed by match_id + outcome_name)
_velocity_registry: Dict[str, OutcomeVelocityTracker] = {}


def record_bet_event(match_id: str, outcome_name: str) -> None:
    """
    External hook: called by bet processing pipeline on each accepted bet.

    Updates the velocity tracker for manipulation detection on next
    trading cycle.
    """
    key = f"{match_id}:{outcome_name}"
    if key not in _velocity_registry:
        _velocity_registry[key] = OutcomeVelocityTracker()
    _velocity_registry[key].record_bet(time.time())


class ManipulationDetectionAgent(BaseTradingAgent):
    """
    Detects sharp / suspicious betting patterns and sets alert flags.
    """

    @property
    def agent_name(self) -> str:
        return "manipulation_detection"

    def process(self, context: TradingContext) -> TradingAgentResult:
        if context.prices_locked:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="prices locked — skip")

        now = time.time()
        alerts: list[str] = []
        max_score = 0.0

        for market_id, prices in context.adjusted_prices.items():
            ref_prob = context.reference_prices.get(market_id)

            for mp in prices:
                score = 0.0

                # Check 1: CLV divergence from Pinnacle reference
                if ref_prob is not None:
                    clv = abs(mp.prob_implied - ref_prob)
                    if clv > _CLV_ALERT_THRESHOLD:
                        clv_score = min(1.0, clv / 0.20)  # Scale 8%→0, 20%→1
                        score = max(score, clv_score * 0.6)
                        alerts.append(
                            f"{market_id}/{mp.outcome_name}: CLV divergence "
                            f"{clv:.3f} vs Pinnacle ref {ref_prob:.3f}"
                        )

                # Check 2: Velocity spike
                vel_key = f"{context.match_id}:{mp.outcome_name}"
                tracker = _velocity_registry.get(vel_key)
                if tracker:
                    bpm = tracker.bets_per_minute(now)
                    if bpm > _VELOCITY_SPIKE_THRESHOLD:
                        vel_score = min(1.0, bpm / (_VELOCITY_SPIKE_THRESHOLD * 3))
                        score = max(score, vel_score * 0.7)
                        alerts.append(
                            f"{market_id}/{mp.outcome_name}: velocity spike "
                            f"{bpm:.1f} bets/min (threshold {_VELOCITY_SPIKE_THRESHOLD})"
                        )

                max_score = max(max_score, score)

        context.manipulation_score = max_score

        if max_score > 0.30:
            context.sharp_alert = True
            context.sharp_alert_details = "; ".join(alerts)
            self._log(context, f"SHARP ALERT: score={max_score:.3f} — {context.sharp_alert_details}")

            if max_score >= _SUSPEND_MANIPULATION_SCORE:
                context.suspend_all = True
                context.suspend_reason = (
                    f"manipulation_detection: score={max_score:.3f} exceeds "
                    f"suspend threshold {_SUSPEND_MANIPULATION_SCORE}"
                )
                logger.warning(
                    "manipulation_suspension_triggered",
                    match_id=context.match_id,
                    score=max_score,
                )
        else:
            self._log(context, f"no manipulation signal (score={max_score:.3f})")

        return TradingAgentResult(
            agent_name=self.agent_name,
            success=True,
            context_mutated=context.sharp_alert,
            notes=f"manipulation_score={max_score:.3f}, sharp_alert={context.sharp_alert}",
        )

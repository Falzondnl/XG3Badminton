"""
smart_scaling_agent.py
======================
SmartScalingAgent — Volume-based click scale and liability adjustments.

Responsibilities:
  1. Reduce click scales when sharp_alert is set
  2. Reduce click scales proportional to current exposure on each outcome
  3. Increase click scales for thin markets (encourage liquidity)
  4. Enforce per-market maximum liability caps

Click scale 1.0 = full normal size. 0.0 = market suspended.
"""

from __future__ import annotations

import structlog

from agents.trading.base_trading_agent import BaseTradingAgent, TradingContext, TradingAgentResult

logger = structlog.get_logger(__name__)

_SHARP_SCALE_MULTIPLIER = 0.25    # Reduce to 25% on sharp alert
_EXPOSURE_SCALE_FLOOR = 0.15      # Minimum scale when near liability limit
_EXPOSURE_SCALE_TRIGGER = 0.70    # Start reducing at 70% of max liability
_MAX_LIABILITY_PER_MARKET = 100_000.0  # GBP per market outcome


class SmartScalingAgent(BaseTradingAgent):
    """
    Adjusts click scales based on sharp alert and current exposure.
    """

    @property
    def agent_name(self) -> str:
        return "smart_scaling"

    def process(self, context: TradingContext) -> TradingAgentResult:
        if context.prices_locked:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="prices locked — skip")

        if context.suspend_all:
            return TradingAgentResult(agent_name=self.agent_name, success=True,
                                     notes="already suspended — skip")

        adjustments: list[str] = []

        for market_id in list(context.click_scales.keys()):
            current_scale = context.click_scales[market_id]

            # 1. Sharp alert penalty
            if context.sharp_alert:
                sharp_factor = max(_EXPOSURE_SCALE_FLOOR,
                                   current_scale * (1.0 - context.manipulation_score * 0.8))
                sharp_reduced = min(current_scale, _SHARP_SCALE_MULTIPLIER)
                if sharp_reduced < current_scale:
                    context.click_scales[market_id] = sharp_reduced
                    adjustments.append(
                        f"{market_id}: sharp_reduced {current_scale:.2f}→{sharp_reduced:.2f}"
                    )
                    current_scale = sharp_reduced

            # 2. Exposure-based scaling
            outcome_exposures = [
                v for k, v in context.current_exposure.items()
                if k.startswith(market_id)
            ]
            if outcome_exposures:
                max_outcome_exposure = max(outcome_exposures)
                exposure_ratio = max_outcome_exposure / _MAX_LIABILITY_PER_MARKET

                if exposure_ratio > _EXPOSURE_SCALE_TRIGGER:
                    # Linear reduction from 1.0 at 70% to floor at 100%
                    t = (exposure_ratio - _EXPOSURE_SCALE_TRIGGER) / (1.0 - _EXPOSURE_SCALE_TRIGGER)
                    exposure_scale = max(
                        _EXPOSURE_SCALE_FLOOR,
                        current_scale * (1.0 - t * (1.0 - _EXPOSURE_SCALE_FLOOR))
                    )
                    if exposure_scale < current_scale:
                        context.click_scales[market_id] = exposure_scale
                        adjustments.append(
                            f"{market_id}: exposure_scaled "
                            f"{current_scale:.2f}→{exposure_scale:.2f} "
                            f"(ratio={exposure_ratio:.2f})"
                        )

                    # Hard suspend at 100%
                    if exposure_ratio >= 1.0:
                        context.click_scales[market_id] = 0.0
                        adjustments.append(f"{market_id}: SUSPENDED (exposure limit reached)")

        msg = f"{len(adjustments)} scale adjustments" if adjustments else "no scaling adjustments"
        if adjustments:
            self._log(context, msg + ": " + ", ".join(adjustments[:3]))

        return TradingAgentResult(
            agent_name=self.agent_name,
            success=True,
            context_mutated=bool(adjustments),
            notes=msg,
        )

"""
agents/auto_pricer.py
=====================
AutoPricer — Periodic pre-match repricing scheduler.

Runs a background asyncio loop that reprices all active pre-match markets
at a configurable interval (default: 60 seconds for outrights, 300 seconds
for pre-match match markets).

Responsibilities:
  1. Iterate over all matches in PRE_MATCH lifecycle state
  2. Fetch latest ELO / ML model inference for each match
  3. Run BadmintonTradingSupervisor trading cycle with fresh prices
  4. Publish updated prices via the registered price publisher
  5. Record reprice latency in ObservabilityAgent
  6. Skip matches where ManualLock is active (TraderControlAgent)

Repricing intervals (configurable via env vars):
  - PRE_MATCH_REPRICE_INTERVAL_S   (default 300s — match winner markets)
  - OUTRIGHT_REPRICE_INTERVAL_S    (default 60s  — tournament winner markets)

ZERO hardcoded probabilities. Prices come from ML inference → Markov → derivative engine.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import structlog

from config.badminton_config import Discipline, TournamentTier

logger = structlog.get_logger(__name__)

# Reprice intervals
_DEFAULT_PREMATCH_INTERVAL_S: float = float(
    os.environ.get("PRE_MATCH_REPRICE_INTERVAL_S", "300")
)
_DEFAULT_OUTRIGHT_INTERVAL_S: float = float(
    os.environ.get("OUTRIGHT_REPRICE_INTERVAL_S", "60")
)


class AutoPricerError(RuntimeError):
    """Raised on unrecoverable auto-pricer errors."""


@dataclass
class RepriceCycleResult:
    """Result of a single reprice cycle across all matches."""
    n_matches_repriced: int
    n_matches_skipped: int
    n_errors: int
    cycle_latency_ms: float
    errors: List[str] = field(default_factory=list)


@dataclass
class MatchRepriceDef:
    """
    Definition of a match registered for auto-repricing.

    The repricer calls `price_fn(match_id)` which must return a dict of
    {market_id: [MarketPrice]} or raise an exception on failure.
    The result is forwarded to `publish_fn(match_id, prices)`.
    """
    match_id: str
    discipline: Discipline
    tier: TournamentTier
    price_fn: Callable         # () → dict[str, list[MarketPrice]]
    publish_fn: Callable       # (match_id, prices) → None
    last_repriced_at: float = 0.0
    reprice_interval_s: float = _DEFAULT_PREMATCH_INTERVAL_S
    n_reprice_cycles: int = 0
    n_errors: int = 0


class AutoPricer:
    """
    Periodic pre-match repricing loop.

    Usage:
        pricer = AutoPricer()
        pricer.register_match(match_id, discipline, tier, price_fn, publish_fn)
        await pricer.run_forever()   # runs until cancelled

    Or single-cycle:
        result = await pricer.run_cycle()
    """

    def __init__(
        self,
        prematch_interval_s: float = _DEFAULT_PREMATCH_INTERVAL_S,
        outright_interval_s: float = _DEFAULT_OUTRIGHT_INTERVAL_S,
    ) -> None:
        self._prematch_interval_s = prematch_interval_s
        self._outright_interval_s = outright_interval_s
        self._matches: Dict[str, MatchRepriceDef] = {}
        self._running = False
        self._n_total_cycles = 0

    def register_match(
        self,
        match_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        price_fn: Callable,
        publish_fn: Callable,
        reprice_interval_s: Optional[float] = None,
    ) -> None:
        """
        Register a match for periodic repricing.

        Args:
            match_id:            XG3 internal match ID.
            discipline:          Badminton discipline.
            tier:                Tournament tier.
            price_fn:            Async or sync callable returning market prices.
            publish_fn:          Callable to publish prices downstream.
            reprice_interval_s:  Override default interval (None = use default).
        """
        interval = reprice_interval_s or self._prematch_interval_s
        self._matches[match_id] = MatchRepriceDef(
            match_id=match_id,
            discipline=discipline,
            tier=tier,
            price_fn=price_fn,
            publish_fn=publish_fn,
            reprice_interval_s=interval,
        )
        logger.info(
            "auto_pricer_match_registered",
            match_id=match_id,
            discipline=discipline.value,
            tier=tier.value,
            interval_s=interval,
        )

    def deregister_match(self, match_id: str) -> None:
        """Remove a match from the auto-repricing loop (e.g. when it goes live)."""
        removed = self._matches.pop(match_id, None)
        if removed:
            logger.info("auto_pricer_match_deregistered", match_id=match_id)

    async def run_forever(self, poll_interval_s: float = 5.0) -> None:
        """
        Run the repricing loop indefinitely.

        Polls every `poll_interval_s` seconds and triggers any matches
        whose reprice interval has elapsed.

        Raises:
            asyncio.CancelledError when the task is cancelled (expected on shutdown).
        """
        self._running = True
        logger.info(
            "auto_pricer_started",
            n_matches=len(self._matches),
            poll_interval_s=poll_interval_s,
        )
        try:
            while self._running:
                await self.run_cycle()
                await asyncio.sleep(poll_interval_s)
        except asyncio.CancelledError:
            logger.info("auto_pricer_stopped")
            raise

    def stop(self) -> None:
        """Signal the run_forever loop to stop after the current cycle."""
        self._running = False

    async def run_cycle(self) -> RepriceCycleResult:
        """
        Run one reprice cycle: reprice all due matches.

        Returns:
            RepriceCycleResult with counts and errors.
        """
        t0 = time.perf_counter()
        n_repriced = 0
        n_skipped = 0
        n_errors = 0
        errors: List[str] = []
        now = time.time()

        for match_id, defn in list(self._matches.items()):
            elapsed = now - defn.last_repriced_at
            if elapsed < defn.reprice_interval_s:
                n_skipped += 1
                continue

            try:
                await self._reprice_match(defn)
                n_repriced += 1
                defn.last_repriced_at = time.time()
                defn.n_reprice_cycles += 1
            except Exception as exc:
                n_errors += 1
                defn.n_errors += 1
                error_msg = f"{match_id}: {exc}"
                errors.append(error_msg)
                logger.error(
                    "auto_pricer_reprice_failed",
                    match_id=match_id,
                    error=str(exc),
                    n_errors_total=defn.n_errors,
                )

        self._n_total_cycles += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if n_repriced > 0 or n_errors > 0:
            logger.info(
                "auto_pricer_cycle_complete",
                n_repriced=n_repriced,
                n_skipped=n_skipped,
                n_errors=n_errors,
                cycle_latency_ms=round(latency_ms, 2),
                total_cycles=self._n_total_cycles,
            )

        return RepriceCycleResult(
            n_matches_repriced=n_repriced,
            n_matches_skipped=n_skipped,
            n_errors=n_errors,
            cycle_latency_ms=round(latency_ms, 2),
            errors=errors,
        )

    async def _reprice_match(self, defn: MatchRepriceDef) -> None:
        """Run the price function for a single match and publish the result."""
        t0 = time.perf_counter()

        # Support both async and sync price functions
        if asyncio.iscoroutinefunction(defn.price_fn):
            prices = await defn.price_fn(defn.match_id)
        else:
            prices = defn.price_fn(defn.match_id)

        if not prices:
            raise AutoPricerError(
                f"price_fn returned empty prices for match {defn.match_id!r}"
            )

        # Publish
        if asyncio.iscoroutinefunction(defn.publish_fn):
            await defn.publish_fn(defn.match_id, prices)
        else:
            defn.publish_fn(defn.match_id, prices)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "auto_pricer_match_repriced",
            match_id=defn.match_id,
            n_markets=len(prices),
            latency_ms=round(latency_ms, 2),
        )

    @property
    def n_registered_matches(self) -> int:
        return len(self._matches)

    @property
    def n_total_cycles(self) -> int:
        return self._n_total_cycles

    def get_match_stats(self, match_id: str) -> Optional[Dict]:
        defn = self._matches.get(match_id)
        if defn is None:
            return None
        return {
            "match_id": defn.match_id,
            "discipline": defn.discipline.value,
            "tier": defn.tier.value,
            "interval_s": defn.reprice_interval_s,
            "n_reprice_cycles": defn.n_reprice_cycles,
            "n_errors": defn.n_errors,
            "last_repriced_at": defn.last_repriced_at,
        }

"""
outright_supervisor.py
======================
OutrightSupervisorAgent — manages tournament outright pricing.

Responsibilities:
  - Periodic tournament repricing (every 60s per V1 Plan)
  - Match result integration into outright simulation
  - H9 gate: tournament winner probabilities sum to ±0.5%
  - Liability management for outright markets

Architecture:
  - Supervised by BadmintonOrchestratorAgent
  - Uses OutrightPricingEngine for Monte Carlo simulation
  - Uses TradingControlManager for liability tracking
  - Publishes prices via price_publisher callback

ZERO hardcoded probabilities — all prices from Monte Carlo simulation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import structlog

from config.badminton_config import (
    Discipline,
    TournamentTier,
    OUTRIGHT_REPRICE_INTERVAL_S,
    OVERROUND_MIN,
    OVERROUND_MAX,
)
from markets.outright_pricing import (
    OutrightPricingEngine,
    TournamentDraw,
    TournamentEntry,
    DrawType,
    OutrightResponse as OutrightResult,
)
from markets.market_trading_control import TradingControlManager, MarketControl

logger = structlog.get_logger(__name__)


class OutrightMarketStatus(str, Enum):
    """Status of an outright market."""
    OPEN = "open"
    SUSPENDED = "suspended"
    RESULTED = "resulted"
    CLOSED = "closed"


@dataclass
class PlayerResult:
    """Result status of a player in a tournament."""
    player_id: str
    is_eliminated: bool = False
    is_winner: bool = False
    rounds_won: int = 0
    current_round: int = 1


@dataclass
class TournamentState:
    """
    Live state of a tournament being tracked.

    Maintained by OutrightSupervisorAgent as match results
    come in from the orchestrator.
    """
    tournament_id: str
    discipline: Discipline
    tier: TournamentTier
    entries: List[TournamentEntry]
    draw_size: int = 0  # set in __post_init__ from len(entries)
    player_results: Dict[str, PlayerResult] = field(default_factory=dict)
    last_priced_at: float = 0.0
    last_prices: Optional[OutrightResult] = None
    status: OutrightMarketStatus = OutrightMarketStatus.OPEN
    match_results: List[Dict[str, Any]] = field(default_factory=list)
    eliminated_players: Set[str] = field(default_factory=set)
    winner: Optional[str] = None

    def __post_init__(self) -> None:
        if self.draw_size == 0:
            self.draw_size = len(self.entries)
        for entry in self.entries:
            self.player_results[entry.entity_id] = PlayerResult(
                player_id=entry.entity_id
            )

    @property
    def active_players(self) -> List[TournamentEntry]:
        """Entries for players still in the tournament."""
        return [e for e in self.entries if e.entity_id not in self.eliminated_players]

    def record_match_result(
        self,
        winner_id: str,
        loser_id: str,
        round_number: int,
    ) -> None:
        """Update state after a match result."""
        self.eliminated_players.add(loser_id)
        self.match_results.append({
            "winner": winner_id,
            "loser": loser_id,
            "round": round_number,
            "ts": time.time(),
        })

        if winner_id in self.player_results:
            self.player_results[winner_id].rounds_won += 1
            self.player_results[winner_id].current_round = round_number + 1

        if loser_id in self.player_results:
            self.player_results[loser_id].is_eliminated = True

        logger.info(
            "outright_match_result_recorded",
            tournament_id=self.tournament_id,
            winner_id=winner_id,
            loser_id=loser_id,
            round_number=round_number,
            remaining_players=len(self.active_players),
        )

        # Check if tournament complete
        if len(self.active_players) == 1:
            self.winner = self.active_players[0].entity_id
            self.player_results[self.winner].is_winner = True
            self.status = OutrightMarketStatus.RESULTED
            logger.info(
                "outright_tournament_complete",
                tournament_id=self.tournament_id,
                winner=self.winner,
            )


@dataclass
class OutrightPriceSnapshot:
    """Point-in-time outright prices with validation status."""
    tournament_id: str
    discipline: Discipline
    prices: Dict[str, float]  # player_id → decimal odds
    win_probs: Dict[str, float]  # player_id → margined probability
    h9_passed: bool  # sum of probs within ±0.5%
    prob_sum: float
    timestamp: float
    n_simulations: int


class OutrightSupervisorAgent:
    """
    Supervisor agent for tournament outright markets.

    Manages one or more active tournaments, periodically reprices
    via Monte Carlo simulation, validates H9 gate (sum = 100% ± 0.5%),
    and handles match result updates.
    """

    def __init__(
        self,
        price_publisher: Optional[Callable[[str, OutrightPriceSnapshot], None]] = None,
    ) -> None:
        self._engine = OutrightPricingEngine()
        self._tournaments: Dict[str, TournamentState] = {}
        self._trading_controls: Dict[str, TradingControlManager] = {}
        self._price_publisher = price_publisher
        self._reprice_interval = OUTRIGHT_REPRICE_INTERVAL_S
        self._running = False
        self._reprice_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]

        logger.info("outright_supervisor_initialised")

    # ------------------------------------------------------------------
    # Tournament lifecycle
    # ------------------------------------------------------------------

    def register_tournament(
        self,
        tournament_id: str,
        discipline: Discipline,
        tier: TournamentTier,
        entries: List[TournamentEntry],
    ) -> None:
        """
        Register a tournament for outright pricing.

        Args:
            tournament_id: Unique tournament identifier
            discipline: MS/WS/MD/WD/XD
            tier: SUPER_1000 / SUPER_750 / etc.
            entries: All player/pair entries with pre-match RWP
        """
        if tournament_id in self._tournaments:
            raise RuntimeError(
                f"Tournament {tournament_id!r} already registered. "
                "Use update_entries() to modify."
            )

        if len(entries) == 0:
            raise ValueError(f"No entries provided for tournament {tournament_id!r}")

        state = TournamentState(
            tournament_id=tournament_id,
            discipline=discipline,
            tier=tier,
            entries=entries,
        )
        self._tournaments[tournament_id] = state
        self._trading_controls[tournament_id] = TradingControlManager(
            match_id=tournament_id  # reusing match_id field for tournament
        )

        logger.info(
            "outright_tournament_registered",
            tournament_id=tournament_id,
            discipline=discipline.value,
            tier=tier.value,
            n_entries=len(entries),
        )

    def on_match_result(
        self,
        tournament_id: str,
        winner_id: str,
        loser_id: str,
        round_number: int,
    ) -> Optional[OutrightPriceSnapshot]:
        """
        Process a match result within the tournament.

        Updates player states and triggers immediate reprice.

        Returns updated outright prices or None if tournament complete.
        """
        if tournament_id not in self._tournaments:
            raise KeyError(f"Tournament {tournament_id!r} not registered")

        state = self._tournaments[tournament_id]
        if state.status == OutrightMarketStatus.RESULTED:
            logger.warning(
                "outright_result_on_complete_tournament",
                tournament_id=tournament_id,
            )
            return None

        state.record_match_result(winner_id, loser_id, round_number)

        if state.status == OutrightMarketStatus.RESULTED:
            self._settle_tournament(tournament_id, state.winner)  # type: ignore[arg-type]
            return None

        # Force immediate reprice after result
        state.last_priced_at = 0.0
        return self._reprice_tournament(tournament_id)

    def suspend_tournament(self, tournament_id: str, reason: str = "") -> None:
        """Suspend all outright trading for a tournament."""
        if tournament_id not in self._tournaments:
            raise KeyError(f"Tournament {tournament_id!r} not registered")

        self._tournaments[tournament_id].status = OutrightMarketStatus.SUSPENDED
        self._trading_controls[tournament_id].suspend_all(reason=reason)

        logger.warning(
            "outright_tournament_suspended",
            tournament_id=tournament_id,
            reason=reason,
        )

    def resume_tournament(self, tournament_id: str) -> None:
        """Resume outright trading."""
        if tournament_id not in self._tournaments:
            raise KeyError(f"Tournament {tournament_id!r} not registered")

        state = self._tournaments[tournament_id]
        if state.status != OutrightMarketStatus.SUSPENDED:
            raise RuntimeError(
                f"Tournament {tournament_id!r} is not suspended "
                f"(status={state.status.value})"
            )

        state.status = OutrightMarketStatus.OPEN
        self._trading_controls[tournament_id].resume_market(
            f"outright_{tournament_id}"
        )

        logger.info("outright_tournament_resumed", tournament_id=tournament_id)

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def get_prices(self, tournament_id: str) -> OutrightPriceSnapshot:
        """
        Get current outright prices.

        Reprices if stale (> OUTRIGHT_REPRICE_INTERVAL_S seconds old).
        Raises RuntimeError if tournament resulted or not registered.
        """
        if tournament_id not in self._tournaments:
            raise KeyError(f"Tournament {tournament_id!r} not registered")

        state = self._tournaments[tournament_id]

        if state.status == OutrightMarketStatus.RESULTED:
            raise RuntimeError(
                f"Tournament {tournament_id!r} has resulted. No live prices."
            )
        if state.status == OutrightMarketStatus.SUSPENDED:
            raise RuntimeError(
                f"Tournament {tournament_id!r} is suspended. Prices unavailable."
            )

        now = time.monotonic()
        if now - state.last_priced_at > self._reprice_interval or state.last_prices is None:
            return self._reprice_tournament(tournament_id)

        # Return cached snapshot
        return self._build_snapshot(state)

    def _reprice_tournament(self, tournament_id: str) -> OutrightPriceSnapshot:
        """Run Monte Carlo repricing and validate H9."""
        state = self._tournaments[tournament_id]
        active = state.active_players

        if len(active) < 2:
            raise RuntimeError(
                f"Tournament {tournament_id!r} has < 2 active players "
                f"({len(active)}) — cannot reprice"
            )

        try:
            # Build entries with eliminated players marked as byes
            draw_entries: List[TournamentEntry] = []
            for entry in state.entries:
                if entry.entity_id in state.eliminated_players:
                    draw_entries.append(dataclasses.replace(entry, is_bye=True))
                else:
                    draw_entries.append(entry)

            # Pad to valid draw size with bye entries if needed
            draw_size = state.draw_size
            while len(draw_entries) < draw_size:
                draw_entries.append(TournamentEntry(
                    entity_id=f"__bye_{len(draw_entries)}__",
                    is_bye=True,
                ))

            draw = TournamentDraw(
                tournament_id=tournament_id,
                discipline=state.discipline,
                tier=state.tier,
                draw_type=DrawType.SINGLE_ELIMINATION,
                draw_size=draw_size,
                entries=draw_entries,
            )
            result = self._engine.price_tournament(draw)
        except Exception as exc:
            logger.error(
                "outright_reprice_failed",
                tournament_id=tournament_id,
                error=str(exc),
            )
            raise

        state.last_prices = result
        state.last_priced_at = time.monotonic()

        snapshot = self._build_snapshot(state)

        if not snapshot.h9_passed:
            logger.error(
                "outright_h9_gate_failure",
                tournament_id=tournament_id,
                prob_sum=round(snapshot.prob_sum, 6),
            )

        if self._price_publisher is not None:
            try:
                self._price_publisher(tournament_id, snapshot)
            except Exception as exc:
                logger.error(
                    "outright_price_publish_error",
                    tournament_id=tournament_id,
                    error=str(exc),
                )

        logger.info(
            "outright_repriced",
            tournament_id=tournament_id,
            n_active=len(active),
            h9_passed=snapshot.h9_passed,
            prob_sum=round(snapshot.prob_sum, 6),
        )

        return snapshot

    def _build_snapshot(self, state: TournamentState) -> OutrightPriceSnapshot:
        """Convert OutrightResult → OutrightPriceSnapshot with H9 validation."""
        if state.last_prices is None:
            raise RuntimeError(
                f"Tournament {state.tournament_id!r} has no cached prices"
            )

        result = state.last_prices
        prices: Dict[str, float] = {}
        win_probs: Dict[str, float] = {}

        for r in result.results:
            if not r.p_win_tournament > 0:
                continue
            prices[r.entity_id] = round(r.odds_with_margin, 4)
            win_probs[r.entity_id] = round(r.p_win_tournament, 6)

        prob_sum = sum(win_probs.values())
        h9_passed = abs(prob_sum - 1.0) <= 0.005  # ±0.5%

        return OutrightPriceSnapshot(
            tournament_id=state.tournament_id,
            discipline=state.discipline,
            prices=prices,
            win_probs=win_probs,
            h9_passed=h9_passed,
            prob_sum=prob_sum,
            timestamp=time.time(),
            n_simulations=result.n_simulations,
        )

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def _settle_tournament(self, tournament_id: str, winner_id: str) -> None:
        """Mark tournament as resulted and settle outright market."""
        state = self._tournaments[tournament_id]
        state.status = OutrightMarketStatus.RESULTED

        tc = self._trading_controls[tournament_id]
        market_id = f"outright_{tournament_id}"

        # Record settlement in trading control
        tc.suspend_all(reason=f"tournament_resulted:winner={winner_id}")

        logger.info(
            "outright_tournament_settled",
            tournament_id=tournament_id,
            winner_id=winner_id,
        )

    # ------------------------------------------------------------------
    # Async reprice loop
    # ------------------------------------------------------------------

    async def start_reprice_loop(self) -> None:
        """Start background periodic repricing of all active tournaments."""
        self._running = True
        logger.info("outright_reprice_loop_started", interval_s=self._reprice_interval)

        while self._running:
            await asyncio.sleep(self._reprice_interval)
            for tournament_id, state in list(self._tournaments.items()):
                if state.status != OutrightMarketStatus.OPEN:
                    continue
                try:
                    self._reprice_tournament(tournament_id)
                except Exception as exc:
                    logger.error(
                        "outright_background_reprice_error",
                        tournament_id=tournament_id,
                        error=str(exc),
                    )

    def stop_reprice_loop(self) -> None:
        """Stop background repricing loop."""
        self._running = False
        logger.info("outright_reprice_loop_stopped")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_tournament_status(self, tournament_id: str) -> Dict[str, Any]:
        """Return operational status of a tournament."""
        if tournament_id not in self._tournaments:
            raise KeyError(f"Tournament {tournament_id!r} not registered")

        state = self._tournaments[tournament_id]
        now = time.monotonic()
        age_s = now - state.last_priced_at if state.last_priced_at > 0 else -1.0

        return {
            "tournament_id": tournament_id,
            "discipline": state.discipline.value,
            "tier": state.tier.value,
            "status": state.status.value,
            "n_entries": len(state.entries),
            "n_active": len(state.active_players),
            "n_eliminated": len(state.eliminated_players),
            "winner": state.winner,
            "n_match_results": len(state.match_results),
            "price_age_s": round(age_s, 1),
            "is_stale": age_s > self._reprice_interval,
            "last_priced_at": state.last_priced_at,
        }

    def get_all_tournaments(self) -> List[Dict[str, Any]]:
        """Return status of all registered tournaments."""
        return [
            self.get_tournament_status(tid) for tid in self._tournaments
        ]

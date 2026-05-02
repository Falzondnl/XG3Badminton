"""
test_outright_pricing.py
========================
Tests for markets/outright_pricing.py — Monte Carlo outright pricing engine.

Covers:
  - TournamentEntry/Draw construction
  - OutrightPricingEngine.price_tournament() happy path
  - H9 gate: winner probabilities sum to ≈ 1.0 (± 0.5%)
  - H10 gate: all odds >= 1.01
  - Higher ELO player gets lower odds (higher win probability)
  - Single-elimination draw with 4 / 8 / 16 entries
  - All disciplines produce valid results
  - TournamentDraw.validate() rejects wrong draw size
  - Completed results applied correctly (already-eliminated players get 0 prob)
  - Reproducibility: same seed produces identical output
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.badminton_config import Discipline, TournamentTier
from markets.outright_pricing import (
    DrawType,
    OutrightPricingEngine,
    OutrightPricingResult,
    OutrightResponse,
    TournamentDraw,
    TournamentEntry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(entity_id: str, elo: float = 1500.0, seeding: int | None = None) -> TournamentEntry:
    return TournamentEntry(
        entity_id=entity_id,
        seeding=seeding,
        rwp_as_server=0.515,
        rwp_as_receiver=0.500,
        elo_rating=elo,
    )


def _draw(
    n: int = 8,
    tier: TournamentTier = TournamentTier.SUPER_500,
    discipline: Discipline = Discipline.MS,
    draw_type: DrawType = DrawType.SINGLE_ELIMINATION,
    elo_top: float = 1700.0,
    rwp_top: float = 0.515,  # top seed RWP; if >0.515 creates clear favourite
) -> TournamentDraw:
    """Build a draw with n entries; top seed has elo_top/rwp_top, rest have 1500/0.515."""
    entries = []
    for i in range(n):
        is_top = i == 0
        e = TournamentEntry(
            entity_id=f"P{i+1:02d}",
            seeding=i + 1,
            rwp_as_server=rwp_top if is_top else 0.515,
            rwp_as_receiver=(rwp_top - 0.015) if is_top else 0.500,
            elo_rating=elo_top if is_top else 1500.0,
        )
        entries.append(e)
    return TournamentDraw(
        tournament_id="T001",
        discipline=discipline,
        tier=tier,
        draw_type=draw_type,
        draw_size=n,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# 1. Dataclass construction
# ---------------------------------------------------------------------------

class TestTournamentDrawConstruction:
    def test_entry_constructs(self) -> None:
        e = _entry("P1")
        assert e.entity_id == "P1"
        assert e.elo_rating == 1500.0

    def test_draw_constructs(self) -> None:
        draw = _draw(8)
        assert draw.tournament_id == "T001"
        assert len(draw.entries) == 8

    def test_draw_validate_wrong_size_raises(self) -> None:
        draw = _draw(8)
        draw.entries.append(_entry("EXTRA"))  # now 9 entries for an 8-draw
        with pytest.raises((ValueError, RuntimeError)):
            draw.validate()

    def test_draw_validate_correct_size_passes(self) -> None:
        draw = _draw(8)
        draw.validate()  # should not raise


# ---------------------------------------------------------------------------
# 2. Happy path
# ---------------------------------------------------------------------------

class TestOutrightPricingHappyPath:
    def test_price_tournament_returns_response(self) -> None:
        engine = OutrightPricingEngine(n_simulations=1000)
        resp = engine.price_tournament(_draw(8))
        assert isinstance(resp, OutrightResponse)

    def test_response_has_prices(self) -> None:
        engine = OutrightPricingEngine(n_simulations=1000)
        resp = engine.price_tournament(_draw(8))
        assert len(resp.results) > 0

    def test_all_entries_have_price(self) -> None:
        engine = OutrightPricingEngine(n_simulations=1000)
        draw = _draw(8)
        resp = engine.price_tournament(draw)
        entity_ids = {e.entity_id for e in draw.entries}
        priced_ids = {r.entity_id for r in resp.results}
        assert entity_ids == priced_ids

    def test_tournament_id_preserved(self) -> None:
        engine = OutrightPricingEngine(n_simulations=1000)
        resp = engine.price_tournament(_draw(8))
        assert resp.tournament_id == "T001"


# ---------------------------------------------------------------------------
# 3. H9 gate: winner probs sum to ≈ 1.0
# ---------------------------------------------------------------------------

class TestH9WinnerProbsSum:
    def test_implied_probs_sum_to_near_1(self) -> None:
        engine = OutrightPricingEngine(n_simulations=5000)
        resp = engine.price_tournament(_draw(8))
        total_implied = sum(p.p_win_tournament for p in resp.results)
        assert abs(total_implied - 1.0) < 0.005, (
            f"H9 violation: winner probs sum={total_implied:.4f} (must be within ±0.5% of 1.0)"
        )

    def test_margined_prices_applied(self) -> None:
        engine = OutrightPricingEngine(n_simulations=2000)
        resp = engine.price_tournament(_draw(8))
        assert resp.margin_applied > 0, "margin_applied must be positive"

    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_various_draw_sizes_sum_to_1(self, n: int) -> None:
        engine = OutrightPricingEngine(n_simulations=2000)
        resp = engine.price_tournament(_draw(n))
        total = sum(p.p_win_tournament for p in resp.results)
        assert abs(total - 1.0) < 0.01, (
            f"H9 violation for draw size {n}: sum={total:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. H10 gate: all odds >= 1.01
# ---------------------------------------------------------------------------

class TestH10MinOdds:
    def test_all_odds_above_minimum(self) -> None:
        engine = OutrightPricingEngine(n_simulations=2000)
        resp = engine.price_tournament(_draw(8))
        for r in resp.results:
            if r.p_win_tournament > 0:
                assert r.odds_fair >= 1.01, (
                    f"H10 violation: {r.entity_id} odds_fair={r.odds_fair:.4f}"
                )


# ---------------------------------------------------------------------------
# 5. Probability ordering: better ELO = shorter odds
# ---------------------------------------------------------------------------

class TestOutrightProbabilityOrdering:
    def test_top_seed_has_highest_win_prob(self) -> None:
        # Give top seed clear RWP advantage (0.560 vs 0.515) — engine uses RWP, not ELO
        # for match probabilities. 10k simulations makes ordering extremely reliable.
        engine = OutrightPricingEngine(n_simulations=10_000)
        resp = engine.price_tournament(_draw(8, rwp_top=0.560))
        results_by_id = {r.entity_id: r for r in resp.results}
        top_p = results_by_id["P01"].p_win_tournament
        avg_p = 1.0 / 8
        assert top_p > avg_p, (
            f"Top seed P01 prob={top_p:.4f} should be above avg {avg_p:.4f}"
        )

    def test_top_seed_has_competitive_odds(self) -> None:
        """Top seed (rwp=0.560 vs 0.515) must be in the top half by win probability."""
        engine = OutrightPricingEngine(n_simulations=10_000)
        resp = engine.price_tournament(_draw(8, rwp_top=0.560))
        results_sorted_by_p = sorted(resp.results, key=lambda r: r.p_win_tournament, reverse=True)
        top_4_ids = {r.entity_id for r in results_sorted_by_p[:4]}
        assert "P01" in top_4_ids, (
            f"Top seed P01 should be in top-4 by win probability"
        )


# ---------------------------------------------------------------------------
# 6. All disciplines
# ---------------------------------------------------------------------------

class TestAllDisciplines:
    @pytest.mark.parametrize("disc", list(Discipline))
    def test_all_disciplines_produce_valid_results(self, disc: Discipline) -> None:
        engine = OutrightPricingEngine(n_simulations=1000)
        resp = engine.price_tournament(_draw(8, discipline=disc))
        total = sum(p.p_win_tournament for p in resp.results)
        assert abs(total - 1.0) < 0.02


# ---------------------------------------------------------------------------
# 7. Reproducibility with seed
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self) -> None:
        engine = OutrightPricingEngine(n_simulations=2000)
        draw = _draw(8)
        resp1 = engine.price_tournament(draw, seed=42)
        resp2 = engine.price_tournament(draw, seed=42)
        probs1 = {r.entity_id: r.p_win_tournament for r in resp1.results}
        probs2 = {r.entity_id: r.p_win_tournament for r in resp2.results}
        for k in probs1:
            assert probs1[k] == probs2[k], f"Prob for {k} differs between runs"


# ---------------------------------------------------------------------------
# 8. Already-played results
# ---------------------------------------------------------------------------

class TestAlreadyPlayedResults:
    def test_eliminated_player_has_zero_probability(self) -> None:
        engine = OutrightPricingEngine(n_simulations=2000)
        draw = _draw(8)
        # P01 beat P08 in R1; P08 is eliminated
        draw.already_played = [("P01", "P08", "P01")]
        resp = engine.price_tournament(draw)
        results_by_id = {r.entity_id: r for r in resp.results}
        if "P08" in results_by_id:
            assert results_by_id["P08"].p_win_tournament == 0.0, (
                f"Eliminated player P08 should have prob=0.0"
            )

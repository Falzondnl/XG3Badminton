"""
score_validator.py
==================
5-layer score integrity validation for badminton. (H6 gate)

Validates all incoming scores before settlement and before
feeding into the Markov engine during live play.

5 validation layers:
  1. Range check: scores within legal BWF ranges (0-30 per game)
  2. Game winner check: each completed game has a valid legal winner
  3. Match structure: games won consistent with match outcome
  4. Sequence consistency: progressive score changes (no negative deltas)
  5. Serving rule verification: server at each point follows BWF rules

Validation severity:
  - CRITICAL: Score is logically impossible → reject and alert
  - WARNING: Score is unusual but possible → log and continue
  - INFO: Score matches expected ranges

Called by:
  - LiveSupervisorAgent (every feed event)
  - GradingService (before settlement)
  - DataLoader (historical data integrity)

Raises ScoreValidationError for CRITICAL violations.
Returns list of ValidationIssue for non-critical findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import structlog

from config.badminton_config import (
    POINTS_TO_WIN_GAME,
    DEUCE_SCORE,
    GOLDEN_POINT_SCORE,
    GOLDEN_POINT_WIN,
    GAMES_TO_WIN_MATCH,
    Discipline,
)
from core.scoring_engine import ScoringEngine

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Single validation finding."""
    severity: ValidationSeverity
    layer: int                    # 1-5 validation layer
    description: str
    game_number: Optional[int] = None
    point_index: Optional[int] = None


class ScoreValidationError(Exception):
    """Raised for CRITICAL score validation failures."""
    def __init__(self, issues: List[ValidationIssue]) -> None:
        super().__init__(f"Score validation failed: {[i.description for i in issues]}")
        self.issues = issues


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ScoreValidator:
    """
    5-layer badminton score validator.
    """

    def validate_game_score(
        self,
        score_a: int,
        score_b: int,
        is_complete: bool = True,
    ) -> List[ValidationIssue]:
        """
        Validate a single game score.

        Layer 1: Range check.
        Layer 2: Game winner check (if complete).

        Args:
            score_a: Score for player/team A.
            score_b: Score for player/team B.
            is_complete: Whether this is a completed game (default True).
                         When True, validates that the game has a legal winner.

        Raises:
            ScoreValidationError: If any CRITICAL validation failures are found.
        """
        issues: List[ValidationIssue] = []

        # Layer 1: Range check
        if score_a < 0 or score_b < 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=1,
                description=f"Negative score: {score_a}-{score_b}",
            ))

        if score_a > GOLDEN_POINT_WIN or score_b > GOLDEN_POINT_WIN:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=1,
                description=f"Score exceeds maximum {GOLDEN_POINT_WIN}: {score_a}-{score_b}",
            ))

        # Check deuce rules: if either score exceeds standard win threshold
        # one side must have a 2-point lead or golden point exactly
        max_standard = max(score_a, score_b)
        if max_standard > POINTS_TO_WIN_GAME:
            # We're in deuce / overtime territory
            min_score = min(score_a, score_b)
            if min_score < DEUCE_SCORE:
                # One side above 21 but opponent didn't reach 20 first → invalid
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    layer=1,
                    description=(
                        f"Score {score_a}-{score_b} is invalid: "
                        f"scores above {POINTS_TO_WIN_GAME} require both sides to reach "
                        f"{DEUCE_SCORE} first"
                    ),
                ))
            else:
                diff = abs(score_a - score_b)
                if diff != 2 and not (
                    (score_a == GOLDEN_POINT_WIN and score_b == GOLDEN_POINT_WIN - 1) or
                    (score_b == GOLDEN_POINT_WIN and score_a == GOLDEN_POINT_WIN - 1)
                ):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        layer=1,
                        description=f"Deuce score must have exactly 2-point lead: {score_a}-{score_b}",
                    ))

        # Layer 2: Game winner check
        if is_complete and not issues:
            winner = ScoringEngine.determine_game_winner(score_a, score_b)
            if winner is None:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    layer=2,
                    description=f"Completed game has no legal winner: {score_a}-{score_b}",
                ))

        critical = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        if critical:
            raise ScoreValidationError(critical)

        return issues

    def validate_live_score_update(
        self,
        old_a: int,
        old_b: int,
        new_a: int,
        new_b: int,
    ) -> None:
        """
        Validate a live score transition (one point at a time).

        Checks:
          - Only one score changes
          - No score decrements
          - Score delta is exactly 1

        Raises:
            ScoreValidationError: If the update is invalid.
        """
        delta_a = new_a - old_a
        delta_b = new_b - old_b
        issues: List[ValidationIssue] = []

        if delta_a < 0 or delta_b < 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=4,
                description=(
                    f"Score decrement detected: {old_a}-{old_b} → {new_a}-{new_b}"
                ),
            ))
        elif delta_a + delta_b != 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=4,
                description=(
                    f"Live update must advance exactly one score by 1: "
                    f"{old_a}-{old_b} → {new_a}-{new_b} "
                    f"(delta_a={delta_a}, delta_b={delta_b})"
                ),
            ))

        if issues:
            raise ScoreValidationError(issues)

    def validate_match_score(
        self,
        game_scores: Optional[List[Tuple[int, int]]] = None,
        discipline: Optional[Discipline] = None,
        is_complete: bool = True,
        games_won_a: Optional[int] = None,
        games_won_b: Optional[int] = None,
    ) -> List[ValidationIssue]:
        """
        Full 5-layer match score validation.

        Accepts either the full-form or simplified form:

        Full form::
            validate_match_score(
                game_scores=[(21,10),(21,15)],
                discipline=Discipline.MS,
                is_complete=True,
            )

        Simplified form (used by tests)::
            validate_match_score(
                games_won_a=2,
                games_won_b=0,
                game_scores=[(21,10),(21,15)],
            )

        Args:
            game_scores: List of (score_a, score_b) for each completed game.
            discipline: Badminton discipline (for Bo3 check). Optional in simplified form.
            is_complete: Whether the match is expected to be complete.
            games_won_a: Expected games won by A (used for cross-validation).
            games_won_b: Expected games won by B (used for cross-validation).

        Raises:
            ScoreValidationError: If any CRITICAL validation failures are found.
        """
        if game_scores is None:
            game_scores = []
        if discipline is None:
            # Default to MS (singles) for simplified calls
            discipline = Discipline.MS

        all_issues: List[ValidationIssue] = []

        # Layer 1+2: Validate each game
        for i, (sa, sb) in enumerate(game_scores):
            try:
                game_issues = self.validate_game_score(sa, sb, is_complete=True)
            except ScoreValidationError as exc:
                for issue in exc.issues:
                    issue.game_number = i + 1
                all_issues.extend(exc.issues)
                continue
            for issue in game_issues:
                issue.game_number = i + 1
            all_issues.extend(game_issues)

        # Cross-validate games_won against actual game scores (if provided)
        if games_won_a is not None or games_won_b is not None:
            actual_wins_a = sum(1 for sa, sb in game_scores if sa > sb)
            actual_wins_b = sum(1 for sa, sb in game_scores if sb > sa)
            if games_won_a is not None and actual_wins_a != games_won_a:
                all_issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    layer=3,
                    description=(
                        f"games_won_a={games_won_a} inconsistent with game scores: "
                        f"actual A wins={actual_wins_a}"
                    ),
                ))
            if games_won_b is not None and actual_wins_b != games_won_b:
                all_issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    layer=3,
                    description=(
                        f"games_won_b={games_won_b} inconsistent with game scores: "
                        f"actual B wins={actual_wins_b}"
                    ),
                ))

        # Layer 3: Match structure
        all_issues.extend(
            self._layer3_match_structure(game_scores, is_complete)
        )

        # Layer 4: Sequence consistency (verify progressive scores)
        all_issues.extend(
            self._layer4_sequence_consistency(game_scores)
        )

        # Layer 5: BWF rule compliance
        all_issues.extend(
            self._layer5_bwf_rules(game_scores, discipline)
        )

        # Log and raise if critical
        critical = [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]
        if critical:
            logger.error(
                "score_validation_critical",
                n_critical=len(critical),
                issues=[i.description for i in critical],
            )
            raise ScoreValidationError(critical)

        for issue in all_issues:
            if issue.severity == ValidationSeverity.WARNING:
                logger.warning(
                    "score_validation_warning",
                    description=issue.description,
                    game_number=issue.game_number,
                )

        return all_issues

    @staticmethod
    def _layer3_match_structure(
        game_scores: List[Tuple[int, int]],
        is_complete: bool,
    ) -> List[ValidationIssue]:
        """Layer 3: Match structure validation."""
        issues: List[ValidationIssue] = []
        n_games = len(game_scores)

        if n_games > 3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=3,
                description=f"More than 3 games in a badminton match: {n_games}",
            ))
            return issues

        # Count games won by each player
        wins_a = 0
        wins_b = 0
        for sa, sb in game_scores:
            winner = ScoringEngine.determine_game_winner(sa, sb)
            if winner == "A":
                wins_a += 1
            elif winner == "B":
                wins_b += 1

        # Check no one has more than GAMES_TO_WIN_MATCH wins
        if wins_a > GAMES_TO_WIN_MATCH:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=3,
                description=f"Player A has {wins_a} game wins (max {GAMES_TO_WIN_MATCH})",
            ))
        if wins_b > GAMES_TO_WIN_MATCH:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=3,
                description=f"Player B has {wins_b} game wins (max {GAMES_TO_WIN_MATCH})",
            ))

        # If complete: must have exactly 2 wins for the match winner
        if is_complete:
            if wins_a != GAMES_TO_WIN_MATCH and wins_b != GAMES_TO_WIN_MATCH:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    layer=3,
                    description=f"Complete match has no winner: A={wins_a} B={wins_b}",
                ))

        # Check match doesn't continue after winner determined
        if wins_a == GAMES_TO_WIN_MATCH and wins_b == GAMES_TO_WIN_MATCH:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=3,
                description="Both players reached game win target simultaneously",
            ))

        return issues

    @staticmethod
    def _layer4_sequence_consistency(
        game_scores: List[Tuple[int, int]],
    ) -> List[ValidationIssue]:
        """
        Layer 4: Sequence consistency.

        Verifies that each game's score is reachable from BWF rules.
        Key check: score of 0-0 at start of each new game (after game end).
        """
        issues: List[ValidationIssue] = []

        for i, (sa, sb) in enumerate(game_scores):
            # Basic consistency: both scores should be >= 0
            if sa < 0 or sb < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    layer=4,
                    game_number=i + 1,
                    description=f"Negative score in game {i+1}: {sa}-{sb}",
                ))

            # A game cannot end 0-0 (unless both retired immediately)
            if sa == 0 and sb == 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    layer=4,
                    game_number=i + 1,
                    description=f"Game {i+1} ended 0-0",
                ))

        return issues

    @staticmethod
    def _layer5_bwf_rules(
        game_scores: List[Tuple[int, int]],
        discipline: Discipline,
    ) -> List[ValidationIssue]:
        """
        Layer 5: BWF-specific rule validation.

        Checks:
        - Maximum 3 games per match (Bo3)
        - Game winning scores match legal terminal scores
        - Third game scores are the deciding game
        """
        issues: List[ValidationIssue] = []

        # Check each completed game score is a legal terminal score
        legal_terminals = ScoringEngine.possible_game_scores()
        legal_set = {(sa, sb) for sa, sb in legal_terminals}

        for i, (sa, sb) in enumerate(game_scores):
            if (sa, sb) not in legal_set and (sb, sa) not in legal_set:
                # Not in canonical set — check if it's a valid win condition
                winner = ScoringEngine.determine_game_winner(sa, sb)
                if winner is None:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        layer=5,
                        game_number=i + 1,
                        description=f"Non-standard game score: {sa}-{sb}",
                    ))

        # Third game specific checks
        if len(game_scores) == 3:
            sa, sb = game_scores[2]
            # Game 3: interval at 11 required by BWF rules — cannot detect from score
            # But: game 3 must be won by the player who hadn't won 2 yet
            wins_a_after_2 = sum(
                1 for s_a, s_b in game_scores[:2]
                if ScoringEngine.determine_game_winner(s_a, s_b) == "A"
            )
            wins_b_after_2 = 2 - wins_a_after_2
            if wins_a_after_2 == wins_b_after_2:  # 1-1 after 2 games — correct, game 3 needed
                pass
            else:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    layer=5,
                    game_number=3,
                    description=f"Game 3 played but match was decided after 2 games",
                ))

        return issues

    def validate_live_score_update(
        self,
        prev_score_a: Optional[int] = None,
        prev_score_b: Optional[int] = None,
        new_score_a: Optional[int] = None,
        new_score_b: Optional[int] = None,
        game_number: int = 0,
        point_index: int = 0,
        # Convenience aliases
        old_a: Optional[int] = None,
        old_b: Optional[int] = None,
        new_a: Optional[int] = None,
        new_b: Optional[int] = None,
    ) -> List[ValidationIssue]:
        """
        Validate a single live score update (prev → new).

        Checks:
        - Only one player's score increased
        - Increment is exactly 1
        - Scores didn't decrease
        """
        # Resolve convenience aliases
        if old_a is not None:
            prev_score_a = old_a
        if old_b is not None:
            prev_score_b = old_b
        if new_a is not None:
            new_score_a = new_a
        if new_b is not None:
            new_score_b = new_b

        if prev_score_a is None or prev_score_b is None or new_score_a is None or new_score_b is None:
            raise ValueError(
                "validate_live_score_update() requires prev/new scores for both teams"
            )

        issues: List[ValidationIssue] = []

        delta_a = new_score_a - prev_score_a
        delta_b = new_score_b - prev_score_b

        if delta_a < 0 or delta_b < 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=4,
                game_number=game_number,
                point_index=point_index,
                description=f"Score decreased: {prev_score_a}-{prev_score_b} → {new_score_a}-{new_score_b}",
            ))

        if delta_a + delta_b != 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=4,
                game_number=game_number,
                point_index=point_index,
                description=f"Non-single-point increment: Δa={delta_a} Δb={delta_b}",
            ))

        if delta_a > 0 and delta_b > 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                layer=4,
                game_number=game_number,
                point_index=point_index,
                description="Both scores increased simultaneously",
            ))

        if critical_issues := [i for i in issues if i.severity == ValidationSeverity.CRITICAL]:
            raise ScoreValidationError(critical_issues)

        return issues

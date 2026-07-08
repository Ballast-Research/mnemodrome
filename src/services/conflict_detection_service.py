"""Conflict detection service for Feature 006 (Fact Invalidation).

Provides complexity scoring and conflict detection for temporal validity management.
Implements progressive automation logic to determine when human validation is required.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from services.neo4j_service import Neo4jService
    from services.validation_history_service import ValidationHistoryService


def calculate_complexity_score(
    old_fact: Any,
    new_fact: dict,
    validation_history: list | None = None,
    competing_facts_count: int | None = None,
    has_dependent_facts: bool | None = None,
) -> int:
    """
    Calculate conflict complexity score (0-100).

    Higher scores indicate more complex conflicts requiring human validation.

    Complexity factors:
    - Age of fact being invalidated (>30 days = +30 points)
    - Multiple competing facts exist (>1 = +20 points)
    - Cross-entity dependencies detected (+25 points)
    - No historical validation pattern match (+25 points)
    - Recent fact creation (<24 hours = -20 points, likely related update)

    Args:
        old_fact: Existing relationship object with created_at attribute
        new_fact: New fact data (dict with entity/relation info)
        validation_history: Optional list of ValidationHistory objects
        competing_facts_count: Optional count of competing facts (default: 0)
        has_dependent_facts: Optional flag for dependencies (default: False)

    Returns:
        Complexity score clamped to 0-100 range

    Algorithm契約:
    - Score 0-30: Simple conflict (auto-resolvable)
    - Score 31-70: Moderate conflict (prompt with recommendation)
    - Score 71-100: Complex conflict (always require validation)
    """
    score = 0

    # Handle None defaults
    if validation_history is None:
        validation_history = []
    if competing_facts_count is None:
        competing_facts_count = 0
    if has_dependent_facts is None:
        has_dependent_facts = False

    # Extract created_at from old_fact
    # Support both object attributes and dict access
    if hasattr(old_fact, "created_at"):
        created_at = old_fact.created_at
    elif isinstance(old_fact, dict) and "created_at" in old_fact:
        created_at = old_fact["created_at"]
    else:
        # Fallback: Do NOT treat as old - this would unfairly penalize
        # Instead, skip age calculation by setting to current time
        logger.warning("old_fact missing created_at, skipping age factor")
        created_at = datetime.now(UTC)  # Age will be 0

    # Ensure created_at is a datetime object
    if isinstance(created_at, str):
        # Parse ISO format datetime string
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

    # Make timezone-aware if needed
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    # Age factor
    now = datetime.now(UTC)
    age_seconds = (now - created_at).total_seconds()
    age_days = age_seconds / 86400  # Convert to days

    # Use > 30 with small epsilon for floating point robustness
    # At exactly 30.0 days (2592000 seconds), should NOT trigger penalty
    if age_seconds > (
        30 * 86400 + 60
    ):  # More than 30 days + 1 minute buffer for floating point
        score += 30
    elif age_seconds < 86400:  # Less than 24 hours (1 day in seconds)
        score -= 20  # Recent = likely related update

    # Competing facts factor
    if competing_facts_count > 1:
        score += 20

    # Dependencies factor
    if has_dependent_facts:
        score += 25

    # Pattern matching factor
    if len(validation_history) == 0:
        score += 25  # Novel conflict type - no historical pattern
    else:
        # Check if ALL validations were approvals (strong pattern)
        all_approvals = all(
            hasattr(v, "human_decision") and v.human_decision == "approve"
            for v in validation_history
        )

        if all_approvals:
            score -= 20  # Strong pattern - likely simple
        # Mixed or all rejections: no score adjustment (neutral)

    # Clamp to 0-100 range
    return max(0, min(100, score))


async def detect_conflicts(
    neo4j_service: Neo4jService, entity1: str, relation: str, entity2_new: str
) -> list[dict]:
    """
    Query Neo4j for existing valid facts that conflict.

    A conflict exists when:
    - Same entity1 and relation
    - Different entity2
    - Relationship is currently valid (invalid_at IS NULL AND expired_at IS NULL)

    Args:
        neo4j_service: Neo4j service instance
        entity1: Source entity name
        relation: Relationship type
        entity2_new: New target entity (to detect conflicts)

    Returns:
        List of conflicting relationship dictionaries
    """
    # D4 (FR-006): return created_at and valid_at as separate fields.
    # D7 (FR-012): scope to native edges (r.type IS NOT NULL).
    cypher = """
    MATCH (source {name: $entity1})-[r:RELATES_TO]->(target)
    WHERE r.type = $relation
      AND r.type IS NOT NULL
      AND r.invalid_at IS NULL
      AND r.expired_at IS NULL
      AND target.name <> $entity2_new
    RETURN
      r.uuid as uuid,
      source.name as entity1,
      r.type as relation,
      target.name as entity2,
      coalesce(r.created_at, r.valid_at) as created_at,
      r.valid_at as valid_at,
      r.context as context
    ORDER BY r.valid_at DESC
    """

    params = {"entity1": entity1, "relation": relation, "entity2_new": entity2_new}

    results = await neo4j_service.query(cypher, params)

    logger.debug(
        f"Conflict detection for {entity1} -{relation}-> {entity2_new}: "
        f"found {len(results)} conflicts"
    )

    return results


def classify_complexity(score: int) -> str:
    """
    Classify complexity score into categories.

    Args:
        score: Complexity score (0-100)

    Returns:
        Classification string: "simple", "moderate", or "complex"

    Thresholds:
    - 0-30: "simple" (auto-resolvable)
    - 31-70: "moderate" (prompt for validation)
    - 71-100: "complex" (always require validation)
    """
    if score <= 30:
        return "simple"
    elif score <= 70:
        return "moderate"
    else:
        return "complex"


# Service instance (can be used as singleton)
class ConflictDetectionService:
    """Service class for conflict detection operations."""

    def __init__(self):
        """Initialize conflict detection service."""
        pass

    async def detect_conflict(
        self, neo4j_service: Neo4jService, entity1: str, relation: str, entity2: str
    ) -> dict:
        """
        Detect conflicts and calculate complexity.

        Args:
            neo4j_service: Neo4j service instance
            entity1: Source entity name
            relation: Relationship type
            entity2: Target entity name

        Returns:
            Conflict detection result with complexity assessment
        """
        conflicts = await detect_conflicts(neo4j_service, entity1, relation, entity2)

        if not conflicts:
            return {
                "conflict_detected": False,
                "existing_facts": [],
                "complexity_score": 0,
                "auto_resolvable": True,
            }

        # For simplicity, assess complexity of first conflict
        # (in practice, might want to assess all and take max)
        old_fact = conflicts[0]

        # Mock validation history for now (will integrate with validation_history_service)
        validation_history = []

        complexity_score = calculate_complexity_score(
            old_fact=old_fact,
            new_fact={"entity1": entity1, "relation": relation, "entity2": entity2},
            validation_history=validation_history,
            competing_facts_count=len(conflicts),
            has_dependent_facts=False,  # Would need dependency check
        )

        complexity_class = classify_complexity(complexity_score)

        return {
            "conflict_detected": True,
            "existing_facts": conflicts,
            "complexity_score": complexity_score,
            "complexity_class": complexity_class,
            "auto_resolvable": complexity_class == "simple",
            "recommendation": "auto_resolve"
            if complexity_class == "simple"
            else "validate",
        }

    async def calculate_complexity_score(
        self,
        old_fact: dict,
        new_fact: dict,
        validation_history: list | None = None,
        competing_facts_count: int | None = None,
        has_dependent_facts: bool | None = None,
    ) -> dict:
        """
        Calculate complexity score with detailed factors.

        Args:
            old_fact: Existing fact dictionary
            new_fact: New fact data
            validation_history: Optional validation history
            competing_facts_count: Optional count of competing facts
            has_dependent_facts: Optional flag for dependencies

        Returns:
            Complexity assessment with score and factors breakdown
        """
        score = calculate_complexity_score(
            old_fact=old_fact,
            new_fact=new_fact,
            validation_history=validation_history,
            competing_facts_count=competing_facts_count,
            has_dependent_facts=has_dependent_facts,
        )

        return {"score": score, "auto_resolvable": score <= 30}

    async def auto_resolve_conflict(
        self,
        neo4j_service: Neo4jService,
        old_fact_uuid: str,
        new_fact: dict,
        reason: str,
    ) -> dict:
        """
        Auto-resolve a simple conflict by invalidating old fact.

        Args:
            neo4j_service: Neo4j service instance
            old_fact_uuid: UUID of old fact to invalidate
            new_fact: New fact data to create
            reason: Reason for auto-resolution

        Returns:
            Resolution result
        """
        # This would be implemented with actual Neo4j transaction
        # For now, return mock success
        return {
            "status": "resolved",
            "old_fact_invalidated": True,
            "new_fact_created": True,
        }

    async def resolve_with_validation(
        self,
        neo4j_service: Neo4jService,
        old_fact_uuid: str,
        new_fact: dict,
        validation_uuid: str,
        reason: str,
    ) -> dict:
        """
        Resolve conflict after human validation.

        Args:
            neo4j_service: Neo4j service instance
            old_fact_uuid: UUID of old fact to invalidate
            new_fact: New fact data to create
            validation_uuid: UUID of validation record
            reason: Reason for resolution

        Returns:
            Resolution result with validation tracking
        """
        # This would be implemented with actual Neo4j transaction
        # For now, return mock success
        return {"status": "resolved", "validation_recorded": True}

    async def calculate_pattern_confidence(
        self,
        conflict_type: str,
        validation_history_service: ValidationHistoryService | None = None,
    ) -> float:
        """
        Calculate confidence based on validation history patterns.

        Args:
            conflict_type: Type of conflict (e.g., "same_relation_different_target")
            validation_history_service: Optional validation history service

        Returns:
            Confidence score (0.0-1.0)
        """
        # Mock implementation - would query validation_history_service
        if validation_history_service is None:
            return 0.0

        # This would be implemented with actual query
        return 0.0

    async def assess_complexity(
        self, old_fact: dict, new_fact: dict, validation_history: list | None = None
    ) -> dict:
        """
        Assess complexity and auto-resolvability (wrapper for contract compatibility).

        Args:
            old_fact: Old fact data
            new_fact: New fact data
            validation_history: Optional validation history

        Returns:
            Dict with complexity_score, auto_resolvable, recommendation, etc.
        """
        score = calculate_complexity_score(
            old_fact=old_fact,
            new_fact=new_fact,
            validation_history=validation_history or [],
        )

        auto_resolvable = score < 50
        complexity_class = classify_complexity(score)

        result = {
            "complexity_score": score,
            "auto_resolvable": auto_resolvable,
            "recommendation": "supersede" if auto_resolvable else "validate",
            "complexity_class": complexity_class,
        }

        # Add rationale for non-auto-resolvable cases
        if not auto_resolvable:
            rationale_parts = []
            if score > 70:
                rationale_parts.append("High complexity conflict")
            if validation_history is None or len(validation_history) == 0:
                rationale_parts.append("No historical validation pattern found")
            result["rationale"] = (
                ". ".join(rationale_parts)
                if rationale_parts
                else f"Complexity score {score}"
            )

        return result

    async def _clamp_score(self, score: int) -> int:
        """
        Clamp score to 0-100 range.

        Args:
            score: Raw score

        Returns:
            Clamped score
        """
        return max(0, min(100, score))


# Global service instance
conflict_detection_service = ConflictDetectionService()

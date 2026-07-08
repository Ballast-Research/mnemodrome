"""
Validation history service for pattern learning and auto-resolution (Feature 006).

This service manages ValidationHistory nodes in Neo4j to enable progressive automation
of conflict resolution through pattern recognition.

Pattern Recognition Strategy:
- Store human validation decisions with conflict metadata
- Analyze patterns across similar conflict types
- Calculate confidence scores for auto-resolution
- Enable smart automation based on historical approvals

Minimum Pattern Recognition Threshold: 10 validations required
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from loguru import logger

if TYPE_CHECKING:
    from src.services.neo4j_service import Neo4jService


async def create_validation_history(
    neo4j_service: Neo4jService,
    conflict_type: str,
    complexity_score: int,
    human_decision: str,
    old_fact_uuid: str,
    new_fact_uuid: str | None = None,
    rationale: str | None = None,
) -> str:
    """
    Create ValidationHistory node in Neo4j to record human validation decision.

    Args:
        neo4j_service: Neo4j service instance for database operations
        conflict_type: Type of conflict (e.g., "infrastructure_relocation")
        complexity_score: Calculated complexity score (0-100)
        human_decision: Human decision - "approve", "reject", or "modify"
        old_fact_uuid: UUID of the old/existing fact
        new_fact_uuid: UUID of the new fact (None if rejected)
        rationale: Optional human explanation for the decision

    Returns:
        UUID of created ValidationHistory node

    Raises:
        Exception if Neo4j write fails
    """
    await neo4j_service.connect()
    assert neo4j_service.driver is not None

    validation_uuid = str(uuid4())
    timestamp = datetime.now(UTC)

    cypher = """
    CREATE (v:ValidationHistory {
        uuid: $uuid,
        conflict_type: $conflict_type,
        complexity_score: $complexity_score,
        human_decision: $human_decision,
        timestamp: $timestamp,
        old_fact_uuid: $old_fact_uuid,
        new_fact_uuid: $new_fact_uuid,
        rationale: $rationale
    })
    RETURN v.uuid as uuid
    """

    params = {
        "uuid": validation_uuid,
        "conflict_type": conflict_type,
        "complexity_score": complexity_score,
        "human_decision": human_decision,
        "timestamp": timestamp.isoformat(),
        "old_fact_uuid": old_fact_uuid,
        "new_fact_uuid": new_fact_uuid,
        "rationale": rationale or "",
    }

    async with neo4j_service.driver.session(database=neo4j_service.database) as session:
        result = await session.run(cypher, params)
        record = await result.single()

        if not record:
            raise Exception(f"Failed to create ValidationHistory: {validation_uuid}")

        logger.info(
            f"Created ValidationHistory: {validation_uuid} "
            f"(type={conflict_type}, decision={human_decision}, score={complexity_score})"
        )

        return record["uuid"]


def find_similar_conflicts(
    validation_history: list[dict], old_fact: dict, new_fact: dict
) -> list[dict]:
    """
    Find similar conflicts in validation history for pattern matching.

    Similarity criteria:
    - Same conflict type (e.g., "infrastructure_relocation")
    - Similar complexity score (±10 points)

    Args:
        validation_history: List of ValidationHistory records (dicts or dataclass objects)
        old_fact: Old/existing fact being considered for invalidation (dict or dataclass)
        new_fact: New fact being proposed (dict or dataclass)

    Returns:
        List of similar validation decisions
    """
    # Determine conflict type for current scenario
    conflict_type = determine_conflict_type(old_fact, new_fact)

    # Calculate complexity score for similarity matching
    complexity_score = _calculate_complexity_for_matching(old_fact, new_fact)

    similar = []
    for validation in validation_history:
        # Handle both dict and dataclass objects
        if hasattr(validation, "__dataclass_fields__"):
            # Dataclass object - use attribute access
            val_conflict_type = validation.conflict_type
            val_complexity_score = validation.complexity_score
        else:
            # Dict object - use dict access
            val_conflict_type = validation.get("conflict_type")
            val_complexity_score = validation.get("complexity_score", 0)

        # Match on conflict type
        if val_conflict_type != conflict_type:
            continue

        # Match on similar complexity (±10 points)
        if abs(val_complexity_score - complexity_score) <= 10:
            similar.append(validation)

    logger.debug(
        f"Found {len(similar)} similar conflicts "
        f"(type={conflict_type}, complexity≈{complexity_score})"
    )

    return similar


def calculate_confidence_score(similar_validations: list[dict]) -> int:
    """
    Calculate confidence score (0-100) for auto-resolution based on approval ratio.

    Confidence = (approvals / total) * 100

    Pattern recognition threshold: Requires 10+ validations for statistical significance.
    If fewer than 10 validations exist, returns 0 (insufficient data).

    Args:
        similar_validations: List of similar validation decisions (dicts or dataclass objects)

    Returns:
        Confidence score 0-100 (0 if insufficient data)
    """
    # Require minimum 10 validations for pattern recognition
    if len(similar_validations) < 10:
        return 0

    # Count approvals
    approvals = 0
    for v in similar_validations:
        # Handle both dict and dataclass objects
        if hasattr(v, "__dataclass_fields__"):
            decision = v.human_decision
        else:
            decision = v.get("human_decision")

        if decision == "approve":
            approvals += 1

    # Calculate approval ratio
    confidence = int((approvals / len(similar_validations)) * 100)

    logger.debug(
        f"Confidence: {confidence}% ({approvals}/{len(similar_validations)} approvals)"
    )

    return confidence


def is_simple_conflict(
    similar_validations: list[dict], confidence_threshold: int = 80
) -> bool:
    """
    Determine if conflict is "simple" enough for auto-resolution.

    Simple conflict criteria:
    - 10+ similar conflicts in validation history
    - Approval ratio >= confidence_threshold (default 80%)
    - No recent rejections (last 5 similar conflicts)

    Args:
        similar_validations: List of similar validation decisions (dicts or dataclass objects)
        confidence_threshold: Minimum approval percentage required (default 80)

    Returns:
        True if conflict can be auto-resolved
    """
    # Calculate confidence
    confidence = calculate_confidence_score(similar_validations)

    # Insufficient data
    if confidence == 0:
        return False

    # Below threshold
    if confidence < confidence_threshold:
        return False

    # Check for recent rejections (pattern change detection)
    # Sort by timestamp (most recent first)
    def get_timestamp(v):
        """Extract timestamp from dict or dataclass."""
        if hasattr(v, "__dataclass_fields__"):
            return v.timestamp
        return v.get("timestamp", "")

    recent_validations = sorted(similar_validations, key=get_timestamp, reverse=True)[
        :5
    ]

    # Any recent rejection indicates pattern change
    has_recent_rejection = False
    for v in recent_validations:
        # Handle both dict and dataclass objects
        if hasattr(v, "__dataclass_fields__"):
            decision = v.human_decision
        else:
            decision = v.get("human_decision")

        if decision == "reject":
            has_recent_rejection = True
            break

    if has_recent_rejection:
        logger.info(
            f"Pattern change detected: Recent rejection found despite {confidence}% confidence"
        )
        return False

    logger.info(
        f"Simple conflict detected: {confidence}% confidence, no recent rejections"
    )

    return True


async def get_validation_history_by_type(
    neo4j_service: Neo4jService, conflict_type: str
) -> list[dict]:
    """
    Query ValidationHistory nodes by conflict type.

    Args:
        neo4j_service: Neo4j service instance
        conflict_type: Conflict type to filter by

    Returns:
        List of validation decisions, ordered by timestamp (newest first)
    """
    await neo4j_service.connect()

    cypher = """
    MATCH (v:ValidationHistory)
    WHERE v.conflict_type = $conflict_type
    RETURN
        v.uuid as uuid,
        v.conflict_type as conflict_type,
        v.complexity_score as complexity_score,
        v.human_decision as human_decision,
        v.timestamp as timestamp,
        v.old_fact_uuid as old_fact_uuid,
        v.new_fact_uuid as new_fact_uuid,
        v.rationale as rationale
    ORDER BY v.timestamp DESC
    """

    params = {"conflict_type": conflict_type}

    results = await neo4j_service.query(cypher, params)

    logger.debug(
        f"Retrieved {len(results)} validation records for type={conflict_type}"
    )

    return results


def determine_conflict_type(old_fact: dict, new_fact: dict) -> str:
    """
    Determine conflict type from fact relationship patterns.

    Conflict types:
    - "infrastructure_relocation": RUNS_ON/HOSTED_ON relationship changes
    - "ownership_transfer": OWNED_BY relationship changes
    - "same_relation_different_target": Same relation, different target entity
    - "temporal_update": Same relationship, different time context

    Args:
        old_fact: Existing fact details (dict or dataclass)
        new_fact: New fact details (dict or dataclass)

    Returns:
        Conflict type identifier string
    """
    # Handle both dict and dataclass objects
    if hasattr(old_fact, "__dataclass_fields__"):
        relation = old_fact.relation
        old_entity2 = old_fact.entity2
    else:
        relation = old_fact.get("relation", "")
        old_entity2 = old_fact.get("entity2")

    if hasattr(new_fact, "__dataclass_fields__"):
        new_entity2 = new_fact.entity2
    else:
        new_entity2 = new_fact.get("entity2")

    # Infrastructure-related conflicts
    if relation in ["RUNS_ON", "HOSTED_ON"]:
        return "infrastructure_relocation"

    # Ownership conflicts
    if relation == "OWNED_BY":
        return "ownership_transfer"

    # Generic relationship target change
    if old_entity2 != new_entity2:
        return "same_relation_different_target"

    # Default: temporal update
    return "temporal_update"


def _calculate_complexity_for_matching(old_fact: dict, new_fact: dict) -> int:
    """
    Calculate simplified complexity score for similarity matching.

    This is a lightweight version focused on grouping similar conflicts.
    Full complexity calculation is in conflict_detection_service.py.

    Factors:
    - Age: >30 days (+30), >7 days (+15), <1 day (-10)
    - Infrastructure relations: -10 (generally simpler)

    Args:
        old_fact: Existing fact (dict or dataclass)
        new_fact: New fact (dict or dataclass)

    Returns:
        Complexity score 0-100
    """
    score = 0

    # Handle both dict and dataclass objects
    if hasattr(old_fact, "__dataclass_fields__"):
        created_at = old_fact.created_at
        relation = old_fact.relation
    else:
        created_at_str = old_fact.get("created_at", "")
        relation = old_fact.get("relation", "")
        created_at = None

    # Parse created_at timestamp
    try:
        # If created_at is already a datetime object
        if isinstance(created_at, datetime):
            now = datetime.now(UTC)
            # Ensure created_at is timezone-aware
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            age_days = (now - created_at).days
        # If created_at is a string
        elif isinstance(created_at, str):
            created_at_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            now = datetime.now(UTC)
            age_days = (now - created_at_dt).days
        else:
            age_days = 0

        # Age factor
        if age_days > 30:
            score += 30
        elif age_days > 7:
            score += 15
        else:
            # Recent facts (1-7 days or < 1 day)
            # Base score of 15 for baseline complexity
            score += 15
    except (ValueError, AttributeError, TypeError):
        # Invalid timestamp, assume neutral
        pass

    # Infrastructure relations are generally simpler
    if relation in ["RUNS_ON", "HOSTED_ON"]:
        score -= 10

    # Clamp to 0-100
    return max(0, min(100, score))


class ValidationHistoryService:
    """
    Service class for validation history operations.

    Provides instance methods for integration with other services.
    """

    async def create_validation_record(
        self,
        neo4j_service: Neo4jService,
        conflict_type: str,
        complexity_score: int,
        human_decision: str,
        old_fact_uuid: str,
        new_fact_uuid: str | None = None,
        rationale: str | None = None,
    ) -> dict[str, Any]:
        """
        Create validation history record (instance method wrapper).

        Returns:
            Dict with uuid and status
        """
        uuid = await create_validation_history(
            neo4j_service=neo4j_service,
            conflict_type=conflict_type,
            complexity_score=complexity_score,
            human_decision=human_decision,
            old_fact_uuid=old_fact_uuid,
            new_fact_uuid=new_fact_uuid,
            rationale=rationale,
        )

        return {"uuid": uuid, "status": "created"}

    async def get_by_conflict_type(
        self, neo4j_service: Neo4jService, conflict_type: str, limit: int = 100
    ) -> list[dict]:
        """
        Get validation history by conflict type (instance method wrapper).

        Args:
            neo4j_service: Neo4j service instance
            conflict_type: Conflict type to filter by
            limit: Maximum records to return

        Returns:
            List of validation records
        """
        results = await get_validation_history_by_type(
            neo4j_service=neo4j_service, conflict_type=conflict_type
        )

        return results[:limit]

    def find_similar(
        self, validation_history: list[dict], old_fact: dict, new_fact: dict
    ) -> list[dict]:
        """
        Find similar conflicts (instance method wrapper).
        """
        return find_similar_conflicts(validation_history, old_fact, new_fact)

    def calculate_confidence(self, similar_validations: list[dict]) -> int:
        """
        Calculate confidence score (instance method wrapper).
        """
        return calculate_confidence_score(similar_validations)

    def is_simple(
        self, similar_validations: list[dict], confidence_threshold: int = 80
    ) -> bool:
        """
        Determine if conflict is simple (instance method wrapper).
        """
        return is_simple_conflict(similar_validations, confidence_threshold)


# Global service instance
validation_history_service = ValidationHistoryService()

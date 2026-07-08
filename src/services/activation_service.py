"""
ACT-R Cognitive Activation Service (Feature 034).

Implements base-level activation scoring, Hebbian co-occurrence strengthening,
and Bayesian confidence scoring based on Anderson's ACT-R cognitive model (1993).

Pure calculation functions (no side effects):
- calculate_activation(): ACT-R base-level activation formula
- compute_final_score(): Weighted blend for retrieval ranking
- bayesian_update(): Odds-ratio Bayesian confidence update

Async service methods (fire-and-forget):
- update_access_tracking(): Increment access_count on retrieved nodes
- strengthen_hebbian_links(): Strengthen co-occurrence weights on existing edges
- update_confidence(): Bayesian confidence adjustment with event log audit
"""

import math
from datetime import UTC, datetime

from loguru import logger

from src.config import (
    ACTR_ACTIVATION_WEIGHT,
    ACTR_CONFIDENCE_DEFAULT,
    ACTR_CONTENT_WEIGHT,
    ACTR_DECAY_EXPONENT,
    ACTR_HEBBIAN_INCREMENT,
    ACTR_HEBBIAN_WEIGHT,
)
from src.services.event_log_service import event_log_service
from src.services.neo4j_service import neo4j_service

# --- Pure Functions (no side effects) ---


def calculate_activation(
    access_count: int,
    last_accessed: datetime,
    decay_exponent: float = ACTR_DECAY_EXPONENT,
) -> float:
    """
    Calculate ACT-R base-level activation.

    Formula: B(M) = ln(n+1) - d * ln(ageDays / (n+1))

    Args:
        access_count: Number of times this memory has been retrieved.
        last_accessed: When this memory was last accessed.
        decay_exponent: Power law decay rate (default 0.5, from Anderson 1993).

    Returns:
        Float activation score (unbounded, typically -3.0 to +6.0).
    """
    now = datetime.now(UTC)
    # Handle timezone-naive datetimes
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=UTC)

    age_days = max((now - last_accessed).total_seconds() / 86400, 0.001)
    n = max(access_count, 0)

    return math.log(n + 1) - decay_exponent * math.log(age_days / (n + 1))


def compute_final_score(
    content_match: float,
    activation: float,
    hebbian_boost: float,
    w1: float = ACTR_CONTENT_WEIGHT,
    w2: float = ACTR_ACTIVATION_WEIGHT,
    w3: float = ACTR_HEBBIAN_WEIGHT,
) -> float:
    """
    Compute weighted blend score for retrieval ranking.

    Normalizes activation via softplus to keep it positive, then blends
    content match, activation, and Hebbian boost with configurable weights.

    Args:
        content_match: Vector similarity or temporal rank position (0.0-1.0).
        activation: From calculate_activation() (unbounded).
        hebbian_boost: From relationship hebbian_weight (0.0+).
        w1: Content match weight (default 0.7, dominant).
        w2: Activation weight (default 0.2, modest).
        w3: Hebbian weight (default 0.1, subtle).

    Returns:
        Float final score for ranking (higher is better).
    """
    # softplus normalization: log(1 + exp(x)), always positive
    normalized_activation = math.log1p(math.exp(activation))

    return (content_match * w1) + (normalized_activation * w2) + (hebbian_boost * w3)


def bayesian_update(
    current_confidence: float,
    evidence_strength: float,
    is_supporting: bool,
) -> float:
    """
    Update confidence using Bayesian odds ratio.

    Converts confidence to odds, multiplies (support) or divides (contradiction)
    by evidence strength, converts back, and clamps to [0.01, 0.99].

    Args:
        current_confidence: Current confidence (0.0-1.0).
        evidence_strength: Likelihood ratio (>1.0 = strong evidence).
        is_supporting: True for supporting evidence, False for contradictory.

    Returns:
        New confidence value clamped to [0.01, 0.99].
    """
    # Convert confidence to odds ratio
    odds = current_confidence / max(1 - current_confidence, 0.001)

    if is_supporting:
        odds *= evidence_strength
    else:
        odds /= evidence_strength

    # Convert back to confidence and clamp
    new_confidence = odds / (odds + 1)
    return min(max(new_confidence, 0.01), 0.99)


# --- Async Service Methods ---


class ActivationService:
    """Service for ACT-R activation dynamics on the Mnemodrome knowledge graph."""

    async def update_access_tracking(self, node_uuids: list[str]) -> None:
        """
        Increment access_count and update last_accessed for retrieved nodes.

        Fire-and-forget: fails silently with warning log (FR-008).
        Uses coalesce for lazy initialization of nodes without activation properties.

        Args:
            node_uuids: UUIDs of nodes that were retrieved.
        """
        if not node_uuids:
            return

        try:
            await neo4j_service.batch_update_access(node_uuids)
        except Exception as e:
            logger.warning(f"Access tracking update failed (non-blocking): {e}")

    async def strengthen_hebbian_links(
        self,
        activated_uuids: list[str],
        increment: float = ACTR_HEBBIAN_INCREMENT,
    ) -> None:
        """
        Strengthen existing RELATES_TO edges between co-activated entities.

        Fire-and-forget: only updates valid edges (invalid_at IS NULL).
        Does NOT create new edges — only strengthens existing associations.

        Args:
            activated_uuids: UUIDs of entities activated in same retrieval context.
            increment: Weight increment per co-activation (default 0.1).
        """
        if len(activated_uuids) < 2:
            return

        # Generate all unique pairs
        pairs = [
            [u1, u2]
            for i, u1 in enumerate(activated_uuids)
            for u2 in activated_uuids[i + 1 :]
        ]

        try:
            # D7 (FR-012): scope to native edges only (r.type IS NOT NULL)
            cypher = """
            UNWIND $pairs AS pair
            MATCH (e1 {uuid: pair[0]})-[r:RELATES_TO]-(e2 {uuid: pair[1]})
            WHERE r.invalid_at IS NULL AND r.type IS NOT NULL
            SET r.hebbian_weight = coalesce(r.hebbian_weight, 0.0) + $increment,
                r.last_coactivation = datetime()
            """
            await neo4j_service.query(cypher, {"pairs": pairs, "increment": increment})
        except Exception as e:
            logger.warning(f"Hebbian link strengthening failed (non-blocking): {e}")

    async def update_confidence(
        self,
        relationship_uuid: str,
        evidence_strength: float,
        is_supporting: bool = True,
    ) -> float:
        """
        Update confidence on a relationship using Bayesian odds ratio.

        Reads current confidence from Neo4j, applies bayesian_update(),
        writes back, and logs the adjustment to EventLogService (FR-012).

        Args:
            relationship_uuid: UUID identifying the RELATES_TO relationship.
            evidence_strength: Likelihood ratio (>1.0 = strong evidence).
            is_supporting: True for supporting, False for contradictory.

        Returns:
            New confidence value after update.

        Raises:
            Exception: If relationship not found.
        """
        # D7 (FR-012) + cardinality guard (Issue 6): scope to native edges and ensure
        # exactly ONE edge matches — duplicate-uuid must never fan-out reads/writes.
        read_cypher = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.uuid = $uuid AND r.type IS NOT NULL
        WITH collect(r) AS rs
        WHERE size(rs) = 1
        WITH rs[0] AS r
        RETURN coalesce(r.confidence, $default_confidence) AS confidence
        """
        result = await neo4j_service.query(
            read_cypher,
            {
                "uuid": relationship_uuid,
                "default_confidence": ACTR_CONFIDENCE_DEFAULT,
            },
        )

        if not result:
            # Distinguish not-found from ambiguous duplicate
            count_result = await neo4j_service.query(
                "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() WHERE r.type IS NOT NULL RETURN count(r) AS cnt",
                {"uuid": relationship_uuid},
            )
            cnt = count_result[0]["cnt"] if count_result else 0
            if cnt == 0:
                raise Exception(
                    f"Relationship not found for confidence update: {relationship_uuid}"
                )
            raise ValueError(
                f"Ambiguous duplicate: {cnt} native edges share uuid={relationship_uuid}. "
                "Cannot safely update confidence — remediate duplicates (D6) first."
            )

        old_confidence = result[0]["confidence"]
        new_confidence = bayesian_update(
            old_confidence, evidence_strength, is_supporting
        )

        # Write updated confidence (D7 + cardinality guard: same pattern as read)
        write_cypher = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.uuid = $uuid AND r.type IS NOT NULL
        WITH collect(r) AS rs
        WHERE size(rs) = 1
        WITH rs[0] AS r
        SET r.confidence = $new_confidence
        """
        await neo4j_service.query(
            write_cypher,
            {
                "uuid": relationship_uuid,
                "new_confidence": new_confidence,
            },
        )

        # Log to event log (FR-012, NFR-004)
        await event_log_service.log_event(
            event_type="memory_operation",
            action="confidence_adjustment",
            actor="activation_service",
            target=relationship_uuid,
            result="success",
            metadata={
                "old_confidence": round(old_confidence, 4),
                "new_confidence": round(new_confidence, 4),
                "evidence_strength": evidence_strength,
                "is_supporting": is_supporting,
            },
        )

        logger.debug(
            f"Confidence updated: {relationship_uuid} "
            f"{old_confidence:.3f} → {new_confidence:.3f} "
            f"({'support' if is_supporting else 'contradict'}, strength={evidence_strength})"
        )

        return new_confidence


# Global singleton instance
activation_service = ActivationService()

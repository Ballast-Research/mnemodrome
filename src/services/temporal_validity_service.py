"""
Temporal Validity Service - Feature 006 Implementation

Manages temporal validity fields for fact invalidation and temporal queries.
Implements the decision tree for setting invalid_at, expired_at, or both.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from services.neo4j_service import Neo4jService


class TemporalValidityService:
    """
    Service for managing temporal validity fields on Neo4j relationships.

    Provides:
    - Field setting logic based on invalidation type
    - Query builder for temporal filtering
    - Relationship update operations

    Feature 006: Fact Invalidation and Temporal Validity
    """

    # Invalidation type constants
    NATURAL = "natural"
    SYSTEM = "system"
    BOTH = "both"

    # Temporal scope constants
    CURRENT = "current"
    HISTORICAL = "historical"
    NATURAL_INVALID = "natural"
    SYSTEM_INVALID = "system"

    def set_temporal_fields(
        self, relationship_uuid: str, invalidation_type: str = "both", reason: str = ""
    ) -> dict[str, datetime]:
        """
        Set temporal validity fields based on invalidation type.

        Decision tree:
        - "natural": Sets only invalid_at (fact ended naturally in real world)
        - "system": Sets only expired_at (system invalidated fact due to error)
        - "both": Sets both fields (default, strongest invalidation)

        Args:
            relationship_uuid: UUID of the relationship being invalidated
            invalidation_type: Type of invalidation ("natural" | "system" | "both")
            reason: Human-readable reason for invalidation

        Returns:
            Dict with fields that were set and their datetime values

        Raises:
            ValueError: If invalidation_type is not one of the valid options

        Example:
            # Natural end (project completed)
            fields = service.set_temporal_fields("uuid-123", "natural", "Project completed")
            # Returns: {"invalid_at": datetime(...)}

            # System correction (duplicate detected)
            fields = service.set_temporal_fields("uuid-456", "system", "Duplicate")
            # Returns: {"expired_at": datetime(...)}

            # Both (default)
            fields = service.set_temporal_fields("uuid-789")
            # Returns: {"invalid_at": datetime(...), "expired_at": datetime(...)}
        """
        now = datetime.now(UTC)
        fields_set = {}

        if invalidation_type == self.NATURAL:
            # Fact ended naturally (real-world change)
            fields_set["invalid_at"] = now
            logger.debug(f"Natural invalidation: {relationship_uuid} - {reason}")

        elif invalidation_type == self.SYSTEM:
            # System invalidated fact (error correction)
            fields_set["expired_at"] = now
            logger.debug(f"System invalidation: {relationship_uuid} - {reason}")

        elif invalidation_type == self.BOTH:
            # Both fields set (default behavior)
            fields_set["invalid_at"] = now
            fields_set["expired_at"] = now
            logger.debug(f"Complete invalidation: {relationship_uuid} - {reason}")

        else:
            raise ValueError(f"Invalid invalidation_type: {invalidation_type}")

        return fields_set

    def build_temporal_query(self, temporal_scope: str = "current") -> str:
        """
        Build Cypher query fragment with temporal filtering.

        Temporal scopes:
        - "current": Only valid relationships (WHERE r.invalid_at IS NULL AND r.expired_at IS NULL)
        - "historical": All relationships with temporal metadata included
        - "natural": Naturally ended facts (WHERE r.invalid_at IS NOT NULL AND r.expired_at IS NULL)
        - "system": System-invalidated facts (WHERE r.expired_at IS NOT NULL)

        Args:
            temporal_scope: Scope of temporal filtering

        Returns:
            Cypher WHERE clause fragment (includes "WHERE" keyword)

        Raises:
            ValueError: If temporal_scope is invalid

        Example:
            # Get only current facts
            query = f"MATCH (a)-[r]->(b) {service.build_temporal_query('current')} RETURN r"
            # Result: "MATCH (a)-[r]->(b) WHERE r.invalid_at IS NULL AND r.expired_at IS NULL RETURN r"

            # Get all facts (no filtering)
            query = f"MATCH (a)-[r]->(b) {service.build_temporal_query('historical')} RETURN r"
            # Result: "MATCH (a)-[r]->(b)  RETURN r" (empty WHERE clause)
        """
        if temporal_scope == self.CURRENT:
            # Only currently valid relationships
            return "WHERE r.invalid_at IS NULL AND r.expired_at IS NULL"

        elif temporal_scope == self.HISTORICAL:
            # All relationships (no temporal filtering)
            return ""

        elif temporal_scope == self.NATURAL_INVALID:
            # Facts that ended naturally (not system errors)
            return "WHERE r.invalid_at IS NOT NULL AND r.expired_at IS NULL"

        elif temporal_scope == self.SYSTEM_INVALID:
            # System-invalidated facts (may still be true in real world)
            return "WHERE r.expired_at IS NOT NULL"

        else:
            raise ValueError(
                f"Invalid temporal_scope: {temporal_scope}. "
                f"Valid options: {self.CURRENT}, {self.HISTORICAL}, "
                f"{self.NATURAL_INVALID}, {self.SYSTEM_INVALID}"
            )

    async def invalidate_relationship(
        self,
        neo4j_service: Neo4jService,
        relationship_uuid: str,
        invalidation_type: str = "both",
        reason: str = "",
        superseded_by: str | None = None,
    ) -> dict:
        """
        Execute relationship invalidation in Neo4j.

        Non-destructive operation: Uses SET to update temporal fields,
        never DELETE. Relationship remains in graph with historical metadata.

        Args:
            neo4j_service: Neo4j service instance for query execution
            relationship_uuid: UUID of the relationship to invalidate
            invalidation_type: Type of invalidation ("natural" | "system" | "both")
            reason: Human-readable reason for invalidation
            superseded_by: Optional UUID of replacement relationship

        Returns:
            Dict with invalidated relationship metadata:
            {
                "status": "success",
                "fact_uuid": "...",
                "invalidated_at": "2025-10-11T...",
                "fields_set": ["invalid_at", "expired_at"],
                "reason": "...",
                "superseded_by": "..." | null
            }

        Raises:
            ValueError: If invalidation_type is invalid
            Exception: If Neo4j operation fails or relationship not found

        Example:
            result = await service.invalidate_relationship(
                neo4j_service,
                "abc-123",
                "natural",
                "Project completed"
            )
        """
        # Get fields to set based on invalidation type
        fields = self.set_temporal_fields(relationship_uuid, invalidation_type, reason)

        # Build Cypher SET clauses
        set_clauses = []
        params = {"uuid": relationship_uuid, "reason": reason}

        for field_name, field_value in fields.items():
            set_clauses.append(f"r.{field_name} = datetime($timestamp_{field_name})")
            params[f"timestamp_{field_name}"] = field_value.isoformat()

        # Always set invalidation reason
        set_clauses.append("r.invalidation_reason = $reason")

        # Optionally set supersession link
        if superseded_by:
            set_clauses.append("r.superseded_by = $superseded_by")
            params["superseded_by"] = superseded_by

        # D3 (FR-002): single-statement, native-scoped, cardinality-guarded.
        # The WHERE size(rs)=1 guard ensures exactly one native edge is matched.
        # If the query returns no row, re-query count to distinguish:
        #   - 0 matches → not found
        #   - >1 matches → ambiguous duplicate (legacy data violation FR-001)
        cypher = f"""
        MATCH ()-[r:RELATES_TO {{uuid: $uuid}}]->()
        WHERE r.type IS NOT NULL
        WITH collect(r) AS rs
        WHERE size(rs) = 1
        WITH rs[0] AS r
        SET {", ".join(set_clauses)}
        RETURN r.uuid as fact_uuid,
               r.invalid_at as invalid_at,
               r.expired_at as expired_at,
               r.invalidation_reason as reason,
               r.superseded_by as superseded_by,
               r.type as relation_type
        """

        logger.debug(
            f"Invalidating relationship: {relationship_uuid} (D3 native-scoped guard)"
        )

        try:
            result = await neo4j_service.query(cypher, params)

            if not result:
                # D3 fail-hard: distinguish 0 vs >1 to give a precise error
                count_result = await neo4j_service.query(
                    """
                    MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
                    WHERE r.type IS NOT NULL
                    RETURN count(r) AS cnt
                    """,
                    {"uuid": relationship_uuid},
                )
                cnt = count_result[0]["cnt"] if count_result else 0
                if cnt == 0:
                    raise ValueError(
                        f"Relationship not found: {relationship_uuid}. "
                        "Verify UUID is correct and relationship exists."
                    )
                raise ValueError(
                    f"Ambiguous duplicate: {cnt} native edges share uuid={relationship_uuid}. "
                    "Remediate duplicates before invalidating (D6)."
                )

            record = result[0]

            # Format response
            response = {
                "status": "success",
                "fact_uuid": record["fact_uuid"],
                "invalidated_at": datetime.now(UTC).isoformat(),
                "fields_set": list(fields.keys()),
                "reason": record["reason"],
                "superseded_by": record.get("superseded_by"),
                "relation_type": record["relation_type"],
            }

            logger.info(
                f"Relationship invalidated: {relationship_uuid} "
                f"(type: {invalidation_type})"
            )

            return response

        except Exception as e:
            logger.error(f"Relationship invalidation failed: {relationship_uuid} - {e}")
            raise

    async def check_relationship_validity(
        self, neo4j_service: Neo4jService, relationship_uuid: str
    ) -> dict:
        """
        Check current validity status of a relationship.

        Args:
            neo4j_service: Neo4j service instance
            relationship_uuid: UUID of relationship to check

        Returns:
            Dict with validity status:
            {
                "is_valid": bool,
                "invalid_at": datetime | null,
                "expired_at": datetime | null,
                "invalidation_reason": str | null
            }

        Example:
            status = await service.check_relationship_validity(neo4j, "uuid-123")
            if status["is_valid"]:
                print("Relationship is currently valid")
        """
        # D7 (FR-012): scope to native edges only (r.type IS NOT NULL)
        cypher = """
        MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
        WHERE r.type IS NOT NULL
        RETURN r.invalid_at as invalid_at,
               r.expired_at as expired_at,
               r.invalidation_reason as reason
        """

        result = await neo4j_service.query(cypher, {"uuid": relationship_uuid})

        if not result:
            raise Exception(f"Relationship not found: {relationship_uuid}")

        record = result[0]

        is_valid = record["invalid_at"] is None and record["expired_at"] is None

        return {
            "is_valid": is_valid,
            "invalid_at": record["invalid_at"],
            "expired_at": record["expired_at"],
            "invalidation_reason": record["reason"],
        }


# Global service instance
temporal_validity_service = TemporalValidityService()

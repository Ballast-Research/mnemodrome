"""
Relationship Factory Service - Temporal Relationship Creation

Creates typed relationships between resolved entities with temporal tracking.
Implements mem0 pattern: valid_at, invalid_at, expired_at fields.
"""

from uuid import uuid4

from loguru import logger

from src.config import ACTR_CONFIDENCE_DEFAULT
from src.services.neo4j_service import neo4j_service
from src.services.predicate_cardinality import is_functional


class RelationshipFactory:
    """
    Factory for creating temporal relationships between entities.

    Creates relationships using entity UUIDs (not names) to prevent ambiguity.
    Tracks temporal validity using mem0 pattern for relationship lifecycle.

    Temporal Fields:
    - valid_at: When relationship became true (timestamp of creation)
    - invalid_at: When relationship became false (NULL = still true)
    - expired_at: When relationship was removed from active graph (NULL = active)
    """

    async def create(
        self,
        source_uuid: str,
        relation_type: str,
        target_uuid: str,
        context: str = "",
        relationship_uuid: str | None = None,
    ) -> dict:
        """
        Create temporal relationship between two entities.

        Args:
            source_uuid: Source entity UUID (from EntityFactory)
            relation_type: Relationship type (e.g., "IMPLEMENTS", "USES")
            target_uuid: Target entity UUID (from EntityFactory)
            context: Optional context/evidence for this relationship

        Returns:
            Dict with relationship details for verification:
            {
                "source_uuid": str,
                "source_name": str,
                "relation_type": str,
                "target_uuid": str,
                "target_name": str,
                "valid_at": datetime,
                "context": str
            }

        Raises:
            Exception: If either entity UUID not found or creation fails

        Example:
            # Entities already resolved via EntityFactory
            example_uuid = "abc-123"
            claude_uuid = "def-456"

            # Create relationship
            rel = await factory.create(
                source_uuid=example_uuid,
                relation_type="COLLABORATES_WITH",
                target_uuid=claude_uuid,
                context="Working on Mnemodrome Phase 4.5"
            )
        """
        logger.debug(
            f"Creating relationship: {source_uuid} -{relation_type}-> {target_uuid}"
        )

        # D1 (FR-011): generate uuid at the capture boundary, never inside Cypher
        if relationship_uuid is None:
            relationship_uuid = str(uuid4())

        # --- Dedup: check for existing active relationship ---
        existing = await self._find_existing(source_uuid, relation_type, target_uuid)
        if existing:
            logger.info(
                f"Duplicate skipped: {existing['source_name']} "
                f"-{existing['relation_type']}-> {existing['target_name']} "
                f"(already exists, valid_at={existing['valid_at']})"
            )
            existing["_dedup"] = "skipped"
            return existing

        # --- Auto-supersede: same source+relation but different target ---
        # ONLY for FUNCTIONAL (single-valued) predicates. Cumulative / many-valued
        # predicates (ORDERED, BOUGHT, OWNS, USES, ...) describe independent facts
        # that must coexist — superseding them on (subject, predicate) alone
        # destroys true facts. See src/services/predicate_cardinality.py.
        # D3: if invalidate() raises (ambiguous duplicate), abort the create — never
        # leave the old fact valid while creating a new one that would conflict with it.
        superseded_targets = []
        if is_functional(relation_type):
            conflicting = await self._find_conflicting(
                source_uuid, relation_type, target_uuid
            )
            if conflicting:
                for old_rel in conflicting:
                    # May raise ValueError on ambiguous duplicates — let it propagate (abort).
                    await self.invalidate(
                        source_uuid, relation_type, old_rel["target_uuid"]
                    )
                    superseded_targets.append(old_rel["target_name"])
                    logger.info(
                        f"Auto-superseded: {old_rel['source_name']} "
                        f"-{old_rel['relation_type']}-> {old_rel['target_name']} "
                        f"(replaced by target {target_uuid})"
                    )
        else:
            logger.debug(
                f"Cumulative predicate '{relation_type}': skipping auto-supersede "
                f"({source_uuid} -{relation_type}-> {target_uuid} coexists with priors)"
            )

        # D1 (FR-011): use app-generated uuid in Cypher param, not randomUUID()
        # Create relationship with temporal tracking
        cypher = """
        MATCH (source {uuid: $source_uuid})
        MATCH (target {uuid: $target_uuid})
        CREATE (source)-[r:RELATES_TO {
            uuid: $relationship_uuid,
            type: $relation_type,
            context: $context,
            valid_at: datetime(),
            created_at: datetime(),
            invalid_at: NULL,
            expired_at: NULL,
            confidence: $confidence_default
        }]->(target)
        RETURN
            source.uuid as source_uuid,
            source.name as source_name,
            source.entity_type as source_type,
            r.uuid as uuid,
            r.uuid as relationship_uuid,
            r.type as relation_type,
            r.valid_at as valid_at,
            r.context as context,
            target.uuid as target_uuid,
            target.name as target_name,
            target.entity_type as target_type
        """

        params = {
            "source_uuid": source_uuid,
            "target_uuid": target_uuid,
            "relation_type": relation_type,
            "context": context,
            "confidence_default": ACTR_CONFIDENCE_DEFAULT,
            "relationship_uuid": relationship_uuid,
        }

        try:
            result = await neo4j_service.query(cypher, params)

            if not result:
                raise Exception(
                    f"Failed to create relationship: {source_uuid} -{relation_type}-> {target_uuid}. "
                    "One or both entity UUIDs may not exist."
                )

            rel_data = result[0]
            if superseded_targets:
                rel_data["_dedup"] = "superseded"
                rel_data["_superseded_targets"] = superseded_targets
            logger.debug(
                f"Relationship created: {rel_data['source_name']} "
                f"-{rel_data['relation_type']}-> {rel_data['target_name']}"
            )

            return rel_data

        except Exception as e:
            logger.error(f"Relationship creation failed: {e}")
            raise

    async def _find_existing(
        self, source_uuid: str, relation_type: str, target_uuid: str
    ) -> dict | None:
        """Check for an active duplicate relationship (same source, relation, target)."""
        cypher = """
        MATCH (source {uuid: $source_uuid})-[r:RELATES_TO {type: $relation_type}]->(target {uuid: $target_uuid})
        WHERE r.invalid_at IS NULL AND r.expired_at IS NULL
        RETURN
            source.uuid as source_uuid,
            source.name as source_name,
            source.entity_type as source_type,
            r.uuid as uuid,
            r.uuid as relationship_uuid,
            r.type as relation_type,
            r.valid_at as valid_at,
            r.context as context,
            target.uuid as target_uuid,
            target.name as target_name,
            target.entity_type as target_type
        LIMIT 1
        """
        result = await neo4j_service.query(
            cypher,
            {
                "source_uuid": source_uuid,
                "relation_type": relation_type,
                "target_uuid": target_uuid,
            },
        )
        return result[0] if result else None

    async def _find_conflicting(
        self, source_uuid: str, relation_type: str, target_uuid: str
    ) -> list[dict]:
        """Find active relationships with same source+relation but different target."""
        cypher = """
        MATCH (source {uuid: $source_uuid})-[r:RELATES_TO {type: $relation_type}]->(target)
        WHERE r.invalid_at IS NULL AND r.expired_at IS NULL
          AND target.uuid <> $target_uuid
        RETURN
            source.uuid as source_uuid,
            source.name as source_name,
            r.type as relation_type,
            target.uuid as target_uuid,
            target.name as target_name
        """
        result = await neo4j_service.query(
            cypher,
            {
                "source_uuid": source_uuid,
                "relation_type": relation_type,
                "target_uuid": target_uuid,
            },
        )
        return result if result else []

    async def invalidate(
        self, source_uuid: str, relation_type: str, target_uuid: str
    ) -> bool:
        """
        Mark relationship as invalid (relationship became false).

        Sets invalid_at timestamp while keeping the relationship in the graph
        for historical queries and temporal analysis.

        Args:
            source_uuid: Source entity UUID
            relation_type: Relationship type
            target_uuid: Target entity UUID

        Returns:
            True if relationship found and invalidated

        Example:
            # Entity used to use Vim, but switched to Claude Code
            await factory.invalidate(example_uuid, "USES", vim_uuid)
        """
        # D3 (FR-002) + D7 (FR-012): native-scoped, cardinality-guarded.
        # Matches exactly the one active native edge for this triple.
        cypher = """
        MATCH (source {uuid: $source_uuid})-[r:RELATES_TO {type: $relation_type}]->(target {uuid: $target_uuid})
        WHERE r.type IS NOT NULL
          AND r.invalid_at IS NULL
        WITH collect(r) AS rs
        WHERE size(rs) = 1
        WITH rs[0] AS r
        SET r.invalid_at = datetime()
        RETURN r.uuid AS invalidated_uuid
        """

        params = {
            "source_uuid": source_uuid,
            "relation_type": relation_type,
            "target_uuid": target_uuid,
        }

        result = await neo4j_service.query(cypher, params)

        if result:
            logger.debug(
                f"Relationship invalidated: {source_uuid} -{relation_type}-> {target_uuid} "
                f"(uuid={result[0]['invalidated_uuid']})"
            )
            return True

        # Distinguish not-found from ambiguous duplicate — raise on ambiguous (D3).
        count_result = await neo4j_service.query(
            """
            MATCH (source {uuid: $source_uuid})-[r:RELATES_TO {type: $relation_type}]->(target {uuid: $target_uuid})
            WHERE r.type IS NOT NULL AND r.invalid_at IS NULL
            RETURN count(r) AS cnt
            """,
            params,
        )
        cnt = count_result[0]["cnt"] if count_result else 0
        if cnt == 0:
            logger.warning(
                f"No active native relationship found to invalidate: {source_uuid} -{relation_type}-> {target_uuid}"
            )
            return False
        raise ValueError(
            f"Ambiguous: {cnt} active native edges match "
            f"{source_uuid} -{relation_type}-> {target_uuid}; none invalidated. "
            "Remediate duplicates (D6) before invalidating."
        )

    async def expire(
        self, source_uuid: str, relation_type: str, target_uuid: str
    ) -> bool:
        """
        Mark relationship as expired (remove from active graph queries).

        Sets expired_at timestamp. Expired relationships are filtered out
        of temporal queries but remain in database for historical analysis.

        Args:
            source_uuid: Source entity UUID
            relation_type: Relationship type
            target_uuid: Target entity UUID

        Returns:
            True if relationship found and expired
        """
        cypher = """
        MATCH (source {uuid: $source_uuid})-[r:RELATES_TO {type: $relation_type}]->(target {uuid: $target_uuid})
        WHERE r.expired_at IS NULL
        SET r.expired_at = datetime()
        RETURN count(r) as expired
        """

        params = {
            "source_uuid": source_uuid,
            "relation_type": relation_type,
            "target_uuid": target_uuid,
        }

        result = await neo4j_service.query(cypher, params)

        if result and result[0]["expired"] > 0:
            logger.debug(
                f"Relationship expired: {source_uuid} -{relation_type}-> {target_uuid}"
            )
            return True

        logger.warning("No relationship found to expire")
        return False

    async def batch_create(
        self, relationships: list[dict], context: str = ""
    ) -> list[dict]:
        """
        Optimization 2: Create multiple relationships in parallel.

        Gang of Four Factory Pattern for batch operations.
        Creates relationships between already-resolved entity UUIDs.

        Args:
            relationships: List of dicts:
                {
                    "source_uuid": str,
                    "relation_type": str,
                    "target_uuid": str,
                }
            context: Optional shared context for all relationships

        Returns:
            List of created relationship data dicts

        Example:
            rels_to_create = [
                {"source_uuid": "abc", "relation_type": "USES", "target_uuid": "def"},
                {"source_uuid": "abc", "relation_type": "IMPLEMENTS", "target_uuid": "ghi"},
            ]
            results = await factory.batch_create(rels_to_create, context="Phase 4.5")
        """
        if not relationships:
            logger.debug("No relationships to create")
            return []

        logger.debug(f"Creating {len(relationships)} relationships in parallel...")
        import asyncio

        # Create all relationships in parallel (no dependencies between them)
        create_tasks = [
            self.create(
                source_uuid=rel["source_uuid"],
                relation_type=rel["relation_type"],
                target_uuid=rel["target_uuid"],
                context=context,
            )
            for rel in relationships
        ]

        try:
            results = await asyncio.gather(*create_tasks, return_exceptions=False)
            logger.debug(
                f"Batch relationship creation complete: {len(results)} created"
            )
            return results
        except Exception as e:
            logger.error(f"Batch relationship creation failed: {e}")
            raise

    async def get_active_relationships(
        self, entity_uuid: str, direction: str = "both"
    ) -> list[dict]:
        """
        Get all active (valid, non-expired) relationships for an entity.

        Args:
            entity_uuid: Entity UUID
            direction: "outgoing", "incoming", or "both" (default)

        Returns:
            List of relationship dicts with entity details
        """
        # D7 (FR-012): scope all relationship reads to native edges (r.type IS NOT NULL)
        if direction == "outgoing":
            cypher = """
            MATCH (source {uuid: $uuid})-[r:RELATES_TO]->(target)
            WHERE r.type IS NOT NULL
              AND r.invalid_at IS NULL
              AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())
            RETURN source, r, target
            """
        elif direction == "incoming":
            cypher = """
            MATCH (source)-[r:RELATES_TO]->(target {uuid: $uuid})
            WHERE r.type IS NOT NULL
              AND r.invalid_at IS NULL
              AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())
            RETURN source, r, target
            """
        else:  # both
            cypher = """
            MATCH (entity {uuid: $uuid})
            OPTIONAL MATCH (entity)-[r_out:RELATES_TO]->(target)
            WHERE r_out.type IS NOT NULL
              AND r_out.invalid_at IS NULL
              AND (r_out.expired_at IS NULL OR datetime(r_out.expired_at) > datetime())
            OPTIONAL MATCH (source)-[r_in:RELATES_TO]->(entity)
            WHERE r_in.type IS NOT NULL
              AND r_in.invalid_at IS NULL
              AND (r_in.expired_at IS NULL OR datetime(r_in.expired_at) > datetime())
            RETURN collect(DISTINCT {source: entity, rel: r_out, target: target}) as outgoing,
                   collect(DISTINCT {source: source, rel: r_in, target: entity}) as incoming
            """

        result = await neo4j_service.query(cypher, {"uuid": entity_uuid})
        return result


# Global factory instance
relationship_factory = RelationshipFactory()

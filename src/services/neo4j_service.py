"""Neo4j database service for graph memory operations."""

import asyncio
from datetime import UTC
from typing import Any
from uuid import uuid4

from loguru import logger
from neo4j import AsyncDriver, AsyncGraphDatabase

from src.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def _normalize_neo4j_value(value: Any) -> Any:
    """Convert neo4j.time temporal types to Python-native types.

    Neo4j returns neo4j.time.DateTime objects which are not JSON-serializable
    and cause type mismatches with Python datetime operations. This normalizes
    them at the service boundary so all downstream code works with standard types.
    """
    # Import neo4j temporal types lazily to avoid import errors if not installed
    try:
        from neo4j.time import (
            Date as Neo4jDate,
        )
        from neo4j.time import (
            DateTime as Neo4jDateTime,
        )
        from neo4j.time import (
            Duration as Neo4jDuration,
        )
    except ImportError:
        return value

    if isinstance(value, Neo4jDateTime):
        native = value.to_native()
        # Ensure timezone-aware
        if native.tzinfo is None:
            native = native.replace(tzinfo=UTC)
        return native
    if isinstance(value, Neo4jDate):
        return value.to_native()
    if isinstance(value, Neo4jDuration):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize_neo4j_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_neo4j_value(item) for item in value]
    return value


class Neo4jService:
    """Service for Neo4j graph database operations."""

    def __init__(self):
        """Initialize Neo4j service (connection established lazily)."""
        # The MCP server and the Admin UI run on different asyncio event loops
        # in the same process (Feature 013 dual-server). An AsyncDriver's
        # connections are bound to the loop that created them, so one driver
        # is kept per loop (Feature 039). ``driver`` resolves to the driver for
        # the running loop; the first one created is the default.
        self._driver: AsyncDriver | None = None
        self._driver_loop: asyncio.AbstractEventLoop | None = None
        self._loop_drivers: dict[asyncio.AbstractEventLoop, AsyncDriver] = {}
        self.uri = NEO4J_URI
        self.user = NEO4J_USER
        self.password = NEO4J_PASSWORD
        self.database = NEO4J_DATABASE

    @staticmethod
    def _running_loop() -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            return None

    @property
    def driver(self) -> AsyncDriver | None:
        """Driver for the running event loop (default driver when unknown)."""
        loop = self._running_loop()
        if (
            loop is not None
            and self._driver_loop is not None
            and loop is not self._driver_loop
        ):
            return self._loop_drivers.get(loop)
        return self._driver

    @driver.setter
    def driver(self, value: AsyncDriver | None) -> None:
        """Set the default driver (tests inject mocks here)."""
        self._driver = value
        self._driver_loop = self._running_loop() if value is not None else None
        self._loop_drivers.clear()

    def _new_driver(self) -> AsyncDriver:
        return AsyncGraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_pool_size=200,  # Increased from default 50 to handle job manager load
            connection_timeout=30.0,  # 30 second timeout for connection acquisition
            max_connection_lifetime=3600,  # Recycle connections after 1 hour
            connection_acquisition_timeout=60.0,  # 60 second timeout waiting for available connection
        )

    async def connect(self) -> None:
        """Establish connection to Neo4j database (one driver per event loop)."""
        if self._driver is None:
            self.driver = self._new_driver()
            logger.info(f"Connected to Neo4j at {self.uri} with pool size 200")
            return
        loop = self._running_loop()
        if (
            loop is not None
            and self._driver_loop is not None
            and loop is not self._driver_loop
            and loop not in self._loop_drivers
        ):
            self._loop_drivers[loop] = self._new_driver()
            logger.info(
                f"Connected to Neo4j at {self.uri} for a second event loop "
                f"({len(self._loop_drivers)} extra driver(s))"
            )

    async def close(self) -> None:
        """Close every Neo4j driver connection."""
        drivers = [d for d in self._loop_drivers.values()]
        if self._driver is not None:
            drivers.append(self._driver)
        for d in drivers:
            try:
                await d.close()
            except Exception as e:  # pragma: no cover - best effort
                logger.warning(f"Error closing Neo4j driver: {e}")
        if drivers:
            self.driver = None
            logger.info("Closed Neo4j connection")

    async def ping(self) -> bool:
        """
        Test Neo4j connectivity.

        Returns:
            True if connection successful

        Raises:
            Exception if connection fails
        """
        await self.connect()
        assert self.driver is not None
        async with self.driver.session(database=self.database) as session:
            result = await session.run("RETURN 1 as ping")
            record = await result.single()
            return record["ping"] == 1

    async def query(
        self, cypher: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            List of result records as dictionaries
        """
        await self.connect()
        assert self.driver is not None
        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return [_normalize_neo4j_value(r) for r in records]

    async def create_entity(self, data: dict[str, Any]) -> str:
        """
        Create an entity node in Neo4j.

        Args:
            data: Entity data (must include 'name' and 'summary')

        Returns:
            UUID of created entity

        Raises:
            Exception if write fails (exception-based verification)
        """
        await self.connect()
        assert self.driver is not None

        # Generate UUID if not provided
        entity_uuid = data.get("uuid", str(uuid4()))

        cypher = """
        CREATE (e:Entity {
            uuid: $uuid,
            name: $name,
            summary: $summary,
            created_at: datetime()
        })
        RETURN e.uuid as uuid
        """

        params = {
            "uuid": entity_uuid,
            "name": data.get("name", ""),
            "summary": data.get("summary", ""),
        }

        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, params)
            record = await result.single()

            if not record:
                raise Exception(f"Failed to create entity: {entity_uuid}")

            logger.debug(f"Created entity: {entity_uuid}")
            return record["uuid"]

    async def create_relationship(
        self,
        entity1_name: str,
        entity1_summary: str,
        relation_type: str,
        entity2_name: str,
        entity2_summary: str,
        context: str = "",
        relationship_uuid: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a relationship between two entities with temporal fields.

        D1 (FR-011): Accepts an app-generated relationship_uuid so the caller
        controls the identity. If not supplied, one is generated here at the
        capture boundary (never inside Cypher via randomUUID()).

        D8: dual-key return — both 'uuid' and 'relationship_uuid' in the
        response so existing consumers that expect either key continue to work.

        Uses MERGE to avoid duplicates - creates entities if they don't exist,
        reuses them if they do. Always creates a new relationship instance.

        Args:
            entity1_name: Source entity name
            entity1_summary: Source entity summary/description
            relation_type: Relationship type (e.g., "IMPLEMENTS", "USES")
            entity2_name: Target entity name
            entity2_summary: Target entity summary/description
            context: Optional context for the relationship
            relationship_uuid: App-generated uuid for the new relationship.
                Generated here at the capture boundary if not provided.

        Returns:
            Dict with entity UUIDs and relationship info (dual-key uuid).

        Raises:
            Exception if relationship creation fails
        """
        await self.connect()
        assert self.driver is not None

        # D1: generate at capture boundary — never rely on randomUUID() in Cypher
        if relationship_uuid is None:
            relationship_uuid = str(uuid4())

        cypher = """
        MERGE (e1:Entity {name: $entity1_name})
        ON CREATE SET
            e1.uuid = randomUUID(),
            e1.summary = $entity1_summary,
            e1.created_at = datetime()
        ON MATCH SET
            e1.summary = $entity1_summary

        MERGE (e2:Entity {name: $entity2_name})
        ON CREATE SET
            e2.uuid = randomUUID(),
            e2.summary = $entity2_summary,
            e2.created_at = datetime()
        ON MATCH SET
            e2.summary = $entity2_summary

        CREATE (e1)-[r:RELATES_TO {
            uuid: $relationship_uuid,
            type: $relation_type,
            context: $context,
            valid_at: datetime(),
            created_at: datetime(),
            invalid_at: NULL,
            expired_at: NULL
        }]->(e2)

        RETURN
            e1.uuid as entity1_uuid,
            e1.name as entity1_name,
            e2.uuid as entity2_uuid,
            e2.name as entity2_name,
            r.uuid as uuid,
            r.uuid as relationship_uuid,
            r.type as relation_type,
            r.valid_at as valid_at
        """

        params = {
            "entity1_name": entity1_name,
            "entity1_summary": entity1_summary,
            "entity2_name": entity2_name,
            "entity2_summary": entity2_summary,
            "relation_type": relation_type,
            "context": context,
            "relationship_uuid": relationship_uuid,
        }

        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, params)
            record = await result.single()

            if not record:
                raise Exception(
                    f"Failed to create relationship: {entity1_name} -{relation_type}-> {entity2_name}"
                )

            logger.debug(
                f"Created relationship: {record['entity1_name']} -{record['relation_type']}-> {record['entity2_name']}"
            )

            # D8: dual-key return — 'uuid' and 'relationship_uuid' both present
            return {
                "entity1_uuid": record["entity1_uuid"],
                "entity1_name": record["entity1_name"],
                "entity2_uuid": record["entity2_uuid"],
                "entity2_name": record["entity2_name"],
                "uuid": record["uuid"],
                "relationship_uuid": record["relationship_uuid"],
                "relation_type": record["relation_type"],
                "valid_at": _normalize_neo4j_value(record["valid_at"]),
            }

    async def delete_entity(self, uuid: str) -> bool:
        """
        Delete an entity by UUID.

        Args:
            uuid: Entity UUID

        Returns:
            True if deleted
        """
        await self.connect()
        assert self.driver is not None

        cypher = """
        MATCH (e:Entity {uuid: $uuid})
        DELETE e
        RETURN count(e) as deleted
        """

        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, {"uuid": uuid})
            record = await result.single()
            deleted = record["deleted"] if record else 0

            logger.debug(f"Deleted entity {uuid}: {deleted > 0}")
            return deleted > 0

    async def temporal_query(
        self, keyword: str | None = None, days: int = 7, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Query recent valid relationships with temporal filtering.

        Filters out:
        - Invalidated relationships (invalid_at IS NOT NULL)
        - Expired relationships (expired_at < now)
        - Old relationships (valid_at older than 'days' ago)

        Args:
            keyword: Optional keyword to filter entity names
            days: Number of days to look back (default 7)
            limit: Maximum results to return (default 10, optimized for token usage)

        Returns:
            List of relationship records with source/target entities
        """
        await self.connect()
        assert self.driver is not None

        # Build keyword filter if provided
        keyword_filter = ""
        if keyword:
            keyword_filter = """
            AND (toLower(source.name) CONTAINS toLower($keyword)
                 OR toLower(target.name) CONTAINS toLower($keyword))
            """

        # Build duration string (ISO 8601 format)
        duration_str = f"P{days}D"

        cypher = f"""
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE r.type IS NOT NULL
          AND r.invalid_at IS NULL
          AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())
          AND r.valid_at IS NOT NULL
          AND datetime(r.valid_at) > datetime() - duration($duration)
          {keyword_filter}
        RETURN
          {{
            name: source.name,
            summary: source.summary,
            uuid: source.uuid
          }} as source,
          r,
          {{
            name: target.name,
            summary: target.summary,
            uuid: target.uuid
          }} as target
        ORDER BY r.valid_at DESC
        LIMIT $limit
        """

        params = {"duration": duration_str, "limit": limit}
        if keyword:
            params["keyword"] = keyword

        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, params)

            # Manual record processing to preserve relationship properties
            # .data() serializes relationships as tuples, losing properties
            formatted_records = []
            async for record in result:
                # Extract relationship properties manually
                rel = record["r"]
                formatted_records.append(
                    {
                        "source": _normalize_neo4j_value(record["source"]),
                        "r": {
                            "type": rel.type if hasattr(rel, "type") else "RELATES_TO",
                            "valid_at": _normalize_neo4j_value(
                                rel.get("valid_at") if hasattr(rel, "get") else None
                            ),
                            "invalid_at": _normalize_neo4j_value(
                                rel.get("invalid_at") if hasattr(rel, "get") else None
                            ),
                            "expired_at": _normalize_neo4j_value(
                                rel.get("expired_at") if hasattr(rel, "get") else None
                            ),
                            "context": rel.get("context")
                            if hasattr(rel, "get")
                            else None,
                        },
                        "target": _normalize_neo4j_value(record["target"]),
                    }
                )

            logger.debug(
                f"Temporal query returned {len(formatted_records)} relationships"
            )
            return formatted_records

    async def temporal_query_by_entities(
        self, entity_uuids: list[str], days: int = 7, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Query recent relationships involving specific entities (Phase 4.5).

        Entity-aware context retrieval using resolved UUIDs for precise filtering.
        Prevents ambiguity from duplicate entity names.

        Filters out:
        - Invalidated relationships (invalid_at IS NOT NULL)
        - Expired relationships (expired_at < now)
        - Old relationships (valid_at older than 'days' ago)

        Args:
            entity_uuids: List of entity UUIDs to filter by (from EntityFactory)
            days: Number of days to look back (default 7)
            limit: Maximum results to return (default 10)

        Returns:
            List of relationship records involving any of the specified entities
        """
        await self.connect()
        assert self.driver is not None

        if not entity_uuids:
            logger.warning(
                "temporal_query_by_entities called with empty entity_uuids list"
            )
            return []

        # Build duration string (ISO 8601 format)
        duration_str = f"P{days}D"

        cypher = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE r.type IS NOT NULL
          AND r.invalid_at IS NULL
          AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())
          AND r.valid_at IS NOT NULL
          AND datetime(r.valid_at) > datetime() - duration($duration)
          AND (source.uuid IN $entity_uuids OR target.uuid IN $entity_uuids)
        RETURN
          {
            name: source.name,
            summary: source.summary,
            uuid: source.uuid,
            entity_type: source.entity_type
          } as source,
          r,
          {
            name: target.name,
            summary: target.summary,
            uuid: target.uuid,
            entity_type: target.entity_type
          } as target
        ORDER BY r.valid_at DESC
        LIMIT $limit
        """

        params = {
            "entity_uuids": entity_uuids,
            "duration": duration_str,
            "limit": limit,
        }

        async with self.driver.session(database=self.database) as session:
            result = await session.run(cypher, params)

            # Manual record processing to preserve relationship properties
            # .data() serializes relationships as tuples, losing properties
            formatted_records = []
            async for record in result:
                # Extract relationship properties manually
                rel = record["r"]
                formatted_records.append(
                    {
                        "source": _normalize_neo4j_value(record["source"]),
                        "r": {
                            "type": rel.type if hasattr(rel, "type") else "RELATES_TO",
                            "valid_at": _normalize_neo4j_value(
                                rel.get("valid_at") if hasattr(rel, "get") else None
                            ),
                            "invalid_at": _normalize_neo4j_value(
                                rel.get("invalid_at") if hasattr(rel, "get") else None
                            ),
                            "expired_at": _normalize_neo4j_value(
                                rel.get("expired_at") if hasattr(rel, "get") else None
                            ),
                            "context": rel.get("context")
                            if hasattr(rel, "get")
                            else None,
                        },
                        "target": _normalize_neo4j_value(record["target"]),
                    }
                )

            logger.debug(
                f"Entity-aware temporal query returned {len(formatted_records)} relationships "
                f"for {len(entity_uuids)} entities"
            )
            return formatted_records

    async def find_relationship(
        self, entity1: str, relation: str, entity2: str
    ) -> dict[str, Any] | None:
        """
        Find a currently valid relationship by entity names and relation type (Feature 006).

        Args:
            entity1: Source entity name
            relation: Relationship type
            entity2: Target entity name

        Returns:
            Relationship dict if found, None otherwise
        """
        await self.connect()

        # D4 (FR-006): return 'created_at' and 'valid_at' as distinct fields.
        # D7 (FR-012): scope to native edges (r.type IS NOT NULL).
        # D7 (Issue 7): also return entity1_uuid, entity2_uuid, relationship_uuid.
        # Raise ValueError on ambiguous triples (>1 active native match) — callers must
        # never silently pick one; they must surface the error (D2, D3 contract).
        cypher = """
        MATCH (source {name: $entity1})-[r:RELATES_TO]->(target {name: $entity2})
        WHERE r.type = $relation
          AND r.type IS NOT NULL
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
        RETURN
          r.uuid as uuid,
          r.uuid as relationship_uuid,
          source.uuid as entity1_uuid,
          source.name as entity1_name,
          r.type as relation_type,
          target.uuid as entity2_uuid,
          target.name as entity2_name,
          coalesce(r.created_at, r.valid_at) as created_at,
          r.valid_at as valid_at,
          r.invalid_at as invalid_at,
          r.expired_at as expired_at,
          r.context as context
        ORDER BY r.valid_at DESC
        LIMIT 1
        """

        params = {"entity1": entity1, "relation": relation, "entity2": entity2}

        # Count first — raise on ambiguous so callers never silently pick one.
        count_cypher = """
        MATCH (source {name: $entity1})-[r:RELATES_TO]->(target {name: $entity2})
        WHERE r.type = $relation
          AND r.type IS NOT NULL
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
        RETURN count(r) AS cnt
        """
        count_result = await self.query(count_cypher, params)
        cnt = count_result[0]["cnt"] if count_result else 0
        if cnt > 1:
            raise ValueError(
                f"Ambiguous triple: {cnt} active native edges for "
                f"{entity1} -{relation}-> {entity2}. "
                "Remediate duplicates (D6) before invalidating or updating."
            )

        results = await self.query(cypher, params)

        if not results:
            return None

        return results[0]

    async def invalidate_relationship(
        self,
        relationship_uuid: str,
        invalidation_type: str = "both",
        reason: str = "",
        superseded_by: str | None = None,
    ) -> dict[str, Any]:
        """
        Invalidate a relationship by setting temporal validity fields (Feature 006).

        This is a convenience wrapper around temporal_validity_service for better encapsulation.

        Args:
            relationship_uuid: UUID of relationship to invalidate
            invalidation_type: Type of invalidation ("natural", "system", or "both")
            reason: Human-readable reason for invalidation
            superseded_by: Optional UUID of replacement relationship

        Returns:
            Dict with invalidation metadata
        """
        from src.services.temporal_validity_service import temporal_validity_service

        result = await temporal_validity_service.invalidate_relationship(
            neo4j_service=self,
            relationship_uuid=relationship_uuid,
            invalidation_type=invalidation_type,
            reason=reason,
            superseded_by=superseded_by,
        )

        return result

    async def execute_update_transaction(
        self,
        old_fact_uuid: str,
        entity1_uuid: str,
        relation_type: str,
        entity2_uuid: str,
        context: str = "",
        new_fact_uuid: str | None = None,
        entity2_old_uuid: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute atomic update transaction: invalidate old fact + create new fact (Feature 006).

        This is a contract-compatible wrapper for atomic fact updates with supersession chains.

        Args:
            old_fact_uuid: UUID of fact to invalidate
            entity1_uuid: Source entity UUID
            relation_type: Relationship type
            entity2_uuid: NEW target entity UUID
            context: Optional context for new fact
            new_fact_uuid: Optional UUID for new fact (generated if not provided)
            entity2_old_uuid: Optional OLD target entity UUID — when provided the old edge
                is bound by (src, uuid, type, old_target) making the match maximally precise.
                Callers should pass this from find_relationship's entity2_uuid return field.

        Returns:
            Dict with transaction results including supersession chain
        """
        from datetime import datetime  # noqa: F811

        if not new_fact_uuid:
            new_fact_uuid = str(uuid4())

        timestamp = datetime.now(UTC)

        # D2 (FR-003): single atomic Cypher for supersession.
        # Old edge bound to source+uuid+type with cardinality guard.
        # When entity2_old_uuid is provided, the old target (x) is also bound for
        # maximum precision (passing old and new target UUIDs distinctly).
        # No partial state can exist after a failure — both SET and CREATE are in
        # a single statement; Neo4j rolls back if anything fails mid-way.
        if entity2_old_uuid:
            old_target_clause = "MATCH (x {uuid: $entity2_old_uuid})"
            old_edge_clause = (
                "MATCH (s)-[old:RELATES_TO {uuid: $old_uuid, type: $rel}]->(x)"
            )
        else:
            old_target_clause = ""
            old_edge_clause = (
                "MATCH (s)-[old:RELATES_TO {uuid: $old_uuid, type: $rel}]->(x)"
            )

        atomic_cypher = f"""
        MATCH (s {{uuid: $entity1_uuid}})
        MATCH (t {{uuid: $entity2_uuid}})
        {old_target_clause}
        {old_edge_clause}
        WHERE old.type IS NOT NULL
        WITH s, t, collect(old) AS olds WHERE size(olds) = 1
        WITH s, t, olds[0] AS old
        SET old.invalid_at = datetime(),
            old.expired_at = datetime(),
            old.invalidation_reason = $reason,
            old.superseded_by = $new_uuid
        CREATE (s)-[new:RELATES_TO {{
            uuid: $new_uuid,
            type: $rel,
            context: $ctx,
            valid_at: datetime(),
            created_at: datetime(),
            invalid_at: NULL,
            expired_at: NULL,
            confidence: $confidence_default
        }}]->(t)
        RETURN old.uuid AS old_uuid, new.uuid AS new_uuid
        """

        try:
            from src.config import ACTR_CONFIDENCE_DEFAULT

            q_params: dict[str, Any] = {
                "entity1_uuid": entity1_uuid,
                "entity2_uuid": entity2_uuid,
                "old_uuid": old_fact_uuid,
                "new_uuid": new_fact_uuid,
                "rel": relation_type,
                "ctx": context,
                "reason": f"Superseded by fact {new_fact_uuid}",
                "confidence_default": ACTR_CONFIDENCE_DEFAULT,
            }
            if entity2_old_uuid:
                q_params["entity2_old_uuid"] = entity2_old_uuid
            result = await self.query(atomic_cypher, q_params)

            if not result:
                raise ValueError(
                    f"Atomic supersession failed: old fact {old_fact_uuid} not found, "
                    f"not unique, or endpoints {entity1_uuid}/{entity2_uuid} missing."
                )

            # Return contract-compliant response
            return {
                "old_fact_uuid": result[0]["old_uuid"],
                "new_fact_uuid": result[0]["new_uuid"],
                "invalidated_at": timestamp.isoformat(),
                "supersession_chain": [result[0]["old_uuid"], result[0]["new_uuid"]],
                "new_fact_context": context,
            }

        except Exception as e:
            logger.error(f"Atomic update transaction failed: {e}")
            raise Exception(f"Transaction failed: {str(e)}")

    async def batch_update_access(self, uuids: list[str]) -> None:
        """
        Batch increment access_count and set last_accessed for multiple nodes.

        Uses UNWIND for efficient single-query batch update.
        Includes coalesce() for lazy initialization of nodes missing access_count.

        Args:
            uuids: List of node UUIDs to update.
        """
        if not uuids:
            return

        cypher = """
        UNWIND $uuids AS uuid
        MATCH (n {uuid: uuid})
        SET n.access_count = coalesce(n.access_count, 0) + 1,
            n.last_accessed = datetime()
        """
        await self.query(cypher, {"uuids": uuids})

    async def filter_valid_facts(self, facts: list[dict[str, Any]]) -> set[int]:
        """
        Return the indices of facts whose underlying RELATES_TO relationship is
        still valid (not invalidated, not expired) in the graph.

        Keeps the vector layer (Qdrant) consistent with the graph's temporal
        validity: a superseded fact is dropped from semantic search even though
        its embedding still lives in Qdrant. Matching is by entity UUID when
        present, falling back to entity name for legacy embeddings that predate
        UUID payloads.

        Args:
            facts: list of dicts, each with keys: idx (int), relation (str),
                e1_uuid (str|None), e2_uuid (str|None),
                e1_name (str|None), e2_name (str|None)

        Returns:
            Set of `idx` values that have at least one currently-valid relationship.
        """
        if not facts:
            return set()

        cypher = """
        UNWIND $facts AS f
        MATCH (s)-[r:RELATES_TO {type: f.relation}]->(o)
        WHERE (s.uuid = f.e1_uuid OR s.name = f.e1_name)
          AND (o.uuid = f.e2_uuid OR o.name = f.e2_name)
          AND r.invalid_at IS NULL
          AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())
        RETURN collect(DISTINCT f.idx) AS valid_idxs
        """
        rows = await self.query(cypher, {"facts": facts})
        if not rows:
            return set()
        return set(rows[0].get("valid_idxs") or [])


# Global service instance
neo4j_service = Neo4jService()

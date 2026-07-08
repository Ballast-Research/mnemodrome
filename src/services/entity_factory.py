"""
Entity Factory Service - Gang of Four Factory Pattern

Implements Amendment 7A: UUID-Based Entity Resolution
Ensures ONE canonical node per real-world entity using MERGE pattern.
"""

from loguru import logger

from src.services.entity_cache import entity_cache
from src.services.neo4j_service import neo4j_service


class EntityFactory:
    """
    Factory for creating or resolving typed entities with UUID-based identity.

    Gang of Four Creational Pattern:
    - Encapsulates entity creation complexity
    - Handles "create vs retrieve existing" logic
    - Single source of truth for entity identity resolution

    Constitutional Authority: Amendment 7A (Entity Identity Principle)
    """

    # Entity type taxonomy (extensible)
    VALID_ENTITY_TYPES = {
        "Person",
        "Project",
        "Organization",
        "Concept",
        "Technology",
        "Event",
        "Location",
        "Document",
        "Entity",  # Generic fallback
    }

    async def create_or_resolve(
        self, name: str, entity_type: str = "Entity", summary: str = ""
    ) -> str:
        """
        Create new entity or resolve existing entity, returning stable UUID.

        Uses Neo4j MERGE pattern with composite natural key (name + type):
        - ON CREATE: Generate new UUID, set initial properties
        - ON MATCH: Update summary if provided (enrich over time)

        Args:
            name: Entity name (e.g., "Alice", "Mnemodrome", "FastMCP")
            entity_type: Entity classification (default: "Entity")
            summary: Optional description (enriches existing entities)

        Returns:
            Entity UUID (stable across all references to this entity)

        Raises:
            ValueError: If entity_type is invalid
            Exception: If Neo4j operation fails

        Example:
            # First call creates entity with new UUID
            uuid1 = await factory.create_or_resolve("Alice", "Person", "Developer")

            # Second call returns SAME UUID (no duplicate)
            uuid2 = await factory.create_or_resolve("Alice", "Person", "Software Architect")

            # uuid1 == uuid2 (one canonical entity)
            # Summary enriched: "Software Architect" replaces "Developer"
        """
        # Validate entity type
        if entity_type not in self.VALID_ENTITY_TYPES:
            logger.warning(
                f"Invalid entity type '{entity_type}', falling back to 'Entity'. "
                f"Valid types: {self.VALID_ENTITY_TYPES}"
            )
            entity_type = "Entity"

        logger.debug(f"Resolving entity: {name} ({entity_type})")

        # Cache-first pattern: check Redis before Neo4j (T062)
        cached_uuid = await entity_cache.get(name, entity_type)
        if cached_uuid:
            logger.debug(f"Entity cache HIT: {name} → {cached_uuid}")
            return cached_uuid

        logger.debug(f"Entity cache MISS: {name}, querying Neo4j")

        # MERGE pattern: natural key is (name + type)
        # This ensures "Carlos:Person" is different from "Carlos:Project"
        cypher = f"""
        MERGE (e:{entity_type} {{name: $name}})
        ON CREATE SET
            e.uuid = randomUUID(),
            e.entity_type = $entity_type,
            e.summary = $summary,
            e.created_at = datetime()
        ON MATCH SET
            e.summary = CASE
                WHEN $summary <> '' THEN $summary
                ELSE e.summary
            END,
            e.last_accessed = datetime()
        RETURN e.uuid as uuid
        """

        params = {
            "name": name,
            "entity_type": entity_type,
            "summary": summary or f"{entity_type}: {name}",
        }

        try:
            result = await neo4j_service.query(cypher, params)

            if not result:
                raise Exception(f"Failed to create/resolve entity: {name}")

            uuid = result[0]["uuid"]
            logger.debug(f"Entity resolved: {name} → {uuid}")

            # Cache the result for future lookups (T062)
            await entity_cache.set(name, uuid, entity_type)

            return uuid

        except Exception as e:
            logger.error(f"Entity resolution failed for {name} ({entity_type}): {e}")
            raise

    async def batch_create_or_resolve(
        self, entities: list[tuple[str, str, str]]
    ) -> dict[str, str]:
        """
        Optimization 2: Resolve multiple entities in parallel with batch cache lookup.

        Uses entity_cache.batch_get() for efficient Redis MGET operation,
        then Neo4j MERGE for cache misses in parallel.

        Args:
            entities: List of (name, entity_type, summary) tuples

        Returns:
            Dict mapping entity_key -> UUID
            Example: {
                "Carlos:Person": "abc-123",
                "Neo4j:Technology": "def-456"
            }

        Example:
            uuids = await factory.batch_create_or_resolve([
                ("Alice", "Person", "Developer"),
                ("Neo4j", "Technology", "Graph Database"),
            ])
        """
        if not entities:
            return {}

        import asyncio

        logger.debug(f"Batch resolving {len(entities)} entities...")

        # Step 1: Batch cache lookup (Redis MGET)
        cache_lookup_tuples = [(name, entity_type) for name, entity_type, _ in entities]
        cache_results = await entity_cache.batch_get(cache_lookup_tuples)

        logger.debug(f"Batch cache: {len(cache_results)} hits out of {len(entities)}")

        # Step 2: Create tasks for cache misses (Neo4j resolution in parallel)
        resolution_tasks = []
        entity_map = {}

        for name, entity_type, summary in entities:
            entity_key = f"{name}:{entity_type}"

            # Check if already cached
            if entity_key in cache_results:
                entity_map[entity_key] = cache_results[entity_key]
                logger.debug(f"Using cached UUID: {entity_key}")
            else:
                # Create async task for Neo4j resolution
                resolution_tasks.append(
                    self._resolve_entity_with_cache(name, entity_type, summary)
                )

        # Step 3: Execute all Neo4j resolutions in parallel
        if resolution_tasks:
            resolved_results = await asyncio.gather(
                *resolution_tasks, return_exceptions=False
            )

            # Collect results
            for entity_key, uuid in resolved_results:
                entity_map[entity_key] = uuid

        logger.debug(f"Batch resolution complete: {len(entity_map)} entities")
        return entity_map

    async def _resolve_entity_with_cache(
        self, name: str, entity_type: str, summary: str
    ) -> tuple[str, str]:
        """
        Helper: Resolve single entity and cache result.

        Returns tuple of (entity_key, uuid) for batch_create_or_resolve.
        """
        uuid = await self.create_or_resolve(name, entity_type, summary)
        entity_key = f"{name}:{entity_type}"
        return entity_key, uuid

    async def get_entity_by_uuid(self, uuid: str) -> dict | None:
        """
        Retrieve entity by UUID.

        Args:
            uuid: Entity UUID

        Returns:
            Entity data dict or None if not found
        """
        cypher = """
        MATCH (e {uuid: $uuid})
        RETURN e.uuid as uuid, e.name as name,
               e.entity_type as entity_type, e.summary as summary
        """

        result = await neo4j_service.query(cypher, {"uuid": uuid})

        if result:
            return result[0]
        return None

    async def get_entity_by_name(
        self, name: str, entity_type: str | None = None
    ) -> dict | None:
        """
        Retrieve entity by name and optional type.

        Args:
            name: Entity name
            entity_type: Optional type filter

        Returns:
            Entity data dict or None if not found
        """
        if entity_type:
            cypher = f"""
            MATCH (e:{entity_type} {{name: $name}})
            RETURN e.uuid as uuid, e.name as name,
                   e.entity_type as entity_type, e.summary as summary
            """
            params = {"name": name}
        else:
            cypher = """
            MATCH (e {name: $name})
            RETURN e.uuid as uuid, e.name as name,
                   e.entity_type as entity_type, e.summary as summary
            LIMIT 1
            """
            params = {"name": name}

        result = await neo4j_service.query(cypher, params)

        if result:
            return result[0]
        return None

    async def enrich_entity(self, uuid: str, summary: str) -> bool:
        """
        Add or update summary for existing entity.

        Args:
            uuid: Entity UUID
            summary: New summary text

        Returns:
            True if enriched successfully
        """
        cypher = """
        MATCH (e {uuid: $uuid})
        SET e.summary = $summary, e.last_updated = datetime()
        RETURN e.uuid as uuid, e.name as name, e.entity_type as entity_type
        """

        result = await neo4j_service.query(cypher, {"uuid": uuid, "summary": summary})

        if result:
            logger.debug(f"Entity enriched: {uuid}")

            # Invalidate cache since entity was modified (T062.5)
            entity_data = result[0]
            await self.invalidate_entity_cache(
                entity_data["name"], entity_data["entity_type"]
            )

            return True

        logger.warning(f"Entity not found for enrichment: {uuid}")
        return False

    async def invalidate_entity_cache(self, name: str, entity_type: str) -> None:
        """
        Invalidate cached entity (T062.5: Cache Invalidation).

        Called when entity is updated/deleted to ensure cache consistency.

        Args:
            name: Entity name
            entity_type: Entity type
        """
        try:
            deleted = await entity_cache.delete(name, entity_type)
            if deleted:
                logger.debug(f"Entity cache invalidated: {name} ({entity_type})")
        except Exception as e:
            logger.error(f"Cache invalidation failed for {name}: {e}")

    def validate_entity_type(self, entity_type: str) -> bool:
        """
        Check if entity type is valid.

        Args:
            entity_type: Type to validate

        Returns:
            True if valid
        """
        return entity_type in self.VALID_ENTITY_TYPES

    @classmethod
    def get_valid_types(cls) -> set:
        """
        Get set of valid entity types.

        Returns:
            Set of valid entity type strings
        """
        return cls.VALID_ENTITY_TYPES.copy()


# Global factory instance
entity_factory = EntityFactory()

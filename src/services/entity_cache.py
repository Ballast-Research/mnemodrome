"""
Entity Cache - Redis-backed caching for entity UUID resolution

Implements cache-first pattern with Neo4j fallback for fast entity lookups.
Reduces entity resolution from 10-15s (LLM classification) to <100ms (cache hit).

Feature 005: Redis Entity Cache
"""

import time

from loguru import logger

from src.services.redis_service import redis_service


class EntityCache:
    """
    Redis-backed entity cache for fast UUID resolution.

    Cache Strategy:
    - Key format: `entity:{name}:{type}`
    - Value: Entity UUID (string)
    - TTL: None (permanent cache, invalidated on updates)
    - Persistence: RDB + AOF for durability

    Performance Targets:
    - Cache hit: <100ms (100x faster than LLM classification)
    - Cache miss: ~10-15s (fallback to Neo4j + LLM)
    - Warm cache startup: <5s for 1000 entities

    Constitutional Authority: Feature 005 (Redis Entity Cache)
    """

    def __init__(self):
        """Initialize entity cache (uses global redis_service)."""
        self.prefix = "entity"
        self.redis = redis_service

    def _make_key(self, name: str, entity_type: str) -> str:
        """
        Generate Redis key for entity.

        Args:
            name: Entity name (e.g., "Alice", "Neo4j")
            entity_type: Entity type (e.g., "Person", "Technology")

        Returns:
            Redis key: "entity:{name}:{type}"

        Example:
            _make_key("Alice", "Person") -> "entity:Alice:Person"
        """
        # Normalize name and type for consistent keys
        name_normalized = name.strip()
        type_normalized = entity_type.strip()
        return f"{self.prefix}:{name_normalized}:{type_normalized}"

    async def get(self, name: str, entity_type: str) -> str | None:
        """
        Get cached entity UUID by name and type.

        Args:
            name: Entity name
            entity_type: Entity type

        Returns:
            Entity UUID if cached, None if cache miss

        Example:
            uuid = await entity_cache.get("Alice", "Person")
            # Returns: "d1844bab-7ca2-4c21-ba45-a583e7a65e22" or None
        """
        if not self.redis.is_connected:
            logger.debug("Redis not connected, cache miss")
            return None

        key = self._make_key(name, entity_type)

        try:
            uuid = await self.redis.get(key)

            if uuid:
                logger.debug(f"Entity cache HIT: {name} ({entity_type}) -> {uuid}")
            else:
                logger.debug(f"Entity cache MISS: {name} ({entity_type})")

            return uuid

        except Exception as e:
            logger.error(f"Entity cache GET error: {e}")
            return None  # Graceful degradation

    async def set(self, name: str, uuid: str, entity_type: str) -> bool:
        """
        Cache entity name→UUID mapping.

        Args:
            name: Entity name
            uuid: Entity UUID
            entity_type: Entity type

        Returns:
            True if cached successfully, False otherwise

        Example:
            await entity_cache.set(
                "Alice",
                "d1844bab-7ca2-4c21-ba45-a583e7a65e22",
                "Person"
            )
        """
        if not self.redis.is_connected:
            logger.debug("Redis not connected, skipping cache write")
            return False

        key = self._make_key(name, entity_type)

        try:
            # No TTL - entities are permanent until explicitly invalidated
            result = await self.redis.set(key, uuid)

            if result:
                logger.debug(f"Entity cached: {name} ({entity_type}) -> {uuid}")
            else:
                logger.warning(f"Entity cache write failed: {name}")

            return result

        except Exception as e:
            logger.error(f"Entity cache SET error: {e}")
            return False  # Graceful degradation

    async def delete(self, name: str, entity_type: str) -> bool:
        """
        Invalidate cached entity (used when entity is updated/deleted).

        Args:
            name: Entity name
            entity_type: Entity type

        Returns:
            True if deleted, False otherwise

        Example:
            await entity_cache.delete("Alice", "Person")
        """
        if not self.redis.is_connected:
            logger.debug("Redis not connected, skipping cache invalidation")
            return False

        key = self._make_key(name, entity_type)

        try:
            result = await self.redis.delete(key)

            if result > 0:
                logger.debug(f"Entity cache invalidated: {name} ({entity_type})")
            else:
                logger.debug(f"Entity cache invalidation skipped (not cached): {name}")

            return result > 0

        except Exception as e:
            logger.error(f"Entity cache DELETE error: {e}")
            return False

    async def exists(self, name: str, entity_type: str) -> bool:
        """
        Check if entity is cached.

        Args:
            name: Entity name
            entity_type: Entity type

        Returns:
            True if cached, False otherwise
        """
        if not self.redis.is_connected:
            return False

        key = self._make_key(name, entity_type)

        try:
            return await self.redis.exists(key)
        except Exception as e:
            logger.error(f"Entity cache EXISTS error: {e}")
            return False

    async def batch_get(self, entities: list[tuple[str, str]]) -> dict[str, str | None]:
        """
        Optimization 2: Get multiple cached entity UUIDs in parallel.

        Redis MGET operation for batch cache lookups (much faster than individual GETs).

        Args:
            entities: List of (name, entity_type) tuples
                Example: [("Alice", "Person"), ("Neo4j", "Technology")]

        Returns:
            Dict mapping entity_key -> UUID (or None if not cached)
            Example: {
                "Carlos:Person": "abc-123",
                "Neo4j:Technology": "def-456"
            }

        Example:
            results = await entity_cache.batch_get([
                ("Alice", "Person"),
                ("Neo4j", "Technology"),
                ("Mnemodrome", "Project")
            ])
        """
        if not self.redis.is_connected:
            logger.debug("Redis not connected, returning empty batch")
            return {}

        if not entities:
            return {}

        # Generate keys for all entities
        keys = [self._make_key(name, entity_type) for name, entity_type in entities]

        try:
            # Use Redis MGET for batch retrieval (atomic operation)
            values = await self.redis.mget(keys)

            # Map results back to entity tuples
            result_map = {}
            for (name, entity_type), uuid in zip(entities, values, strict=False):
                entity_key = f"{name}:{entity_type}"
                if uuid:
                    result_map[entity_key] = uuid
                    logger.debug(f"Batch cache HIT: {entity_key} -> {uuid}")
                else:
                    logger.debug(f"Batch cache MISS: {entity_key}")

            return result_map

        except Exception as e:
            logger.error(f"Entity batch cache GET error: {e}")
            return {}

    async def batch_set(self, entities: dict[str, tuple[str, str]]) -> int:
        """
        Optimization 2: Set multiple cached entity UUIDs in parallel.

        Redis MSET operation for batch cache writes (much faster than individual SETs).

        Args:
            entities: Dict mapping entity_key -> (name, entity_type, uuid)
                     Actually dict of "name:type" -> uuid for MSET

        Returns:
            Number of entities cached

        Example:
            await entity_cache.batch_set({
                "Carlos:Person": "abc-123",
                "Neo4j:Technology": "def-456"
            })
        """
        if not self.redis.is_connected:
            logger.debug("Redis not connected, skipping batch cache write")
            return 0

        if not entities:
            return 0

        try:
            # Use Redis MSET for batch write (atomic operation)
            result = await self.redis.mset(entities)

            if result:
                logger.debug(f"Batch cached: {len(entities)} entities")
            else:
                logger.warning("Batch cache write failed")

            return len(entities) if result else 0

        except Exception as e:
            logger.error(f"Entity batch cache SET error: {e}")
            return 0

    async def warm_cache(self, neo4j_service) -> dict:
        """
        Warm cache on startup by loading existing entities from Neo4j.

        Idempotent: Checks if cache already populated before warming.

        Returns:
            Statistics: {cached: int, skipped: int, total: int, duration_ms: float}
        """
        if not self.redis.is_connected:
            logger.warning("Redis not connected, skipping cache warm")
            return {"cached": 0, "skipped": 0, "total": 0, "duration_ms": 0}

        start_time = time.time()

        try:
            # IDEMPOTENCY CHECK: Is cache already warm?
            existing_keys = await self.redis.keys("entity:*")
            if len(existing_keys) > 1000:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"✅ Cache already warm ({len(existing_keys)} entities), "
                    f"skipping re-warm ({duration_ms:.1f}ms check)"
                )
                return {
                    "cached": 0,
                    "skipped": len(existing_keys),
                    "total": len(existing_keys),
                    "duration_ms": duration_ms,
                }

            logger.info(
                f"Cache incomplete ({len(existing_keys)} entities), warming from Neo4j..."
            )

            # Query all entities from Neo4j
            query = """
            MATCH (e)
            WHERE e.uuid IS NOT NULL AND e.name IS NOT NULL AND e.entity_type IS NOT NULL
            RETURN e.uuid AS uuid, e.name AS name, e.entity_type AS type
            """

            results = await neo4j_service.query(query)
            total_count = len(results)

            # Batch cache writes
            cache_mapping = {}
            for record in results:
                uuid = record["uuid"]
                name = record["name"]
                entity_type = record["type"]

                key = self._make_key(name, entity_type)
                cache_mapping[key] = uuid

            # Use MSET for batch write (faster than individual SETs)
            if cache_mapping:
                await self.redis.mset(cache_mapping)

            entity_count = len(cache_mapping)
            duration_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Entity cache warmed: {entity_count} entities cached "
                f"({duration_ms:.1f}ms)"
            )

            return {
                "cached": entity_count,
                "skipped": 0,
                "total": entity_count,
                "duration_ms": duration_ms,
            }

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Cache warm failed: {e}")
            return {"cached": 0, "skipped": 0, "total": 0, "duration_ms": duration_ms}


# Global service instance
entity_cache = EntityCache()

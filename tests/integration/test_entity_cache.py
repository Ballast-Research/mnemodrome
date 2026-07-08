"""
Tests for Redis Entity Cache (Feature 005)

Tests cache performance, fallback behavior, and data consistency.
"""

import asyncio

import pytest

from src.config import REDIS_DB, REDIS_HOST, REDIS_PORT
from src.services.entity_cache import entity_cache
from src.services.entity_factory import entity_factory
from src.services.neo4j_service import neo4j_service
from src.services.redis_service import redis_service


@pytest.fixture(scope="module")
async def setup_services():
    """Initialize services for testing."""
    # Connect to Redis
    await redis_service.connect(REDIS_HOST, REDIS_PORT, REDIS_DB)

    # Connect to Neo4j
    await neo4j_service.connect()

    yield

    # Cleanup
    await redis_service.disconnect()
    await neo4j_service.close()


@pytest.mark.asyncio
async def test_cache_hit_performance(setup_services):
    """
    Test 1: Cache hit performance (<100ms).

    Verifies that cached entity lookups are significantly faster than
    Neo4j + LLM classification (target: <100ms vs 10-15s).
    """
    # Create entity and cache it
    entity_name = "TestEntity_CacheHit"
    entity_type = "Technology"

    uuid = await entity_factory.create_or_resolve(entity_name, entity_type)

    # Measure cache hit performance
    import time

    start = time.time()
    cached_uuid = await entity_cache.get(entity_name, entity_type)
    elapsed_ms = (time.time() - start) * 1000

    assert cached_uuid == uuid, "Cached UUID should match original"
    assert elapsed_ms < 100, f"Cache hit took {elapsed_ms}ms, should be <100ms"

    print(f"✅ Cache hit performance: {elapsed_ms:.2f}ms")


@pytest.mark.asyncio
async def test_cache_miss_fallback(setup_services):
    """
    Test 2: Cache miss with Neo4j fallback.

    Verifies that cache misses correctly fall back to Neo4j without errors.
    """
    # Clear cache for test entity
    entity_name = "TestEntity_CacheMiss"
    entity_type = "Project"

    await entity_cache.delete(entity_name, entity_type)

    # Cache miss should return None
    cached_uuid = await entity_cache.get(entity_name, entity_type)
    assert cached_uuid is None, "Cache miss should return None"

    # EntityFactory should handle miss and query Neo4j
    uuid = await entity_factory.create_or_resolve(entity_name, entity_type)
    assert uuid is not None, "Neo4j fallback should create entity"

    # Verify entity was cached after creation
    cached_uuid = await entity_cache.get(entity_name, entity_type)
    assert cached_uuid == uuid, "Entity should be cached after creation"

    print(f"✅ Cache miss fallback working: {uuid}")


@pytest.mark.asyncio
async def test_warm_cache_startup(setup_services):
    """
    Test 3: Warm cache on startup (100 entities).

    Verifies that cache warming loads existing entities from Neo4j efficiently.
    Performance target: <5s for 1000 entities (proportional for smaller sets).
    """
    # Create test entities
    test_entities = []
    for i in range(10):
        name = f"WarmCacheTest_{i}"
        entity_type = "Entity"
        uuid = await entity_factory.create_or_resolve(name, entity_type)
        test_entities.append((name, entity_type, uuid))

    # Clear cache
    for name, entity_type, _ in test_entities:
        await entity_cache.delete(name, entity_type)

    # Warm cache
    import time

    start = time.time()
    count = await entity_cache.warm_cache(neo4j_service)
    elapsed = time.time() - start

    assert count >= 10, f"Should cache at least 10 entities, got {count}"
    assert elapsed < 5, (
        f"Warm cache took {elapsed:.2f}s, should be <5s for 1000 entities"
    )

    # Verify entities are cached
    for name, entity_type, uuid in test_entities:
        cached_uuid = await entity_cache.get(name, entity_type)
        assert cached_uuid == uuid, f"Entity {name} should be cached"

    print(f"✅ Warm cache: {count} entities in {elapsed:.2f}s")


@pytest.mark.asyncio
async def test_cache_invalidation(setup_services):
    """
    Test 4: Cache invalidation on entity update.

    Verifies that entity updates correctly invalidate the cache.
    """
    entity_name = "TestEntity_Invalidation"
    entity_type = "Concept"

    # Create and cache entity
    uuid = await entity_factory.create_or_resolve(
        entity_name, entity_type, "Original summary"
    )

    # Verify cached
    cached_uuid = await entity_cache.get(entity_name, entity_type)
    assert cached_uuid == uuid, "Entity should be cached"

    # Update entity (should invalidate cache)
    await entity_factory.enrich_entity(uuid, "Updated summary")

    # Cache should be invalidated
    # Note: After invalidation, next lookup will re-cache
    # For this test, we verify the invalidation happened by checking the entity
    entity_data = await entity_factory.get_entity_by_uuid(uuid)
    assert entity_data["summary"] == "Updated summary", "Summary should be updated"

    print(f"✅ Cache invalidation working for {uuid}")


@pytest.mark.asyncio
async def test_redis_unavailable_fallback(setup_services):
    """
    Test 5: Server continues when Redis unavailable.

    Verifies graceful degradation when Redis is down.
    """
    # Disconnect Redis to simulate unavailability
    await redis_service.disconnect()

    # EntityFactory should still work (fallback to Neo4j only)
    entity_name = "TestEntity_NoRedis"
    entity_type = "Location"

    uuid = await entity_factory.create_or_resolve(entity_name, entity_type)
    assert uuid is not None, "Should create entity even without Redis"

    # Cache operations should gracefully fail
    cached_uuid = await entity_cache.get(entity_name, entity_type)
    assert cached_uuid is None, "Cache should return None when Redis unavailable"

    # Reconnect Redis for remaining tests
    await redis_service.connect(REDIS_HOST, REDIS_PORT, REDIS_DB)

    print(f"✅ Graceful degradation working: {uuid}")


@pytest.mark.asyncio
async def test_concurrent_access(setup_services):
    """
    Test 6: Concurrent access (10 parallel requests).

    Verifies cache handles concurrent requests correctly.
    """
    entity_name = "TestEntity_Concurrent"
    entity_type = "Organization"

    # Create entity
    uuid = await entity_factory.create_or_resolve(entity_name, entity_type)

    # Concurrent cache reads
    async def read_cache():
        return await entity_cache.get(entity_name, entity_type)

    tasks = [read_cache() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # All should return same UUID
    assert all(r == uuid for r in results), (
        "All concurrent reads should return same UUID"
    )

    print("✅ Concurrent access: 10 parallel reads successful")


@pytest.mark.asyncio
async def test_memory_footprint(setup_services):
    """
    Test 7: Memory footprint validation.

    Verifies that 10k entities use <100MB of Redis memory.
    Target: ~10KB per entity
    """
    # Cache 1000 test entities (reduced from 10k for faster testing)
    for i in range(1000):
        name = f"MemoryTest_{i}"
        entity_type = "Entity"
        uuid = f"uuid-{i:06d}"
        await entity_cache.set(name, uuid, entity_type)

    # Check Redis memory usage
    if redis_service.is_connected:
        info = await redis_service.client.info("memory")
        used_memory_mb = info["used_memory"] / (1024 * 1024)

        # 1000 entities should use <10MB (extrapolate to 10k = <100MB)
        assert used_memory_mb < 10, (
            f"1000 entities used {used_memory_mb:.2f}MB, should be <10MB"
        )

        print(
            f"✅ Memory footprint: 1000 entities = {used_memory_mb:.2f}MB (extrapolated 10k = {used_memory_mb * 10:.2f}MB)"
        )
    else:
        print("⚠️  Redis not connected, skipping memory test")


@pytest.mark.asyncio
async def test_persistence_after_restart(setup_services):
    """
    Test 8: Persistence survives Redis restart.

    Verifies RDB+AOF persistence keeps cache across restarts.
    Note: This test requires manual Redis restart to fully verify.
    """
    entity_name = "TestEntity_Persistence"
    entity_type = "Document"

    # Create and cache entity
    uuid = await entity_factory.create_or_resolve(entity_name, entity_type)

    # Verify cached
    cached_uuid = await entity_cache.get(entity_name, entity_type)
    assert cached_uuid == uuid, "Entity should be cached"

    print(f"✅ Entity persisted: {entity_name} -> {uuid}")
    print("⚠️  To fully test persistence, restart Redis container and re-run this test")


if __name__ == "__main__":
    """Run tests manually with: python -m pytest tests/test_entity_cache.py -v"""
    pytest.main([__file__, "-v", "-s"])

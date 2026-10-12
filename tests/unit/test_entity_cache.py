"""Unit tests for Entity Cache Service.

Tests for the EntityCache class covering:
- Key formatting and namespacing
- Cache get/set/delete/exists operations with mocked Redis
- Batch get/set operations
- Cache warming from Neo4j
- Graceful degradation when Redis is unavailable
- Error handling on all operations
"""

from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest

from src.services.entity_cache import EntityCache


class TestEntityCacheKeyFormatting:
    """Tests for _make_key and key namespacing."""

    def setup_method(self):
        self.cache = EntityCache()

    def test_make_key_basic(self):
        """Key follows entity:{name}:{type} format."""
        assert self.cache._make_key("Alice", "Person") == "entity:Alice:Person"

    def test_make_key_strips_whitespace(self):
        """Leading/trailing whitespace in name and type is stripped."""
        assert self.cache._make_key("  Alice  ", "  Person  ") == "entity:Alice:Person"

    def test_make_key_preserves_internal_spaces(self):
        """Internal spaces in names are preserved."""
        assert (
            self.cache._make_key("Neo4j Database", "Technology")
            == "entity:Neo4j Database:Technology"
        )

    def test_make_key_uses_prefix(self):
        """Key uses the configured prefix."""
        assert self.cache.prefix == "entity"
        key = self.cache._make_key("X", "Y")
        assert key.startswith("entity:")

    def test_make_key_different_types_different_keys(self):
        """Same name with different types produces different keys."""
        key1 = self.cache._make_key("Python", "Language")
        key2 = self.cache._make_key("Python", "Snake")
        assert key1 != key2

    def test_make_key_case_sensitive(self):
        """Keys are case-sensitive for name and type."""
        key1 = self.cache._make_key("alice", "person")
        key2 = self.cache._make_key("Alice", "Person")
        assert key1 != key2


class TestEntityCacheDisconnected:
    """Tests for graceful degradation when Redis is not connected."""

    def setup_method(self):
        self.cache = EntityCache()
        # Patch the redis_service on the instance
        self.mock_redis = MagicMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=False)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_get_returns_none_when_disconnected(self):
        result = await self.cache.get("Alice", "Person")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_returns_false_when_disconnected(self):
        result = await self.cache.set("Alice", "uuid-123", "Person")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_returns_false_when_disconnected(self):
        result = await self.cache.delete("Alice", "Person")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_disconnected(self):
        result = await self.cache.exists("Alice", "Person")
        assert result is False

    @pytest.mark.asyncio
    async def test_batch_get_returns_empty_when_disconnected(self):
        result = await self.cache.batch_get([("Alice", "Person")])
        assert result == {}

    @pytest.mark.asyncio
    async def test_batch_set_returns_zero_when_disconnected(self):
        result = await self.cache.batch_set({"entity:Alice:Person": "uuid-123"})
        assert result == 0

    @pytest.mark.asyncio
    async def test_warm_cache_returns_zeros_when_disconnected(self):
        neo4j = AsyncMock()
        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 0
        assert result["total"] == 0
        assert "duration_ms" in result


class TestEntityCacheGet:
    """Tests for cache get operations."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_get_cache_hit(self):
        """Returns UUID on cache hit."""
        self.mock_redis.get = AsyncMock(return_value="uuid-abc-123")
        result = await self.cache.get("Alice", "Person")
        assert result == "uuid-abc-123"
        self.mock_redis.get.assert_awaited_once_with("entity:Alice:Person")

    @pytest.mark.asyncio
    async def test_get_cache_miss(self):
        """Returns None on cache miss."""
        self.mock_redis.get = AsyncMock(return_value=None)
        result = await self.cache.get("Unknown", "Person")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_error_returns_none(self):
        """Returns None on Redis error (graceful degradation)."""
        self.mock_redis.get = AsyncMock(side_effect=Exception("Connection lost"))
        result = await self.cache.get("Alice", "Person")
        assert result is None


class TestEntityCacheSet:
    """Tests for cache set operations."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_set_success(self):
        """Returns True on successful cache write."""
        self.mock_redis.set = AsyncMock(return_value=True)
        result = await self.cache.set("Alice", "uuid-abc-123", "Person")
        assert result is True
        self.mock_redis.set.assert_awaited_once_with(
            "entity:Alice:Person", "uuid-abc-123"
        )

    @pytest.mark.asyncio
    async def test_set_failure(self):
        """Returns False when Redis set returns falsy."""
        self.mock_redis.set = AsyncMock(return_value=False)
        result = await self.cache.set("Alice", "uuid-abc-123", "Person")
        assert result is False

    @pytest.mark.asyncio
    async def test_set_error_returns_false(self):
        """Returns False on Redis error (graceful degradation)."""
        self.mock_redis.set = AsyncMock(side_effect=Exception("Write error"))
        result = await self.cache.set("Alice", "uuid-abc-123", "Person")
        assert result is False


class TestEntityCacheDelete:
    """Tests for cache delete/invalidation operations."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_delete_existing_key(self):
        """Returns True when key existed and was deleted."""
        self.mock_redis.delete = AsyncMock(return_value=1)
        result = await self.cache.delete("Alice", "Person")
        assert result is True
        self.mock_redis.delete.assert_awaited_once_with("entity:Alice:Person")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self):
        """Returns False when key did not exist."""
        self.mock_redis.delete = AsyncMock(return_value=0)
        result = await self.cache.delete("Unknown", "Person")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_error_returns_false(self):
        """Returns False on Redis error (graceful degradation)."""
        self.mock_redis.delete = AsyncMock(side_effect=Exception("Delete error"))
        result = await self.cache.delete("Alice", "Person")
        assert result is False


class TestEntityCacheExists:
    """Tests for cache existence checks."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Returns True when key exists."""
        self.mock_redis.exists = AsyncMock(return_value=True)
        result = await self.cache.exists("Alice", "Person")
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Returns False when key does not exist."""
        self.mock_redis.exists = AsyncMock(return_value=False)
        result = await self.cache.exists("Alice", "Person")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_error_returns_false(self):
        """Returns False on Redis error (graceful degradation)."""
        self.mock_redis.exists = AsyncMock(side_effect=Exception("Exists error"))
        result = await self.cache.exists("Alice", "Person")
        assert result is False


class TestEntityCacheBatchGet:
    """Tests for batch get operations."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_batch_get_all_hits(self):
        """Returns all UUIDs when all cached."""
        self.mock_redis.mget = AsyncMock(return_value=["uuid-1", "uuid-2"])
        entities = [("Alice", "Person"), ("Neo4j", "Technology")]
        result = await self.cache.batch_get(entities)
        assert result == {"Alice:Person": "uuid-1", "Neo4j:Technology": "uuid-2"}
        self.mock_redis.mget.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_batch_get_partial_hits(self):
        """Only includes hits in result dict, skips misses."""
        self.mock_redis.mget = AsyncMock(return_value=["uuid-1", None])
        entities = [("Alice", "Person"), ("Unknown", "Person")]
        result = await self.cache.batch_get(entities)
        assert result == {"Alice:Person": "uuid-1"}
        assert "Unknown:Person" not in result

    @pytest.mark.asyncio
    async def test_batch_get_all_misses(self):
        """Returns empty dict when no hits."""
        self.mock_redis.mget = AsyncMock(return_value=[None, None])
        entities = [("A", "X"), ("B", "Y")]
        result = await self.cache.batch_get(entities)
        assert result == {}

    @pytest.mark.asyncio
    async def test_batch_get_empty_list(self):
        """Returns empty dict for empty input without calling Redis."""
        result = await self.cache.batch_get([])
        assert result == {}
        self.mock_redis.mget.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_get_error_returns_empty(self):
        """Returns empty dict on Redis error (graceful degradation)."""
        self.mock_redis.mget = AsyncMock(side_effect=Exception("MGET error"))
        result = await self.cache.batch_get([("A", "X")])
        assert result == {}

    @pytest.mark.asyncio
    async def test_batch_get_correct_keys(self):
        """Sends correctly formatted keys to Redis mget."""
        self.mock_redis.mget = AsyncMock(return_value=[None])
        await self.cache.batch_get([("Alice", "Person")])
        called_keys = self.mock_redis.mget.call_args[0][0]
        assert called_keys == ["entity:Alice:Person"]


class TestEntityCacheBatchSet:
    """Tests for batch set operations."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_batch_set_success(self):
        """Returns count of entities cached on success."""
        self.mock_redis.mset = AsyncMock(return_value=True)
        mapping = {"Alice:Person": "uuid-1", "Neo4j:Technology": "uuid-2"}
        result = await self.cache.batch_set(mapping)
        assert result == 2
        self.mock_redis.mset.assert_awaited_once_with(mapping)

    @pytest.mark.asyncio
    async def test_batch_set_failure(self):
        """Returns 0 when mset returns falsy."""
        self.mock_redis.mset = AsyncMock(return_value=False)
        result = await self.cache.batch_set({"X:Y": "uuid-1"})
        assert result == 0

    @pytest.mark.asyncio
    async def test_batch_set_empty_dict(self):
        """Returns 0 for empty input without calling Redis."""
        result = await self.cache.batch_set({})
        assert result == 0
        self.mock_redis.mset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_set_error_returns_zero(self):
        """Returns 0 on Redis error (graceful degradation)."""
        self.mock_redis.mset = AsyncMock(side_effect=Exception("MSET error"))
        result = await self.cache.batch_set({"X:Y": "uuid-1"})
        assert result == 0


class TestEntityCacheWarmCache:
    """Tests for cache warming from Neo4j."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_warm_cache_populates_from_neo4j(self):
        """Queries Neo4j and caches all entities via mset."""
        self.mock_redis.keys = AsyncMock(return_value=[])  # Cache empty
        self.mock_redis.mset = AsyncMock(return_value=True)

        neo4j = AsyncMock()
        neo4j.query = AsyncMock(
            return_value=[
                {"uuid": "uuid-1", "name": "Alice", "type": "Person"},
                {"uuid": "uuid-2", "name": "Neo4j", "type": "Technology"},
            ]
        )

        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 2
        assert result["skipped"] == 0
        assert result["total"] == 2
        assert result["duration_ms"] >= 0

        # Verify mset called with correct key mapping
        mset_call = self.mock_redis.mset.call_args[0][0]
        assert mset_call["entity:Alice:Person"] == "uuid-1"
        assert mset_call["entity:Neo4j:Technology"] == "uuid-2"

    @pytest.mark.asyncio
    async def test_warm_cache_skips_when_already_warm(self):
        """Does not re-warm when >1000 keys already exist (idempotency)."""
        self.mock_redis.keys = AsyncMock(return_value=["k"] * 1500)

        neo4j = AsyncMock()
        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 0
        assert result["skipped"] == 1500
        assert result["total"] == 1500
        # Neo4j should not be queried
        neo4j.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_warm_cache_partial_warm_triggers_refresh(self):
        """Re-warms when <1000 keys exist."""
        self.mock_redis.keys = AsyncMock(return_value=["k"] * 500)
        self.mock_redis.mset = AsyncMock(return_value=True)

        neo4j = AsyncMock()
        neo4j.query = AsyncMock(
            return_value=[
                {"uuid": "uuid-1", "name": "A", "type": "X"},
            ]
        )

        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 1
        neo4j.query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_warm_cache_empty_neo4j(self):
        """Handles empty Neo4j result gracefully."""
        self.mock_redis.keys = AsyncMock(return_value=[])
        self.mock_redis.mset = AsyncMock(return_value=True)

        neo4j = AsyncMock()
        neo4j.query = AsyncMock(return_value=[])

        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 0
        assert result["total"] == 0
        # mset should not be called with empty mapping
        self.mock_redis.mset.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_warm_cache_neo4j_error(self):
        """Returns zeros on Neo4j query failure (graceful degradation)."""
        self.mock_redis.keys = AsyncMock(return_value=[])

        neo4j = AsyncMock()
        neo4j.query = AsyncMock(side_effect=Exception("Neo4j unavailable"))

        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 0
        assert result["total"] == 0
        assert result["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_warm_cache_redis_keys_error(self):
        """Returns zeros when Redis keys() call fails."""
        self.mock_redis.keys = AsyncMock(side_effect=Exception("Redis error"))

        neo4j = AsyncMock()
        result = await self.cache.warm_cache(neo4j)
        assert result["cached"] == 0
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_warm_cache_duration_tracked(self):
        """Duration is always reported even on failure."""
        self.mock_redis.keys = AsyncMock(return_value=[])

        neo4j = AsyncMock()
        neo4j.query = AsyncMock(side_effect=Exception("fail"))

        result = await self.cache.warm_cache(neo4j)
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], float)


class TestEntityCacheRoundTrip:
    """Integration-style tests using mock_redis_client fixture for round-trip behavior."""

    def setup_method(self):
        self.cache = EntityCache()
        self.mock_redis = AsyncMock()
        type(self.mock_redis).is_connected = PropertyMock(return_value=True)
        self.cache.redis = self.mock_redis

    @pytest.mark.asyncio
    async def test_set_then_get(self):
        """Value set can be retrieved."""
        stored = {}

        async def mock_set(key, value):
            stored[key] = value
            return True

        async def mock_get(key):
            return stored.get(key)

        self.mock_redis.set = AsyncMock(side_effect=mock_set)
        self.mock_redis.get = AsyncMock(side_effect=mock_get)

        await self.cache.set("Alice", "uuid-123", "Person")
        result = await self.cache.get("Alice", "Person")
        assert result == "uuid-123"

    @pytest.mark.asyncio
    async def test_set_then_delete_then_get(self):
        """Deleted key returns None on subsequent get."""
        stored = {}

        async def mock_set(key, value):
            stored[key] = value
            return True

        async def mock_get(key):
            return stored.get(key)

        async def mock_delete(key):
            return 1 if stored.pop(key, None) is not None else 0

        self.mock_redis.set = AsyncMock(side_effect=mock_set)
        self.mock_redis.get = AsyncMock(side_effect=mock_get)
        self.mock_redis.delete = AsyncMock(side_effect=mock_delete)

        await self.cache.set("Alice", "uuid-123", "Person")
        deleted = await self.cache.delete("Alice", "Person")
        assert deleted is True
        result = await self.cache.get("Alice", "Person")
        assert result is None

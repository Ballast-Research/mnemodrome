"""Unit tests for Redis Service.

Tests for the RedisService class covering:
- Connection management and pooling
- Health checks (ping)
- Key-value operations (get, set, delete, exists)
- Batch operations (mget, mset)
- Connection state handling
- Error handling and graceful degradation
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.redis_service import RedisService


class TestRedisService:
    """Tests for the RedisService class."""

    @pytest.fixture
    def service(self):
        """Provide a fresh Redis service instance."""
        return RedisService()

    @pytest.fixture
    def mock_client(self):
        """Provide a mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def mock_pool(self):
        """Provide a mock connection pool."""
        return AsyncMock()

    # ========================
    # Connection Tests
    # ========================

    @pytest.mark.asyncio
    async def test_connect_success(self, service, mock_client):
        """Test successful connection to Redis."""
        mock_client.ping = AsyncMock()

        with patch(
            "src.services.redis_service.redis.ConnectionPool"
        ) as mock_pool_class:
            with patch("src.services.redis_service.redis.Redis") as mock_redis_class:
                mock_pool = AsyncMock()
                mock_pool_class.return_value = mock_pool
                mock_redis_class.return_value = mock_client

                await service.connect(host="localhost", port=6379, db=0)

                assert service.client is not None
                assert service.pool is not None
                assert service._connected is True
                mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_custom_params(self, service, mock_client):
        """Test connection with custom parameters."""
        mock_client.ping = AsyncMock()

        with patch(
            "src.services.redis_service.redis.ConnectionPool"
        ) as mock_pool_class:
            with patch("src.services.redis_service.redis.Redis") as mock_redis_class:
                mock_pool = AsyncMock()
                mock_pool_class.return_value = mock_pool
                mock_redis_class.return_value = mock_client

                await service.connect(
                    host="redis.example.com",
                    port=6380,
                    db=1,
                    max_connections=20,
                    decode_responses=False,
                )

                # Verify pool was created with correct params
                mock_pool_class.assert_called_once()
                call_args = mock_pool_class.call_args
                assert call_args[1]["host"] == "redis.example.com"
                assert call_args[1]["port"] == 6380
                assert call_args[1]["db"] == 1
                assert call_args[1]["max_connections"] == 20

    @pytest.mark.asyncio
    async def test_connect_failure(self, service, mock_client):
        """Test connection failure handling."""
        mock_client.ping = AsyncMock(side_effect=Exception("Connection refused"))

        with patch("src.services.redis_service.redis.ConnectionPool"):
            with patch("src.services.redis_service.redis.Redis") as mock_redis_class:
                mock_redis_class.return_value = mock_client

                with pytest.raises(Exception, match="Connection refused"):
                    await service.connect()

                assert service._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_success(self, service, mock_client, mock_pool):
        """Test successful disconnection."""
        service.client = mock_client
        service.pool = mock_pool

        await service.disconnect()

        mock_client.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()
        assert service._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_no_client(self, service):
        """Test disconnect when client not initialized."""
        service.client = None

        await service.disconnect()

        assert service._connected is False

    # ========================
    # Ping Tests
    # ========================

    @pytest.mark.asyncio
    async def test_ping_success(self, service, mock_client):
        """Test successful ping."""
        service.client = mock_client
        mock_client.ping = AsyncMock()

        result = await service.ping()

        assert result is True
        mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_ping_failure(self, service, mock_client):
        """Test ping failure."""
        service.client = mock_client
        mock_client.ping = AsyncMock(side_effect=Exception("Connection lost"))

        result = await service.ping()

        assert result is False

    @pytest.mark.asyncio
    async def test_ping_no_client(self, service):
        """Test ping when client not initialized."""
        service.client = None

        result = await service.ping()

        assert result is False

    # ========================
    # Basic Operations Tests
    # ========================

    @pytest.mark.asyncio
    async def test_get_hit(self, service, mock_client):
        """Test GET operation with cache hit."""
        service.client = mock_client
        service._connected = True
        mock_client.get = AsyncMock(return_value="test-value")

        result = await service.get("test-key")

        assert result == "test-value"
        mock_client.get.assert_called_once_with("test-key")

    @pytest.mark.asyncio
    async def test_get_miss(self, service, mock_client):
        """Test GET operation with cache miss."""
        service.client = mock_client
        service._connected = True
        mock_client.get = AsyncMock(return_value=None)

        result = await service.get("test-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_not_connected(self, service):
        """Test GET returns None when not connected."""
        service._connected = False

        result = await service.get("test-key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_error(self, service, mock_client):
        """Test GET error handling."""
        service.client = mock_client
        service._connected = True
        mock_client.get = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(Exception, match="Redis error"):
            await service.get("test-key")

    @pytest.mark.asyncio
    async def test_set_success(self, service, mock_client):
        """Test successful SET operation."""
        service.client = mock_client
        service._connected = True
        mock_client.set = AsyncMock(return_value=True)

        result = await service.set("test-key", "test-value")

        assert result is True
        mock_client.set.assert_called_once_with(
            "test-key", "test-value", ex=None, nx=False
        )

    @pytest.mark.asyncio
    async def test_set_with_expiration(self, service, mock_client):
        """Test SET with expiration time."""
        service.client = mock_client
        service._connected = True
        mock_client.set = AsyncMock(return_value=True)

        result = await service.set("test-key", "test-value", ex=3600)

        assert result is True
        mock_client.set.assert_called_once_with(
            "test-key", "test-value", ex=3600, nx=False
        )

    @pytest.mark.asyncio
    async def test_set_nx_flag(self, service, mock_client):
        """Test SET with nx flag (only set if not exists)."""
        service.client = mock_client
        service._connected = True
        mock_client.set = AsyncMock(return_value=True)

        result = await service.set("test-key", "test-value", nx=True)

        assert result is True
        mock_client.set.assert_called_once_with(
            "test-key", "test-value", ex=None, nx=True
        )

    @pytest.mark.asyncio
    async def test_set_not_connected(self, service):
        """Test SET returns False when not connected."""
        service._connected = False

        result = await service.set("test-key", "test-value")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_success(self, service, mock_client):
        """Test successful DELETE operation."""
        service.client = mock_client
        service._connected = True
        mock_client.delete = AsyncMock(return_value=1)

        result = await service.delete("test-key")

        assert result == 1
        mock_client.delete.assert_called_once_with("test-key")

    @pytest.mark.asyncio
    async def test_delete_key_not_found(self, service, mock_client):
        """Test DELETE on non-existent key."""
        service.client = mock_client
        service._connected = True
        mock_client.delete = AsyncMock(return_value=0)

        result = await service.delete("test-key")

        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_not_connected(self, service):
        """Test DELETE returns 0 when not connected."""
        service._connected = False

        result = await service.delete("test-key")

        assert result == 0

    @pytest.mark.asyncio
    async def test_exists_true(self, service, mock_client):
        """Test EXISTS when key exists."""
        service.client = mock_client
        service._connected = True
        mock_client.exists = AsyncMock(return_value=1)

        result = await service.exists("test-key")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_false(self, service, mock_client):
        """Test EXISTS when key doesn't exist."""
        service.client = mock_client
        service._connected = True
        mock_client.exists = AsyncMock(return_value=0)

        result = await service.exists("test-key")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_not_connected(self, service):
        """Test EXISTS returns False when not connected."""
        service._connected = False

        result = await service.exists("test-key")

        assert result is False

    # ========================
    # Batch Operations Tests
    # ========================

    @pytest.mark.asyncio
    async def test_mget_success(self, service, mock_client):
        """Test successful MGET operation."""
        service.client = mock_client
        service._connected = True
        mock_client.mget = AsyncMock(return_value=["value1", None, "value3"])

        result = await service.mget(["key1", "key2", "key3"])

        assert result == ["value1", None, "value3"]
        mock_client.mget.assert_called_once_with(["key1", "key2", "key3"])

    @pytest.mark.asyncio
    async def test_mget_not_connected(self, service):
        """Test MGET returns list of None when not connected."""
        service._connected = False

        result = await service.mget(["key1", "key2", "key3"])

        assert result == [None, None, None]

    @pytest.mark.asyncio
    async def test_mset_success(self, service, mock_client):
        """Test successful MSET operation."""
        service.client = mock_client
        service._connected = True
        mock_client.mset = AsyncMock(return_value=True)

        mapping = {"key1": "value1", "key2": "value2"}
        result = await service.mset(mapping)

        assert result is True
        mock_client.mset.assert_called_once_with(mapping)

    @pytest.mark.asyncio
    async def test_mset_not_connected(self, service):
        """Test MSET returns False when not connected."""
        service._connected = False

        result = await service.mset({"key1": "value1"})

        assert result is False

    # ========================
    # Keys Pattern Tests
    # ========================

    @pytest.mark.asyncio
    async def test_keys_pattern_match(self, service, mock_client):
        """Test KEYS with pattern matching."""
        service.client = mock_client
        service._connected = True
        mock_client.keys = AsyncMock(return_value=["entity:1", "entity:2", "entity:3"])

        result = await service.keys("entity:*")

        assert result == ["entity:1", "entity:2", "entity:3"]
        mock_client.keys.assert_called_once_with("entity:*")

    @pytest.mark.asyncio
    async def test_keys_no_match(self, service, mock_client):
        """Test KEYS with no matches."""
        service.client = mock_client
        service._connected = True
        mock_client.keys = AsyncMock(return_value=[])

        result = await service.keys("nonexistent:*")

        assert result == []

    @pytest.mark.asyncio
    async def test_keys_not_connected(self, service):
        """Test KEYS returns empty list when not connected."""
        service._connected = False

        result = await service.keys("*")

        assert result == []

    @pytest.mark.asyncio
    async def test_keys_with_bytes(self, service, mock_client):
        """Test KEYS properly decodes byte responses."""
        service.client = mock_client
        service._connected = True
        mock_client.keys = AsyncMock(return_value=[b"entity:1", "entity:2"])

        result = await service.keys("entity:*")

        # Should decode bytes to strings
        assert all(isinstance(k, str) for k in result)

    # ========================
    # Connection State Tests
    # ========================

    @pytest.mark.asyncio
    async def test_is_connected_property(self, service):
        """Test is_connected property."""
        service._connected = False
        assert service.is_connected is False

        service._connected = True
        assert service.is_connected is True

    @pytest.mark.asyncio
    async def test_error_recovery(self, service, mock_client):
        """Test error handling doesn't permanently break connection state."""
        service.client = mock_client
        service._connected = True
        mock_client.get = AsyncMock(side_effect=Exception("Connection error"))

        with pytest.raises(Exception):
            await service.get("test-key")

        # Should still be marked as connected (error is transient)
        assert service.is_connected is True

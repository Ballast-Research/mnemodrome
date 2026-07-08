"""
Redis Service - Connection pooling and key-value operations

Provides connection management, health checks, and graceful degradation
for Redis-based entity caching (Feature 005).
"""

import redis.asyncio as redis
from loguru import logger


class RedisService:
    """
    Async Redis service with connection pooling and graceful degradation.

    Features:
    - Connection pool management
    - Health checks via ping()
    - Graceful error handling
    - Auto-reconnection support

    Constitutional Authority: Feature 005 (Redis Entity Cache)
    """

    def __init__(self):
        """Initialize Redis service (connection created lazily)."""
        self.client: redis.Redis | None = None
        self.pool: redis.ConnectionPool | None = None
        self._connected = False

    async def connect(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_connections: int = 10,
        decode_responses: bool = True,
    ) -> None:
        """
        Connect to Redis with connection pooling.

        Args:
            host: Redis host (default: localhost)
            port: Redis port (default: 6379)
            db: Redis database number (default: 0)
            max_connections: Max connections in pool (default: 10)
            decode_responses: Decode byte responses to strings (default: True)

        Raises:
            redis.ConnectionError: If connection fails
        """
        try:
            logger.info(f"Connecting to Redis at {host}:{port} (db={db})")

            # Create connection pool
            self.pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                max_connections=max_connections,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )

            # Create Redis client from pool
            self.client = redis.Redis(connection_pool=self.pool)

            # Verify connection with ping
            await self.client.ping()

            self._connected = True
            logger.info(
                f"Redis connected: {host}:{port} (pool size: {max_connections})"
            )

        except Exception as e:
            self._connected = False
            logger.error(f"Redis connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """
        Gracefully disconnect from Redis and close connection pool.

        Ensures all connections are properly released.
        """
        if not self.client:
            logger.debug("Redis client not initialized, skipping disconnect")
            return

        try:
            logger.info("Disconnecting from Redis")

            # Close client connection
            await self.client.close()

            # Close connection pool
            if self.pool:
                await self.pool.disconnect()

            self._connected = False
            logger.info("Redis disconnected successfully")

        except Exception as e:
            logger.error(f"Error during Redis disconnect: {e}")
            self._connected = False

    async def ping(self) -> bool:
        """
        Health check: verify Redis connection is alive.

        Returns:
            True if ping successful, False otherwise
        """
        if not self.client:
            logger.warning("Redis client not initialized")
            return False

        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    async def get(self, key: str) -> str | None:
        """
        Get value by key.

        Args:
            key: Redis key

        Returns:
            Value if exists, None otherwise

        Raises:
            redis.RedisError: If operation fails
        """
        if not self._connected:
            logger.warning("Redis not connected, returning None")
            return None

        try:
            value = await self.client.get(key)
            logger.debug(f"Redis GET: {key} -> {'HIT' if value else 'MISS'}")
            return value
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            raise

    async def set(
        self, key: str, value: str, ex: int | None = None, nx: bool = False
    ) -> bool:
        """
        Set key-value pair with optional expiration.

        Args:
            key: Redis key
            value: Value to store
            ex: Expiration time in seconds (None = no expiration)
            nx: Only set if key doesn't exist (default: False)

        Returns:
            True if set successful, False otherwise

        Raises:
            redis.RedisError: If operation fails
        """
        if not self._connected:
            logger.warning("Redis not connected, skipping SET")
            return False

        try:
            result = await self.client.set(key, value, ex=ex, nx=nx)
            logger.debug(f"Redis SET: {key} -> {result}")
            return bool(result)
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            raise

    async def delete(self, key: str) -> int:
        """
        Delete key from Redis.

        Args:
            key: Redis key to delete

        Returns:
            Number of keys deleted (0 or 1)

        Raises:
            redis.RedisError: If operation fails
        """
        if not self._connected:
            logger.warning("Redis not connected, skipping DELETE")
            return 0

        try:
            result = await self.client.delete(key)
            logger.debug(f"Redis DELETE: {key} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Redis DELETE error for key '{key}': {e}")
            raise

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.

        Args:
            key: Redis key to check

        Returns:
            True if key exists, False otherwise

        Raises:
            redis.RedisError: If operation fails
        """
        if not self._connected:
            logger.warning("Redis not connected, returning False")
            return False

        try:
            result = await self.client.exists(key)
            logger.debug(f"Redis EXISTS: {key} -> {bool(result)}")
            return bool(result)
        except Exception as e:
            logger.error(f"Redis EXISTS error for key '{key}': {e}")
            raise

    async def mget(self, keys: list[str]) -> list[str | None]:
        """
        Get multiple values by keys (batch operation).

        Args:
            keys: List of Redis keys

        Returns:
            List of values (None for missing keys)

        Raises:
            redis.RedisError: If operation fails
        """
        if not self._connected:
            logger.warning("Redis not connected, returning empty list")
            return [None] * len(keys)

        try:
            values = await self.client.mget(keys)
            logger.debug(
                f"Redis MGET: {len(keys)} keys -> {len([v for v in values if v])} hits"
            )
            return values
        except Exception as e:
            logger.error(f"Redis MGET error: {e}")
            raise

    async def mset(self, mapping: dict[str, str]) -> bool:
        """
        Set multiple key-value pairs (batch operation).

        Args:
            mapping: Dictionary of key-value pairs

        Returns:
            True if successful, False otherwise

        Raises:
            redis.RedisError: If operation fails
        """
        if not self._connected:
            logger.warning("Redis not connected, skipping MSET")
            return False

        try:
            result = await self.client.mset(mapping)
            logger.debug(f"Redis MSET: {len(mapping)} keys -> {result}")
            return bool(result)
        except Exception as e:
            logger.error(f"Redis MSET error: {e}")
            raise

    async def keys(self, pattern: str) -> list[str]:
        """
        Get all keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "entity:*")

        Returns:
            List of matching keys
        """
        if not self._connected:
            return []

        try:
            keys = await self.client.keys(pattern)
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        except Exception as e:
            logger.error(f"Redis KEYS error: {e}")
            return []

    @property
    def is_connected(self) -> bool:
        """Check if Redis is currently connected."""
        return self._connected


# Global service instance
redis_service = RedisService()

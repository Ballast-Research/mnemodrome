"""PostgreSQL connection and utilities for Mnemodrome event logging.

Provides async PostgreSQL support using asyncpg with connection pooling.
This module is only imported when EVENT_LOG_BACKEND=postgres; asyncpg is
an optional dependency and must not be imported for the default SQLite path.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import os
import re
from typing import Any

from loguru import logger


def _rewrite_placeholders(query: str) -> str:
    """Rewrite SQLite-style ? placeholders to asyncpg-style $N positional params.

    Replaces each standalone `?` in order with $1, $2, ... The service only
    uses `?` as parameter markers (never as literals), so a sequential replace
    is safe.
    """
    counter = 0

    def replace(match: re.Match) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    # Match standalone `?` — not inside string literals; since the service SQL
    # never contains quoted `?` this simple pattern is sufficient.
    return re.sub(r"\?", replace, query)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class _LoopAwareAcquire:
    """Async context manager that acquires from the running loop's pool."""

    def __init__(self, owner: PostgreSQLConnection) -> None:
        self._owner = owner
        self._cm = None

    async def __aenter__(self):
        pool = await self._owner._pool_for_running_loop()
        self._cm = pool.acquire()
        return await self._cm.__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        return await self._cm.__aexit__(exc_type, exc, tb)


class PostgreSQLConnection:
    """Async PostgreSQL connection manager with connection pooling.

    Exposes the same async interface as SQLiteConnection:
    connect / disconnect / execute / fetch / fetchrow / fetchval / executemany
    plus the shared methods: get_size_mb / vacuum / backend_description.
    """

    def __init__(self) -> None:
        """Initialize PostgreSQL connection manager from pinned env vars."""
        # DSN takes priority; fall back to individual PG_* vars
        self._dsn: str | None = os.getenv("EVENT_LOG_POSTGRES_DSN")

        self.host: str = os.getenv("PG_HOST", "localhost")
        self.port: int = int(os.getenv("PG_PORT", "5432"))
        self.database: str = os.getenv("PG_DATABASE", "event_logs")
        self.user: str = os.getenv("PG_USER", "event_log_user")
        self._password: str | None = os.getenv("PG_PASSWORD")

        self._min_size: int = int(os.getenv("PG_POOL_MIN", "5"))
        self._max_size: int = int(os.getenv("PG_POOL_MAX", "20"))

        self.pool = None  # asyncpg.Pool, typed as Any to avoid static import
        self._connected: bool = False
        # asyncpg pools are bound to the event loop that created them. The MCP
        # server and the Admin UI run on different loops in one process, so a
        # second loop gets its own small pool (Feature 039).
        self._pool_loop: asyncio.AbstractEventLoop | None = None
        self._loop_pools: dict[asyncio.AbstractEventLoop, Any] = {}
        self._loop_pool_lock: asyncio.Lock | None = None

    async def connect(self) -> None:
        """Establish asyncpg connection pool to PostgreSQL.

        Uses EVENT_LOG_POSTGRES_DSN when set, otherwise builds the DSN from
        PG_HOST / PG_PORT / PG_DATABASE / PG_USER / PG_PASSWORD.

        Sets search_path=event_logs,public so bare table names (sessions /
        events) resolve to the event_logs schema without any SQL changes.
        """
        if self._connected:
            return

        try:
            self.pool = await self._create_pool(self._min_size, self._max_size)
            self._pool_loop = _running_loop()
            self._connected = True
            logger.info(
                f"PostgreSQL connection pool established: {self.backend_description}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def _create_pool(self, min_size: int, max_size: int):
        # Import here so asyncpg is only loaded for the postgres backend
        import asyncpg  # noqa: PLC0415

        pool_kwargs: dict[str, Any] = dict(
            min_size=min_size,
            max_size=max_size,
            command_timeout=10,
            server_settings={"search_path": "event_logs,public"},
        )
        if self._dsn:
            return await asyncpg.create_pool(dsn=self._dsn, **pool_kwargs)
        return await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self._password,
            **pool_kwargs,
        )

    async def _pool_for_running_loop(self):
        """Pool for the running loop; creates a small one for a second loop."""
        loop = _running_loop()
        if self._pool_loop is None or loop is None or loop is self._pool_loop:
            return self.pool
        pool = self._loop_pools.get(loop)
        if pool is not None:
            return pool
        if self._loop_pool_lock is None:
            self._loop_pool_lock = asyncio.Lock()
        async with self._loop_pool_lock:
            pool = self._loop_pools.get(loop)
            if pool is None:
                pool = await self._create_pool(1, max(2, self._max_size // 4))
                self._loop_pools[loop] = pool
                logger.info("PostgreSQL pool created for a second event loop")
        return pool

    async def disconnect(self) -> None:
        """Close every connection pool."""
        for pool in list(self._loop_pools.values()):
            try:
                await pool.close()
            except Exception as e:  # pragma: no cover - best effort
                logger.warning(f"Error closing secondary PostgreSQL pool: {e}")
        self._loop_pools.clear()
        if self.pool:
            await self.pool.close()
            self._connected = False
            self._pool_loop = None
            logger.info("PostgreSQL connection pool closed")

    def _acquire(self):
        """Return a connection context manager from the right pool."""
        if not self._connected:
            raise RuntimeError("Connection pool not initialized. Call connect() first.")
        if self._pool_loop is None or _running_loop() in (None, self._pool_loop):
            return self.pool.acquire()
        return _LoopAwareAcquire(self)

    @staticmethod
    def _coerce_arg(val: Any) -> Any:
        """Coerce a Python value to a type that asyncpg can bind.

        asyncpg is strict about parameter types for PostgreSQL column types:

        - **TIMESTAMPTZ**: requires a ``datetime`` object, not an ISO string.
          The service layer always passes ``.isoformat()`` strings, so we parse
          them here.
        - **JSONB**: requires a JSON *string*, not a Python dict/list.
          If the caller already passes a JSON string the value is left as-is.
          If a dict/list arrives (e.g. from a test helper) it is serialised.
        - All other types are passed through unchanged.
        """
        if isinstance(val, str):
            # Try ISO datetime parse first (the service always uses .isoformat())
            try:
                return _dt.datetime.fromisoformat(val)
            except ValueError:
                pass
            # JSON strings stay as-is; asyncpg accepts them for JSONB columns.
            return val
        if isinstance(val, (dict, list)):
            # Serialise to JSON string for JSONB columns
            return json.dumps(val)
        return val

    @staticmethod
    def _coerce_row(row: Any) -> dict[str, Any]:
        """Convert an asyncpg Record to a plain dict.

        - TIMESTAMPTZ values are coerced to ISO strings so that the service
          layer can treat them uniformly with the SQLite backend.
        - JSONB values (returned as dicts/lists by asyncpg) are serialised back
          to JSON strings so that the service's ``json.loads()`` calls succeed.
        """
        result: dict[str, Any] = {}
        for key in row.keys():
            val = row[key]
            if isinstance(val, (_dt.datetime, _dt.date)):
                val = val.isoformat()
            elif isinstance(val, (dict, list)):
                # JSONB column — serialise back to string so service-layer
                # json.loads() works identically to the SQLite path.
                val = json.dumps(val)
            result[key] = val
        return result

    async def execute(self, query: str, *args: Any) -> None:
        """Execute a query without returning results."""
        pg_query = _rewrite_placeholders(query)
        pg_args = tuple(self._coerce_arg(a) for a in args)
        async with self._acquire() as conn:
            await conn.execute(pg_query, *pg_args)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Execute a query and return all rows as list of dicts."""
        pg_query = _rewrite_placeholders(query)
        pg_args = tuple(self._coerce_arg(a) for a in args)
        async with self._acquire() as conn:
            rows = await conn.fetch(pg_query, *pg_args)
            return [self._coerce_row(row) for row in rows]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """Execute a query and return a single row as dict."""
        pg_query = _rewrite_placeholders(query)
        pg_args = tuple(self._coerce_arg(a) for a in args)
        async with self._acquire() as conn:
            row = await conn.fetchrow(pg_query, *pg_args)
            return self._coerce_row(row) if row else None

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Execute a query and return a single scalar value."""
        pg_query = _rewrite_placeholders(query)
        pg_args = tuple(self._coerce_arg(a) for a in args)
        async with self._acquire() as conn:
            return await conn.fetchval(pg_query, *pg_args)

    async def executemany(self, query: str, args: list) -> None:
        """Execute a query multiple times with different argument tuples."""
        pg_query = _rewrite_placeholders(query)
        pg_args_list = [tuple(self._coerce_arg(a) for a in row) for row in args]
        async with self._acquire() as conn:
            await conn.executemany(pg_query, pg_args_list)

    # ------------------------------------------------------------------
    # Shared interface methods (also present on SQLiteConnection)
    # ------------------------------------------------------------------

    async def get_size_mb(self) -> float:
        """Return the total size of the event_logs schema tables in MB."""
        try:
            result = await self.fetchval(
                "SELECT (pg_total_relation_size('event_logs.events') "
                "+ pg_total_relation_size('event_logs.sessions')) / 1048576.0"
            )
            return float(result) if result is not None else 0.0
        except Exception as e:
            logger.warning(f"Failed to get PostgreSQL size: {e}")
            return 0.0

    async def vacuum(self) -> None:
        """No-op: PostgreSQL uses autovacuum."""

    @property
    def backend_description(self) -> str:
        """Human-readable description of this backend."""
        return f"PostgreSQL: {self.host}:{self.port}/{self.database}"

"""SQLite connection and utilities for Mnemodrome event logging.

Provides async SQLite support using aiosqlite with automatic schema
initialization. Lightweight, zero-configuration default for the
event log layer — no external database service required.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

from src import config


class SQLiteConnection:
    """Async SQLite connection manager.

    Unlike PostgreSQL, SQLite requires no external service — the database
    file is created automatically on first connect. Schema initialization
    is handled via CREATE TABLE IF NOT EXISTS on every connect, making
    the event log fully self-bootstrapping.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize SQLite connection manager.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to EVENT_LOG_DB_PATH config value.
        """
        self.db_path = db_path or config.EVENT_LOG_DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._connected = False

    async def connect(self) -> None:
        """Open SQLite database and initialize schema.

        Creates parent directories and database file if they don't exist.
        Runs schema initialization on every connect (idempotent).
        """
        if self._connected:
            return

        try:
            # Ensure parent directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row

            # Enable WAL mode for better concurrent read performance
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")

            # Initialize schema
            await self._init_schema()

            self._connected = True
            logger.info(f"✅ SQLite event log connected: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise

    async def _init_schema(self) -> None:
        """Create event log tables if they don't exist."""
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                session_id TEXT NOT NULL,
                actor TEXT,
                action TEXT NOT NULL,
                target TEXT,
                context TEXT,
                result TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_session_id
                ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_event_type
                ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_action
                ON events(action);
        """)
        await self._db.commit()

    async def disconnect(self) -> None:
        """Close SQLite connection."""
        if self._db:
            await self._db.close()
            self._db = None
            self._connected = False
            logger.info("✅ SQLite event log connection closed")

    async def execute(self, query: str, *args) -> None:
        """Execute a query without returning results."""
        if not self._connected:
            raise RuntimeError("SQLite not connected. Call connect() first.")
        await self._db.execute(query, args)
        await self._db.commit()

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute a query and return all rows as list of dicts."""
        if not self._connected:
            raise RuntimeError("SQLite not connected. Call connect() first.")
        cursor = await self._db.execute(query, args)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        """Execute a query and return a single row as dict."""
        if not self._connected:
            raise RuntimeError("SQLite not connected. Call connect() first.")
        cursor = await self._db.execute(query, args)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        """Execute a query and return a single scalar value."""
        if not self._connected:
            raise RuntimeError("SQLite not connected. Call connect() first.")
        cursor = await self._db.execute(query, args)
        row = await cursor.fetchone()
        return row[0] if row else None

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected

    # ------------------------------------------------------------------
    # Shared interface methods (also present on PostgreSQLConnection)
    # ------------------------------------------------------------------

    async def get_size_mb(self) -> float:
        """Return the size of the SQLite database file (plus WAL) in MB."""
        try:
            size_bytes = 0
            if os.path.exists(self.db_path):
                size_bytes += os.path.getsize(self.db_path)
                wal_path = self.db_path + "-wal"
                if os.path.exists(wal_path):
                    size_bytes += os.path.getsize(wal_path)
            return size_bytes / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Failed to get SQLite size: {e}")
            return 0.0

    async def vacuum(self) -> None:
        """Run VACUUM to reclaim disk space (SQLite doesn't shrink on DELETE)."""
        await self.execute("VACUUM")

    @property
    def backend_description(self) -> str:
        """Human-readable description of this backend."""
        return f"SQLite: {self.db_path}"


# Global SQLite connection instance
sqlite_connection = SQLiteConnection()

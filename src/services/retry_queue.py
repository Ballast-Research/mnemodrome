"""Persistent retry queue for failed storage operations."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import RETRY_QUEUE_PATH


class RetryQueue:
    """JSON-based persistent queue for failed write operations."""

    def __init__(self, queue_path: Path | None = None):
        """
        Initialize retry queue.

        Args:
            queue_path: Path to queue file (defaults to config)
        """
        self.queue_path = queue_path or RETRY_QUEUE_PATH
        self._lock = asyncio.Lock()

    async def _read_queue(self) -> list[dict[str, Any]]:
        """Read queue from disk."""
        if not self.queue_path.exists():
            return []

        try:
            with open(self.queue_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in retry queue: {self.queue_path}")
            return []

    async def _write_queue(self, queue: list[dict[str, Any]]) -> None:
        """Write queue to disk."""
        with open(self.queue_path, "w") as f:
            json.dump(queue, f, indent=2, default=str)

    async def add(self, item: dict[str, Any]) -> None:
        """
        Add item to retry queue.

        Args:
            item: Queue item (must include 'layer', 'timestamp', 'error')
        """
        async with self._lock:
            queue = await self._read_queue()
            queue.append(item)
            await self._write_queue(queue)
            logger.info(
                f"Added to retry queue: {item['layer']} - {len(queue)} items total"
            )

    async def load(self) -> list[dict[str, Any]]:
        """
        Load all items from retry queue.

        Returns:
            List of queued items
        """
        async with self._lock:
            return await self._read_queue()

    async def remove(self, item: dict[str, Any]) -> None:
        """
        Remove item from retry queue.

        Args:
            item: Item to remove (matched by timestamp and layer)
        """
        async with self._lock:
            queue = await self._read_queue()

            # Find and remove matching item
            queue = [
                q
                for q in queue
                if not (
                    q.get("timestamp") == item.get("timestamp")
                    and q.get("layer") == item.get("layer")
                )
            ]

            await self._write_queue(queue)
            logger.debug(f"Removed from retry queue: {item['layer']}")

    async def clear(self) -> None:
        """Clear all items from retry queue."""
        async with self._lock:
            await self._write_queue([])
            logger.info("Cleared retry queue")

    async def size(self) -> int:
        """Get number of items in queue."""
        queue = await self.load()
        return len(queue)


# Global retry queue instance
retry_queue = RetryQueue()

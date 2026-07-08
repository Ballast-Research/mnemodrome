"""Unit tests for retry queue service (src/services/retry_queue.py).

Tests the RetryQueue class which manages persistent JSON-based retry queue
for failed write operations.
"""

import json

import pytest

from src.services.retry_queue import RetryQueue


@pytest.fixture
def tmp_queue_path(tmp_path):
    """Provide a temporary queue file path."""
    return tmp_path / "retry_queue.json"


@pytest.fixture
def retry_queue(tmp_queue_path):
    """Provide a RetryQueue instance using temporary file."""
    return RetryQueue(queue_path=tmp_queue_path)


@pytest.mark.asyncio
class TestRetryQueueAdd:
    """Tests for add method."""

    async def test_add_item_creates_queue(self, retry_queue, tmp_queue_path):
        """Test adding item creates queue file."""
        item = {
            "layer": "graph",
            "timestamp": "2025-03-12T10:00:00Z",
            "error": "Connection timeout",
            "data": {"entity": "test"},
        }

        await retry_queue.add(item)

        assert tmp_queue_path.exists()
        with open(tmp_queue_path) as f:
            queue = json.load(f)
        assert len(queue) == 1
        assert queue[0]["layer"] == "graph"

    async def test_add_multiple_items(self, retry_queue, tmp_queue_path):
        """Test adding multiple items."""
        items = [
            {"layer": "graph", "timestamp": "2025-03-12T10:00:00Z", "error": "Error 1"},
            {
                "layer": "vector",
                "timestamp": "2025-03-12T10:01:00Z",
                "error": "Error 2",
            },
        ]

        for item in items:
            await retry_queue.add(item)

        with open(tmp_queue_path) as f:
            queue = json.load(f)
        assert len(queue) == 2

    async def test_add_item_preserves_existing(self, retry_queue, tmp_queue_path):
        """Test that adding item preserves existing items."""
        # Add first item
        await retry_queue.add({"layer": "graph", "timestamp": "t1", "error": "e1"})

        # Add second item
        await retry_queue.add({"layer": "vector", "timestamp": "t2", "error": "e2"})

        with open(tmp_queue_path) as f:
            queue = json.load(f)

        assert len(queue) == 2
        assert queue[0]["layer"] == "graph"
        assert queue[1]["layer"] == "vector"


@pytest.mark.asyncio
class TestRetryQueueLoad:
    """Tests for load method."""

    async def test_load_empty_queue(self, retry_queue, tmp_queue_path):
        """Test loading when queue file doesn't exist."""
        result = await retry_queue.load()
        assert result == []

    async def test_load_existing_queue(self, retry_queue, tmp_queue_path):
        """Test loading existing queue."""
        items = [
            {"layer": "graph", "timestamp": "t1", "error": "e1"},
            {"layer": "vector", "timestamp": "t2", "error": "e2"},
        ]

        # Write queue to file
        with open(tmp_queue_path, "w") as f:
            json.dump(items, f)

        result = await retry_queue.load()

        assert len(result) == 2
        assert result[0]["layer"] == "graph"

    async def test_load_corrupted_queue(self, retry_queue, tmp_queue_path):
        """Test loading corrupted queue returns empty list."""
        # Write invalid JSON
        with open(tmp_queue_path, "w") as f:
            f.write("{invalid json")

        result = await retry_queue.load()

        assert result == []


@pytest.mark.asyncio
class TestRetryQueueRemove:
    """Tests for remove method."""

    async def test_remove_item_by_timestamp_and_layer(
        self, retry_queue, tmp_queue_path
    ):
        """Test removing item by timestamp and layer."""
        # Add items
        items = [
            {"layer": "graph", "timestamp": "2025-03-12T10:00:00Z", "error": "e1"},
            {"layer": "vector", "timestamp": "2025-03-12T10:01:00Z", "error": "e2"},
        ]

        for item in items:
            await retry_queue.add(item)

        # Remove first item
        await retry_queue.remove(
            {"layer": "graph", "timestamp": "2025-03-12T10:00:00Z"}
        )

        result = await retry_queue.load()
        assert len(result) == 1
        assert result[0]["layer"] == "vector"

    async def test_remove_nonexistent_item(self, retry_queue, tmp_queue_path):
        """Test removing non-existent item doesn't fail."""
        await retry_queue.add({"layer": "graph", "timestamp": "t1", "error": "e1"})

        # Try to remove non-existent item
        await retry_queue.remove({"layer": "nonexistent", "timestamp": "nonexistent"})

        # Original item should still be there
        result = await retry_queue.load()
        assert len(result) == 1

    async def test_remove_all_items(self, retry_queue, tmp_queue_path):
        """Test removing all items one by one."""
        items = [
            {"layer": "graph", "timestamp": "t1", "error": "e1"},
            {"layer": "vector", "timestamp": "t2", "error": "e2"},
        ]

        for item in items:
            await retry_queue.add(item)

        # Remove all items
        for item in items:
            await retry_queue.remove(item)

        result = await retry_queue.load()
        assert len(result) == 0


@pytest.mark.asyncio
class TestRetryQueueClear:
    """Tests for clear method."""

    async def test_clear_queue(self, retry_queue, tmp_queue_path):
        """Test clearing entire queue."""
        # Add items
        await retry_queue.add({"layer": "graph", "timestamp": "t1", "error": "e1"})
        await retry_queue.add({"layer": "vector", "timestamp": "t2", "error": "e2"})

        # Clear queue
        await retry_queue.clear()

        result = await retry_queue.load()
        assert len(result) == 0

    async def test_clear_empty_queue(self, retry_queue, tmp_queue_path):
        """Test clearing empty queue doesn't fail."""
        # Should not raise
        await retry_queue.clear()

        result = await retry_queue.load()
        assert len(result) == 0


@pytest.mark.asyncio
class TestRetryQueueSize:
    """Tests for size method."""

    async def test_size_empty_queue(self, retry_queue):
        """Test size of empty queue."""
        result = await retry_queue.size()
        assert result == 0

    async def test_size_with_items(self, retry_queue):
        """Test size with items."""
        await retry_queue.add({"layer": "graph", "timestamp": "t1", "error": "e1"})
        await retry_queue.add({"layer": "vector", "timestamp": "t2", "error": "e2"})

        result = await retry_queue.size()
        assert result == 2

    async def test_size_after_remove(self, retry_queue):
        """Test size after removing items."""
        await retry_queue.add({"layer": "graph", "timestamp": "t1", "error": "e1"})
        await retry_queue.add({"layer": "vector", "timestamp": "t2", "error": "e2"})

        await retry_queue.remove({"layer": "graph", "timestamp": "t1"})

        result = await retry_queue.size()
        assert result == 1


@pytest.mark.asyncio
class TestRetryQueueConcurrency:
    """Tests for concurrent access with lock."""

    async def test_concurrent_add_operations(self, retry_queue):
        """Test concurrent add operations are serialized."""
        import asyncio

        async def add_item(layer, timestamp):
            await retry_queue.add(
                {"layer": layer, "timestamp": timestamp, "error": "test"}
            )

        # Run concurrent adds
        await asyncio.gather(
            add_item("graph", "t1"),
            add_item("vector", "t2"),
            add_item("file", "t3"),
        )

        result = await retry_queue.load()
        assert len(result) == 3

    async def test_concurrent_read_write(self, retry_queue):
        """Test concurrent read and write operations."""
        import asyncio

        async def write_items():
            for i in range(5):
                await retry_queue.add(
                    {"layer": "graph", "timestamp": f"t{i}", "error": f"e{i}"}
                )

        async def read_items():
            results = []
            for _ in range(5):
                result = await retry_queue.load()
                results.append(len(result))
            return results

        # Run concurrent read/write
        await asyncio.gather(write_items(), read_items())

        final_result = await retry_queue.load()
        assert len(final_result) == 5


@pytest.mark.asyncio
class TestRetryQueueIntegration:
    """Integration tests for retry queue operations."""

    async def test_typical_workflow(self, retry_queue):
        """Test typical retry queue workflow."""
        # Add failed operations
        await retry_queue.add(
            {
                "layer": "graph",
                "timestamp": "2025-03-12T10:00:00Z",
                "error": "Connection timeout",
                "operation": "create_entity",
            }
        )

        await retry_queue.add(
            {
                "layer": "vector",
                "timestamp": "2025-03-12T10:01:00Z",
                "error": "Service unavailable",
                "operation": "upsert_embedding",
            }
        )

        # Check queue size
        size = await retry_queue.size()
        assert size == 2

        # Load and process
        items = await retry_queue.load()
        assert len(items) == 2

        # Remove successfully processed item
        await retry_queue.remove(items[0])

        # Verify
        remaining = await retry_queue.load()
        assert len(remaining) == 1
        assert remaining[0]["layer"] == "vector"

    async def test_queue_persistence(self, tmp_path):
        """Test that queue persists across instances."""
        queue_path = tmp_path / "persistent_queue.json"

        # First instance
        queue1 = RetryQueue(queue_path=queue_path)
        await queue1.add({"layer": "graph", "timestamp": "t1", "error": "e1"})

        # Second instance with same path
        queue2 = RetryQueue(queue_path=queue_path)
        items = await queue2.load()

        assert len(items) == 1
        assert items[0]["layer"] == "graph"

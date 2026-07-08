"""Sync service for best-effort dual-write to Neo4j and markdown storage.

Implements resilient dual-write pattern:
- Never discard data - queue failed writes for retry
- At least one layer must succeed (fail if BOTH fail)
- Persistent retry queue survives restarts
- Automatic retry processor runs until queue empty
"""

import asyncio
from typing import Any

from loguru import logger

from src.services.file_service import file_service
from src.services.neo4j_service import neo4j_service
from src.services.retry_queue import retry_queue


class SyncService:
    """Service for synchronized dual-write with retry queue."""

    def __init__(self):
        """Initialize sync service."""
        self.retry_task: asyncio.Task | None = None

    async def dual_write_with_retry(
        self, neo4j_data: dict[str, Any], markdown_content: str, filename: str
    ) -> dict[str, Any]:
        """
        Best-effort dual-write to Neo4j and markdown storage.

        Strategy:
        - Attempt both writes
        - Queue failures for retry
        - Return success if AT LEAST ONE succeeds
        - Raise exception if BOTH fail

        Args:
            neo4j_data: Data for Neo4j write (must include operation type)
            markdown_content: Content for markdown file
            filename: Filename for markdown (will be placed in storage path)

        Returns:
            Status dict with layer statuses and any queued items

        Raises:
            Exception if BOTH layers fail
        """
        neo4j_success = False
        markdown_success = False
        queued_items = []

        # Attempt Neo4j write
        try:
            operation = neo4j_data.get("operation")
            if operation == "create_entity":
                await neo4j_service.create_entity(neo4j_data["data"])
                neo4j_success = True
                logger.debug("Neo4j write succeeded")
            elif operation == "create_relationship":
                await neo4j_service.create_relationship(**neo4j_data["data"])
                neo4j_success = True
                logger.debug("Neo4j relationship created")
            else:
                raise ValueError(f"Unknown Neo4j operation: {operation}")

        except Exception as e:
            logger.warning(f"Neo4j write failed, queuing for retry: {e}")
            queued_item = {"layer": "neo4j", "data": neo4j_data, "error": str(e)}
            await retry_queue.add(queued_item)
            queued_items.append(queued_item)

        # Attempt markdown write
        try:
            file_path = file_service.write_file(filename, markdown_content)
            markdown_success = True
            logger.debug(f"Markdown write succeeded: {file_path}")

        except Exception as e:
            logger.warning(f"Markdown write failed, queuing for retry: {e}")
            queued_item = {
                "layer": "markdown",
                "filename": filename,
                "content": markdown_content,
                "error": str(e),
            }
            await retry_queue.add(queued_item)
            queued_items.append(queued_item)

        # Evaluate overall status
        if not neo4j_success and not markdown_success:
            raise Exception(
                "Dual-write failed: BOTH layers failed. "
                f"Queued {len(queued_items)} items for retry."
            )

        # Start retry processor if queue not empty
        if queued_items:
            await self._start_retry_processor()

        # Determine status
        if neo4j_success and markdown_success:
            status = "complete"
        else:
            status = "partial"

        return {
            "status": status,
            "neo4j": neo4j_success,
            "markdown": markdown_success,
            "queued": len(queued_items),
        }

    async def _start_retry_processor(self) -> None:
        """
        Start retry processor if not already running.

        Processor runs in background until queue is empty.
        """
        if self.retry_task and not self.retry_task.done():
            logger.debug("Retry processor already running")
            return

        logger.info("Starting retry processor...")
        self.retry_task = asyncio.create_task(self._process_retry_queue())

    async def _process_retry_queue(self) -> None:
        """
        Process retry queue until empty.

        Runs in background, attempting queued writes every 60 seconds.
        Exits when queue is empty.
        """
        retry_count = 0

        while True:
            queue_items = await retry_queue.load()

            if not queue_items:
                logger.info(
                    f"Retry queue empty after {retry_count} cycles, exiting processor"
                )
                return

            logger.info(
                f"Retry cycle {retry_count + 1}: Processing {len(queue_items)} queued items"
            )

            for item in queue_items:
                layer = item.get("layer")

                try:
                    if layer == "neo4j":
                        # Retry Neo4j write
                        neo4j_data = item["data"]
                        operation = neo4j_data.get("operation")

                        if operation == "create_entity":
                            await neo4j_service.create_entity(neo4j_data["data"])
                            logger.info("Retry succeeded: Neo4j entity created")
                            await retry_queue.remove(item)
                        elif operation == "create_relationship":
                            await neo4j_service.create_relationship(
                                **neo4j_data["data"]
                            )
                            logger.info("Retry succeeded: Neo4j relationship created")
                            await retry_queue.remove(item)
                        else:
                            logger.error(
                                f"Unknown Neo4j operation in retry: {operation}"
                            )

                    elif layer == "markdown":
                        # Retry markdown write
                        filename = item["filename"]
                        content = item["content"]
                        file_path = file_service.write_file(filename, content)
                        logger.info(f"Retry succeeded: Markdown written to {file_path}")
                        await retry_queue.remove(item)

                    else:
                        logger.error(f"Unknown layer in retry queue: {layer}")

                except Exception as e:
                    logger.warning(f"Retry failed for {layer}: {e} (will retry in 60s)")

            retry_count += 1
            await asyncio.sleep(60)  # Wait 60 seconds before next retry cycle


# Global service instance
sync_service = SyncService()

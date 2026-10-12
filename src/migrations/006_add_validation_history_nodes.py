"""
Migration 006: Create ValidationHistory node type and indexes.

This migration creates the ValidationHistory node type for tracking human
validation decisions. Used for pattern learning in progressive automation.

Node structure:
- ValidationHistory {
    uuid: string (unique identifier)
    conflict_type: string
    complexity_score: integer 0-100
    human_decision: string (approve/reject/modify)
    timestamp: datetime
    old_fact_uuid: string
    new_fact_uuid: string | null
    rationale: string | null
  }

Run: python -m src.migrations.006_add_validation_history_nodes
"""

import asyncio

from loguru import logger

from src.services.neo4j_service import neo4j_service


async def migrate_add_validation_history():
    """
    Create ValidationHistory node type with indexes for efficient querying.

    Indexes created:
    - Unique constraint on ValidationHistory.uuid
    - Index on ValidationHistory.conflict_type (for pattern matching)
    - Index on ValidationHistory.timestamp (for temporal queries)
    """
    logger.info("Starting migration 006: Create ValidationHistory node type")

    await neo4j_service.connect()

    try:
        # Step 1: Create unique constraint on uuid
        constraint_query = """
        CREATE CONSTRAINT validation_history_uuid IF NOT EXISTS
        FOR (v:ValidationHistory)
        REQUIRE v.uuid IS UNIQUE
        """

        await neo4j_service.query(constraint_query)
        logger.info("Created unique constraint on ValidationHistory.uuid")

        # Step 2: Create index on conflict_type for pattern matching
        conflict_type_index = """
        CREATE INDEX validation_history_conflict_type IF NOT EXISTS
        FOR (v:ValidationHistory)
        ON (v.conflict_type)
        """

        await neo4j_service.query(conflict_type_index)
        logger.info("Created index on ValidationHistory.conflict_type")

        # Step 3: Create index on timestamp for temporal queries
        timestamp_index = """
        CREATE INDEX validation_history_timestamp IF NOT EXISTS
        FOR (v:ValidationHistory)
        ON (v.timestamp)
        """

        await neo4j_service.query(timestamp_index)
        logger.info("Created index on ValidationHistory.timestamp")

        # Step 4: Verify node type is ready
        verify_query = """
        MATCH (v:ValidationHistory)
        RETURN count(v) as count
        """

        result = await neo4j_service.query(verify_query)
        count = result[0]["count"] if result else 0
        logger.info(f"ValidationHistory nodes in database: {count}")

        logger.success("ValidationHistory node type and indexes created successfully")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    finally:
        await neo4j_service.close()

    logger.info("Migration 006 ValidationHistory complete")


if __name__ == "__main__":
    asyncio.run(migrate_add_validation_history())

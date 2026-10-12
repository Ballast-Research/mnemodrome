"""
Migration 034: Create indexes for ACT-R cognitive activation properties.

This migration creates indexes on activation-related properties added by
Amendment 9 to support efficient activation-based retrieval ranking.

Indexes Created:
- idx_entity_access_count: Entity(access_count) - frequency-based filtering
- idx_entity_last_accessed: Entity(last_accessed) - recency-based filtering
- idx_relates_to_hebbian: RELATES_TO(hebbian_weight) - co-occurrence filtering
- idx_relates_to_confidence: RELATES_TO(confidence) - confidence-based filtering

Run: python -m src.migrations.034_activation_indexes
"""

import asyncio

from loguru import logger

from src.services.neo4j_service import neo4j_service


async def migrate_activation_indexes():
    """
    Create indexes for ACT-R activation properties on existing schema.

    No constraints added — these are optional enrichment properties
    initialized lazily on first access.
    """
    logger.info("Starting migration 034: Create activation property indexes")

    await neo4j_service.connect()

    try:
        indexes = [
            (
                "idx_entity_access_count",
                "CREATE INDEX idx_entity_access_count IF NOT EXISTS FOR (e:Entity) ON (e.access_count)",
                "Entity.access_count",
            ),
            (
                "idx_entity_last_accessed",
                "CREATE INDEX idx_entity_last_accessed IF NOT EXISTS FOR (e:Entity) ON (e.last_accessed)",
                "Entity.last_accessed",
            ),
            (
                "idx_relates_to_hebbian",
                "CREATE INDEX idx_relates_to_hebbian IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.hebbian_weight)",
                "RELATES_TO.hebbian_weight",
            ),
            (
                "idx_relates_to_confidence",
                "CREATE INDEX idx_relates_to_confidence IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.confidence)",
                "RELATES_TO.confidence",
            ),
        ]

        for idx_name, idx_query, field_desc in indexes:
            try:
                await neo4j_service.query(idx_query)
                logger.success(f"  Index created on {field_desc}")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"  Index on {field_desc} already exists (idempotent)")
                else:
                    raise

        logger.success("Migration 034 complete - activation indexes ready")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    finally:
        await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(migrate_activation_indexes())

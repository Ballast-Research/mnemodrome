"""
Migration 006: Add temporal validity fields to existing relationships.

This migration ensures all relationships in the graph have the required
temporal validity fields. New relationships already have these fields via
create_relationship(), but this backfills older relationships.

Fields added:
- invalid_at: datetime | null (when fact stopped being true)
- expired_at: datetime | null (when system invalidated fact)
- invalidation_reason: string | null
- superseded_by: string (UUID) | null
- complexity_score: integer 0-100 | null
- validation_decision: string | null

Run: python -m src.migrations.006_add_temporal_fields
"""

import asyncio

from loguru import logger

from src.services.neo4j_service import neo4j_service


async def migrate_add_temporal_fields():
    """
    Add temporal validity fields to all existing relationships.

    Sets NULL values for backward compatibility:
    - NULL invalid_at and expired_at = currently valid
    - Preserves existing valid_at timestamps
    """
    logger.info("Starting migration 006: Add temporal validity fields")

    await neo4j_service.connect()

    # Step 1: Count relationships that need migration
    count_query = """
    MATCH ()-[r]->()
    WHERE NOT EXISTS(r.invalid_at)
       OR NOT EXISTS(r.expired_at)
    RETURN count(r) as count
    """

    result = await neo4j_service.query(count_query)
    count = result[0]["count"] if result else 0
    logger.info(f"Found {count} relationships needing temporal field migration")

    if count == 0:
        logger.info("No relationships need migration - all up to date")
        return

    # Step 2: Add temporal fields to relationships that don't have them
    migration_query = """
    MATCH ()-[r]->()
    WHERE NOT EXISTS(r.invalid_at)
       OR NOT EXISTS(r.expired_at)
    SET r.invalid_at = CASE WHEN NOT EXISTS(r.invalid_at) THEN NULL ELSE r.invalid_at END,
        r.expired_at = CASE WHEN NOT EXISTS(r.expired_at) THEN NULL ELSE r.expired_at END,
        r.invalidation_reason = CASE WHEN NOT EXISTS(r.invalidation_reason) THEN NULL ELSE r.invalidation_reason END,
        r.superseded_by = CASE WHEN NOT EXISTS(r.superseded_by) THEN NULL ELSE r.superseded_by END,
        r.complexity_score = CASE WHEN NOT EXISTS(r.complexity_score) THEN NULL ELSE r.complexity_score END,
        r.validation_decision = CASE WHEN NOT EXISTS(r.validation_decision) THEN NULL ELSE r.validation_decision END
    RETURN count(r) as migrated
    """

    try:
        result = await neo4j_service.query(migration_query)
        migrated = result[0]["migrated"] if result else 0
        logger.success(
            f"Successfully migrated {migrated} relationships with temporal fields"
        )

        # Step 3: Verify migration
        verify_query = """
        MATCH ()-[r]->()
        WHERE NOT EXISTS(r.invalid_at) OR NOT EXISTS(r.expired_at)
        RETURN count(r) as remaining
        """

        result = await neo4j_service.query(verify_query)
        remaining = result[0]["remaining"] if result else 0

        if remaining == 0:
            logger.success(
                "Migration verified - all relationships now have temporal fields"
            )
        else:
            logger.warning(
                f"Migration incomplete - {remaining} relationships still missing fields"
            )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    finally:
        await neo4j_service.close()

    logger.info("Migration 006 complete")


if __name__ == "__main__":
    asyncio.run(migrate_add_temporal_fields())

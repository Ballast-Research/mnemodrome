"""
Migration 007: Backfill uuid and created_at on RELATES_TO relationships.

Feature 006 (conflict detection / invalidation / update) looks up relationships
by r.uuid, but the capture-time creation paths historically created RELATES_TO
relationships with neither uuid nor created_at -- leaving them impossible to
invalidate (find_relationship returned uuid=None -> "Relationship not found:
None") and breaking conflict detection (null created_at).

The creation paths now set both fields at creation time (RelationshipFactory.create
and Neo4jService.create_relationship). This migration backfills the relationships
that predate that fix.

Single-pass and idempotent -- coalesce preserves any value that already exists,
so re-running is a no-op:
  - uuid:       randomUUID() where missing
  - created_at: existing created_at, else valid_at, else now()

For a ~30k-relationship graph this completes in seconds. If the graph ever grows
large enough that a single transaction is a concern, wrap the SET in
`CALL { WITH r ... } IN TRANSACTIONS OF 10000 ROWS` (requires an auto-commit
session rather than neo4j_service.query()).

Run: python -m src.migrations.007_add_relationship_uuid_and_created_at
"""

import asyncio

from loguru import logger

from src.services.neo4j_service import neo4j_service


async def migrate_add_uuid_and_created_at():
    """Backfill uuid and created_at on every RELATES_TO relationship missing them."""
    logger.info(
        "Starting migration 007: backfill uuid/created_at on RELATES_TO relationships"
    )

    await neo4j_service.connect()

    try:
        # Step 1: Count relationships needing the backfill
        count_query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.uuid IS NULL OR r.created_at IS NULL
        RETURN count(r) AS count
        """
        result = await neo4j_service.query(count_query)
        count = result[0]["count"] if result else 0
        logger.info(f"Found {count} relationships needing uuid/created_at backfill")

        if count == 0:
            logger.info("No relationships need migration - all up to date")
            return

        # Step 2: Single-pass idempotent backfill
        backfill_query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.uuid IS NULL OR r.created_at IS NULL
        SET r.uuid = coalesce(r.uuid, randomUUID()),
            r.created_at = coalesce(r.created_at, r.valid_at, datetime())
        RETURN count(r) AS migrated
        """
        result = await neo4j_service.query(backfill_query)
        migrated = result[0]["migrated"] if result else 0
        logger.success(f"Backfilled {migrated} relationships with uuid and created_at")

        # Step 3: Verify nothing remains
        verify_query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.uuid IS NULL OR r.created_at IS NULL
        RETURN count(r) AS remaining
        """
        result = await neo4j_service.query(verify_query)
        remaining = result[0]["remaining"] if result else 0

        if remaining == 0:
            logger.success(
                "✅ Migration verified - all RELATES_TO relationships now have uuid and created_at"
            )
        else:
            logger.warning(
                f"⚠️  Migration incomplete - {remaining} relationships still missing fields"
            )

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    finally:
        await neo4j_service.close()

    logger.info("Migration 007 complete")


if __name__ == "__main__":
    asyncio.run(migrate_add_uuid_and_created_at())

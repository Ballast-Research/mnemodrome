"""
Migration 032: Create EvolutionCandidate schema and relationships.

This migration sets up the graph schema for Evolving Strategies (Feature 032),
enabling the system to propose instruction changes based on high-confidence patterns.

Nodes Created:
- EvolutionCandidate: Proposed instruction changes with evidence and metadata

Relationships Created:
- BASED_ON: EvolutionCandidate -> Pattern (evidence links)
- TARGETS: EvolutionCandidate -> Entity (skill tracking)
- GENERATED_CANDIDATE: Session -> EvolutionCandidate

Constraints:
- EvolutionCandidate.uuid must be unique

Indexes:
- EvolutionCandidate.status (query by status)
- EvolutionCandidate.skill_path (skill-specific lookups)
- EvolutionCandidate.created_at (temporal queries)
- EvolutionCandidate.confidence_score (ranking)

Run: python -m src.migrations.032_evolution_candidate_schema
"""

import asyncio

from loguru import logger

from src.services.neo4j_service import neo4j_service


async def migrate_evolution_candidate_schema():
    """
    Create EvolutionCandidate node schema and relationships.

    Strategy:
    1. Create unique constraint on EvolutionCandidate.uuid
    2. Create performance indexes on status, skill_path, created_at, confidence_score
    3. Verify schema is in place
    """
    logger.info("Starting migration 032: Create EvolutionCandidate schema")

    await neo4j_service.connect()

    try:
        # Step 1: Create unique constraint on uuid
        logger.info("Creating unique constraint on EvolutionCandidate.uuid...")
        constraint_query = """
        CREATE CONSTRAINT evolution_candidate_uuid_unique
        IF NOT EXISTS
        FOR (ec:EvolutionCandidate)
        REQUIRE ec.uuid IS UNIQUE
        """

        try:
            await neo4j_service.query(constraint_query)
            logger.success("✓ Unique constraint created on EvolutionCandidate.uuid")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("✓ Constraint already exists (idempotent)")
            else:
                raise

        # Step 2: Create indexes for query performance
        logger.info("Creating performance indexes...")

        indexes = [
            (
                "evolution_status",
                "CREATE INDEX evolution_status IF NOT EXISTS FOR (ec:EvolutionCandidate) ON (ec.status)",
                "status",
            ),
            (
                "evolution_skill",
                "CREATE INDEX evolution_skill IF NOT EXISTS FOR (ec:EvolutionCandidate) ON (ec.skill_path)",
                "skill_path",
            ),
            (
                "evolution_created",
                "CREATE INDEX evolution_created IF NOT EXISTS FOR (ec:EvolutionCandidate) ON (ec.created_at)",
                "created_at",
            ),
            (
                "evolution_confidence",
                "CREATE INDEX evolution_confidence IF NOT EXISTS FOR (ec:EvolutionCandidate) ON (ec.confidence_score)",
                "confidence_score",
            ),
        ]

        for idx_name, idx_query, field_name in indexes:
            try:
                await neo4j_service.query(idx_query)
                logger.success(f"✓ Index created on {field_name}")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"✓ Index on {field_name} already exists (idempotent)")
                else:
                    raise

        # Step 3: Verify schema is in place
        logger.info("Verifying schema...")

        verify_query = """
        SHOW CONSTRAINTS
        WHERE name IN ['evolution_candidate_uuid_unique']
        """

        result = await neo4j_service.query(verify_query)

        if result:
            logger.success("✓ EvolutionCandidate constraint verified")
        else:
            logger.warning("⚠️  Could not verify constraint (may not be visible yet)")

        # Step 4: Document schema structure
        logger.info("""
        EvolutionCandidate Schema:
        ├── uuid: String (UNIQUE)
        ├── skill_path: String (e.g., ~/.claude/skills/video-transcription/SKILL.md)
        ├── skill_name: String (human-readable skill name)
        ├── proposed_diff: String (unified diff format)
        ├── rationale: String (why this change is proposed)
        ├── pattern_uuids: [String] (evidence pattern UUIDs)
        ├── pattern_summary: String (human-readable evidence summary)
        ├── status: String ("pending" | "approved" | "rejected" | "skipped" | "applied")
        ├── rejection_reason: String (optional)
        ├── confidence_score: Float (0.0-1.0)
        ├── occurrence_count: Integer (total occurrences)
        ├── success_rate: Float (0.0-1.0)
        ├── created_at: DateTime
        ├── reviewed_at: DateTime (optional)
        ├── applied_at: DateTime (optional)
        └── reviewed_by: String (optional, "user")

        Relationships:
        ├── BASED_ON → Pattern (evidence links with pattern metadata)
        ├── TARGETS → Entity (skill name tracking)
        └── GENERATED_CANDIDATE ← Session (creation tracking)
        """)

        logger.success("✅ Migration 032 complete - EvolutionCandidate schema ready")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

    finally:
        await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(migrate_evolution_candidate_schema())

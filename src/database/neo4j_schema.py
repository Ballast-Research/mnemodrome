"""Neo4j schema initialization for Mnemodrome.

Consolidates all constraints and indexes into a single idempotent function
that runs on every server startup. Every statement uses IF NOT EXISTS,
making it safe to re-run at any time.

This replaces the need to manually run individual migration scripts.
"""

from __future__ import annotations

from loguru import logger

# All constraints and indexes required by Mnemodrome, consolidated from
# migrations 006, 026, 031, 032, and 034.
#
# Each statement is idempotent (IF NOT EXISTS). Order doesn't matter
# because there are no cross-dependencies between these DDL statements.

CONSTRAINTS = [
    # ValidationHistory (Feature 006: Conflict Detection)
    """CREATE CONSTRAINT validation_history_uuid IF NOT EXISTS
       FOR (v:ValidationHistory) REQUIRE v.uuid IS UNIQUE""",
    # Pattern (Feature 026: Cross-Session Pattern Recognition)
    """CREATE CONSTRAINT pattern_uuid IF NOT EXISTS
       FOR (p:Pattern) REQUIRE p.uuid IS UNIQUE""",
    # Procedure (Feature 031: Procedural Memory Layer)
    """CREATE CONSTRAINT procedure_uuid IF NOT EXISTS
       FOR (p:Procedure) REQUIRE p.uuid IS UNIQUE""",
    # EvolutionCandidate (Feature 032: Evolving Strategies)
    """CREATE CONSTRAINT evolution_candidate_uuid_unique IF NOT EXISTS
       FOR (ec:EvolutionCandidate) REQUIRE ec.uuid IS UNIQUE""",
]

INDEXES = [
    # --- ValidationHistory indexes ---
    """CREATE INDEX validation_history_conflict_type IF NOT EXISTS
       FOR (v:ValidationHistory) ON (v.conflict_type)""",
    """CREATE INDEX validation_history_timestamp IF NOT EXISTS
       FOR (v:ValidationHistory) ON (v.timestamp)""",
    # --- Pattern indexes (Feature 026) ---
    """CREATE INDEX pattern_type IF NOT EXISTS
       FOR (p:Pattern) ON (p.pattern_type)""",
    """CREATE INDEX pattern_success_rate IF NOT EXISTS
       FOR (p:Pattern) ON (p.success_rate)""",
    """CREATE INDEX pattern_occurrence_count IF NOT EXISTS
       FOR (p:Pattern) ON (p.occurrence_count)""",
    """CREATE INDEX pattern_created_at IF NOT EXISTS
       FOR (p:Pattern) ON (p.created_at)""",
    """CREATE INDEX pattern_archived IF NOT EXISTS
       FOR (p:Pattern) ON (p.archived)""",
    """CREATE INDEX pattern_latest_occurrence IF NOT EXISTS
       FOR (p:Pattern) ON (p.latest_occurrence_date)""",
    # --- Procedure indexes (Feature 031) ---
    """CREATE INDEX procedure_problem_type IF NOT EXISTS
       FOR (p:Procedure) ON (p.problem_type)""",
    """CREATE INDEX procedure_outcome IF NOT EXISTS
       FOR (p:Procedure) ON (p.outcome)""",
    """CREATE INDEX procedure_created IF NOT EXISTS
       FOR (p:Procedure) ON (p.created_at)""",
    # --- EvolutionCandidate indexes (Feature 032) ---
    """CREATE INDEX evolution_status IF NOT EXISTS
       FOR (ec:EvolutionCandidate) ON (ec.status)""",
    """CREATE INDEX evolution_skill IF NOT EXISTS
       FOR (ec:EvolutionCandidate) ON (ec.skill_path)""",
    """CREATE INDEX evolution_created IF NOT EXISTS
       FOR (ec:EvolutionCandidate) ON (ec.created_at)""",
    """CREATE INDEX evolution_confidence IF NOT EXISTS
       FOR (ec:EvolutionCandidate) ON (ec.confidence_score)""",
    # --- ACT-R Activation indexes (Feature 034) ---
    """CREATE INDEX idx_entity_access_count IF NOT EXISTS
       FOR (e:Entity) ON (e.access_count)""",
    """CREATE INDEX idx_entity_last_accessed IF NOT EXISTS
       FOR (e:Entity) ON (e.last_accessed)""",
    """CREATE INDEX idx_relates_to_hebbian IF NOT EXISTS
       FOR ()-[r:RELATES_TO]-() ON (r.hebbian_weight)""",
    """CREATE INDEX idx_relates_to_confidence IF NOT EXISTS
       FOR ()-[r:RELATES_TO]-() ON (r.confidence)""",
]


async def init_neo4j_schema(neo4j_service) -> None:
    """Initialize all Neo4j constraints and indexes.

    Runs on every server startup. All statements are idempotent
    (IF NOT EXISTS), so this is safe to call repeatedly.

    Args:
        neo4j_service: Connected Neo4jService instance with a query() method.
    """
    logger.info("Initializing Neo4j schema (constraints + indexes)...")

    errors = []

    # Create constraints first (they implicitly create indexes)
    for stmt in CONSTRAINTS:
        try:
            await neo4j_service.query(stmt)
        except Exception as e:
            if "already exists" in str(e).lower():
                continue  # Expected on subsequent runs
            errors.append(f"Constraint error: {e}")
            logger.warning(f"Schema constraint warning: {e}")

    # Create indexes
    for stmt in INDEXES:
        try:
            await neo4j_service.query(stmt)
        except Exception as e:
            if "already exists" in str(e).lower():
                continue
            errors.append(f"Index error: {e}")
            logger.warning(f"Schema index warning: {e}")

    if errors:
        logger.warning(
            f"Neo4j schema init completed with {len(errors)} warnings "
            f"({len(CONSTRAINTS)} constraints, {len(INDEXES)} indexes attempted)"
        )
    else:
        logger.info(
            f"✅ Neo4j schema initialized "
            f"({len(CONSTRAINTS)} constraints, {len(INDEXES)} indexes)"
        )

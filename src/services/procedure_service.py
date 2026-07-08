"""
Procedure Service (Feature 031: Procedural Memory Layer)

Captures and retrieves diagnostic procedures - structured problem-solving
pathways: symptom → diagnostic steps → root cause → fix → verification.

Architecture:
- Core write (Neo4j) is synchronous (<200ms)
- Enrichment (Qdrant embedding, markdown, entities, events) is fire-and-forget
- Deduplication via Qdrant similarity search at 0.92 threshold
- Embedding strategy: "{symptom} {root_cause}" concatenation
"""

import asyncio
import json
import time
import uuid as uuid_module
from datetime import datetime

from loguru import logger

from src.config import (
    PROCEDURE_DEDUP_THRESHOLD,
    PROCEDURE_SEARCH_DEFAULT_LIMIT,
    MemoryType,
)
from src.models.procedure_models import Procedure, ProcedureMatch
from src.services.entity_factory import entity_factory
from src.services.event_log_service import event_log_service
from src.services.file_service import file_service
from src.services.neo4j_service import neo4j_service
from src.services.vector_service import get_vector_service

# Qdrant collection name for procedures
COLLECTION_PROCEDURES = "mnemodrome_procedures"


class ProcedureService:
    """
    Service for capturing and retrieving procedural memory.

    Follows the same patterns as capture_learning in mcp_server.py:
    - Synchronous core write to Neo4j
    - Fire-and-forget enrichment (Qdrant, markdown, entities, events)
    - Deduplication via semantic similarity
    """

    async def create_procedure(
        self,
        procedure: Procedure,
        session_uuid: str | None = None,
    ) -> tuple[str, bool]:
        """
        Create a new procedure or return existing duplicate.

        Steps:
        1. Check for duplicates via Qdrant (0.92 threshold)
        2. If duplicate found, return existing UUID
        3. Otherwise, write Neo4j node and return new UUID

        Args:
            procedure: Validated Procedure model
            session_uuid: Optional session UUID for CAPTURED_IN relationship

        Returns:
            Tuple of (procedure_uuid, is_new) where is_new=False means duplicate
        """
        # Generate UUID if not provided
        if not procedure.procedure_uuid:
            procedure.procedure_uuid = str(uuid_module.uuid4())

        # Check for duplicates
        try:
            existing_uuid = await self._check_duplicate(
                symptom=procedure.symptom,
                root_cause=procedure.root_cause,
                problem_type=procedure.problem_type,
            )
            if existing_uuid:
                logger.info(
                    f"Duplicate procedure detected: {existing_uuid} "
                    f"(symptom similarity >={PROCEDURE_DEDUP_THRESHOLD})"
                )
                return existing_uuid, False
        except Exception as e:
            # Dedup check failure is non-fatal - proceed with creation
            logger.warning(f"Dedup check failed, proceeding with creation: {e}")

        # Write to Neo4j (synchronous, <200ms target)
        await self._write_neo4j_node(procedure)

        logger.info(
            f"Procedure created: {procedure.procedure_uuid} ({procedure.problem_type})"
        )
        return procedure.procedure_uuid, True

    async def _write_neo4j_node(self, procedure: Procedure) -> None:
        """
        Create Procedure node in Neo4j.

        Args:
            procedure: Validated Procedure model

        Raises:
            Exception: If Neo4j write fails
        """
        cypher = """
        CREATE (p:Procedure {
            uuid: $uuid,
            problem_type: $problem_type,
            symptom: $symptom,
            diagnostic_steps: $diagnostic_steps,
            root_cause: $root_cause,
            fix: $fix,
            verification: $verification,
            outcome: $outcome,
            technologies: $technologies,
            pattern_uuid: $pattern_uuid,
            session_uuid: $session_uuid,
            created_at: datetime()
        })
        RETURN p.uuid as uuid
        """
        params = {
            "uuid": procedure.procedure_uuid,
            "problem_type": procedure.problem_type,
            "symptom": procedure.symptom,
            "diagnostic_steps": json.dumps(procedure.diagnostic_steps),
            "root_cause": procedure.root_cause,
            "fix": procedure.fix,
            "verification": procedure.verification,
            "outcome": procedure.outcome,
            "technologies": json.dumps(procedure.technologies),
            "pattern_uuid": procedure.pattern_uuid or "",
            "session_uuid": procedure.session_uuid or "",
        }

        result = await neo4j_service.query(cypher, params)
        if not result:
            raise Exception(
                f"Failed to create Procedure node: {procedure.procedure_uuid}"
            )

    async def _check_duplicate(
        self,
        symptom: str,
        root_cause: str,
        problem_type: str,
    ) -> str | None:
        """
        Check for duplicate procedures via Qdrant semantic similarity.

        Uses embedding of "{symptom} {root_cause}" with 0.92 threshold,
        filtered by matching problem_type.

        Args:
            symptom: Problem symptom text
            root_cause: Root cause text
            problem_type: Problem category filter

        Returns:
            UUID of existing duplicate, or None if no duplicate found
        """
        vector_service = get_vector_service()
        if not vector_service.enabled or not vector_service._initialized:
            return None

        embed_text = f"{symptom} {root_cause}"
        embedding = await vector_service.embed_text(embed_text)
        if not embedding:
            return None

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="problem_type",
                        match=MatchValue(value=problem_type),
                    )
                ]
            )

            results = await asyncio.to_thread(
                vector_service.client.query_points,
                collection_name=COLLECTION_PROCEDURES,
                query=embedding,
                query_filter=query_filter,
                limit=1,
                score_threshold=PROCEDURE_DEDUP_THRESHOLD,
            )

            if results.points:
                hit = results.points[0]
                existing_uuid = hit.payload.get("uuid")
                logger.debug(
                    f"Duplicate found: {existing_uuid} "
                    f"(score={hit.score:.4f}, threshold={PROCEDURE_DEDUP_THRESHOLD})"
                )
                return existing_uuid

        except Exception as e:
            # Collection may not exist yet on first run
            logger.debug(f"Qdrant dedup search failed (may be first procedure): {e}")

        return None

    async def enrich_procedure(
        self,
        procedure: Procedure,
        session_uuid: str | None = None,
    ) -> None:
        """
        Background enrichment pipeline for a procedure.

        Runs as fire-and-forget after core Neo4j write:
        1. Store Qdrant embedding
        2. Write markdown file
        3. Create Technology entities + USES_TECHNOLOGY relationships
        4. Create INSTANTIATES relationship (if pattern_uuid)
        5. Create CAPTURED_IN relationship (if session_uuid)
        6. Log event

        Args:
            procedure: Procedure model with UUID set
            session_uuid: Optional session UUID
        """
        enrichment_start = time.time()
        proc_uuid = procedure.procedure_uuid

        # 1. Store Qdrant embedding
        async def store_embedding():
            try:
                vector_service = get_vector_service()
                if not vector_service.enabled or not vector_service._initialized:
                    logger.debug("Vector service unavailable, skipping embedding")
                    return

                embed_text = f"{procedure.symptom} {procedure.root_cause}"
                embedding = await vector_service.embed_text(embed_text)
                if not embedding:
                    logger.warning(f"Empty embedding for procedure {proc_uuid}")
                    return

                point_id = uuid_module.UUID(proc_uuid)
                payload = {
                    "uuid": proc_uuid,
                    "problem_type": procedure.problem_type,
                    "symptom": procedure.symptom,
                    "root_cause": procedure.root_cause,
                    "technologies": procedure.technologies,
                    "outcome": procedure.outcome,
                    "created_at": procedure.created_at.isoformat(),
                }

                from qdrant_client.models import PointStruct

                await asyncio.to_thread(
                    vector_service.client.upsert,
                    collection_name=COLLECTION_PROCEDURES,
                    points=[
                        PointStruct(
                            id=str(point_id),
                            vector=embedding,
                            payload=payload,
                        )
                    ],
                )
                logger.debug(f"Procedure embedding stored: {proc_uuid}")
            except Exception as e:
                logger.warning(f"Procedure embedding failed: {e}")

        # 2. Write markdown file
        async def write_markdown():
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                safe_symptom = (
                    procedure.symptom[:40].lower().replace(" ", "-").replace("/", "-")
                )
                filename = f"procedure-{safe_symptom}-{timestamp}"

                steps_md = "\n".join(
                    f"  {i + 1}. {step}"
                    for i, step in enumerate(procedure.diagnostic_steps)
                )
                techs_md = (
                    ", ".join(procedure.technologies)
                    if procedure.technologies
                    else "N/A"
                )

                content = f"""# Procedure: {procedure.symptom[:80]}

**Problem Type**: {procedure.problem_type}
**Outcome**: {procedure.outcome}
**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**UUID**: `{proc_uuid}`

---

## Symptom
{procedure.symptom}

## Diagnostic Steps
{steps_md}

## Root Cause
{procedure.root_cause}

## Fix
{procedure.fix}

## Verification
{procedure.verification}

## Technologies
{techs_md}

---

*Captured via Mnemodrome MCP capture_procedure tool*
"""
                file_service.write_file(filename, content, MemoryType.PROCEDURE)
                logger.debug(f"Procedure markdown written: {filename}")
            except Exception as e:
                logger.warning(f"Procedure markdown write failed: {e}")

        # 3. Create Technology entities + relationships
        async def create_technology_links():
            try:
                if not procedure.technologies:
                    return

                for tech in procedure.technologies:
                    tech_uuid = await entity_factory.create_or_resolve(
                        name=tech, entity_type="Technology"
                    )
                    # Create USES_TECHNOLOGY relationship
                    cypher = """
                    MATCH (p:Procedure {uuid: $proc_uuid})
                    MATCH (t:Technology {uuid: $tech_uuid})
                    MERGE (p)-[:USES_TECHNOLOGY]->(t)
                    """
                    await neo4j_service.query(
                        cypher,
                        {
                            "proc_uuid": proc_uuid,
                            "tech_uuid": tech_uuid,
                        },
                    )
                logger.debug(f"Technology links created for procedure {proc_uuid}")
            except Exception as e:
                logger.warning(f"Technology linking failed: {e}")

        # 4. Create INSTANTIATES relationship
        async def link_pattern():
            try:
                if not procedure.pattern_uuid:
                    return
                cypher = """
                MATCH (p:Procedure {uuid: $proc_uuid})
                MATCH (pat:Pattern {uuid: $pattern_uuid})
                MERGE (p)-[:INSTANTIATES]->(pat)
                """
                await neo4j_service.query(
                    cypher,
                    {
                        "proc_uuid": proc_uuid,
                        "pattern_uuid": procedure.pattern_uuid,
                    },
                )
                logger.debug(
                    f"Pattern link created: {proc_uuid} -> {procedure.pattern_uuid}"
                )
            except Exception as e:
                logger.warning(f"Pattern linking failed: {e}")

        # 5. Create CAPTURED_IN relationship
        async def link_session():
            try:
                s_uuid = session_uuid or procedure.session_uuid
                if not s_uuid:
                    return
                cypher = """
                MATCH (p:Procedure {uuid: $proc_uuid})
                MATCH (s:Session {uuid: $session_uuid})
                MERGE (s)-[:PRODUCED_PROCEDURE]->(p)
                """
                await neo4j_service.query(
                    cypher,
                    {
                        "proc_uuid": proc_uuid,
                        "session_uuid": s_uuid,
                    },
                )
                logger.debug(f"Session link created: {s_uuid} -> {proc_uuid}")
            except Exception as e:
                logger.warning(f"Session linking failed: {e}")

        # 6. Log event
        async def log_event():
            try:
                await event_log_service.log_event(
                    event_type="procedure_capture",
                    action="captured",
                    actor="system",
                    target=proc_uuid,
                    context=f"problem_type={procedure.problem_type}, outcome={procedure.outcome}",
                    result="success",
                )
            except Exception as e:
                logger.warning(f"Event logging failed: {e}")

        # Execute all enrichment in parallel
        try:
            await asyncio.gather(
                store_embedding(),
                write_markdown(),
                create_technology_links(),
                link_pattern(),
                link_session(),
                log_event(),
                return_exceptions=True,
            )
            elapsed = (time.time() - enrichment_start) * 1000
            logger.info(
                f"Procedure enrichment complete: {proc_uuid} in {elapsed:.0f}ms"
            )
        except Exception as e:
            logger.warning(f"Procedure enrichment failed: {e}")

    async def find_procedures(
        self,
        query_text: str,
        problem_type: str | None = None,
        min_success_rate: float = 0.0,
        limit: int = PROCEDURE_SEARCH_DEFAULT_LIMIT,
    ) -> list[ProcedureMatch]:
        """
        Find procedures by semantic similarity to a problem description.

        Uses Qdrant vector search with optional problem_type filter,
        then enriches from Neo4j for full procedure details.

        Args:
            query_text: Problem description / symptom text
            problem_type: Optional filter by problem category
            min_success_rate: Minimum success rate (not yet used - reserved)
            limit: Maximum results

        Returns:
            List of ProcedureMatch objects ordered by similarity
        """
        vector_service = get_vector_service()

        # Try Qdrant semantic search first
        if vector_service.enabled and vector_service._initialized:
            try:
                return await self._search_qdrant(
                    query_text=query_text,
                    problem_type=problem_type,
                    limit=limit,
                )
            except Exception as e:
                logger.warning(
                    f"Qdrant procedure search failed, falling back to Neo4j: {e}"
                )

        # Fallback: Neo4j text search
        return await self._search_neo4j_fallback(
            query_text=query_text,
            problem_type=problem_type,
            limit=limit,
        )

    async def _search_qdrant(
        self,
        query_text: str,
        problem_type: str | None = None,
        limit: int = 5,
    ) -> list[ProcedureMatch]:
        """Semantic search via Qdrant, enriched from Neo4j."""
        vector_service = get_vector_service()

        embedding = await vector_service.embed_text(query_text)
        if not embedding:
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_filter = None
        if problem_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="problem_type",
                        match=MatchValue(value=problem_type),
                    )
                ]
            )

        results = await asyncio.to_thread(
            vector_service.client.query_points,
            collection_name=COLLECTION_PROCEDURES,
            query=embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=0.5,  # Lower threshold for search (not dedup)
        )

        if not results.points:
            return []

        # Enrich from Neo4j
        matches = []
        for hit in results.points:
            proc_uuid = hit.payload.get("uuid")
            if not proc_uuid:
                continue

            neo4j_data = await self._get_procedure_from_neo4j(proc_uuid)
            if not neo4j_data:
                # Use Qdrant payload as fallback
                matches.append(
                    ProcedureMatch(
                        procedure_uuid=proc_uuid,
                        problem_type=hit.payload.get("problem_type", "other"),
                        symptom=hit.payload.get("symptom", ""),
                        diagnostic_steps=[],
                        root_cause=hit.payload.get("root_cause", ""),
                        fix="",
                        verification="",
                        outcome=hit.payload.get("outcome", "SUCCESS"),
                        technologies=hit.payload.get("technologies", []),
                        similarity_score=hit.score,
                        created_at=datetime.fromisoformat(hit.payload["created_at"])
                        if hit.payload.get("created_at")
                        else datetime.utcnow(),
                    )
                )
                continue

            # Parse JSON fields from Neo4j
            diag_steps = neo4j_data.get("diagnostic_steps", "[]")
            if isinstance(diag_steps, str):
                diag_steps = json.loads(diag_steps)

            techs = neo4j_data.get("technologies", "[]")
            if isinstance(techs, str):
                techs = json.loads(techs)

            # Get linked pattern name if available
            pattern_uuid = neo4j_data.get("pattern_uuid", "")
            pattern_name = None
            if pattern_uuid:
                pattern_name = await self._get_pattern_name(pattern_uuid)

            created_at = neo4j_data.get("created_at")
            if isinstance(created_at, datetime):
                pass  # Already a datetime (normalized by neo4j_service)
            elif isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            else:
                created_at = datetime.utcnow()

            matches.append(
                ProcedureMatch(
                    procedure_uuid=proc_uuid,
                    problem_type=neo4j_data.get("problem_type", "other"),
                    symptom=neo4j_data.get("symptom", ""),
                    diagnostic_steps=diag_steps,
                    root_cause=neo4j_data.get("root_cause", ""),
                    fix=neo4j_data.get("fix", ""),
                    verification=neo4j_data.get("verification", ""),
                    outcome=neo4j_data.get("outcome", "SUCCESS"),
                    technologies=techs,
                    similarity_score=hit.score,
                    created_at=created_at,
                    linked_pattern_uuid=pattern_uuid or None,
                    linked_pattern_name=pattern_name,
                )
            )

        logger.info(f"Found {len(matches)} procedures for query: {query_text[:50]}...")
        return matches

    async def _search_neo4j_fallback(
        self,
        query_text: str,
        problem_type: str | None = None,
        limit: int = 5,
    ) -> list[ProcedureMatch]:
        """Fallback text search via Neo4j when Qdrant is unavailable."""
        # Extract keywords from query for CONTAINS matching
        keywords = [w.lower() for w in query_text.split() if len(w) > 3][:5]
        if not keywords:
            return []

        # Build WHERE clause with keyword matching
        conditions = []
        params = {"limit": limit}

        for i, kw in enumerate(keywords):
            param_name = f"kw{i}"
            conditions.append(
                f"(toLower(p.symptom) CONTAINS ${param_name} OR "
                f"toLower(p.root_cause) CONTAINS ${param_name})"
            )
            params[param_name] = kw

        where_clause = " OR ".join(conditions)

        if problem_type:
            where_clause = f"p.problem_type = $problem_type AND ({where_clause})"
            params["problem_type"] = problem_type

        cypher = f"""
        MATCH (p:Procedure)
        WHERE {where_clause}
        RETURN p
        ORDER BY p.created_at DESC
        LIMIT $limit
        """

        results = await neo4j_service.query(cypher, params)
        if not results:
            return []

        matches = []
        for row in results:
            p = row["p"]

            diag_steps = p.get("diagnostic_steps", "[]")
            if isinstance(diag_steps, str):
                diag_steps = json.loads(diag_steps)

            techs = p.get("technologies", "[]")
            if isinstance(techs, str):
                techs = json.loads(techs)

            created_at = p.get("created_at")
            if isinstance(created_at, datetime):
                pass  # Already a datetime (normalized by neo4j_service)
            elif isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            else:
                created_at = datetime.utcnow()

            matches.append(
                ProcedureMatch(
                    procedure_uuid=p.get("uuid", ""),
                    problem_type=p.get("problem_type", "other"),
                    symptom=p.get("symptom", ""),
                    diagnostic_steps=diag_steps,
                    root_cause=p.get("root_cause", ""),
                    fix=p.get("fix", ""),
                    verification=p.get("verification", ""),
                    outcome=p.get("outcome", "SUCCESS"),
                    technologies=techs,
                    similarity_score=0.0,  # No semantic score for text fallback
                    created_at=created_at,
                )
            )

        logger.info(f"Found {len(matches)} procedures via Neo4j fallback")
        return matches

    async def _get_procedure_from_neo4j(self, proc_uuid: str) -> dict | None:
        """Fetch full procedure data from Neo4j by UUID."""
        cypher = """
        MATCH (p:Procedure {uuid: $uuid})
        RETURN p.uuid as uuid, p.problem_type as problem_type,
               p.symptom as symptom, p.diagnostic_steps as diagnostic_steps,
               p.root_cause as root_cause, p.fix as fix,
               p.verification as verification, p.outcome as outcome,
               p.technologies as technologies, p.pattern_uuid as pattern_uuid,
               p.created_at as created_at
        """
        result = await neo4j_service.query(cypher, {"uuid": proc_uuid})
        return result[0] if result else None

    async def _get_pattern_name(self, pattern_uuid: str) -> str | None:
        """Get pattern name by UUID for enrichment."""
        if not pattern_uuid:
            return None
        cypher = """
        MATCH (pat:Pattern {uuid: $uuid})
        RETURN pat.name as name
        """
        result = await neo4j_service.query(cypher, {"uuid": pattern_uuid})
        return result[0]["name"] if result else None


# Global singleton
procedure_service = ProcedureService()

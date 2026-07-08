"""
Reasoning-time knowledge validation service for Feature 028.

Orchestrates claim extraction, parallel knowledge retrieval, conflict validation,
and synthesis into a unified validation result. Never raises — always returns a result.

Constitutional Principle: Reasoning must be validated against stored knowledge
before delivery to ensure consistency and correctness.
"""

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from loguru import logger

from src.config import (
    REASONING_VALIDATION_LEARNING_LIMIT,
    REASONING_VALIDATION_PER_QUERY_TIMEOUT_MS,
)
from src.models.reasoning_validation_models import (
    ConflictSummary,
    LearningReference,
    PatternReference,
    ReasoningValidationResult,
)
from src.services.claim_extraction_service import ClaimExtractionService
from src.services.conflict_validation_service import ConflictValidationService
from src.services.neo4j_fact_retrieval_service import Neo4jFactRetrievalService
from src.services.pattern_query_service import PatternQueryService

if TYPE_CHECKING:
    from src.services.llm_service import LLMService
    from src.services.neo4j_service import Neo4jService
    from src.services.redis_service import RedisService
    from src.services.vector_service import VectorService


class ValidationSynthesizer:
    """
    Merge conflict, learning, and pattern results into a unified validation result.

    Thin logic — no additional LLM calls. Pure data transformation.
    """

    def synthesize(
        self,
        conflicts: list,
        learnings: list[dict],
        patterns: list,
        elapsed_ms: float,
        degraded_queries: list[str],
        claims_analyzed: int = 0,
        entities_checked: int = 0,
    ) -> ReasoningValidationResult:
        """
        Synthesize validation components into a unified result.

        Args:
            conflicts: List of ConflictResult from Feature 025
            learnings: List of dicts from VectorService.search()
            patterns: List of PatternMatch from Feature 026
            elapsed_ms: Time elapsed for validation
            degraded_queries: List of query names that failed or timed out
            claims_analyzed: Number of assertions analyzed
            entities_checked: Number of unique entities checked

        Returns:
            ReasoningValidationResult with status determined by content
        """
        # Determine validation status (conflicts have highest priority)
        if conflicts:
            validation_status = "conflicts_detected"
        elif learnings or patterns:
            validation_status = "reinforced"
        else:
            validation_status = "clean"

        # Map ConflictResult to ConflictSummary
        conflict_summaries = []
        for conflict in conflicts:
            try:
                summary = ConflictSummary(
                    conflict_type=conflict.conflict_type,
                    severity=conflict.severity,
                    claim=conflict.claim.sentence
                    if hasattr(conflict.claim, "sentence")
                    else str(conflict.claim),
                    stored_fact=f"{conflict.stored_fact.entity1_name} {conflict.stored_fact.relation} {conflict.stored_fact.entity2_name}"
                    if hasattr(conflict.stored_fact, "entity1_name")
                    else str(conflict.stored_fact),
                    recommendation=conflict.recommendation,
                    correction_suggestion=conflict.correction_suggestion,
                )
                conflict_summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to map conflict result: {e}")
                continue

        # Map learning dicts to LearningReference
        learning_references = []
        for learning in learnings:
            try:
                # Extract excerpt (truncate to 200 chars)
                content = learning.get("content", "")
                excerpt = content[:200] if isinstance(content, str) else ""

                ref = LearningReference(
                    uuid=learning.get("uuid", ""),
                    title=learning.get("title", ""),
                    similarity_score=learning.get("score", 0.0),
                    created_at=learning.get("created_at"),
                    excerpt=excerpt,
                )
                learning_references.append(ref)
            except Exception as e:
                logger.warning(f"Failed to map learning: {e}")
                continue

        # Map PatternMatch to PatternReference
        pattern_references = []
        for pattern in patterns:
            try:
                ref = PatternReference(
                    pattern_uuid=pattern.pattern_uuid,
                    name=pattern.name,
                    pattern_type=pattern.pattern_type,
                    similarity_score=pattern.similarity_score,
                    success_rate=pattern.success_rate,
                    occurrence_count=pattern.occurrence_count,
                    relevance_explanation=pattern.relevance_explanation,
                )
                pattern_references.append(ref)
            except Exception as e:
                logger.warning(f"Failed to map pattern: {e}")
                continue

        # Build summary string
        if validation_status == "conflicts_detected":
            summary = (
                f"{len(conflicts)} conflict(s) detected. Review before responding."
            )
        elif validation_status == "reinforced":
            summary = f"Response consistent with {len(learning_references)} learning(s) and {len(pattern_references)} pattern(s)."
        else:
            summary = "No conflicts or relevant context found."

        # Append degraded_queries if present
        if degraded_queries:
            summary += f" (partial: {', '.join(degraded_queries)} degraded)"

        # Build and return result
        return ReasoningValidationResult(
            validation_status=validation_status,
            conflicts=conflict_summaries,
            relevant_learnings=learning_references,
            relevant_patterns=pattern_references,
            claims_analyzed=claims_analyzed,
            entities_checked=entities_checked,
            elapsed_ms=elapsed_ms,
            degraded_queries=degraded_queries,
            summary=summary,
        )


class ReasoningValidationService:
    """
    Orchestrates reasoning-time knowledge validation.

    Composes Feature 025 (claim extraction + conflict validation) and
    Feature 026 (pattern detection) with parallel execution to stay
    within 500ms soft / 1000ms hard latency budget.

    Never raises — degrades gracefully if any upstream query fails or times out.
    """

    def __init__(
        self,
        llm_service: "LLMService",
        neo4j_service: "Neo4jService",
        vector_service: "VectorService",
        redis_service: Optional["RedisService"] = None,
    ):
        """
        Initialize ReasoningValidationService.

        Args:
            llm_service: LLMService instance for claim extraction
            neo4j_service: Neo4jService instance for fact retrieval
            vector_service: VectorService instance for learning search
            redis_service: Optional RedisService for claim extraction caching
        """
        self.claim_extraction = ClaimExtractionService(
            llm_service=llm_service,
            redis_service=redis_service,
        )
        self.conflict_validation = ConflictValidationService()
        self.fact_retrieval = Neo4jFactRetrievalService(neo4j_service)
        self.pattern_query = PatternQueryService()
        self.vector_service = vector_service
        self.synthesizer = ValidationSynthesizer()

        logger.info("ReasoningValidationService initialized")

    async def validate_response(
        self,
        proposed_response: str,
        timeout_ms: int = 1000,
    ) -> ReasoningValidationResult:
        """
        Validate proposed_response against Mnemodrome's full knowledge base.

        Phases:
          1. Extract claims from proposed_response (LLM, sequential — must complete first)
          2. Run three parallel queries (Neo4j facts, vector learnings, patterns)
          3. Validate claims against retrieved facts
          4. Synthesize into ReasoningValidationResult

        Args:
            proposed_response: The text Claude intends to deliver to the user.
            timeout_ms: Hard deadline for validation (default 1000ms).
                        Phase 1 (claim extraction) gets 500ms of this budget.

        Returns:
            ReasoningValidationResult always (never raises).
        """
        start_time = time.time()
        degraded_queries: list[str] = []

        try:
            # Phase 1: Extract claims with 500ms budget
            logger.debug("Phase 1: Extracting claims (500ms budget)")
            try:
                claims = await asyncio.wait_for(
                    self.claim_extraction.extract_claims(proposed_response),
                    timeout=0.5,
                )
            except TimeoutError:
                logger.warning("Claim extraction timed out (exceeded 500ms budget)")
                elapsed_ms = (time.time() - start_time) * 1000
                return ReasoningValidationResult(
                    validation_status="clean",
                    elapsed_ms=elapsed_ms,
                    summary="Claim extraction timed out",
                    claims_analyzed=0,
                    entities_checked=0,
                )
            except Exception as e:
                logger.error(f"Claim extraction failed: {e}")
                elapsed_ms = (time.time() - start_time) * 1000
                return ReasoningValidationResult(
                    validation_status="clean",
                    elapsed_ms=elapsed_ms,
                    summary=f"Claim extraction error: {str(e)[:100]}",
                    claims_analyzed=0,
                    entities_checked=0,
                )

            if not claims:
                logger.info("No claims extracted from proposed response")
                elapsed_ms = (time.time() - start_time) * 1000
                return ReasoningValidationResult(
                    validation_status="clean",
                    elapsed_ms=elapsed_ms,
                    summary="No assertions found to validate",
                    claims_analyzed=0,
                    entities_checked=0,
                )

            # Filter to assertions only
            assertions = [c for c in claims if c.claim_type == "assertion"]
            if not assertions:
                logger.info(f"No assertions found among {len(claims)} claims")
                elapsed_ms = (time.time() - start_time) * 1000
                return ReasoningValidationResult(
                    validation_status="clean",
                    elapsed_ms=elapsed_ms,
                    summary="No factual assertions found.",
                    claims_analyzed=0,
                    entities_checked=0,
                )

            logger.debug(
                f"Filtered {len(assertions)} assertions from {len(claims)} total claims"
            )

            # Extract entity names from assertions (inline, no LLM call)
            entity_names = list(
                {name for c in assertions for name in [c.entity1, c.entity2] if name}
            )
            if not entity_names:
                logger.info("No entities found in assertions")
                elapsed_ms = (time.time() - start_time) * 1000
                return ReasoningValidationResult(
                    validation_status="clean",
                    elapsed_ms=elapsed_ms,
                    summary="No entities found to validate against.",
                    claims_analyzed=len(assertions),
                    entities_checked=0,
                )

            logger.debug(
                f"Extracted {len(entity_names)} unique entity names from assertions"
            )

            # Build query summary for vector and pattern search
            query_summary = " ".join(
                entity_names
                + [c.sentence for c in assertions if hasattr(c, "sentence")]
            )

            # Phase 2: Run parallel queries
            logger.debug("Phase 2: Running parallel queries")
            facts, learnings, patterns = await self._run_parallel_queries(
                entity_names,
                query_summary,
                REASONING_VALIDATION_PER_QUERY_TIMEOUT_MS,
                degraded_queries,
            )

            # Phase 3: Validate claims against stored facts
            logger.debug("Phase 3: Validating claims against facts")
            conflicts = []
            if facts:
                try:
                    conflicts = await self.conflict_validation.validate_claims_batch(
                        assertions, facts
                    )
                except Exception as e:
                    logger.warning(
                        f"Conflict validation failed (graceful degradation): {e}"
                    )
                    conflicts = []

            # Phase 4: Synthesize results
            logger.debug("Phase 4: Synthesizing results")
            elapsed_ms = (time.time() - start_time) * 1000
            result = self.synthesizer.synthesize(
                conflicts=conflicts,
                learnings=learnings,
                patterns=patterns,
                elapsed_ms=elapsed_ms,
                degraded_queries=degraded_queries,
                claims_analyzed=len(assertions),
                entities_checked=len(entity_names),
            )

            logger.info(
                f"Validation complete: status={result.validation_status}, "
                f"conflicts={len(conflicts)}, learnings={len(learnings)}, "
                f"patterns={len(patterns)}, elapsed={elapsed_ms:.0f}ms"
            )

            return result

        except Exception as e:
            # Outer exception handler - return safe degraded result
            logger.error(f"Unexpected error during validation: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ReasoningValidationResult(
                validation_status="clean",
                elapsed_ms=elapsed_ms,
                summary=f"Validation error: {str(e)[:100]}",
                claims_analyzed=0,
                entities_checked=0,
                degraded_queries=degraded_queries,
            )

    async def _run_parallel_queries(
        self,
        entity_names: list[str],
        query_summary: str,
        per_query_timeout_ms: int,
        degraded_queries: list,
    ) -> tuple[list, list[dict], list]:
        """
        Run fact retrieval, vector search, and pattern search in parallel.

        Each query gets its own asyncio.wait_for timeout. Failures return
        empty lists rather than propagating exceptions (graceful degradation).

        Args:
            entity_names: List of entity names to query
            query_summary: Query text for vector and pattern search
            per_query_timeout_ms: Timeout per query in milliseconds
            degraded_queries: Mutable list to track failed queries

        Returns:
            (stored_facts, similar_learnings, pattern_matches)
        """

        async def safe_facts():
            """Safely retrieve facts with timeout and exception handling."""
            try:
                # get_facts_for_entities expects list[str]
                return await asyncio.wait_for(
                    self.fact_retrieval.get_facts_for_entities(entity_names),
                    timeout=per_query_timeout_ms / 1000.0,
                )
            except TimeoutError:
                logger.warning(
                    f"Fact retrieval timed out after {per_query_timeout_ms}ms"
                )
                degraded_queries.append("fact_retrieval")
                return []
            except Exception as e:
                logger.warning(f"Fact retrieval degraded: {e}")
                degraded_queries.append("fact_retrieval")
                return []

        async def safe_learnings():
            """Safely search learnings with timeout and exception handling."""
            try:
                # VectorService.search uses query_text= and search_type=
                return await asyncio.wait_for(
                    self.vector_service.search(
                        query_text=query_summary,
                        search_type="learnings",
                        limit=REASONING_VALIDATION_LEARNING_LIMIT,
                    ),
                    timeout=per_query_timeout_ms / 1000.0,
                )
            except TimeoutError:
                logger.warning(
                    f"Learning search timed out after {per_query_timeout_ms}ms"
                )
                degraded_queries.append("learning_search")
                return []
            except Exception as e:
                logger.warning(f"Learning search degraded: {e}")
                degraded_queries.append("learning_search")
                return []

        async def safe_patterns():
            """Safely search patterns with timeout and exception handling."""
            try:
                return await asyncio.wait_for(
                    self.pattern_query.search_patterns_by_problem(query_summary),
                    timeout=per_query_timeout_ms / 1000.0,
                )
            except TimeoutError:
                logger.warning(
                    f"Pattern search timed out after {per_query_timeout_ms}ms"
                )
                degraded_queries.append("pattern_search")
                return []
            except Exception as e:
                logger.warning(f"Pattern search degraded: {e}")
                degraded_queries.append("pattern_search")
                return []

        # Run all three queries in parallel
        results = await asyncio.gather(
            safe_facts(),
            safe_learnings(),
            safe_patterns(),
            return_exceptions=True,
        )

        # Extract results, handling any exceptions
        facts = results[0] if not isinstance(results[0], Exception) else []
        learnings = results[1] if not isinstance(results[1], Exception) else []
        patterns = results[2] if not isinstance(results[2], Exception) else []

        logger.debug(
            f"Parallel query results: facts={len(facts)}, "
            f"learnings={len(learnings)}, patterns={len(patterns)}"
        )

        return facts, learnings, patterns

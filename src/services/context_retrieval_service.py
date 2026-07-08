"""
Context Retrieval Service (Feature 022: Semantic Vector Layer)

Implements the retrieval cascade:
1. Vector similarity search (Qdrant)
2. Entity extraction from matched content (Redis cache)
3. Graph traversal for relationships (Neo4j)
4. Rich content retrieval (Markdown files)

Constitutional Principle: Always-on observation
Automatically searches for relevant context on every user prompt.
"""

import asyncio
from datetime import datetime
from typing import Any

from loguru import logger

from src.services.procedure_service import procedure_service

from .neo4j_service import neo4j_service
from .redis_service import redis_service
from .vector_service import get_vector_service

# Keywords that suggest the user prompt describes a problem (Feature 031)
PROBLEM_KEYWORDS = {
    "error",
    "slow",
    "failing",
    "broken",
    "timeout",
    "issue",
    "bug",
    "crash",
    "hanging",
    "stuck",
    "not working",
    "wrong",
    "problem",
    "failed",
    "exception",
    "latency",
    "regression",
    "fix",
    "debug",
    "troubleshoot",
    "investigate",
    "diagnose",
}


def _prompt_describes_problem(prompt: str) -> bool:
    """Check if prompt appears to describe a problem using keyword heuristic."""
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in PROBLEM_KEYWORDS)


class ContextRetrievalService:
    """
    Orchestrates multi-layer context retrieval cascade.

    Flow:
    1. Embed user prompt → search vector DB for similar content
    2. Extract entities from similar content → resolve via Redis cache
    3. Query Neo4j graph for entity relationships and context
    4. Retrieve markdown files for rich learnings/sessions
    5. Format and return unified context
    """

    def __init__(self):
        """Initialize context retrieval service."""
        self.vector_service = get_vector_service()
        self.redis_service = redis_service
        self.neo4j_service = neo4j_service

        # Performance tracking
        self.stats = {
            "total_retrievals": 0,
            "vector_hits": 0,
            "vector_misses": 0,
            "avg_retrieval_time_ms": 0,
        }

    async def initialize(self):
        """Initialize all dependent services."""
        logger.info("Initializing context retrieval service...")

        # Initialize vector service
        await self.vector_service.initialize()

        # Redis and Neo4j services initialize on first use
        logger.success("Context retrieval service initialized")

    async def retrieve_context_for_prompt(
        self,
        prompt: str,
        max_results: int = 5,
        include_learnings: bool = True,
        include_facts: bool = True,
        include_sessions: bool = False,
        include_procedures: bool = True,
    ) -> dict[str, Any]:
        """
        Retrieve relevant context for user prompt using cascade.

        Args:
            prompt: User's prompt/query
            max_results: Maximum similar items to retrieve at each stage
            include_learnings: Include similar learnings in results
            include_facts: Include related facts from graph
            include_sessions: Include similar session summaries

        Returns:
            Unified context dictionary with:
            - is_greenfield: True if no relevant history found
            - similar_prompts: List of semantically similar past prompts
            - similar_learnings: List of related learnings
            - entity_context: Graph context from entities
            - facts: Related facts from knowledge graph
            - retrieval_time_ms: Performance metric
        """
        start_time = datetime.utcnow()
        self.stats["total_retrievals"] += 1

        context = {
            "prompt": prompt,
            "is_greenfield": True,
            "similar_prompts": [],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [],
            "retrieval_time_ms": 0,
        }

        try:
            # Stage 1: Vector similarity search
            logger.info(f"[Cascade] Stage 1: Vector search for: '{prompt[:80]}...'")
            similar_prompts = await self.vector_service.search_similar_prompts(
                query_text=prompt, limit=max_results
            )

            if similar_prompts:
                self.stats["vector_hits"] += 1
                context["is_greenfield"] = False
                context["similar_prompts"] = similar_prompts
                logger.info(
                    f"[Cascade] Vector: Found {len(similar_prompts)} similar prompts"
                )
            else:
                self.stats["vector_misses"] += 1
                logger.info("[Cascade] Vector: No similar prompts (greenfield topic)")

            # Stage 2: Search similar learnings if enabled
            if include_learnings:
                logger.info("[Cascade] Stage 2: Searching learnings...")
                similar_learnings = await self.vector_service.search_similar_learnings(
                    query_text=prompt, limit=max_results
                )
                if similar_learnings:
                    context["is_greenfield"] = False
                    context["similar_learnings"] = similar_learnings
                    logger.info(f"[Cascade] Learnings: Found {len(similar_learnings)}")

            # Stage 3: Extract entities and query graph
            if include_facts and not context["is_greenfield"]:
                logger.info("[Cascade] Stage 3: Extracting entities...")
                entities = await self._extract_entities_from_matches(
                    similar_prompts + context.get("similar_learnings", [])
                )

                if entities:
                    logger.info(
                        f"[Cascade] Entities: Extracted {len(entities)} unique entities"
                    )

                    # Query Neo4j for entity relationships + related facts in parallel
                    logger.info("[Cascade] Stage 4: Querying knowledge graph...")
                    entity_context, facts = await asyncio.gather(
                        self._query_entity_context(entities),
                        self._get_related_facts(entities),
                    )
                    context["entity_context"] = entity_context
                    context["facts"] = facts

            # Stage 5: Procedure search (Feature 031 - only for problem-like prompts)
            if include_procedures and _prompt_describes_problem(prompt):
                try:
                    procedure_matches = await procedure_service.find_procedures(
                        query_text=prompt,
                        limit=3,
                    )
                    if procedure_matches:
                        context["procedures"] = procedure_matches
                        logger.debug(
                            f"Stage 5: Found {len(procedure_matches)} matching procedures"
                        )
                except Exception as e:
                    logger.warning(f"Stage 5 procedure search failed (non-fatal): {e}")

            # Calculate performance
            end_time = datetime.utcnow()
            retrieval_time = (end_time - start_time).total_seconds() * 1000
            context["retrieval_time_ms"] = round(retrieval_time, 2)

            # Update average
            total = self.stats["total_retrievals"]
            current_avg = self.stats["avg_retrieval_time_ms"]
            self.stats["avg_retrieval_time_ms"] = (
                current_avg * (total - 1) + retrieval_time
            ) / total

            logger.success(
                f"[Cascade] Complete in {retrieval_time:.0f}ms "
                f"(greenfield={context['is_greenfield']})"
            )

            return context

        except Exception as e:
            logger.error(f"[Cascade] Context retrieval failed: {e}")
            context["error"] = str(e)
            return context

    async def retrieve_context_for_prompt_with_embedding(
        self,
        prompt: str,
        embedding: list[float],
        max_results: int = 5,
        include_learnings: bool = True,
        include_facts: bool = True,
        include_sessions: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve relevant context using a pre-computed embedding.

        Task 2 Optimization: Reuses embedding to avoid redundant GPU calls.
        Executes cascade using cached embedding for search operations.

        Args:
            prompt: User's prompt/query (for logging)
            embedding: Pre-computed embedding vector (no GPU call needed)
            max_results: Maximum similar items to retrieve at each stage
            include_learnings: Include similar learnings in results
            include_facts: Include related facts from graph
            include_sessions: Include similar session summaries

        Returns:
            Unified context dictionary (same structure as retrieve_context_for_prompt)
        """
        start_time = datetime.utcnow()
        self.stats["total_retrievals"] += 1

        context = {
            "prompt": prompt,
            "is_greenfield": True,
            "similar_prompts": [],
            "similar_learnings": [],
            "entity_context": [],
            "facts": [],
            "retrieval_time_ms": 0,
        }

        try:
            # Stage 1: Vector similarity search using pre-computed embedding
            logger.info(
                f"[Cascade-Optimized] Stage 1: Vector search (cached embedding) for: '{prompt[:80]}...'"
            )
            similar_prompts = (
                await self.vector_service.search_similar_prompts_by_embedding(
                    query_embedding=embedding, limit=max_results
                )
            )

            if similar_prompts:
                self.stats["vector_hits"] += 1
                context["is_greenfield"] = False
                context["similar_prompts"] = similar_prompts
                logger.info(
                    f"[Cascade-Optimized] Vector: Found {len(similar_prompts)} similar prompts (cached embedding)"
                )
            else:
                self.stats["vector_misses"] += 1
                logger.info(
                    "[Cascade-Optimized] Vector: No similar prompts (greenfield topic)"
                )

            # Stage 2: Search similar learnings using same embedding if enabled
            if include_learnings:
                logger.info(
                    "[Cascade-Optimized] Stage 2: Searching learnings (cached embedding)..."
                )
                similar_learnings = (
                    await self.vector_service.search_similar_learnings_by_embedding(
                        query_embedding=embedding, limit=max_results
                    )
                )
                if similar_learnings:
                    context["is_greenfield"] = False
                    context["similar_learnings"] = similar_learnings
                    logger.info(
                        f"[Cascade-Optimized] Learnings: Found {len(similar_learnings)}"
                    )

            # Stage 3: Extract entities and query graph
            if include_facts and not context["is_greenfield"]:
                logger.info("[Cascade-Optimized] Stage 3: Extracting entities...")
                entities = await self._extract_entities_from_matches(
                    similar_prompts + context.get("similar_learnings", [])
                )

                if entities:
                    logger.info(
                        f"[Cascade-Optimized] Entities: Extracted {len(entities)} unique entities"
                    )

                    # Query Neo4j for entity relationships + related facts in parallel
                    logger.info(
                        "[Cascade-Optimized] Stage 4: Querying knowledge graph..."
                    )
                    entity_context, facts = await asyncio.gather(
                        self._query_entity_context(entities),
                        self._get_related_facts(entities),
                    )
                    context["entity_context"] = entity_context
                    context["facts"] = facts

            # Calculate performance
            end_time = datetime.utcnow()
            retrieval_time = (end_time - start_time).total_seconds() * 1000
            context["retrieval_time_ms"] = round(retrieval_time, 2)

            # Update average
            total = self.stats["total_retrievals"]
            current_avg = self.stats["avg_retrieval_time_ms"]
            self.stats["avg_retrieval_time_ms"] = (
                current_avg * (total - 1) + retrieval_time
            ) / total

            logger.success(
                f"[Cascade-Optimized] Complete in {retrieval_time:.0f}ms "
                f"(greenfield={context['is_greenfield']}, embedding_reuse=true)"
            )

            return context

        except Exception as e:
            logger.error(f"[Cascade-Optimized] Context retrieval failed: {e}")
            context["error"] = str(e)
            return context

    async def _extract_entities_from_matches(
        self, matches: list[dict[str, Any]]
    ) -> set[str]:
        """
        Extract unique entity UUIDs from vector search matches.

        Args:
            matches: List of similar prompts/learnings with neo4j_uuid

        Returns:
            Set of entity UUIDs
        """
        entities = set()
        for match in matches:
            neo4j_uuid = match.get("neo4j_uuid")
            if neo4j_uuid:
                entities.add(neo4j_uuid)

        return entities

    async def _query_entity_context(
        self, entity_uuids: set[str]
    ) -> list[dict[str, Any]]:
        """
        Query Neo4j for entity context (properties, relationships).

        Args:
            entity_uuids: Set of entity UUIDs to query

        Returns:
            List of entity context dictionaries
        """
        if not entity_uuids:
            return []

        try:
            # Query Neo4j for entity nodes and their relationships
            # D7: scope to native RELATES_TO edges (r.type IS NOT NULL) so foreign
            # null-type edges (mem0/Graphiti legacy) are never read.
            query = """
            MATCH (e)
            WHERE e.uuid IN $uuids
            OPTIONAL MATCH (e)-[r:RELATES_TO]->(related)
            WHERE r.type IS NOT NULL
            RETURN e, collect({rel: type(r), target: related, context: r.context}) as relationships
            LIMIT 50
            """

            results = await self.neo4j_service.query(
                query, {"uuids": list(entity_uuids)}
            )

            context = []
            for record in results:
                entity = record["e"]
                relationships = record["relationships"]

                context.append(
                    {
                        "uuid": entity.get("uuid"),
                        "type": list(entity.labels)[0] if entity.labels else "Unknown",
                        "name": entity.get("name", entity.get("title", "Unknown")),
                        "properties": dict(entity),
                        "relationships": [
                            {
                                "type": rel["rel"],
                                "target": rel["target"].get("name", "Unknown")
                                if rel["target"]
                                else None,
                                # Per-relationship provenance — previously discarded (issue #97).
                                "context": rel.get("context")
                                if hasattr(rel, "get")
                                else None,
                            }
                            for rel in relationships
                            if rel["target"]
                        ],
                    }
                )

            return context

        except Exception as e:
            logger.error(f"Failed to query entity context: {e}")
            return []

    async def _get_related_facts(self, entity_uuids: set[str]) -> list[dict[str, Any]]:
        """
        Get facts (relationships) involving these entities.

        Args:
            entity_uuids: Entity UUIDs to find facts for

        Returns:
            List of facts as dictionaries
        """
        if not entity_uuids:
            return []

        try:
            # D7: scope to native RELATES_TO edges (r.type IS NOT NULL).
            query = """
            MATCH (e1)-[r:RELATES_TO]->(e2)
            WHERE r.type IS NOT NULL
              AND (e1.uuid IN $uuids OR e2.uuid IN $uuids)
              AND r.invalid_at IS NULL
              AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())
            RETURN e1.name as entity1, type(r) as relation, e2.name as entity2,
                   r.created_at as created_at, r.context as context
            ORDER BY r.created_at DESC
            LIMIT 20
            """

            results = await self.neo4j_service.query(
                query, {"uuids": list(entity_uuids)}
            )

            facts = []
            for record in results:
                facts.append(
                    {
                        "entity1": record["entity1"],
                        "relation": record["relation"],
                        "entity2": record["entity2"],
                        "created_at": record["created_at"],
                        "context": record.get("context", ""),
                    }
                )

            return facts

        except Exception as e:
            logger.error(f"Failed to get related facts: {e}")
            return []

    async def store_prompt_for_future_retrieval(
        self, prompt: str, prompt_id: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Store prompt embedding for future similarity searches.

        Args:
            prompt: User prompt text
            prompt_id: Unique identifier
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        try:
            return await self.vector_service.store_prompt_embedding(
                text=prompt, prompt_id=prompt_id, metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to store prompt for retrieval: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get retrieval statistics.

        Returns:
            Statistics dictionary
        """
        hit_rate = 0
        if self.stats["total_retrievals"] > 0:
            hit_rate = (
                self.stats["vector_hits"] / self.stats["total_retrievals"]
            ) * 100

        return {
            **self.stats,
            "hit_rate_percent": round(hit_rate, 2),
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of all retrieval cascade components.

        Returns:
            Health status dictionary
        """
        vector_health = await self.vector_service.health_check()

        return {
            "status": "healthy" if vector_health["status"] == "healthy" else "degraded",
            "vector_layer": vector_health,
            "stats": self.get_stats(),
        }


# Singleton instance
_context_retrieval_service: ContextRetrievalService | None = None


def get_context_retrieval_service() -> ContextRetrievalService:
    """Get or create singleton context retrieval service."""
    global _context_retrieval_service

    if _context_retrieval_service is None:
        _context_retrieval_service = ContextRetrievalService()

    assert _context_retrieval_service is not None
    return _context_retrieval_service

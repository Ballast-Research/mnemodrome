"""
Mnemodrome Code-First API

Direct Python interface for Claude Code following Anthropic's code-first pattern.
Achieves 98.7% token reduction vs MCP by loading only needed functionality on-demand.

Basic Usage:
    ```python
    from src.interfaces.code_first import MnemodromeAPI

    # Initialize API
    api = MnemodromeAPI()
    await api.initialize()

    # Load context
    context = await api.load_context("Ollama infrastructure", days=7)

    # Capture facts
    await api.capture_fact(
        "Ollama",
        "RUNS_AS",
        "user-level service",
        context="Needs S-drive access for models"
    )

    # Cleanup
    await api.close()
    ```

Async Context Manager Usage:
    ```python
    from src.interfaces.code_first import MnemodromeAPI

    async with MnemodromeAPI() as api:
        context = await api.load_context("current topic")
        await api.capture_fact("Entity1", "RELATES_TO", "Entity2")
    ```

Error Handling:
    ```python
    from src.interfaces.code_first import MnemodromeAPI

    api = MnemodromeAPI()
    try:
        await api.initialize()
        # ... use API ...
    except RuntimeError as e:
        print(f"Initialization failed: {e}")
    finally:
        await api.close()
    ```
"""

import asyncio
import time
from typing import Any
from uuid import uuid4

from loguru import logger

from src.config import REDIS_DB, REDIS_HOST, REDIS_PORT
from src.services.classification_service import classification_service
from src.services.context_formatter import format_context_markdown
from src.services.entity_factory import entity_factory
from src.services.file_service import file_service
from src.services.graph_factory import graph_factory
from src.services.keyword_extractor import extract_keywords
from src.services.neo4j_service import neo4j_service
from src.services.redis_service import redis_service
from src.services.relationship_factory import relationship_factory
from src.services.temporal_validity_service import temporal_validity_service


class MnemodromeAPI:
    """
    Code-first Python API for Mnemodrome dual-memory system.

    Thin wrapper over Mnemodrome's service layer providing direct Python access
    without MCP protocol overhead. Designed for token-efficient usage in
    Claude Code sessions.

    Token Efficiency:
        - MCP Protocol: ~150,000 tokens (all tools loaded upfront)
        - Code-First: ~2,000 tokens (98.7% reduction)

    Architecture:
        - Delegates to existing service layer (no business logic duplication)
        - Shares Neo4j and Redis connections with MCP server
        - Supports graceful degradation when Redis unavailable

    Lifecycle:
        1. Create instance: `api = MnemodromeAPI()`
        2. Initialize services: `await api.initialize()`
        3. Use methods: `await api.load_context(...)`
        4. Cleanup: `await api.close()`

    Attributes:
        _initialized: Tracks whether services have been initialized
    """

    def __init__(self) -> None:
        """
        Create MnemodromeAPI instance.

        Note: Services are NOT initialized until initialize() is called.
        This allows configuration override before connection.
        """
        self._initialized = False
        logger.debug("MnemodromeAPI instance created (not yet initialized)")

    async def initialize(
        self,
        redis_host: str = REDIS_HOST,
        redis_port: int = REDIS_PORT,
        redis_db: int = REDIS_DB,
        redis_max_connections: int = 10,
    ) -> None:
        """
        Initialize service connections (Neo4j, Redis).

        Connects to Neo4j (required) and Redis (optional with graceful degradation).
        Must be called before using any other methods.

        Args:
            redis_host: Redis host (default from config)
            redis_port: Redis port (default from config)
            redis_db: Redis database number (default from config)
            redis_max_connections: Redis connection pool size

        Raises:
            RuntimeError: If already initialized or Neo4j connection fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()
            ```

            With custom Redis configuration:
            ```python
            api = MnemodromeAPI()
            await api.initialize(redis_host="localhost", redis_port=6380)
            ```
        """
        if self._initialized:
            raise RuntimeError("MnemodromeAPI already initialized")

        logger.info("Initializing Mnemodrome services...")

        # Connect to Neo4j (required)
        logger.info("Connecting to Neo4j...")
        await neo4j_service.connect()

        # Connect to Redis with graceful degradation
        logger.info(f"Connecting to Redis at {redis_host}:{redis_port}...")
        try:
            await redis_service.connect(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                max_connections=redis_max_connections,
            )
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️  Redis unavailable (graceful degradation): {e}")

        # Verify Neo4j connectivity
        try:
            await neo4j_service.ping()
            logger.info("✅ Neo4j connectivity verified")
        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            # Cleanup and re-raise
            await self.close()
            raise RuntimeError(f"Neo4j connection failed: {e}") from e

        self._initialized = True
        logger.info("✅ Mnemodrome services initialized")

    async def close(self) -> None:
        """
        Close service connections and cleanup resources.

        Safe to call multiple times. Idempotent operation.

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()
            try:
                # ... use API ...
            finally:
                await api.close()
            ```
        """
        if not self._initialized:
            logger.debug("close() called but services not initialized")
            return

        logger.info("Closing Mnemodrome services...")

        # Close Redis connection (graceful)
        if redis_service.is_connected:
            try:
                await redis_service.disconnect()
                logger.debug("✅ Redis disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting Redis: {e}")

        # Close Neo4j connection (required)
        try:
            await neo4j_service.close()
            logger.debug("✅ Neo4j closed")
        except Exception as e:
            logger.warning(f"Error closing Neo4j: {e}")

        self._initialized = False
        logger.info("✅ Mnemodrome services closed")

    def _check_initialized(self) -> None:
        """
        Verify services are initialized before method calls.

        Raises:
            RuntimeError: If initialize() has not been called
        """
        if not self._initialized:
            raise RuntimeError(
                "MnemodromeAPI not initialized. Call await api.initialize() first."
            )

    async def load_context(
        self,
        prompt: str = "",
        days: int = 7,
        limit: int = 10,
    ) -> str:
        """
        Load recent context using temporal-first strategy.

        Extracts entities from prompt and retrieves recent related relationships
        from Mnemodrome's temporal graph memory. Uses entity-aware filtering when
        entities are found, otherwise falls back to keyword search.

        Args:
            prompt: Optional prompt to extract entities/keywords for filtering
            days: Number of days to look back for temporal context (default: 7)
            limit: Maximum number of relationships to return (default: 10)

        Returns:
            Formatted markdown with recent context from graph memory

        Raises:
            RuntimeError: If not initialized
            Exception: If context loading fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            # Load context about a topic
            context = await api.load_context("Ollama infrastructure", days=7)
            print(context)
            ```

            Without prompt (all recent context):
            ```python
            # Load all recent activity
            recent = await api.load_context(days=3, limit=20)
            ```
        """
        self._check_initialized()

        start_time = time.time()
        logger.info(
            f"Loading context (prompt={'provided' if prompt else 'none'}, "
            f"days={days}, limit={limit})"
        )

        # Extract entities from prompt using graph factory
        entity_uuids = []
        if prompt:
            try:
                # Extract entities with LLM
                extraction = await graph_factory._extract_with_llm(prompt)

                if extraction["entities"]:
                    logger.debug(
                        f"Extracted {len(extraction['entities'])} entities from prompt"
                    )

                    # T057: Batch classify all entities in single LLM call
                    entities_to_classify = [
                        {
                            "name": entity["name"],
                            "context": prompt,
                            "role": entity.get("role", "entity"),
                        }
                        for entity in extraction["entities"]
                    ]
                    entity_types = await classification_service.classify_batch(
                        entities_to_classify, context=prompt
                    )

                    # Resolve entities to UUIDs (deduplication-aware)
                    resolution_tasks = [
                        entity_factory.create_or_resolve(
                            name=entity["name"],
                            entity_type=entity_type,
                            summary=prompt[:200],
                        )
                        for entity, entity_type in zip(
                            extraction["entities"], entity_types, strict=False
                        )
                    ]
                    entity_uuids = await asyncio.gather(*resolution_tasks)

                    logger.debug(f"Resolved {len(entity_uuids)} entity UUIDs")

            except Exception as e:
                logger.warning(f"Entity extraction failed, using keyword fallback: {e}")

        # Fallback: Extract keyword if entity extraction failed
        keyword = None
        if not entity_uuids and prompt:
            keyword = extract_keywords(prompt)
            if keyword:
                logger.debug(f"Using keyword filter: {keyword}")

        # Query temporal memory
        try:
            if entity_uuids:
                # Entity-aware query
                results = await neo4j_service.temporal_query_by_entities(
                    entity_uuids=entity_uuids,
                    days=days,
                    limit=limit,
                )
            else:
                # Keyword-based query
                results = await neo4j_service.temporal_query(
                    keyword=keyword,
                    days=days,
                    limit=limit,
                )

            # Format as markdown
            context_markdown = format_context_markdown(results)

            query_time = (time.time() - start_time) * 1000
            logger.info(
                f"Context loaded in {query_time:.1f}ms ({len(results)} relationships)"
            )

            return context_markdown

        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            raise Exception(f"Context loading failed: {e}") from e

    async def capture_fact(
        self,
        entity1: str,
        relation: str,
        entity2: str,
        context: str = "",
        entity1_type: str | None = None,
        entity2_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Capture a factual relationship between two entities.

        Creates relationship in Neo4j graph and markdown documentation.
        Uses entity factory for deduplication (one canonical node per entity).
        Automatically classifies entity types if not provided.

        Args:
            entity1: Source entity name (e.g., "Mnemodrome", "Alice")
            relation: Relationship type (e.g., "IMPLEMENTS", "USES")
            entity2: Target entity name (e.g., "Neo4j", "FastMCP")
            context: Optional context/evidence for this fact
            entity1_type: Optional entity type override (Person, Project, Technology, etc.)
            entity2_type: Optional entity type override

        Returns:
            Dict with capture results:
            {
                "status": "complete" | "partial",
                "entity1_uuid": str,
                "entity2_uuid": str,
                "relationship": dict,
                "markdown_path": str | None,
                "duration_ms": float
            }

        Raises:
            RuntimeError: If not initialized
            Exception: If both storage layers fail

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            # Capture a fact
            result = await api.capture_fact(
                "Ollama",
                "RUNS_ON",
                "server-01",
                context="User-level service for S-drive access"
            )

            print(f"Captured: {result['entity1_uuid']} → {result['entity2_uuid']}")
            ```

            With type overrides:
            ```python
            result = await api.capture_fact(
                "Alice",
                "DEVELOPS",
                "Mnemodrome",
                entity1_type="Person",
                entity2_type="Project"
            )
            ```
        """
        self._check_initialized()

        start_time = time.time()
        logger.info(f"Capturing fact: {entity1} -{relation}-> {entity2}")

        # T057: Batch classify entities if types not provided
        if not entity1_type or not entity2_type:
            entities_to_classify = []
            entity_indices = {}

            if not entity1_type:
                entity_indices["entity1"] = len(entities_to_classify)
                entities_to_classify.append(
                    {"name": entity1, "role": "source", "context": context}
                )

            if not entity2_type:
                entity_indices["entity2"] = len(entities_to_classify)
                entities_to_classify.append(
                    {"name": entity2, "role": "target", "context": context}
                )

            # Batch classify all missing types
            classified_types = await classification_service.classify_batch(
                entities_to_classify, context=context
            )

            # Assign types back
            if "entity1" in entity_indices:
                entity1_type = classified_types[entity_indices["entity1"]]
                logger.debug(f"Entity1 classified: {entity1} → {entity1_type}")

            if "entity2" in entity_indices:
                entity2_type = classified_types[entity_indices["entity2"]]
                logger.debug(f"Entity2 classified: {entity2} → {entity2_type}")

        # Resolve entities to UUIDs (creates or retrieves existing)
        entity1_uuid = await entity_factory.create_or_resolve(
            name=entity1,
            entity_type=entity1_type,
            summary=context or f"{entity1_type}: {entity1}",
        )
        logger.debug(f"Entity1 resolved: {entity1} → {entity1_uuid}")

        entity2_uuid = await entity_factory.create_or_resolve(
            name=entity2,
            entity_type=entity2_type,
            summary=context or f"{entity2_type}: {entity2}",
        )
        logger.debug(f"Entity2 resolved: {entity2} → {entity2_uuid}")

        # Create relationship using UUIDs
        relationship_data = await relationship_factory.create(
            source_uuid=entity1_uuid,
            relation_type=relation,
            target_uuid=entity2_uuid,
            context=context,
        )
        logger.debug(
            f"Relationship created: {relationship_data['source_name']} "
            f"-{relationship_data['relation_type']}-> {relationship_data['target_name']}"
        )

        # Create markdown documentation
        markdown_content = f"""# Fact: {entity1} {relation} {entity2}

**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}

## Relationship
- **Source**: {entity1} ({entity1_type})
  - UUID: `{entity1_uuid}`
- **Relation**: {relation}
- **Target**: {entity2} ({entity2_type})
  - UUID: `{entity2_uuid}`

## Context
{context or "No additional context provided"}

## Technical Details
- Entity resolution via EntityFactory (prevents duplicates)
- Automatic classification via ClassificationService
- Temporal relationship via RelationshipFactory

---
*Captured via Mnemodrome Code-First API*
"""

        # Write markdown file
        markdown_path = None
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"fact-{entity1.lower().replace(' ', '-')}-{timestamp}.md"
            markdown_path = file_service.write_file(filename, markdown_content)
            logger.info(f"✅ Markdown documentation: {markdown_path}")
            markdown_success = True
        except Exception as e:
            logger.warning(f"Markdown write failed: {e}")
            markdown_success = False

        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Fact captured in {duration_ms:.1f}ms")

        # Return structured result
        return {
            "status": "complete" if markdown_success else "partial",
            "entity1_uuid": entity1_uuid,
            "entity1_type": entity1_type,
            "entity2_uuid": entity2_uuid,
            "entity2_type": entity2_type,
            "relationship": relationship_data,
            "markdown_path": str(markdown_path) if markdown_path else None,
            "duration_ms": duration_ms,
        }

    async def capture_context(self, context: str) -> dict[str, Any]:
        """
        Extract and capture multi-entity graph from rich natural language.

        Automatically extracts entities, relationships, and creates complete subgraph.
        Perfect for complex scenarios with multiple entities and relationships.

        Args:
            context: Natural language description with multiple entities and relationships

        Returns:
            Dict with extraction results:
            {
                "entities_created": int,
                "relationships_created": int,
                "entity_uuids": dict[str, str],
                "extraction_time_ms": float
            }

        Raises:
            RuntimeError: If not initialized
            Exception: If graph creation fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_context(
                "Alice and Bob collaborated with Charlie to implement Phase 4a "
                "using FastMCP and Neo4j, which resolved the entity duplication bug"
            )

            print(f"Created {result['entities_created']} entities")
            ```
        """
        self._check_initialized()

        start_time = time.time()
        logger.info(f"Capturing graph from context: {context[:100]}...")

        # Delegate to graph factory
        result = await graph_factory.build_from_extraction(context)

        extraction_time = (time.time() - start_time) * 1000
        logger.info(
            f"Graph captured in {extraction_time:.1f}ms: "
            f"{result['entities_created']} entities, "
            f"{result['relationships_created']} relationships"
        )

        return {
            "entities_created": result["entities_created"],
            "relationships_created": result["relationships_created"],
            "entity_uuids": result["entity_uuids"],
            "extraction_time_ms": extraction_time,
        }

    async def capture_learning(
        self,
        title: str,
        content: str,
        category: str = "general",
    ) -> dict[str, Any]:
        """
        Capture structured learning or insight in dual memory layers.

        Creates Learning node in Neo4j for relationship discovery and markdown
        file in storage vault for human review and Obsidian integration.

        Args:
            title: Learning title (brief, descriptive)
            content: Full learning content (markdown supported)
            category: Learning category (e.g., insight, solution, pattern, breakthrough)

        Returns:
            Dict with capture results:
            {
                "uuid": str,
                "file_path": str,
                "duration_ms": float
            }

        Raises:
            RuntimeError: If not initialized
            Exception: If capture fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_learning(
                title="Factory Pattern Eliminates Entity Duplication",
                content="Using Gang of Four Factory Pattern...",
                category="breakthrough"
            )
            ```
        """
        self._check_initialized()

        start_time = time.time()
        logger.info(f"Capturing learning: {title} (category: {category})")

        # Generate UUID
        learning_uuid = str(uuid4())

        # Create learning node in Neo4j
        cypher = """
        CREATE (l:Learning {
            uuid: $uuid,
            title: $title,
            content: $content,
            category: $category,
            created_at: datetime(),
            last_accessed: datetime()
        })
        RETURN l.uuid as uuid
        """

        result = await neo4j_service.query(
            cypher,
            {
                "uuid": learning_uuid,
                "title": title,
                "content": content,
                "category": category,
            },
        )

        if not result:
            raise Exception("Failed to create learning node in Neo4j")

        # Create markdown file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_title = title.lower().replace(" ", "-").replace("/", "-")[:50]
        filename = f"learning-{safe_title}-{timestamp}.md"

        markdown_content = f"""# Learning: {title}

**Category**: {category}
**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**UUID**: `{learning_uuid}`

---

{content}

---

*Captured via Mnemodrome Code-First API*
"""

        try:
            file_path = file_service.write_file(filename, markdown_content)
        except Exception as e:
            # Cleanup Neo4j node if file write fails
            await neo4j_service.query(
                "MATCH (l:Learning {uuid: $uuid}) DELETE l", {"uuid": learning_uuid}
            )
            raise Exception(f"Failed to write learning file: {e}") from e

        capture_time = (time.time() - start_time) * 1000
        logger.info(f"Learning captured in {capture_time:.1f}ms")

        return {
            "uuid": learning_uuid,
            "file_path": str(file_path),
            "duration_ms": capture_time,
        }

    async def capture_session(
        self,
        conversation_summary: str,
        key_insights: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Capture end-of-session knowledge with graph integration.

        Creates Session node in Neo4j linked to key entities discussed.
        Captures high-level summary while maintaining connections to specifics.

        Args:
            conversation_summary: High-level summary of what was accomplished
            key_insights: Optional list of key insights/takeaways

        Returns:
            Dict with capture results:
            {
                "uuid": str,
                "file_path": str,
                "entity_count": int,
                "duration_ms": float
            }

        Raises:
            RuntimeError: If not initialized
            Exception: If capture fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_session(
                conversation_summary="Implemented Phase 4.5 factory pattern",
                key_insights=[
                    "Gang of Four pattern eliminated duplicates",
                    "UUID-based resolution ensures single canonical node"
                ]
            )
            ```
        """
        self._check_initialized()

        start_time = time.time()
        logger.info("Capturing session knowledge...")

        # Generate UUID
        session_uuid = str(uuid4())

        # Create session node
        cypher = """
        CREATE (s:Session {
            uuid: $uuid,
            summary: $summary,
            key_insights: $insights,
            created_at: datetime()
        })
        RETURN s.uuid as uuid
        """

        result = await neo4j_service.query(
            cypher,
            {
                "uuid": session_uuid,
                "summary": conversation_summary,
                "insights": key_insights or [],
            },
        )

        if not result:
            raise Exception("Failed to create session node in Neo4j")

        # Create markdown file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"session-{timestamp}.md"

        insights_md = "\n".join([f"- {insight}" for insight in (key_insights or [])])

        markdown_content = f"""# Session Summary

**Captured**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**UUID**: `{session_uuid}`

## Summary

{conversation_summary}

## Key Insights

{insights_md if insights_md else "*No insights captured*"}

---

*Captured via Mnemodrome Code-First API*
"""

        try:
            file_path = file_service.write_file(filename, markdown_content)
        except Exception as e:
            # Cleanup Neo4j node if file write fails
            await neo4j_service.query(
                "MATCH (s:Session {uuid: $uuid}) DELETE s", {"uuid": session_uuid}
            )
            raise Exception(f"Failed to write session file: {e}") from e

        capture_time = (time.time() - start_time) * 1000
        logger.info(f"Session captured in {capture_time:.1f}ms")

        return {
            "uuid": session_uuid,
            "file_path": str(file_path),
            "entity_count": 0,  # TODO: Extract and link entities
            "duration_ms": capture_time,
        }

    async def invalidate_fact(
        self,
        entity1: str,
        relation: str,
        entity2: str,
        reason: str,
        invalidation_type: str = "both",
    ) -> dict[str, Any]:
        """
        Mark a fact as no longer valid (temporal validity).

        Non-destructive operation that preserves historical record.

        Args:
            entity1: Source entity name
            relation: Relationship type
            entity2: Target entity name
            reason: Human-readable reason for invalidation
            invalidation_type: Type of invalidation ("natural", "system", or "both")

        Returns:
            Dict with invalidation results:
            {
                "status": "success",
                "fact_invalidated": bool,
                "reason": str
            }

        Raises:
            RuntimeError: If not initialized
            Exception: If invalidation fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            result = await api.invalidate_fact(
                "Mnemodrome",
                "RUNS_ON",
                "server-02",
                reason="Migrated to server-01",
                invalidation_type="natural"
            )
            ```
        """
        self._check_initialized()

        logger.info(f"Invalidating fact: {entity1} -{relation}-> {entity2}")

        # D8 (FR-002, FR-012): route through the hardened temporal_validity_service
        # which implements the D3 cardinality guard and D7 native-scope.
        # First find the relationship uuid (hardened find_relationship uses D4 provenance
        # and D7 scope).
        old_rel = await neo4j_service.find_relationship(
            entity1=entity1,
            relation=relation,
            entity2=entity2,
        )

        if not old_rel:
            logger.info(
                f"No active native fact found: {entity1} -{relation}-> {entity2}"
            )
            return {
                "status": "not_found",
                "fact_invalidated": False,
                "reason": reason,
                "relationships_updated": 0,
            }

        try:
            result = await temporal_validity_service.invalidate_relationship(
                neo4j_service=neo4j_service,
                relationship_uuid=old_rel["uuid"],
                invalidation_type=invalidation_type,
                reason=reason,
            )
            logger.info(f"Invalidated 1 relationship (uuid={old_rel['uuid']})")
            return {
                "status": "success",
                "fact_invalidated": True,
                "reason": reason,
                "relationships_updated": 1,
                "fact_uuid": result.get("fact_uuid"),
            }
        except ValueError as e:
            logger.error(f"Invalidation blocked (D3): {e}")
            return {
                "status": "error",
                "fact_invalidated": False,
                "reason": reason,
                "error": str(e),
                "relationships_updated": 0,
            }

    async def update_fact(
        self,
        entity1: str,
        relation: str,
        entity2_old: str,
        entity2_new: str,
        context: str = "",
        conflict_resolution: str = "prompt",
    ) -> dict[str, Any]:
        """
        Update a fact by atomically superseding it with a new fact.

        Creates supersession chain: old fact invalidated, new fact created, linked.

        Args:
            entity1: Source entity name
            relation: Relationship type
            entity2_old: Old target entity (being replaced)
            entity2_new: New target entity (replacement)
            context: Optional context for the new fact
            conflict_resolution: Conflict handling ("prompt", "auto", or "error")

        Returns:
            Dict with update results:
            {
                "status": "success",
                "invalidated_fact_uuid": str,
                "new_fact_uuid": str
            }

        Raises:
            RuntimeError: If not initialized
            Exception: If update fails

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            result = await api.update_fact(
                "Mnemodrome",
                "RUNS_ON",
                entity2_old="server-02",
                entity2_new="server-01",
                context="Infrastructure migration"
            )
            ```
        """
        self._check_initialized()

        logger.info(
            f"Updating fact: {entity1} -{relation}-> {entity2_old} to {entity2_new}"
        )

        # D8 (FR-003, FR-005): route through hardened atomic supersession.
        # First locate the old fact.
        old_rel = await neo4j_service.find_relationship(
            entity1=entity1, relation=relation, entity2=entity2_old
        )
        if not old_rel:
            logger.warning(f"Old fact not found: {entity1} -{relation}-> {entity2_old}")
            return {
                "status": "not_found",
                "invalidated_fact": False,
                "new_fact_uuid": None,
                "new_relationship": None,
            }

        # Resolve entity UUIDs for the atomic update.
        # entity1_uuid and entity2_uuid (old target) come from find_relationship return
        # so we pass them distinctly (D2: old and new target UUIDs passed separately).
        source_uuid = old_rel.get("entity1_uuid")
        if not source_uuid:
            source_uuid = await entity_factory.create_or_resolve(entity1)

        entity2_old_uuid = old_rel.get("entity2_uuid")
        target_uuid = await entity_factory.create_or_resolve(entity2_new)

        old_uuid = old_rel["uuid"]
        new_uuid = str(uuid4())

        try:
            tx_result = await neo4j_service.execute_update_transaction(
                old_fact_uuid=old_uuid,
                entity1_uuid=source_uuid,
                relation_type=relation,
                entity2_uuid=target_uuid,
                context=context,
                new_fact_uuid=new_uuid,
                entity2_old_uuid=entity2_old_uuid,
            )
        except Exception as e:
            logger.error(f"D8 atomic update failed: {e}")
            raise

        # D2b: embed new fact in vector store (best-effort)
        try:
            from src.services.vector_service import get_vector_service

            vs = get_vector_service()
            if vs and vs.is_enabled():
                await vs.store_fact_embedding(
                    entity1=entity1,
                    relation=relation,
                    entity2=entity2_new,
                    context=context,
                    fact_uuid=new_uuid,
                    metadata={"operation": "code_first_update", "supersedes": old_uuid},
                )
        except Exception as embed_err:
            logger.warning(
                f"D2b vector embedding after code_first.update_fact failed: {embed_err}"
            )

        logger.info("Fact updated successfully (D8 atomic)")

        return {
            "status": "success",
            "invalidated_fact": True,
            "invalidated_fact_uuid": tx_result["old_fact_uuid"],
            "new_fact_uuid": tx_result["new_fact_uuid"],
            "supersession_chain": tx_result["supersession_chain"],
            "new_relationship": {"uuid": new_uuid, "type": relation},
        }

    async def health(self) -> dict[str, Any]:
        """
        Health check endpoint for monitoring and orchestration.

        Returns:
            Dict with health status:
            {
                "status": "healthy" | "degraded",
                "neo4j": "connected" | "disconnected",
                "redis": "connected" | "disconnected",
                "timestamp": float
            }

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            status = await api.health()
            print(f"Status: {status['status']}")
            ```
        """
        # Check Neo4j connectivity
        neo4j_status = "disconnected"
        try:
            await neo4j_service.ping()
            neo4j_status = "connected"
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")

        # Check Redis connectivity
        redis_status = "disconnected"
        if redis_service.is_connected:
            redis_status = "connected"

        # Build health response
        return {
            "status": "healthy" if neo4j_status == "connected" else "degraded",
            "neo4j": neo4j_status,
            "redis": redis_status,
            "timestamp": time.time(),
        }

    async def discover_schema(self) -> dict[str, Any]:
        """
        Discover actual Neo4j schema (source of truth).

        Queries production Neo4j to discover relationship and entity properties.

        Returns:
            Dict with schema information:
            {
                "sample_relationships": list[dict],
                "sample_entities": list[dict]
            }

        Raises:
            RuntimeError: If not initialized

        Example:
            ```python
            api = MnemodromeAPI()
            await api.initialize()

            schema = await api.discover_schema()
            print(f"Found {len(schema['sample_entities'])} sample entities")
            ```
        """
        self._check_initialized()

        logger.info("Discovering Neo4j schema...")

        # D7: scope to native edges only — never read foreign null-type edges.
        rel_query = """
        MATCH ()-[r:RELATES_TO]->()
        WHERE r.type IS NOT NULL
        RETURN r LIMIT 5
        """
        sample_rels = await neo4j_service.query(rel_query)

        # Get sample entities
        entity_query = """
        MATCH (e)
        RETURN e LIMIT 5
        """
        sample_entities = await neo4j_service.query(entity_query)

        logger.info(
            f"Schema discovered: {len(sample_rels)} sample relationships, "
            f"{len(sample_entities)} sample entities"
        )

        return {
            "sample_relationships": sample_rels,
            "sample_entities": sample_entities,
        }

    async def __aenter__(self) -> "MnemodromeAPI":
        """Async context manager entry - initialize services."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - cleanup services."""
        await self.close()

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "initialized" if self._initialized else "not initialized"
        return f"<MnemodromeAPI {status}>"

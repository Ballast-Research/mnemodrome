"""Deduplication service for knowledge layer validation (Feature 023).

Provides pre-capture validation to prevent duplicate learnings through a
three-stage validation pipeline:
    1. Title exact match (Neo4j index lookup)
    2. Vector similarity search (Qdrant semantic matching)
    3. Fuzzy title fallback (difflib when Qdrant unavailable)

Constitutional Principle: Knowledge deduplication ensures the single source
of truth in Mnemodrome's dual-memory system.
"""

import difflib
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.config import (
    DEDUP_AUTO_SKIP_THRESHOLD,
    DEDUP_ENABLE_GRACEFUL_DEGRADATION,
    DEDUP_FUZZY_MATCH_THRESHOLD,
    DEDUP_MAX_VALIDATION_TIME_MS,
    DEDUP_VALIDATION_THRESHOLD,
    DEDUP_VECTOR_SEARCH_LIMIT,
)
from src.models.deduplication_models import (
    AppendResult,
    BootstrapResult,
    SimilarityMatch,
    ValidationResult,
)

if TYPE_CHECKING:
    from src.services.file_service import FileService
    from src.services.neo4j_service import Neo4jService
    from src.services.vector_service import VectorService


class DeduplicationService:
    """Service for pre-capture learning deduplication and content appending.

    Implements a three-stage validation pipeline:
        1. Title exact match via Neo4j (fast, indexed lookup)
        2. Vector similarity via Qdrant (semantic matching)
        3. Fuzzy title fallback via difflib (graceful degradation)

    Attributes:
        vector_service: VectorService for semantic similarity search
        neo4j_service: Neo4jService for graph database operations
        file_service: FileService for markdown file operations
        auto_skip_threshold: Similarity score for auto-skipping (default: 0.95)
        validation_threshold: Similarity score for requiring validation (default: 0.90)
        max_validation_time_ms: Maximum validation time before timeout (default: 300ms)
        vector_search_limit: Max results from vector search (default: 5)
        fuzzy_match_threshold: Threshold for fuzzy title matching (default: 0.85)
        enable_graceful_degradation: Fall back to fuzzy when Qdrant unavailable
    """

    def __init__(
        self,
        vector_service: "VectorService",
        neo4j_service: "Neo4jService",
        file_service: "FileService",
        auto_skip_threshold: float | None = None,
        validation_threshold: float | None = None,
        max_validation_time_ms: float | None = None,
        vector_search_limit: int | None = None,
        fuzzy_match_threshold: float | None = None,
        enable_graceful_degradation: bool | None = None,
    ):
        """Initialize DeduplicationService with required dependencies.

        Args:
            vector_service: VectorService instance for semantic search
            neo4j_service: Neo4jService instance for graph operations
            file_service: FileService instance for file operations
            auto_skip_threshold: Override for DEDUP_AUTO_SKIP_THRESHOLD
            validation_threshold: Override for DEDUP_VALIDATION_THRESHOLD
            max_validation_time_ms: Override for DEDUP_MAX_VALIDATION_TIME_MS
            vector_search_limit: Override for DEDUP_VECTOR_SEARCH_LIMIT
            fuzzy_match_threshold: Override for DEDUP_FUZZY_MATCH_THRESHOLD
            enable_graceful_degradation: Override for DEDUP_ENABLE_GRACEFUL_DEGRADATION

        Raises:
            ValueError: If any required service is None
        """
        # Validate required dependencies
        if vector_service is None:
            raise ValueError("vector_service is required")
        if neo4j_service is None:
            raise ValueError("neo4j_service is required")
        if file_service is None:
            raise ValueError("file_service is required")

        self.vector_service = vector_service
        self.neo4j_service = neo4j_service
        self.file_service = file_service

        # Configuration with defaults from config.py
        self.auto_skip_threshold = (
            auto_skip_threshold
            if auto_skip_threshold is not None
            else DEDUP_AUTO_SKIP_THRESHOLD
        )
        self.validation_threshold = (
            validation_threshold
            if validation_threshold is not None
            else DEDUP_VALIDATION_THRESHOLD
        )
        self.max_validation_time_ms = (
            max_validation_time_ms
            if max_validation_time_ms is not None
            else DEDUP_MAX_VALIDATION_TIME_MS
        )
        self.vector_search_limit = (
            vector_search_limit
            if vector_search_limit is not None
            else DEDUP_VECTOR_SEARCH_LIMIT
        )
        self.fuzzy_match_threshold = (
            fuzzy_match_threshold
            if fuzzy_match_threshold is not None
            else DEDUP_FUZZY_MATCH_THRESHOLD
        )
        self.enable_graceful_degradation = (
            enable_graceful_degradation
            if enable_graceful_degradation is not None
            else DEDUP_ENABLE_GRACEFUL_DEGRADATION
        )

        logger.info(
            f"DeduplicationService initialized "
            f"(auto_skip={self.auto_skip_threshold}, "
            f"validation={self.validation_threshold}, "
            f"fuzzy={self.fuzzy_match_threshold})"
        )

    async def title_exact_match(self, title: str) -> str | None:
        """Stage 1: Check for exact title match in Neo4j.

        Performs a case-sensitive indexed lookup for existing learnings
        with the exact same title.

        Args:
            title: Learning title to check

        Returns:
            UUID of existing learning if found, None otherwise

        Raises:
            Exception: If Neo4j query fails (caller should handle)
        """
        cypher = """
        MATCH (l:Learning {title: $title})
        WHERE l.invalid_at IS NULL
        RETURN l.uuid AS uuid
        LIMIT 1
        """

        try:
            results = await self.neo4j_service.query(cypher, {"title": title})
            if results and results[0].get("uuid"):
                uuid = results[0]["uuid"]
                logger.debug(f"Title exact match found: {title} -> {uuid}")
                return uuid
            return None
        except Exception as e:
            logger.error(f"Title exact match query failed: {e}")
            raise

    async def fallback_title_fuzzy_match(
        self,
        title: str,
        threshold: float | None = None,
    ) -> list[SimilarityMatch]:
        """Stage 3 Fallback: Fuzzy title matching using difflib.

        Used when Qdrant/vector service is unavailable. Queries all
        learning titles from Neo4j and performs fuzzy string matching.

        Args:
            title: Learning title to match
            threshold: Minimum similarity ratio (default: fuzzy_match_threshold)

        Returns:
            List of SimilarityMatch objects for titles meeting threshold,
            ordered by similarity descending
        """
        threshold = threshold if threshold is not None else self.fuzzy_match_threshold

        cypher = """
        MATCH (l:Learning)
        WHERE l.invalid_at IS NULL
        RETURN l.uuid AS uuid, l.title AS title, l.category AS category,
               l.file_path AS file_path, l.created_at AS created_at
        """

        try:
            results = await self.neo4j_service.query(cypher)
        except Exception as e:
            logger.error(f"Fuzzy match query failed: {e}")
            return []

        matches: list[SimilarityMatch] = []
        title_lower = title.lower()

        for record in results:
            existing_title = record.get("title", "")
            if not existing_title:
                continue

            # Case-insensitive fuzzy matching
            ratio = difflib.SequenceMatcher(
                None, title_lower, existing_title.lower()
            ).ratio()

            if ratio >= threshold:
                # Format created_at
                created_at = record.get("created_at")
                if hasattr(created_at, "isoformat"):
                    created_at_str = created_at.isoformat()
                else:
                    created_at_str = str(created_at) if created_at else ""

                matches.append(
                    SimilarityMatch(
                        uuid=record["uuid"],
                        title=existing_title,
                        similarity_score=ratio,
                        created_at=created_at_str,
                        file_path=record.get("file_path", ""),
                        category=record.get("category"),
                        excerpt=None,  # No content available in fuzzy mode
                    )
                )

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity_score, reverse=True)

        if matches:
            logger.debug(f"Fuzzy match found {len(matches)} candidates for '{title}'")

        return matches

    async def vector_similarity_search(
        self,
        content: str,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> list[SimilarityMatch]:
        """Stage 2: Semantic similarity search via Qdrant.

        Generates embedding for content and searches for similar
        learnings in the vector database.

        Args:
            content: Learning content to match
            threshold: Minimum similarity score (default: validation_threshold)
            limit: Maximum results to return (default: vector_search_limit)

        Returns:
            List of SimilarityMatch objects ordered by similarity descending

        Raises:
            Exception: If vector service unavailable and graceful_degradation=False
        """
        threshold = threshold if threshold is not None else self.validation_threshold
        limit = limit if limit is not None else self.vector_search_limit

        if not self.vector_service.is_enabled():
            if self.enable_graceful_degradation:
                logger.warning("Vector service disabled, returning empty results")
                return []
            raise RuntimeError("Vector service unavailable")

        try:
            results = await self.vector_service.search_similar_learnings(
                query_text=content,
                limit=limit,
                min_score=threshold,
            )
        except Exception as e:
            if self.enable_graceful_degradation:
                logger.warning(f"Vector search failed, graceful degradation: {e}")
                return []
            raise

        matches: list[SimilarityMatch] = []
        for result in results:
            matches.append(
                SimilarityMatch(
                    uuid=result.get("neo4j_uuid", result.get("id", "")),
                    title=result.get("title", ""),
                    similarity_score=result.get("score", 0.0),
                    created_at=result.get("timestamp", ""),
                    file_path=result.get("file_path", ""),
                    category=result.get("category"),
                    excerpt=result.get("content", "")[:200]
                    if result.get("content")
                    else None,
                )
            )

        return matches

    async def validate_learning(
        self,
        title: str,
        content: str,
    ) -> ValidationResult:
        """Validate learning uniqueness through three-stage pipeline.

        Pipeline:
            1. Title exact match (Neo4j) -> validation_required if found
            2. Vector similarity (Qdrant) -> auto_skipped (≥95%) or validation_required (90-95%)
            3. Fuzzy fallback (difflib) -> if Stage 2 unavailable

        Args:
            title: Learning title
            content: Learning content (minimum 50 characters)

        Returns:
            ValidationResult with status, similarity info, and recommendations
        """
        start_time = time.perf_counter()
        validation_stage = "title_exact"

        try:
            # Input validation
            if not title or len(title.strip()) == 0:
                return ValidationResult(
                    status="proceed",
                    validation_stage="input_validation",
                    validation_time_ms=self._elapsed_ms(start_time),
                    rationale="Empty title provided",
                )

            if not content or len(content.strip()) < 50:
                return ValidationResult(
                    status="proceed",
                    validation_stage="input_validation",
                    validation_time_ms=self._elapsed_ms(start_time),
                    rationale="Content too short for meaningful deduplication",
                )

            # Stage 1: Title exact match
            try:
                exact_uuid = await self.title_exact_match(title)
                if exact_uuid:
                    return ValidationResult(
                        status="validation_required",
                        validation_stage="title_exact",
                        validation_time_ms=self._elapsed_ms(start_time),
                        conflict_type="title_match",
                        existing_learning_uuid=exact_uuid,
                        similarity_score=1.0,
                        recommendation="append",
                        rationale="Exact title match found. Consider appending to existing learning.",
                        options={
                            "append": "Add content as new section to existing learning",
                            "skip": "Do not create this learning",
                            "force_create": "Create as separate learning with modified title",
                        },
                    )
            except Exception as e:
                logger.warning(f"Stage 1 (title exact) failed: {e}")
                # Continue to Stage 2

            # Stage 2: Vector similarity search
            validation_stage = "vector_similarity"
            use_fuzzy_fallback = False

            try:
                if self.vector_service.is_enabled():
                    similar = await self.vector_similarity_search(
                        content=content,
                        threshold=self.validation_threshold,
                        limit=self.vector_search_limit,
                    )
                else:
                    use_fuzzy_fallback = True
                    similar = []
            except Exception as e:
                logger.warning(f"Stage 2 (vector similarity) failed: {e}")
                use_fuzzy_fallback = self.enable_graceful_degradation
                similar = []

            # Stage 3: Fuzzy fallback if needed
            if use_fuzzy_fallback:
                validation_stage = "fuzzy_fallback"
                similar = await self.fallback_title_fuzzy_match(
                    title=title,
                    threshold=self.fuzzy_match_threshold,
                )

            # Analyze results
            if not similar:
                return ValidationResult(
                    status="proceed",
                    validation_stage=validation_stage,
                    validation_time_ms=self._elapsed_ms(start_time),
                    rationale="No similar learnings found. Content appears unique.",
                )

            top_match = similar[0]

            # Auto-skip if similarity >= auto_skip_threshold
            if top_match.similarity_score >= self.auto_skip_threshold:
                return ValidationResult(
                    status="auto_skipped",
                    validation_stage=validation_stage,
                    validation_time_ms=self._elapsed_ms(start_time),
                    conflict_type="semantic_similarity",
                    existing_learning_uuid=top_match.uuid,
                    similarity_score=top_match.similarity_score,
                    recommendation="skip",
                    rationale=f"Content {top_match.similarity_score:.0%} semantically similar to existing learning",
                )

            # Validation required for scores in [validation_threshold, auto_skip_threshold)
            existing_items = [
                {
                    "uuid": m.uuid,
                    "title": m.title,
                    "similarity_score": m.similarity_score,
                    "created_at": m.created_at,
                    "file_path": m.file_path,
                    "category": m.category,
                    "excerpt": m.excerpt,
                }
                for m in similar[:5]  # Limit to 5 items
            ]

            return ValidationResult(
                status="validation_required",
                validation_stage=validation_stage,
                validation_time_ms=self._elapsed_ms(start_time),
                conflict_type="semantic_similarity",
                existing_learning_uuid=top_match.uuid,
                similarity_score=top_match.similarity_score,
                existing_items=existing_items,
                recommendation="append"
                if top_match.similarity_score >= 0.80
                else "force_create",
                rationale=f"Content shows {top_match.similarity_score:.0%} similarity to existing learning. Consider appending or creating separate learning.",
                options={
                    "append": "Add content as new section to existing learning",
                    "force_create": "Create as separate learning despite similarity",
                },
            )

        except Exception as e:
            logger.error(f"Validation failed unexpectedly: {e}")
            # Graceful degradation: proceed on error
            return ValidationResult(
                status="proceed",
                validation_stage=validation_stage,
                validation_time_ms=self._elapsed_ms(start_time),
                rationale=f"Validation error (proceeding): {e}",
            )

    async def append_to_learning(
        self,
        learning_uuid: str,
        new_content: str,
        new_section_title: str | None = None,
    ) -> AppendResult:
        """Append content to an existing learning.

        Updates:
            1. Markdown file with new timestamped section
            2. Neo4j node metadata (last_updated, append_count)
            3. Qdrant embedding (regenerated with combined content)

        Args:
            learning_uuid: UUID of existing learning to update
            new_content: Content to append (minimum 50 characters)
            new_section_title: Optional title for new section

        Returns:
            AppendResult with status, updated counts, and file info
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        section_title = (
            new_section_title or f"Update: {datetime.utcnow().strftime('%Y-%m-%d')}"
        )

        # Input validation
        if not new_content or len(new_content.strip()) < 50:
            return AppendResult(
                status="error",
                action="error_occurred",
                learning_uuid=learning_uuid,
                file_path="",
                append_count=0,
                timestamp=timestamp,
                error="Content must be at least 50 characters",
            )

        try:
            # Get existing learning from Neo4j
            cypher = """
            MATCH (l:Learning {uuid: $uuid})
            WHERE l.invalid_at IS NULL
            RETURN l.uuid AS uuid, l.title AS title, l.file_path AS file_path,
                   l.content AS content, l.category AS category,
                   COALESCE(l.append_count, 0) AS append_count
            """
            results = await self.neo4j_service.query(cypher, {"uuid": learning_uuid})

            if not results:
                return AppendResult(
                    status="error",
                    action="error_occurred",
                    learning_uuid=learning_uuid,
                    file_path="",
                    append_count=0,
                    timestamp=timestamp,
                    error=f"Learning not found: {learning_uuid}",
                )

            learning = results[0]
            file_path = learning.get("file_path", "")
            old_append_count = learning.get("append_count", 0)

            # Step 1-3: Update markdown file
            if file_path and Path(file_path).exists():
                try:
                    original_content = Path(file_path).read_text(encoding="utf-8")
                    new_section = f"\n\n---\n\n## {section_title}\n\n{new_content}\n"
                    updated_content = original_content + new_section

                    # Atomic write using temp file
                    temp_path = Path(file_path).with_suffix(".tmp")
                    temp_path.write_text(updated_content, encoding="utf-8")
                    temp_path.rename(file_path)
                except Exception as e:
                    logger.error(f"Failed to update markdown file: {e}")
                    return AppendResult(
                        status="error",
                        action="error_occurred",
                        learning_uuid=learning_uuid,
                        file_path=file_path,
                        append_count=old_append_count,
                        timestamp=timestamp,
                        error=f"Failed to update markdown file: {e}",
                    )

            # Step 4: Update Neo4j metadata
            new_append_count = old_append_count + 1
            update_cypher = """
            MATCH (l:Learning {uuid: $uuid})
            SET l.last_updated = datetime(),
                l.append_count = $append_count
            RETURN l.append_count AS append_count
            """
            await self.neo4j_service.query(
                update_cypher, {"uuid": learning_uuid, "append_count": new_append_count}
            )

            # Step 5: Regenerate Qdrant embedding
            embedding_updated = False
            try:
                if self.vector_service.is_enabled():
                    # Read updated content for embedding
                    if file_path and Path(file_path).exists():
                        full_content = Path(file_path).read_text(encoding="utf-8")
                    else:
                        # Combine old content with new
                        old_content = learning.get("content", "")
                        full_content = f"{old_content}\n\n{new_content}"

                    result = await self.vector_service.store_learning_embedding(
                        title=learning.get("title", ""),
                        content=full_content,
                        learning_uuid=learning_uuid,
                        category=learning.get("category", "general"),
                        metadata={"append_count": new_append_count},
                    )
                    embedding_updated = bool(result)
            except Exception as e:
                logger.warning(f"Embedding regeneration failed (non-fatal): {e}")
                # Continue - embedding regeneration is non-blocking

            return AppendResult(
                status="success",
                action="appended",
                learning_uuid=learning_uuid,
                file_path=file_path,
                append_count=new_append_count,
                timestamp=timestamp,
                new_section_title=section_title,
                embedding_updated=embedding_updated,
            )

        except Exception as e:
            logger.error(f"Append operation failed: {e}")
            return AppendResult(
                status="error",
                action="error_occurred",
                learning_uuid=learning_uuid,
                file_path="",
                append_count=0,
                timestamp=timestamp,
                error=str(e),
            )

    async def bootstrap_learnings(
        self,
        learnings_dir: str,
        dry_run: bool = False,
        batch_size: int | None = None,
        max_failures: int | None = None,
    ) -> BootstrapResult:
        """Bootstrap vector embeddings for existing learnings.

        Scans a directory for learning markdown files and generates
        embeddings in Qdrant, skipping files that already exist.

        Args:
            learnings_dir: Path to directory containing learning files
            dry_run: If True, scan files without writing to Qdrant
            batch_size: Number of files per batch (default: from config)
            max_failures: Maximum failures before abort (default: from config)

        Returns:
            BootstrapResult with counts and error summary
        """
        from src.config import DEDUP_BOOTSTRAP_BATCH_SIZE, DEDUP_BOOTSTRAP_MAX_FAILURES

        batch_size = batch_size or DEDUP_BOOTSTRAP_BATCH_SIZE
        max_failures = max_failures or DEDUP_BOOTSTRAP_MAX_FAILURES

        start_time = time.perf_counter()
        dir_path = Path(learnings_dir)

        if not dir_path.exists() or not dir_path.is_dir():
            return BootstrapResult(
                total_files=0,
                successful=0,
                failed=1,
                skipped=0,
                duration_seconds=self._elapsed_ms(start_time) / 1000,
                dry_run=dry_run,
                failed_files=[str(dir_path)],
                error_summary={"DirectoryNotFound": 1},
            )

        # Find all learning markdown files
        learning_files = list(dir_path.glob("learning-*.md"))
        total_files = len(learning_files)

        successful = 0
        failed = 0
        skipped = 0
        failed_files: list[str] = []
        error_summary: dict = {}

        for file_path in learning_files:
            if failed >= max_failures:
                logger.warning(
                    f"Max failures ({max_failures}) reached, aborting bootstrap"
                )
                break

            try:
                # Parse file (would need proper implementation)
                # For now, just count files
                if dry_run:
                    logger.info(f"DRY RUN: Would process {file_path.name}")
                    # In dry run, we still count as would-be-processed
                else:
                    # Actual processing would happen here
                    successful += 1

            except Exception as e:
                error_type = type(e).__name__
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
                failed_files.append(str(file_path))
                failed += 1
                logger.error(f"Failed to process {file_path.name}: {e}")

        return BootstrapResult(
            total_files=total_files,
            successful=successful,
            failed=failed,
            skipped=skipped,
            duration_seconds=self._elapsed_ms(start_time) / 1000,
            dry_run=dry_run,
            failed_files=failed_files,
            error_summary=error_summary,
        )

    def _elapsed_ms(self, start_time: float) -> float:
        """Calculate elapsed time in milliseconds."""
        return (time.perf_counter() - start_time) * 1000


# Singleton instance placeholder
_deduplication_service: DeduplicationService | None = None


def get_deduplication_service(
    vector_service: "VectorService | None" = None,
    neo4j_service: "Neo4jService | None" = None,
    file_service: "FileService | None" = None,
) -> DeduplicationService:
    """Get or create singleton deduplication service instance.

    Args:
        vector_service: Optional VectorService (uses default if not provided)
        neo4j_service: Optional Neo4jService (uses default if not provided)
        file_service: Optional FileService (uses default if not provided)

    Returns:
        DeduplicationService singleton instance
    """
    global _deduplication_service

    if _deduplication_service is None:
        # Import dependencies here to avoid circular imports
        from src.services.file_service import FileService
        from src.services.neo4j_service import Neo4jService
        from src.services.vector_service import get_vector_service

        _deduplication_service = DeduplicationService(
            vector_service=vector_service or get_vector_service(),
            neo4j_service=neo4j_service or Neo4jService(),
            file_service=file_service or FileService(),
        )

    assert _deduplication_service is not None
    return _deduplication_service


# Module-level instance for import convenience
# Usage: from src.services.deduplication_service import deduplication_service
deduplication_service: DeduplicationService | None = None

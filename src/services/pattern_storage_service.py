"""
Pattern Storage Service for Feature 026.

Handles storing extracted patterns in Neo4j with deduplication and relationship management.
Creates Pattern nodes and links them to sessions with outcome tracking.
"""

from datetime import UTC, datetime

from loguru import logger

from src.config import PATTERN_RECOGNITION_ENABLED, PATTERN_SIMILARITY_THRESHOLD
from src.models.pattern_models import ExtractedPattern, StoredPattern
from src.services.neo4j_service import Neo4jService


class PatternStorageService:
    """
    Store patterns in Neo4j with full relationship tracking.

    Manages:
    - Pattern node creation and updates
    - ATTEMPTED_WITH relationships (Problem → Pattern)
    - DISCOVERED_PATTERN relationships (Session → Pattern)
    - Occurrence and success/failure count tracking
    """

    def __init__(self):
        """Initialize pattern storage service with Neo4j dependency."""
        self.neo4j_service = Neo4jService()
        self.enabled = PATTERN_RECOGNITION_ENABLED

    async def check_for_duplicate_pattern(
        self,
        problem: str,
        solution: str,
        similarity_threshold: float = PATTERN_SIMILARITY_THRESHOLD,
    ) -> str | None:
        """
        Check if a pattern already exists with similar problem/solution.

        Uses semantic similarity (Levenshtein distance via Neo4j) to find duplicates.
        If a duplicate is found, returns its UUID so occurrence count can be incremented.

        Args:
            problem: Problem description to check
            solution: Solution approach to check
            similarity_threshold: Minimum similarity score for duplicate (0.0-1.0)

        Returns:
            UUID of duplicate pattern if found (>= similarity_threshold), None otherwise

        Raises:
            ValueError: If problem or solution is empty
        """
        if not problem or not solution:
            raise ValueError("problem and solution are required")

        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        try:
            # Use Levenshtein similarity to find candidates
            # We check both problem and solution for high similarity
            cypher = """
            MATCH (p:Pattern)
            WHERE p.archived = false
            WITH p,
                 apoc.text.levenshteinSimilarity(p.solution, $solution) as sol_similarity,
                 apoc.text.levenshteinSimilarity(p.problem, $problem) as prob_similarity
            WHERE (sol_similarity >= $threshold OR prob_similarity >= $threshold)
            WITH p, sol_similarity, prob_similarity, (sol_similarity * 0.6 + prob_similarity * 0.4) as combined
            ORDER BY combined DESC
            LIMIT 1
            RETURN p.uuid as uuid, combined as similarity
            """

            results = await self.neo4j_service.query(
                cypher,
                {
                    "problem": problem,
                    "solution": solution,
                    "threshold": similarity_threshold,
                },
            )

            if results:
                duplicate_uuid = results[0].get("uuid")
                similarity = results[0].get("similarity", 0.0)
                logger.debug(
                    f"Found duplicate pattern {duplicate_uuid} "
                    f"with {similarity:.0%} similarity"
                )
                return duplicate_uuid

            logger.debug("No duplicate patterns found")
            return None

        except Exception as e:
            logger.error(f"Error checking for duplicate patterns: {e}")
            return None

    async def store_pattern(
        self, pattern: ExtractedPattern, session_uuid: str
    ) -> StoredPattern | None:
        """
        Store a single extracted pattern in Neo4j.

        Creates or updates a Pattern node and establishes relationships
        to the session that discovered it.

        Args:
            pattern: ExtractedPattern to store
            session_uuid: UUID of session that discovered this pattern

        Returns:
            StoredPattern with full metadata, or None if storage failed

        Raises:
            ValueError: If pattern or session_uuid is invalid
        """
        if not self.enabled:
            logger.debug("Pattern recognition disabled, skipping storage")
            return None

        if not pattern or not session_uuid:
            raise ValueError("pattern and session_uuid are required")

        try:
            # Check for duplicate pattern using semantic similarity (Feature 026 - T026-008)
            duplicate_uuid = await self.check_for_duplicate_pattern(
                problem=pattern.problem_description, solution=pattern.solution_approach
            )

            if duplicate_uuid:
                # Pattern already exists - update occurrence count instead
                success = await self.update_pattern_on_repeat(
                    pattern_uuid=duplicate_uuid,
                    outcome=pattern.outcome,
                    session_uuid=session_uuid,
                )

                if success:
                    logger.info(
                        f"Pattern {duplicate_uuid} already exists, "
                        f"incremented occurrence count instead of creating new"
                    )
                    # Return the updated pattern
                    stored = await self._get_pattern_details(duplicate_uuid)
                    return stored
                else:
                    logger.warning(
                        f"Failed to update duplicate pattern {duplicate_uuid}"
                    )
                    return None

            # Generate pattern UUID for new pattern
            pattern_uuid = await self._generate_pattern_uuid()

            # Create pattern name from solution approach
            pattern_name = self._generate_pattern_name(pattern.solution_approach)

            # Build Cypher query to create Pattern node
            cypher = """
            CREATE (p:Pattern {
                uuid: $pattern_uuid,
                name: $pattern_name,
                pattern_type: $pattern_type,
                problem: $problem,
                solution: $solution,
                occurrence_count: 1,
                success_count: CASE WHEN $outcome = 'SUCCESS' THEN 1 ELSE 0 END,
                failure_count: CASE WHEN $outcome = 'FAILED' THEN 1 ELSE 0 END,
                success_rate: CASE WHEN $outcome = 'SUCCESS' THEN 1.0 ELSE 0.0 END,
                confidence: $confidence,
                consecutive_failures: CASE WHEN $outcome = 'FAILED' THEN 1 ELSE 0 END,
                first_occurrence_date: datetime($now),
                latest_occurrence_date: datetime($now),
                created_at: datetime($now),
                updated_at: datetime($now),
                archived: false
            })
            WITH p
            MATCH (s:Session {uuid: $session_uuid})
            CREATE (s)-[rel:DISCOVERED_PATTERN {
                method: 'extraction',
                created_at: datetime($now)
            }]->(p)
            CREATE (s)-[attempt:ATTEMPTED_WITH {
                date: datetime($now),
                outcome: $outcome,
                session_uuid: $session_uuid,
                confidence: $confidence
            }]->(p)
            RETURN p.uuid as pattern_uuid
            """

            result = await self.neo4j_service.query(
                cypher,
                {
                    "pattern_uuid": pattern_uuid,
                    "pattern_name": pattern_name,
                    "pattern_type": pattern.pattern_type,
                    "problem": pattern.problem_description,
                    "solution": pattern.solution_approach,
                    "outcome": pattern.outcome,
                    "confidence": pattern.confidence,
                    "session_uuid": session_uuid,
                    "now": datetime.now(UTC).isoformat(),
                },
            )

            if result:
                logger.info(f"Stored pattern {pattern_uuid}: {pattern_name}")

                # Build and return StoredPattern object
                now = datetime.now(UTC)
                stored = StoredPattern(
                    pattern_uuid=pattern_uuid,
                    name=pattern_name,
                    pattern_type=pattern.pattern_type,
                    problem=pattern.problem_description,
                    solution=pattern.solution_approach,
                    occurrence_count=1,
                    success_count=1 if pattern.outcome == "SUCCESS" else 0,
                    failure_count=1 if pattern.outcome == "FAILED" else 0,
                    success_rate=1.0 if pattern.outcome == "SUCCESS" else 0.0,
                    confidence=pattern.confidence,
                    consecutive_failures=1 if pattern.outcome == "FAILED" else 0,
                    first_occurrence_date=now,
                    latest_occurrence_date=now,
                    created_at=now,
                    updated_at=now,
                    supporting_sessions=[session_uuid],
                    archived=False,
                )
                return stored
            else:
                logger.error(f"Failed to create pattern node for {pattern_name}")
                return None

        except Exception as e:
            logger.error(f"Error storing pattern: {e}")
            return None

    async def update_pattern_on_repeat(
        self, pattern_uuid: str, outcome: str, session_uuid: str
    ) -> bool:
        """
        Update pattern occurrence and success/failure counts when repeated.

        Args:
            pattern_uuid: UUID of existing pattern
            outcome: Result of this application (SUCCESS, FAILED, PARTIAL, INCONCLUSIVE)
            session_uuid: UUID of session applying this pattern

        Returns:
            True if update successful, False otherwise
        """
        try:
            now = datetime.now(UTC)

            cypher = """
            MATCH (p:Pattern {uuid: $pattern_uuid})
            SET p.occurrence_count = p.occurrence_count + 1,
                p.success_count = p.success_count + CASE WHEN $outcome = 'SUCCESS' THEN 1 ELSE 0 END,
                p.failure_count = p.failure_count + CASE WHEN $outcome = 'FAILED' THEN 1 ELSE 0 END,
                p.latest_occurrence_date = datetime($now),
                p.updated_at = datetime($now),
                p.success_rate =
                  CASE WHEN p.occurrence_count + 1 > 0
                  THEN (p.success_count + CASE WHEN $outcome = 'SUCCESS' THEN 1 ELSE 0 END)
                       / CAST(p.occurrence_count + 1 AS FLOAT)
                  ELSE 0.0
                  END,
                p.consecutive_failures =
                  CASE WHEN $outcome = 'FAILED'
                  THEN p.consecutive_failures + 1
                  ELSE 0
                  END
            WITH p
            MATCH (s:Session {uuid: $session_uuid})
            CREATE (s)-[attempt:ATTEMPTED_WITH {
                date: datetime($now),
                outcome: $outcome,
                session_uuid: $session_uuid,
                confidence: 1.0
            }]->(p)
            RETURN p.uuid as pattern_uuid
            """

            result = await self.neo4j_service.query(
                cypher,
                {
                    "pattern_uuid": pattern_uuid,
                    "outcome": outcome,
                    "session_uuid": session_uuid,
                    "now": now.isoformat(),
                },
            )

            if result:
                logger.debug(f"Updated pattern {pattern_uuid} with outcome {outcome}")
                return True
            else:
                logger.warning(f"Pattern {pattern_uuid} not found for update")
                return False

        except Exception as e:
            logger.error(f"Error updating pattern {pattern_uuid}: {e}")
            return False

    async def get_similar_patterns(
        self, pattern_uuid: str, similarity_threshold: float = 0.90
    ) -> list[tuple[str, float]]:
        """
        Find semantically similar patterns for deduplication.

        Args:
            pattern_uuid: UUID of pattern to find duplicates for
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of (similar_pattern_uuid, similarity_score) tuples
        """
        try:
            cypher = """
            MATCH (p1:Pattern {uuid: $pattern_uuid})
            MATCH (p2:Pattern)
            WHERE p1 <> p2
            AND p1.pattern_type = p2.pattern_type
            AND p1.archived = false
            AND p2.archived = false
            WITH p1, p2,
                 apoc.text.levenshteinSimilarity(p1.solution, p2.solution) as similarity
            WHERE similarity >= $threshold
            RETURN p2.uuid as uuid, similarity
            ORDER BY similarity DESC
            """

            results = await self.neo4j_service.query(
                cypher,
                {"pattern_uuid": pattern_uuid, "threshold": similarity_threshold},
            )

            return (
                [(row["uuid"], row["similarity"]) for row in results] if results else []
            )

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    async def archive_pattern(self, pattern_uuid: str) -> bool:
        """
        Soft-delete a pattern by marking it archived.

        Args:
            pattern_uuid: UUID of pattern to archive

        Returns:
            True if successful, False otherwise
        """
        try:
            cypher = """
            MATCH (p:Pattern {uuid: $pattern_uuid})
            SET p.archived = true,
                p.updated_at = datetime($now)
            RETURN p.uuid as uuid
            """

            result = await self.neo4j_service.query(
                cypher,
                {
                    "pattern_uuid": pattern_uuid,
                    "now": datetime.now(UTC).isoformat(),
                },
            )

            if result:
                logger.info(f"Archived pattern {pattern_uuid}")
                return True
            else:
                logger.warning(f"Pattern {pattern_uuid} not found for archival")
                return False

        except Exception as e:
            logger.error(f"Error archiving pattern: {e}")
            return False

    async def _generate_pattern_uuid(self) -> str:
        """
        Generate a unique UUID for pattern using Neo4j randomUUID.

        Returns:
            UUID string
        """
        result = await self.neo4j_service.query("RETURN randomUUID() as uuid")
        return result[0]["uuid"] if result else None

    async def _get_pattern_details(self, pattern_uuid: str) -> StoredPattern | None:
        """
        Retrieve full pattern details by UUID.

        Args:
            pattern_uuid: UUID of pattern to retrieve

        Returns:
            StoredPattern with all metadata, or None if not found
        """
        try:
            cypher = """
            MATCH (p:Pattern {uuid: $pattern_uuid})
            RETURN p
            """

            results = await self.neo4j_service.query(
                cypher, {"pattern_uuid": pattern_uuid}
            )

            if not results:
                return None

            pattern_node = results[0].get("p")
            if not pattern_node:
                return None

            # Convert node to StoredPattern
            now = datetime.now(UTC)

            # Parse datetime fields
            created_at = pattern_node.get("created_at")
            updated_at = pattern_node.get("updated_at")
            first_occurrence = pattern_node.get("first_occurrence_date")
            latest_occurrence = pattern_node.get("latest_occurrence_date")

            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if isinstance(first_occurrence, str):
                first_occurrence = datetime.fromisoformat(
                    first_occurrence.replace("Z", "+00:00")
                )
            if isinstance(latest_occurrence, str):
                latest_occurrence = datetime.fromisoformat(
                    latest_occurrence.replace("Z", "+00:00")
                )

            return StoredPattern(
                pattern_uuid=pattern_node.get("uuid"),
                name=pattern_node.get("name", "Unknown"),
                pattern_type=pattern_node.get("pattern_type"),
                problem=pattern_node.get("problem", ""),
                solution=pattern_node.get("solution", ""),
                occurrence_count=int(pattern_node.get("occurrence_count", 1)),
                success_count=int(pattern_node.get("success_count", 0)),
                failure_count=int(pattern_node.get("failure_count", 0)),
                success_rate=float(pattern_node.get("success_rate", 0.0)),
                confidence=float(pattern_node.get("confidence", 1.0)),
                consecutive_failures=int(pattern_node.get("consecutive_failures", 0)),
                first_occurrence_date=first_occurrence or now,
                latest_occurrence_date=latest_occurrence or now,
                created_at=created_at or now,
                updated_at=updated_at or now,
                supporting_sessions=pattern_node.get("supporting_sessions", []),
                archived=bool(pattern_node.get("archived", False)),
            )

        except Exception as e:
            logger.error(f"Error retrieving pattern details {pattern_uuid}: {e}")
            return None

    def _generate_pattern_name(
        self, solution_approach: str, max_length: int = 100
    ) -> str:
        """
        Generate a concise pattern name from solution approach.

        Args:
            solution_approach: Full solution description
            max_length: Maximum name length

        Returns:
            Cleaned pattern name
        """
        # Take first 100 characters, clean up
        name = solution_approach[:max_length].strip()

        # Remove trailing punctuation
        while name and name[-1] in ".,:;!?":
            name = name[:-1].strip()

        return name if name else "Unknown Pattern"

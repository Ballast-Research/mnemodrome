"""
Pattern Query Service for Feature 026.

Handles querying stored patterns from Neo4j with flexible filtering and ranking.
Supports filtering by problem type, success rate, and other metadata.
"""

from datetime import UTC, datetime, timedelta

from loguru import logger

from src.config import PATTERN_RECOGNITION_ENABLED
from src.models.pattern_models import PatternMatch, StoredPattern
from src.services.neo4j_service import Neo4jService


class PatternQueryService:
    """
    Query patterns stored in Neo4j with flexible filtering and ranking.

    Supports:
    - Filtering by problem type and success rate
    - Ranking by success rate, occurrence count, and confidence
    - Archival status checking (excludes archived patterns by default)
    - UUID-based retrieval for specific patterns
    """

    def __init__(self):
        """Initialize pattern query service with Neo4j dependency."""
        self.neo4j_service = Neo4jService()
        self.enabled = PATTERN_RECOGNITION_ENABLED

    async def query_patterns(
        self,
        problem_type: str = "",
        min_success_rate: float = 0.0,
        limit: int = 10,
        exclude_archived: bool = True,
    ) -> list[StoredPattern]:
        """
        Query patterns with optional filters.

        Args:
            problem_type: Filter by pattern type (empty string = all types)
            min_success_rate: Minimum success rate threshold (0.0-1.0)
            limit: Maximum patterns to return
            exclude_archived: Exclude archived patterns (default: True)

        Returns:
            List of StoredPattern objects sorted by success rate descending,
            then occurrence count descending

        Raises:
            ValueError: If parameters are invalid
        """
        if not self.enabled:
            logger.debug("Pattern recognition disabled, returning empty list")
            return []

        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0.0 and 1.0")

        if limit < 1:
            raise ValueError("limit must be >= 1")

        try:
            # Build Cypher query with optional filters
            where_clauses = []

            if exclude_archived:
                where_clauses.append("p.archived = false")

            if problem_type:
                where_clauses.append("p.pattern_type = $problem_type")

            if min_success_rate > 0.0:
                where_clauses.append("p.success_rate >= $min_success_rate")

            # Combine WHERE clauses
            where_clause = " AND ".join(where_clauses) if where_clauses else "true"

            cypher = f"""
            MATCH (p:Pattern)
            WHERE {where_clause}
            RETURN p
            ORDER BY p.success_rate DESC, p.occurrence_count DESC
            LIMIT $limit
            """

            # Build parameters
            params = {"limit": limit}
            if problem_type:
                params["problem_type"] = problem_type
            if min_success_rate > 0.0:
                params["min_success_rate"] = min_success_rate

            # Execute query
            results = await self.neo4j_service.query(cypher, params)

            if not results:
                logger.debug("No patterns found matching query filters")
                return []

            # Convert results to StoredPattern objects
            patterns = []
            for row in results:
                try:
                    pattern_data = row.get("p")
                    if pattern_data:
                        stored_pattern = self._node_to_stored_pattern(pattern_data)
                        patterns.append(stored_pattern)
                except Exception as e:
                    logger.warning(f"Error converting pattern node: {e}")
                    continue

            logger.debug(f"Found {len(patterns)} patterns matching filters")
            return patterns

        except Exception as e:
            logger.error(f"Error querying patterns: {e}")
            raise

    async def get_pattern_by_uuid(self, uuid: str) -> StoredPattern | None:
        """
        Retrieve a specific pattern by its UUID.

        Args:
            uuid: UUID of the pattern

        Returns:
            StoredPattern if found, None otherwise

        Raises:
            ValueError: If uuid is empty
        """
        if not uuid:
            raise ValueError("uuid is required")

        try:
            cypher = """
            MATCH (p:Pattern {uuid: $uuid})
            RETURN p
            """

            results = await self.neo4j_service.query(cypher, {"uuid": uuid})

            if not results:
                logger.debug(f"Pattern {uuid} not found")
                return None

            pattern_data = results[0].get("p")
            if pattern_data:
                stored_pattern = self._node_to_stored_pattern(pattern_data)
                logger.debug(f"Retrieved pattern {uuid}")
                return stored_pattern

            return None

        except Exception as e:
            logger.error(f"Error retrieving pattern {uuid}: {e}")
            raise

    async def search_patterns_by_problem(
        self, problem_text: str, limit: int = 10
    ) -> list[PatternMatch]:
        """
        Search for patterns by problem description using text similarity.

        Uses Neo4j's apoc.text.levenshteinSimilarity for string matching.

        Args:
            problem_text: Problem description to search for
            limit: Maximum results to return

        Returns:
            List of PatternMatch objects with similarity scores, ranked by
            similarity then success rate

        Raises:
            ValueError: If problem_text is empty
        """
        if not problem_text or len(problem_text.strip()) < 3:
            raise ValueError("problem_text must be at least 3 characters")

        try:
            cypher = """
            MATCH (p:Pattern)
            WHERE p.archived = false
            WITH p,
                 apoc.text.levenshteinSimilarity(p.problem, $problem_text) as similarity
            WHERE similarity > 0.5
            RETURN p, similarity
            ORDER BY similarity DESC, p.success_rate DESC
            LIMIT $limit
            """

            results = await self.neo4j_service.query(
                cypher, {"problem_text": problem_text, "limit": limit}
            )

            if not results:
                logger.debug(f"No similar patterns found for: {problem_text[:50]}")
                return []

            matches = []
            for row in results:
                try:
                    pattern_data = row.get("p")
                    similarity = row.get("similarity", 0.0)

                    if pattern_data:
                        stored = self._node_to_stored_pattern(pattern_data)
                        match = PatternMatch(
                            pattern_uuid=stored.pattern_uuid,
                            name=stored.name,
                            pattern_type=stored.pattern_type,
                            similarity_score=similarity,
                            success_rate=stored.success_rate,
                            occurrence_count=stored.occurrence_count,
                            confidence=stored.confidence,
                            relevance_explanation=(
                                f"Problem similarity: {similarity:.0%}. "
                                f"Success rate: {stored.success_rate:.0%}. "
                                f"Occurred {stored.occurrence_count} times."
                            ),
                            consecutive_failures=stored.consecutive_failures,
                            last_occurred=stored.latest_occurrence_date,
                        )
                        matches.append(match)
                except Exception as e:
                    logger.warning(f"Error converting pattern match: {e}")
                    continue

            logger.debug(f"Found {len(matches)} similar patterns")
            return matches

        except Exception as e:
            logger.error(f"Error searching patterns: {e}")
            raise

    async def get_recent_patterns(
        self, days: int = 30, limit: int = 10
    ) -> list[StoredPattern]:
        """
        Get patterns discovered recently.

        Args:
            days: Number of days to look back
            limit: Maximum patterns to return

        Returns:
            List of StoredPattern objects from recent sessions, ordered by
            most recent first
        """
        if days < 1:
            raise ValueError("days must be >= 1")

        try:
            cutoff_date = (datetime.now(UTC) - timedelta(days=days)).isoformat()

            cypher = """
            MATCH (p:Pattern)
            WHERE p.archived = false
            AND p.latest_occurrence_date >= datetime($cutoff_date)
            RETURN p
            ORDER BY p.latest_occurrence_date DESC
            LIMIT $limit
            """

            results = await self.neo4j_service.query(
                cypher, {"cutoff_date": cutoff_date, "limit": limit}
            )

            if not results:
                logger.debug(f"No patterns found in last {days} days")
                return []

            patterns = []
            for row in results:
                try:
                    pattern_data = row.get("p")
                    if pattern_data:
                        stored = self._node_to_stored_pattern(pattern_data)
                        patterns.append(stored)
                except Exception as e:
                    logger.warning(f"Error converting pattern: {e}")
                    continue

            logger.debug(f"Found {len(patterns)} recent patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error retrieving recent patterns: {e}")
            raise

    async def get_high_success_patterns(
        self, min_success_rate: float = 0.80, min_occurrences: int = 3, limit: int = 10
    ) -> list[StoredPattern]:
        """
        Get patterns with high success rates and proven track records.

        Args:
            min_success_rate: Minimum success rate (0.0-1.0, default: 0.80)
            min_occurrences: Minimum number of times pattern must occur
            limit: Maximum patterns to return

        Returns:
            List of high-confidence StoredPattern objects
        """
        if not 0.0 <= min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be between 0.0 and 1.0")

        if min_occurrences < 1:
            raise ValueError("min_occurrences must be >= 1")

        try:
            cypher = """
            MATCH (p:Pattern)
            WHERE p.archived = false
            AND p.success_rate >= $min_success_rate
            AND p.occurrence_count >= $min_occurrences
            RETURN p
            ORDER BY p.success_rate DESC, p.occurrence_count DESC
            LIMIT $limit
            """

            results = await self.neo4j_service.query(
                cypher,
                {
                    "min_success_rate": min_success_rate,
                    "min_occurrences": min_occurrences,
                    "limit": limit,
                },
            )

            if not results:
                logger.debug(
                    f"No patterns found with >{min_success_rate:.0%} success "
                    f"and {min_occurrences}+ occurrences"
                )
                return []

            patterns = []
            for row in results:
                try:
                    pattern_data = row.get("p")
                    if pattern_data:
                        stored = self._node_to_stored_pattern(pattern_data)
                        patterns.append(stored)
                except Exception as e:
                    logger.warning(f"Error converting pattern: {e}")
                    continue

            logger.debug(f"Found {len(patterns)} high-success patterns")
            return patterns

        except Exception as e:
            logger.error(f"Error retrieving high-success patterns: {e}")
            raise

    def _node_to_stored_pattern(self, node: dict) -> StoredPattern:
        """
        Convert Neo4j node to StoredPattern object.

        Args:
            node: Neo4j node data

        Returns:
            StoredPattern object

        Raises:
            ValueError: If node is missing required fields
        """
        try:
            # Parse datetime fields
            created_at = node.get("created_at")
            updated_at = node.get("updated_at")
            first_occurrence = node.get("first_occurrence_date")
            latest_occurrence = node.get("latest_occurrence_date")

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

            # Get supporting sessions (relationships)
            supporting_sessions = node.get("supporting_sessions", [])
            if not isinstance(supporting_sessions, list):
                supporting_sessions = (
                    list(supporting_sessions) if supporting_sessions else []
                )

            return StoredPattern(
                pattern_uuid=node.get("uuid"),
                name=node.get("name", "Unknown"),
                pattern_type=node.get("pattern_type"),
                problem=node.get("problem", ""),
                solution=node.get("solution", ""),
                occurrence_count=int(node.get("occurrence_count", 1)),
                success_count=int(node.get("success_count", 0)),
                failure_count=int(node.get("failure_count", 0)),
                success_rate=float(node.get("success_rate", 0.0)),
                confidence=float(node.get("confidence", 1.0)),
                consecutive_failures=int(node.get("consecutive_failures", 0)),
                first_occurrence_date=first_occurrence or datetime.now(UTC),
                latest_occurrence_date=latest_occurrence or datetime.now(UTC),
                created_at=created_at or datetime.now(UTC),
                updated_at=updated_at or datetime.now(UTC),
                supporting_sessions=supporting_sessions,
                archived=bool(node.get("archived", False)),
            )
        except Exception as e:
            logger.error(f"Error converting node to StoredPattern: {e}")
            raise ValueError(f"Invalid pattern node: {e}")

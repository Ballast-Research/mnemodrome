"""Unit tests for PatternQueryService."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.models.pattern_models import PatternMatch
from src.services.pattern_query_service import PatternQueryService


@pytest.fixture
def pattern_query_service():
    """Provide PatternQueryService with mocked Neo4j dependency."""
    with patch("src.services.pattern_query_service.Neo4jService"):
        service = PatternQueryService()
        service.neo4j_service = AsyncMock()
        return service


class TestPatternQueryServiceInit:
    """Test service initialization."""

    def test_init_with_defaults(self):
        """Test service initializes with defaults."""
        with patch("src.services.pattern_query_service.Neo4jService"):
            with patch(
                "src.services.pattern_query_service.PATTERN_RECOGNITION_ENABLED", True
            ):
                service = PatternQueryService()
                assert service.enabled is True
                assert service.neo4j_service is not None


class TestQueryPatterns:
    """Test pattern querying with filters."""

    @pytest.mark.asyncio
    async def test_query_returns_empty_when_disabled(self, pattern_query_service):
        """Test returns empty list when feature disabled."""
        pattern_query_service.enabled = False

        result = await pattern_query_service.query_patterns()

        assert result == []

    @pytest.mark.asyncio
    async def test_query_validates_min_success_rate(self, pattern_query_service):
        """Test validates min_success_rate bounds."""
        pattern_query_service.enabled = True

        with pytest.raises(
            ValueError, match="min_success_rate must be between 0.0 and 1.0"
        ):
            await pattern_query_service.query_patterns(min_success_rate=1.5)

    @pytest.mark.asyncio
    async def test_query_validates_limit(self, pattern_query_service):
        """Test validates limit is positive."""
        pattern_query_service.enabled = True

        with pytest.raises(ValueError, match="limit must be >= 1"):
            await pattern_query_service.query_patterns(limit=0)

    @pytest.mark.asyncio
    async def test_query_builds_correct_cypher_with_filters(
        self, pattern_query_service
    ):
        """Test correct Cypher query is built with all filters."""
        pattern_query_service.enabled = True
        pattern_query_service.neo4j_service.query = AsyncMock(return_value=[])

        await pattern_query_service.query_patterns(
            problem_type="infrastructure",
            min_success_rate=0.8,
            limit=5,
            exclude_archived=True,
        )

        # Verify query was called with correct parameters
        call_args = pattern_query_service.neo4j_service.query.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]

        assert "p.archived = false" in cypher
        assert "p.pattern_type = $problem_type" in cypher
        assert "p.success_rate >= $min_success_rate" in cypher
        assert "ORDER BY p.success_rate DESC, p.occurrence_count DESC" in cypher
        assert params["problem_type"] == "infrastructure"
        assert params["min_success_rate"] == 0.8
        assert params["limit"] == 5

    @pytest.mark.asyncio
    async def test_query_converts_results_to_stored_patterns(
        self, pattern_query_service
    ):
        """Test query results are converted to StoredPattern objects."""
        pattern_query_service.enabled = True

        mock_node = {
            "uuid": "pattern-1",
            "name": "Test Pattern",
            "pattern_type": "code",
            "problem": "Memory leak",
            "solution": "Implement garbage collection",
            "occurrence_count": 3,
            "success_count": 2,
            "failure_count": 1,
            "success_rate": 0.67,
            "confidence": 0.85,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC),
            "latest_occurrence_date": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "supporting_sessions": ["session1", "session2"],
            "archived": False,
        }

        pattern_query_service.neo4j_service.query = AsyncMock(
            return_value=[{"p": mock_node}]
        )

        result = await pattern_query_service.query_patterns()

        assert len(result) == 1
        assert result[0].pattern_uuid == "pattern-1"
        assert result[0].success_rate == 0.67


class TestGetPatternByUuid:
    """Test pattern retrieval by UUID."""

    @pytest.mark.asyncio
    async def test_get_pattern_validates_uuid(self, pattern_query_service):
        """Test raises ValueError on empty UUID."""
        pattern_query_service.enabled = True

        with pytest.raises(ValueError, match="uuid is required"):
            await pattern_query_service.get_pattern_by_uuid("")

    @pytest.mark.asyncio
    async def test_get_pattern_returns_none_if_not_found(self, pattern_query_service):
        """Test returns None if pattern not found."""
        pattern_query_service.enabled = True
        pattern_query_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_query_service.get_pattern_by_uuid("nonexistent-uuid")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_pattern_returns_stored_pattern_if_found(
        self, pattern_query_service
    ):
        """Test returns StoredPattern if found."""
        pattern_query_service.enabled = True

        mock_node = {
            "uuid": "pattern-1",
            "name": "Cache Pattern",
            "pattern_type": "infrastructure",
            "problem": "Slow API",
            "solution": "Redis caching",
            "occurrence_count": 5,
            "success_count": 4,
            "failure_count": 1,
            "success_rate": 0.8,
            "confidence": 0.9,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC),
            "latest_occurrence_date": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "supporting_sessions": [],
            "archived": False,
        }

        pattern_query_service.neo4j_service.query = AsyncMock(
            return_value=[{"p": mock_node}]
        )

        result = await pattern_query_service.get_pattern_by_uuid("pattern-1")

        assert result is not None
        assert result.pattern_uuid == "pattern-1"
        assert result.success_rate == 0.8


class TestSearchPatternsByProblem:
    """Test pattern search by problem description."""

    @pytest.mark.asyncio
    async def test_search_validates_problem_text_length(self, pattern_query_service):
        """Test validates problem text is at least 3 characters."""
        pattern_query_service.enabled = True

        with pytest.raises(
            ValueError, match="problem_text must be at least 3 characters"
        ):
            await pattern_query_service.search_patterns_by_problem("ab")

    @pytest.mark.asyncio
    async def test_search_returns_pattern_matches(self, pattern_query_service):
        """Test search returns PatternMatch objects with similarity scores."""
        pattern_query_service.enabled = True

        mock_node = {
            "uuid": "pattern-1",
            "name": "Connection Pool Pattern",
            "pattern_type": "infrastructure",
            "problem": "Database connection timeout",
            "solution": "Implement connection pooling",
            "occurrence_count": 2,
            "success_count": 2,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.9,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC),
            "latest_occurrence_date": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "supporting_sessions": ["session1"],
            "archived": False,
        }

        pattern_query_service.neo4j_service.query = AsyncMock(
            return_value=[{"p": mock_node, "similarity": 0.85}]
        )

        result = await pattern_query_service.search_patterns_by_problem(
            "database connection issues", limit=5
        )

        assert len(result) == 1
        assert isinstance(result[0], PatternMatch)
        assert result[0].similarity_score == 0.85
        assert result[0].success_rate == 1.0

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_no_matches(self, pattern_query_service):
        """Test search returns empty list if no matches."""
        pattern_query_service.enabled = True
        pattern_query_service.neo4j_service.query = AsyncMock(return_value=[])

        result = await pattern_query_service.search_patterns_by_problem(
            "obscure problem"
        )

        assert result == []


class TestGetRecentPatterns:
    """Test retrieval of recently discovered patterns."""

    @pytest.mark.asyncio
    async def test_get_recent_validates_days(self, pattern_query_service):
        """Test validates days parameter."""
        pattern_query_service.enabled = True

        with pytest.raises(ValueError, match="days must be >= 1"):
            await pattern_query_service.get_recent_patterns(days=0)

    @pytest.mark.asyncio
    async def test_get_recent_builds_correct_query(self, pattern_query_service):
        """Test correct query with cutoff date."""
        pattern_query_service.enabled = True
        pattern_query_service.neo4j_service.query = AsyncMock(return_value=[])

        await pattern_query_service.get_recent_patterns(days=7, limit=10)

        call_args = pattern_query_service.neo4j_service.query.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]

        assert "p.latest_occurrence_date >= datetime($cutoff_date)" in cypher
        assert "ORDER BY p.latest_occurrence_date DESC" in cypher
        assert "cutoff_date" in params
        assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_get_recent_returns_stored_patterns(self, pattern_query_service):
        """Test returns StoredPattern objects."""
        pattern_query_service.enabled = True

        mock_node = {
            "uuid": "recent-pattern",
            "name": "Recent Pattern",
            "pattern_type": "testing",
            "problem": "Flaky tests",
            "solution": "Add test retries",
            "occurrence_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "success_rate": 1.0,
            "confidence": 0.8,
            "consecutive_failures": 0,
            "first_occurrence_date": datetime.now(UTC),
            "latest_occurrence_date": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "supporting_sessions": [],
            "archived": False,
        }

        pattern_query_service.neo4j_service.query = AsyncMock(
            return_value=[{"p": mock_node}]
        )

        result = await pattern_query_service.get_recent_patterns(days=30)

        assert len(result) == 1
        assert result[0].pattern_type == "testing"


class TestGetHighSuccessPatterns:
    """Test retrieval of high-success-rate patterns."""

    @pytest.mark.asyncio
    async def test_get_high_success_validates_success_rate(self, pattern_query_service):
        """Test validates success rate bounds."""
        pattern_query_service.enabled = True

        with pytest.raises(
            ValueError, match="min_success_rate must be between 0.0 and 1.0"
        ):
            await pattern_query_service.get_high_success_patterns(min_success_rate=1.5)

    @pytest.mark.asyncio
    async def test_get_high_success_validates_occurrences(self, pattern_query_service):
        """Test validates minimum occurrences."""
        pattern_query_service.enabled = True

        with pytest.raises(ValueError, match="min_occurrences must be >= 1"):
            await pattern_query_service.get_high_success_patterns(min_occurrences=0)

    @pytest.mark.asyncio
    async def test_get_high_success_builds_correct_query(self, pattern_query_service):
        """Test correct query with success filters."""
        pattern_query_service.enabled = True
        pattern_query_service.neo4j_service.query = AsyncMock(return_value=[])

        await pattern_query_service.get_high_success_patterns(
            min_success_rate=0.9, min_occurrences=5, limit=10
        )

        call_args = pattern_query_service.neo4j_service.query.call_args
        cypher = call_args[0][0]
        params = call_args[0][1]

        assert "p.success_rate >= $min_success_rate" in cypher
        assert "p.occurrence_count >= $min_occurrences" in cypher
        assert params["min_success_rate"] == 0.9
        assert params["min_occurrences"] == 5

    @pytest.mark.asyncio
    async def test_get_high_success_returns_patterns_in_order(
        self, pattern_query_service
    ):
        """Test returns patterns ordered by success rate then occurrences."""
        pattern_query_service.enabled = True

        mock_nodes = [
            {
                "p": {
                    "uuid": "pattern-1",
                    "name": "High Success",
                    "pattern_type": "code",
                    "problem": "Problem 1",
                    "solution": "Solution 1",
                    "occurrence_count": 10,
                    "success_count": 10,
                    "failure_count": 0,
                    "success_rate": 1.0,
                    "confidence": 0.95,
                    "consecutive_failures": 0,
                    "first_occurrence_date": datetime.now(UTC),
                    "latest_occurrence_date": datetime.now(UTC),
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC),
                    "supporting_sessions": [],
                    "archived": False,
                }
            }
        ]

        pattern_query_service.neo4j_service.query = AsyncMock(return_value=mock_nodes)

        result = await pattern_query_service.get_high_success_patterns()

        assert len(result) == 1
        assert result[0].success_rate == 1.0


class TestNodeToStoredPattern:
    """Test conversion of Neo4j nodes to StoredPattern objects."""

    def test_converts_node_with_all_fields(self, pattern_query_service):
        """Test node with all fields converts correctly."""
        node = {
            "uuid": "pattern-1",
            "name": "Complete Pattern",
            "pattern_type": "architecture",
            "problem": "Monolith too large",
            "solution": "Microservices migration",
            "occurrence_count": 3,
            "success_count": 2,
            "failure_count": 1,
            "success_rate": 0.67,
            "confidence": 0.88,
            "consecutive_failures": 1,
            "first_occurrence_date": datetime.now(UTC),
            "latest_occurrence_date": datetime.now(UTC),
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "supporting_sessions": ["s1", "s2", "s3"],
            "archived": False,
        }

        result = pattern_query_service._node_to_stored_pattern(node)

        assert result.pattern_uuid == "pattern-1"
        assert result.success_rate == 0.67
        assert result.occurrence_count == 3
        assert len(result.supporting_sessions) == 3

    def test_handles_missing_optional_fields(self, pattern_query_service):
        """Test handles missing optional fields with defaults."""
        node = {
            "uuid": "pattern-2",
            "name": "Minimal Pattern",
            "pattern_type": "testing",
        }

        result = pattern_query_service._node_to_stored_pattern(node)

        assert result.pattern_uuid == "pattern-2"
        assert result.success_rate == 0.0
        assert result.occurrence_count == 1
        assert result.supporting_sessions == []

    def test_parses_iso_datetime_strings(self, pattern_query_service):
        """Test parsing ISO format datetime strings."""
        now = datetime.now(UTC)
        node = {
            "uuid": "pattern-3",
            "name": "Datetime Pattern",
            "pattern_type": "deployment",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "first_occurrence_date": now.isoformat(),
            "latest_occurrence_date": now.isoformat(),
        }

        result = pattern_query_service._node_to_stored_pattern(node)

        assert isinstance(result.created_at, datetime)
        assert isinstance(result.first_occurrence_date, datetime)

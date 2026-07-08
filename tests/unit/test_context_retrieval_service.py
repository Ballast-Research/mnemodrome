"""Unit tests for ContextRetrievalService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.context_retrieval_service import (
    ContextRetrievalService,
    _prompt_describes_problem,
)


@pytest.fixture
def context_retrieval_service():
    """Provide ContextRetrievalService with mocked dependencies."""
    with patch("src.services.context_retrieval_service.get_vector_service"):
        with patch("src.services.context_retrieval_service.redis_service"):
            with patch("src.services.context_retrieval_service.neo4j_service"):
                service = ContextRetrievalService()
                service.vector_service = AsyncMock()
                service.redis_service = MagicMock()
                service.neo4j_service = AsyncMock()
                return service


class TestPromptDescribesProblem:
    """Test problem detection heuristic."""

    def test_detects_error_keywords(self):
        """Test detects error-related keywords."""
        assert _prompt_describes_problem("Why is there an error?")
        assert _prompt_describes_problem("Something is broken")
        assert _prompt_describes_problem("Fix the timeout issue")

    def test_detects_problem_keywords(self):
        """Test detects problem-related keywords."""
        assert _prompt_describes_problem("Help debug this")
        assert _prompt_describes_problem("The service is slow")
        assert _prompt_describes_problem("Investigate the crash")

    def test_returns_false_for_non_problem_prompts(self):
        """Test returns False for non-problem prompts."""
        assert not _prompt_describes_problem("Tell me about Redis")
        assert not _prompt_describes_problem("What's a factory pattern?")
        assert not _prompt_describes_problem("Explain Neo4j relationships")

    def test_case_insensitive_detection(self):
        """Test case-insensitive keyword matching."""
        assert _prompt_describes_problem("ERROR FOUND")
        assert _prompt_describes_problem("There is a BUG")
        assert _prompt_describes_problem("TIMEOUT occurred")


class TestContextRetrievalServiceInit:
    """Test service initialization."""

    def test_init_initializes_services(self):
        """Test initializes all dependencies."""
        with patch("src.services.context_retrieval_service.get_vector_service"):
            with patch("src.services.context_retrieval_service.redis_service"):
                with patch("src.services.context_retrieval_service.neo4j_service"):
                    service = ContextRetrievalService()
                    assert service.vector_service is not None
                    assert service.redis_service is not None
                    assert service.neo4j_service is not None

    def test_init_sets_up_stats(self):
        """Test initializes statistics tracking."""
        with patch("src.services.context_retrieval_service.get_vector_service"):
            with patch("src.services.context_retrieval_service.redis_service"):
                with patch("src.services.context_retrieval_service.neo4j_service"):
                    service = ContextRetrievalService()
                    assert service.stats["total_retrievals"] == 0
                    assert service.stats["vector_hits"] == 0
                    assert service.stats["vector_misses"] == 0


class TestRetrieveContextForPrompt:
    """Test main context retrieval method."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_structure(self, context_retrieval_service):
        """Test returns correct response structure."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[]
        )
        context_retrieval_service.vector_service.search_similar_learnings = AsyncMock(
            return_value=[]
        )

        result = await context_retrieval_service.retrieve_context_for_prompt(
            "test prompt"
        )

        assert "prompt" in result
        assert "is_greenfield" in result
        assert "similar_prompts" in result
        assert "similar_learnings" in result
        assert "entity_context" in result
        assert "facts" in result
        assert "retrieval_time_ms" in result

    @pytest.mark.asyncio
    async def test_retrieve_sets_greenfield_true_on_no_results(
        self, context_retrieval_service
    ):
        """Test sets is_greenfield=True when no similar content found."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[]
        )
        context_retrieval_service.vector_service.search_similar_learnings = AsyncMock(
            return_value=[]
        )

        result = await context_retrieval_service.retrieve_context_for_prompt(
            "unique prompt"
        )

        assert result["is_greenfield"] is True

    @pytest.mark.asyncio
    async def test_retrieve_sets_greenfield_false_on_hits(
        self, context_retrieval_service
    ):
        """Test sets is_greenfield=False when similar content found."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[{"text": "similar prompt", "neo4j_uuid": "uuid1"}]
        )

        result = await context_retrieval_service.retrieve_context_for_prompt(
            "test prompt"
        )

        assert result["is_greenfield"] is False

    @pytest.mark.asyncio
    async def test_retrieve_searches_learnings_when_enabled(
        self, context_retrieval_service
    ):
        """Test searches learnings when include_learnings=True."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[]
        )
        context_retrieval_service.vector_service.search_similar_learnings = AsyncMock(
            return_value=[{"title": "Learning", "neo4j_uuid": "uuid1"}]
        )

        result = await context_retrieval_service.retrieve_context_for_prompt(
            "test", include_learnings=True
        )

        context_retrieval_service.vector_service.search_similar_learnings.assert_called_once()
        assert len(result["similar_learnings"]) > 0

    @pytest.mark.asyncio
    async def test_retrieve_skips_learnings_when_disabled(
        self, context_retrieval_service
    ):
        """Test skips learnings when include_learnings=False."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[]
        )

        result = await context_retrieval_service.retrieve_context_for_prompt(
            "test", include_learnings=False
        )

        context_retrieval_service.vector_service.search_similar_learnings.assert_not_called()

    @pytest.mark.asyncio
    async def test_retrieve_tracks_statistics(self, context_retrieval_service):
        """Test retrieval statistics are tracked."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[]
        )
        context_retrieval_service.vector_service.search_similar_learnings = AsyncMock(
            return_value=[]
        )

        await context_retrieval_service.retrieve_context_for_prompt("test")
        await context_retrieval_service.retrieve_context_for_prompt("test2")

        assert context_retrieval_service.stats["total_retrievals"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_calculates_retrieval_time(self, context_retrieval_service):
        """Test calculates and returns retrieval time."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            return_value=[]
        )
        context_retrieval_service.vector_service.search_similar_learnings = AsyncMock(
            return_value=[]
        )

        result = await context_retrieval_service.retrieve_context_for_prompt("test")

        assert result["retrieval_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_retrieve_handles_cascade_errors_gracefully(
        self, context_retrieval_service
    ):
        """Test handles errors gracefully and continues."""
        context_retrieval_service.vector_service.search_similar_prompts = AsyncMock(
            side_effect=Exception("Vector service error")
        )

        result = await context_retrieval_service.retrieve_context_for_prompt("test")

        assert "error" in result
        assert isinstance(result["retrieval_time_ms"], (int, float))


class TestRetrieveContextForPromptWithEmbedding:
    """Test context retrieval with pre-computed embeddings."""

    @pytest.mark.asyncio
    async def test_retrieve_with_embedding_uses_cached_embedding(
        self, context_retrieval_service
    ):
        """Test uses cached embedding instead of computing new one."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        context_retrieval_service.vector_service.search_similar_prompts_by_embedding = (
            AsyncMock(return_value=[])
        )
        context_retrieval_service.vector_service.search_similar_learnings_by_embedding = AsyncMock(
            return_value=[]
        )

        result = (
            await context_retrieval_service.retrieve_context_for_prompt_with_embedding(
                "test prompt", embedding, include_learnings=True
            )
        )

        # Verify embedding-based methods were called
        context_retrieval_service.vector_service.search_similar_prompts_by_embedding.assert_called()
        context_retrieval_service.vector_service.search_similar_learnings_by_embedding.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_with_embedding_returns_structure(
        self, context_retrieval_service
    ):
        """Test returns correct response structure."""
        embedding = [0.1, 0.2, 0.3]
        context_retrieval_service.vector_service.search_similar_prompts_by_embedding = (
            AsyncMock(return_value=[])
        )
        context_retrieval_service.vector_service.search_similar_learnings_by_embedding = AsyncMock(
            return_value=[]
        )

        result = (
            await context_retrieval_service.retrieve_context_for_prompt_with_embedding(
                "test", embedding
            )
        )

        assert "retrieval_time_ms" in result
        assert "is_greenfield" in result


class TestExtractEntitiesFromMatches:
    """Test entity extraction from matches."""

    @pytest.mark.asyncio
    async def test_extracts_unique_entity_uuids(self, context_retrieval_service):
        """Test extracts unique UUIDs from matches."""
        matches = [
            {"neo4j_uuid": "uuid1", "text": "match1"},
            {"neo4j_uuid": "uuid2", "text": "match2"},
            {"neo4j_uuid": "uuid1", "text": "match3"},
        ]

        result = await context_retrieval_service._extract_entities_from_matches(matches)

        assert len(result) == 2
        assert "uuid1" in result
        assert "uuid2" in result

    @pytest.mark.asyncio
    async def test_handles_missing_uuids(self, context_retrieval_service):
        """Test ignores matches without neo4j_uuid."""
        matches = [
            {"text": "match without uuid"},
            {"neo4j_uuid": "uuid1", "text": "match with uuid"},
        ]

        result = await context_retrieval_service._extract_entities_from_matches(matches)

        assert len(result) == 1
        assert "uuid1" in result


class TestQueryEntityContext:
    """Test entity context querying."""

    @pytest.mark.asyncio
    async def test_query_returns_empty_for_empty_uuids(self, context_retrieval_service):
        """Test returns empty list for empty UUID set."""
        result = await context_retrieval_service._query_entity_context(set())

        assert result == []

    @pytest.mark.asyncio
    async def test_query_returns_entity_context(self, context_retrieval_service):
        """Test returns entity context dictionaries."""
        mock_entity = MagicMock()
        mock_entity.get = MagicMock(
            side_effect=lambda k, d=None: {"uuid": "uuid1", "name": "Entity1"}.get(k, d)
        )
        mock_entity.labels = ["Node"]

        mock_record = {"e": mock_entity, "relationships": []}
        context_retrieval_service.neo4j_service.query = AsyncMock(
            return_value=[mock_record]
        )

        result = await context_retrieval_service._query_entity_context({"uuid1"})

        assert len(result) == 1
        assert result[0]["uuid"] == "uuid1"
        assert result[0]["name"] == "Entity1"


class TestGetRelatedFacts:
    """Test related facts retrieval."""

    @pytest.mark.asyncio
    async def test_get_facts_returns_empty_for_empty_uuids(
        self, context_retrieval_service
    ):
        """Test returns empty list for empty UUID set."""
        result = await context_retrieval_service._get_related_facts(set())

        assert result == []

    @pytest.mark.asyncio
    async def test_get_facts_returns_fact_dictionaries(self, context_retrieval_service):
        """Test returns facts as dictionaries."""
        mock_record = {
            "entity1": "Entity1",
            "relation": "RELATES_TO",
            "entity2": "Entity2",
            "created_at": "2025-01-15",
            "context": "Some context",
        }
        context_retrieval_service.neo4j_service.query = AsyncMock(
            return_value=[mock_record]
        )

        result = await context_retrieval_service._get_related_facts({"uuid1"})

        assert len(result) == 1
        assert result[0]["entity1"] == "Entity1"
        assert result[0]["relation"] == "RELATES_TO"


class TestStorePromptForFutureRetrieval:
    """Test prompt storage for future retrieval."""

    @pytest.mark.asyncio
    async def test_store_calls_vector_service(self, context_retrieval_service):
        """Test calls vector service to store prompt."""
        context_retrieval_service.vector_service.store_prompt_embedding = AsyncMock(
            return_value=True
        )

        result = await context_retrieval_service.store_prompt_for_future_retrieval(
            "test prompt", "prompt-id-1", metadata={"session": "session1"}
        )

        assert result is True
        context_retrieval_service.vector_service.store_prompt_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_handles_errors(self, context_retrieval_service):
        """Test returns False on error."""
        context_retrieval_service.vector_service.store_prompt_embedding = AsyncMock(
            side_effect=Exception("Storage error")
        )

        result = await context_retrieval_service.store_prompt_for_future_retrieval(
            "test", "prompt-id-1"
        )

        assert result is False


class TestGetStats:
    """Test statistics retrieval."""

    def test_get_stats_returns_dict(self, context_retrieval_service):
        """Test returns statistics dictionary."""
        stats = context_retrieval_service.get_stats()

        assert isinstance(stats, dict)
        assert "total_retrievals" in stats
        assert "vector_hits" in stats
        assert "hit_rate_percent" in stats

    def test_get_stats_calculates_hit_rate(self, context_retrieval_service):
        """Test calculates hit rate percentage."""
        context_retrieval_service.stats["total_retrievals"] = 10
        context_retrieval_service.stats["vector_hits"] = 5

        stats = context_retrieval_service.get_stats()

        assert stats["hit_rate_percent"] == 50.0

    def test_get_stats_returns_zero_hit_rate_on_no_retrievals(
        self, context_retrieval_service
    ):
        """Test returns 0 hit rate when no retrievals."""
        context_retrieval_service.stats["total_retrievals"] = 0

        stats = context_retrieval_service.get_stats()

        assert stats["hit_rate_percent"] == 0


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_calls_vector_service(self, context_retrieval_service):
        """Test calls vector service health check."""
        context_retrieval_service.vector_service.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )

        result = await context_retrieval_service.health_check()

        assert result is not None
        context_retrieval_service.vector_service.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check_returns_structure(self, context_retrieval_service):
        """Test health check returns correct structure."""
        context_retrieval_service.vector_service.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )

        result = await context_retrieval_service.health_check()

        assert "status" in result
        assert "vector_layer" in result
        assert "stats" in result


class TestNeo4jMethodRegression:
    """Regression: neo4j_service.query() must be used, not execute_read().

    Bug: context_retrieval_service.py called neo4j_service.execute_read()
    which does not exist (only query() exists), crashing the hot retrieval path.
    Fixed 2026-03-26.
    """

    @pytest.mark.asyncio
    async def test_query_entity_context_calls_query_not_execute_read(
        self, context_retrieval_service
    ):
        """Regression: _query_entity_context must call neo4j_service.query()."""
        context_retrieval_service.neo4j_service.query = AsyncMock(return_value=[])

        await context_retrieval_service._query_entity_context({"uuid1"})

        context_retrieval_service.neo4j_service.query.assert_called_once()
        assert (
            not hasattr(context_retrieval_service.neo4j_service, "execute_read")
            or not context_retrieval_service.neo4j_service.execute_read.called
        )

    @pytest.mark.asyncio
    async def test_get_related_facts_calls_query_not_execute_read(
        self, context_retrieval_service
    ):
        """Regression: _get_related_facts must call neo4j_service.query()."""
        context_retrieval_service.neo4j_service.query = AsyncMock(return_value=[])

        await context_retrieval_service._get_related_facts({"uuid1"})

        context_retrieval_service.neo4j_service.query.assert_called_once()
        assert (
            not hasattr(context_retrieval_service.neo4j_service, "execute_read")
            or not context_retrieval_service.neo4j_service.execute_read.called
        )

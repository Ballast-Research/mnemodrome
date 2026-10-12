"""
Tests for Mnemodrome Code-First API

Tests the MnemodromeAPI class structure, initialization, and lifecycle management.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.interfaces.code_first import MnemodromeAPI


class TestMnemodromeAPIStructure:
    """Test class structure and initialization."""

    def test_create_instance(self) -> None:
        """Can create MnemodromeAPI instance."""
        api = MnemodromeAPI()
        assert api is not None
        assert isinstance(api, MnemodromeAPI)

    def test_initial_state_not_initialized(self) -> None:
        """New instance starts in uninitialized state."""
        api = MnemodromeAPI()
        assert api._initialized is False

    def test_repr_not_initialized(self) -> None:
        """String representation shows initialization state."""
        api = MnemodromeAPI()
        assert "not initialized" in repr(api)

    @pytest.mark.asyncio
    async def test_close_when_not_initialized_safe(self) -> None:
        """Calling close() when not initialized is safe (idempotent)."""
        api = MnemodromeAPI()
        # Should not raise
        await api.close()

    def test_check_initialized_raises_when_not_initialized(self) -> None:
        """_check_initialized() raises RuntimeError when not initialized."""
        api = MnemodromeAPI()
        with pytest.raises(RuntimeError, match="not initialized"):
            api._check_initialized()


class TestMnemodromeAPILifecycle:
    """Test initialization and cleanup lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_changes_state(self) -> None:
        """initialize() changes _initialized flag and connects to services."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock successful connections
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            api = MnemodromeAPI()
            await api.initialize()

            # Verify state changed
            assert api._initialized is True

            # Verify connections attempted
            mock_neo4j.connect.assert_called_once()
            mock_neo4j.ping.assert_called_once()
            mock_redis.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialize_raises(self) -> None:
        """Calling initialize() twice raises RuntimeError."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            api = MnemodromeAPI()
            await api.initialize()

            # Second initialization should raise
            with pytest.raises(RuntimeError, match="already initialized"):
                await api.initialize()

    @pytest.mark.asyncio
    async def test_redis_graceful_degradation(self) -> None:
        """Redis connection failure doesn't prevent initialization."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock Neo4j success, Redis failure
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock(side_effect=Exception("Redis unavailable"))

            api = MnemodromeAPI()
            await api.initialize()

            # Should still be initialized despite Redis failure
            assert api._initialized is True

    @pytest.mark.asyncio
    async def test_neo4j_failure_raises_and_cleans_up(self) -> None:
        """Neo4j connection failure raises RuntimeError and cleans up."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock Neo4j ping failure
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock(side_effect=Exception("Neo4j down"))
            mock_neo4j.close = AsyncMock()
            mock_redis.connect = AsyncMock()
            mock_redis.disconnect = AsyncMock()
            mock_redis.is_connected = False

            api = MnemodromeAPI()

            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Neo4j connection failed"):
                await api.initialize()

            # Should not be initialized
            assert api._initialized is False

    @pytest.mark.asyncio
    async def test_close_disconnects_services(self) -> None:
        """close() disconnects both Neo4j and Redis."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_neo4j.close = AsyncMock()
            mock_redis.connect = AsyncMock()
            mock_redis.disconnect = AsyncMock()
            mock_redis.is_connected = True

            api = MnemodromeAPI()
            await api.initialize()
            await api.close()

            # Verify disconnections
            mock_redis.disconnect.assert_called_once()
            mock_neo4j.close.assert_called_once()
            assert api._initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Can use as async context manager."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock services
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_neo4j.close = AsyncMock()
            mock_redis.connect = AsyncMock()
            mock_redis.disconnect = AsyncMock()
            mock_redis.is_connected = True

            async with MnemodromeAPI() as api:
                assert api._initialized is True

            # Should be closed after context exit
            assert api._initialized is False
            mock_neo4j.close.assert_called_once()


class TestMnemodromeAPIMethods:
    """Test API methods."""

    @pytest.mark.asyncio
    async def test_load_context_requires_initialization(self) -> None:
        """load_context() raises if not initialized."""
        api = MnemodromeAPI()
        with pytest.raises(RuntimeError, match="not initialized"):
            await api.load_context()

    @pytest.mark.asyncio
    async def test_load_context_with_prompt(self) -> None:
        """load_context() extracts entities and queries temporal memory."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.graph_factory") as mock_graph,
            patch("src.interfaces.code_first.classification_service") as mock_classify,
            patch("src.interfaces.code_first.entity_factory") as mock_entity,
            patch("src.interfaces.code_first.format_context_markdown") as mock_format,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock entity extraction
            mock_graph._extract_with_llm = AsyncMock(
                return_value={"entities": [{"name": "Ollama", "role": "entity"}]}
            )
            mock_classify.classify_batch = AsyncMock(return_value=["Technology"])
            mock_entity.create_or_resolve = AsyncMock(return_value="uuid-123")

            # Mock temporal query
            mock_neo4j.temporal_query_by_entities = AsyncMock(
                return_value=[
                    {"entity1": "Ollama", "relation": "RUNS_ON", "entity2": "server-01"}
                ]
            )
            mock_format.return_value = "# Context\n\nOllama RUNS_ON server-01"

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.load_context("Ollama infrastructure", days=7, limit=10)

            # Verify entity extraction called
            mock_graph._extract_with_llm.assert_called_once()

            # Verify temporal query called with entity UUIDs
            mock_neo4j.temporal_query_by_entities.assert_called_once_with(
                entity_uuids=["uuid-123"], days=7, limit=10
            )

            # Verify result formatted
            assert result == "# Context\n\nOllama RUNS_ON server-01"

    @pytest.mark.asyncio
    async def test_load_context_without_prompt(self) -> None:
        """load_context() queries all recent context when no prompt."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.format_context_markdown") as mock_format,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock temporal query
            mock_neo4j.temporal_query = AsyncMock(
                return_value=[
                    {"entity1": "Test", "relation": "TEST", "entity2": "Data"}
                ]
            )
            mock_format.return_value = "# Recent Context\n\nTest TEST Data"

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.load_context(days=3, limit=20)

            # Verify temporal query called with no keyword
            mock_neo4j.temporal_query.assert_called_once_with(
                keyword=None, days=3, limit=20
            )

            assert result == "# Recent Context\n\nTest TEST Data"

    @pytest.mark.asyncio
    async def test_load_context_entity_extraction_fallback(self) -> None:
        """load_context() falls back to keyword when entity extraction fails."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.graph_factory") as mock_graph,
            patch("src.interfaces.code_first.extract_keywords") as mock_keywords,
            patch("src.interfaces.code_first.format_context_markdown") as mock_format,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock entity extraction failure
            mock_graph._extract_with_llm = AsyncMock(side_effect=Exception("LLM error"))

            # Mock keyword extraction
            mock_keywords.return_value = "infrastructure"

            # Mock temporal query
            mock_neo4j.temporal_query = AsyncMock(return_value=[])
            mock_format.return_value = "# No results"

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.load_context("infrastructure")

            # Verify keyword fallback used
            mock_keywords.assert_called_once()
            mock_neo4j.temporal_query.assert_called_once_with(
                keyword="infrastructure", days=7, limit=10
            )

    @pytest.mark.asyncio
    async def test_capture_fact_requires_initialization(self) -> None:
        """capture_fact() raises if not initialized."""
        api = MnemodromeAPI()
        with pytest.raises(RuntimeError, match="not initialized"):
            await api.capture_fact("Entity1", "RELATES_TO", "Entity2")

    @pytest.mark.asyncio
    async def test_capture_fact_with_context(self) -> None:
        """capture_fact() classifies, resolves, and creates relationship."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.classification_service") as mock_classify,
            patch("src.interfaces.code_first.entity_factory") as mock_entity,
            patch(
                "src.interfaces.code_first.relationship_factory"
            ) as mock_relationship,
            patch("src.interfaces.code_first.file_service") as mock_pai,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock classification
            mock_classify.classify_batch = AsyncMock(
                return_value=["Technology", "Server"]
            )

            # Mock entity resolution
            mock_entity.create_or_resolve = AsyncMock(
                side_effect=["uuid-ollama", "uuid-server-01"]
            )

            # Mock relationship creation
            mock_relationship.create = AsyncMock(
                return_value={
                    "source_uuid": "uuid-ollama",
                    "source_name": "Ollama",
                    "relation_type": "RUNS_ON",
                    "target_uuid": "uuid-server-01",
                    "target_name": "server-01",
                }
            )

            # Mock markdown write
            from pathlib import Path

            mock_pai.write_file.return_value = Path("/test/fact-ollama-20250101.md")

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_fact(
                "Ollama", "RUNS_ON", "server-01", context="User-level service"
            )

            # Verify classification called
            assert mock_classify.classify_batch.call_count == 1

            # Verify entity resolution called
            assert mock_entity.create_or_resolve.call_count == 2

            # Verify relationship created
            mock_relationship.create.assert_called_once_with(
                source_uuid="uuid-ollama",
                relation_type="RUNS_ON",
                target_uuid="uuid-server-01",
                context="User-level service",
            )

            # Verify result structure
            assert result["status"] == "complete"
            assert result["entity1_uuid"] == "uuid-ollama"
            assert result["entity2_uuid"] == "uuid-server-01"
            assert result["entity1_type"] == "Technology"
            assert result["entity2_type"] == "Server"
            assert result["markdown_path"] is not None

    @pytest.mark.asyncio
    async def test_capture_fact_with_type_overrides(self) -> None:
        """capture_fact() skips classification when types provided."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.classification_service") as mock_classify,
            patch("src.interfaces.code_first.entity_factory") as mock_entity,
            patch(
                "src.interfaces.code_first.relationship_factory"
            ) as mock_relationship,
            patch("src.interfaces.code_first.file_service") as mock_pai,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock entity resolution
            mock_entity.create_or_resolve = AsyncMock(
                side_effect=["uuid-alice", "uuid-mnemodrome"]
            )

            # Mock relationship creation
            mock_relationship.create = AsyncMock(
                return_value={
                    "source_uuid": "uuid-alice",
                    "source_name": "Alice",
                    "relation_type": "DEVELOPS",
                    "target_uuid": "uuid-mnemodrome",
                    "target_name": "Mnemodrome",
                }
            )

            # Mock markdown write
            from pathlib import Path

            mock_pai.write_file.return_value = Path("/test/fact.md")

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_fact(
                "Alice",
                "DEVELOPS",
                "Mnemodrome",
                entity1_type="Person",
                entity2_type="Project",
            )

            # Verify classification NOT called (types provided)
            mock_classify.classify_entity.assert_not_called()

            # Verify result has correct types
            assert result["entity1_type"] == "Person"
            assert result["entity2_type"] == "Project"

    @pytest.mark.asyncio
    async def test_capture_fact_markdown_failure_partial_success(self) -> None:
        """capture_fact() returns partial status when markdown fails."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.classification_service") as mock_classify,
            patch("src.interfaces.code_first.entity_factory") as mock_entity,
            patch(
                "src.interfaces.code_first.relationship_factory"
            ) as mock_relationship,
            patch("src.interfaces.code_first.file_service") as mock_pai,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock services
            mock_classify.classify_batch = AsyncMock(return_value=["Type1", "Type2"])
            mock_entity.create_or_resolve = AsyncMock(side_effect=["uuid1", "uuid2"])
            mock_relationship.create = AsyncMock(
                return_value={
                    "source_uuid": "uuid1",
                    "source_name": "Test",
                    "relation_type": "TEST",
                    "target_uuid": "uuid2",
                    "target_name": "Data",
                }
            )

            # Mock markdown write failure
            mock_pai.write_file.side_effect = Exception("Disk full")

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_fact("Test", "TEST", "Data")

            # Should return partial success (Neo4j succeeded, markdown failed)
            assert result["status"] == "partial"
            assert result["markdown_path"] is None

    @pytest.mark.asyncio
    async def test_capture_context_basic(self) -> None:
        """capture_context() delegates to graph_factory."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.graph_factory") as mock_graph,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock graph extraction
            mock_graph.build_from_extraction = AsyncMock(
                return_value={
                    "entities_created": 3,
                    "relationships_created": 2,
                    "entity_uuids": {
                        "Alice": "uuid1",
                        "Mnemodrome": "uuid2",
                        "Neo4j": "uuid3",
                    },
                }
            )

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_context("Alice builds Mnemodrome using Neo4j")

            # Verify delegation
            mock_graph.build_from_extraction.assert_called_once()
            assert result["entities_created"] == 3
            assert result["relationships_created"] == 2

    @pytest.mark.asyncio
    async def test_capture_learning_basic(self) -> None:
        """capture_learning() creates Neo4j node and markdown file."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.file_service") as mock_pai,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock Neo4j learning creation
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "learning-uuid"}])

            # Mock markdown write
            from pathlib import Path

            mock_pai.write_file.return_value = Path("/test/learning.md")

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_learning(
                "Test Learning", "Learning content", category="insight"
            )

            # Verify result
            assert "uuid" in result
            assert result["file_path"] == "/test/learning.md"

    @pytest.mark.asyncio
    async def test_capture_session_basic(self) -> None:
        """capture_session() creates Neo4j node and markdown file."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.file_service") as mock_pai,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock Neo4j session creation
            mock_neo4j.query = AsyncMock(return_value=[{"uuid": "session-uuid"}])

            # Mock markdown write
            from pathlib import Path

            mock_pai.write_file.return_value = Path("/test/session.md")

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.capture_session(
                "Session summary", key_insights=["Insight 1", "Insight 2"]
            )

            # Verify result
            assert "uuid" in result
            assert result["file_path"] == "/test/session.md"

    @pytest.mark.asyncio
    async def test_invalidate_fact_basic(self) -> None:
        """invalidate_fact() routes through D8 hardened primitives."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.temporal_validity_service") as mock_tvs,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # D8: invalidate_fact now calls find_relationship first, then tvs.invalidate_relationship
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "rel-uuid-1",
                    "entity1_name": "Entity1",
                    "relation_type": "RELATES_TO",
                    "entity2_name": "Entity2",
                    "invalid_at": None,
                    "expired_at": None,
                }
            )
            mock_tvs.invalidate_relationship = AsyncMock(
                return_value={
                    "status": "success",
                    "fact_uuid": "rel-uuid-1",
                    "invalidated_at": "2026-06-25T00:00:00Z",
                    "fields_set": ["invalid_at", "expired_at"],
                    "reason": "Test invalidation",
                    "superseded_by": None,
                    "relation_type": "RELATES_TO",
                }
            )

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.invalidate_fact(
                "Entity1", "RELATES_TO", "Entity2", reason="Test invalidation"
            )

            # Verify result
            assert result["status"] == "success"
            assert result["fact_invalidated"] is True
            mock_neo4j.find_relationship.assert_called_once()
            mock_tvs.invalidate_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_fact_basic(self) -> None:
        """update_fact() uses D8 atomic supersession via execute_update_transaction."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
            patch("src.interfaces.code_first.entity_factory") as mock_entity,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # D8: update_fact calls find_relationship, entity_factory.get_or_create, execute_update_transaction
            # Mock includes entity2_uuid (new in Issue 7 fix) so code_first.update_fact
            # can pass old target UUID distinctly to execute_update_transaction (D2).
            mock_neo4j.find_relationship = AsyncMock(
                return_value={
                    "uuid": "old-uuid-1",
                    "relationship_uuid": "old-uuid-1",
                    "entity1_uuid": "src-uuid-1",
                    "entity1_name": "Entity1",
                    "relation_type": "RELATES_TO",
                    "entity2_uuid": "old-tgt-uuid-1",
                    "entity2_name": "Entity2Old",
                    "invalid_at": None,
                    "expired_at": None,
                }
            )
            mock_entity.create_or_resolve = AsyncMock(return_value="tgt-uuid-new")
            mock_neo4j.execute_update_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": "old-uuid-1",
                    "new_fact_uuid": "new-uuid-1",
                    "invalidated_at": "2026-06-25T00:00:00Z",
                    "supersession_chain": ["old-uuid-1", "new-uuid-1"],
                    "new_fact_context": "",
                }
            )

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.update_fact(
                "Entity1",
                "RELATES_TO",
                entity2_old="Entity2Old",
                entity2_new="Entity2New",
            )

            # Verify result
            assert result["status"] == "success"
            assert result["invalidated_fact"] is True
            mock_neo4j.find_relationship.assert_called_once()
            mock_neo4j.execute_update_transaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        """health() returns service connectivity status."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()  # Will succeed
            mock_redis.connect = AsyncMock()
            mock_redis.is_connected = True

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.health()

            # Verify health check result
            assert result["status"] == "healthy"
            assert result["neo4j"] == "connected"
            assert result["redis"] == "connected"
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_discover_schema(self) -> None:
        """discover_schema() queries Neo4j for schema information."""
        with (
            patch("src.interfaces.code_first.neo4j_service") as mock_neo4j,
            patch("src.interfaces.code_first.redis_service") as mock_redis,
        ):
            # Mock initialization
            mock_neo4j.connect = AsyncMock()
            mock_neo4j.ping = AsyncMock()
            mock_redis.connect = AsyncMock()

            # Mock schema queries
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [{"r": {"type": "RELATES_TO"}}],  # sample relationships
                    [{"e": {"uuid": "test", "name": "Test"}}],  # sample entities
                ]
            )

            api = MnemodromeAPI()
            await api.initialize()

            result = await api.discover_schema()

            # Verify schema discovery
            assert "sample_relationships" in result
            assert "sample_entities" in result
            assert len(result["sample_relationships"]) == 1
            assert len(result["sample_entities"]) == 1

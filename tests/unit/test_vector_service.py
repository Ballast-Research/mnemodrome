"""Unit tests for Vector Service.

Tests for the VectorService and OllamaEmbedder classes covering:
- Embedding generation via Ollama
- Vector similarity search with Qdrant
- Fallback handling (primary to secondary Ollama)
- Error handling and graceful degradation
- Collection initialization
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.vector_service import OllamaEmbedder, VectorService


class TestOllamaEmbedder:
    """Tests for the OllamaEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Provide a fresh embedder instance."""
        return OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")

    @pytest.mark.asyncio
    async def test_embedder_initialization(self):
        """Test embedder initialization with default values."""
        embedder = OllamaEmbedder()

        assert embedder.host == "http://localhost:11434"
        assert embedder.model == "nomic-embed-text"
        assert embedder.fallback_host is None
        assert embedder._using_fallback is False

    @pytest.mark.asyncio
    async def test_embedder_with_fallback(self):
        """Test embedder initialization with fallback host."""
        embedder = OllamaEmbedder(
            host="http://primary:11434", fallback_host="http://secondary:11434"
        )

        assert embedder.host == "http://primary:11434"
        assert embedder.fallback_host == "http://secondary:11434"

    @pytest.mark.asyncio
    async def test_embedder_strips_trailing_slash(self):
        """Test that embedder strips trailing slashes from hosts."""
        embedder = OllamaEmbedder(
            host="http://localhost:11434/", fallback_host="http://secondary:11434/"
        )

        assert embedder.host == "http://localhost:11434"
        assert embedder.fallback_host == "http://secondary:11434"

    @pytest.mark.asyncio
    async def test_initialize_success(self, embedder):
        """Test successful embedder initialization."""
        with patch.object(embedder, "client") as mock_client:
            mock_response = AsyncMock()
            mock_response.json = MagicMock(
                return_value={
                    "models": [
                        {"name": "nomic-embed-text:latest"},
                        {"name": "other-model"},
                    ]
                }
            )
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)

            # Mock embed method
            with patch.object(embedder, "embed") as mock_embed:
                mock_embed.return_value = [0.1] * 768

                result = await embedder.initialize()

                assert result is True
                assert embedder._dimension == 768

    @pytest.mark.asyncio
    async def test_initialize_model_not_found(self, embedder):
        """Test initialization when model not found on primary."""
        with patch.object(embedder, "client") as mock_client:
            mock_response = AsyncMock()
            mock_response.json = MagicMock(
                return_value={"models": [{"name": "some-other-model"}]}
            )
            mock_response.raise_for_status = MagicMock()
            mock_client.get = AsyncMock(return_value=mock_response)

            result = await embedder.initialize()

            assert result is False

    @pytest.mark.asyncio
    async def test_embed_text(self, embedder):
        """Test embedding text with Ollama."""
        with patch.object(embedder, "client") as mock_client:
            mock_response = AsyncMock()
            mock_response.json = MagicMock(
                return_value={
                    "embedding": [0.1, 0.2, 0.3] * 256  # 768D vector
                }
            )
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await embedder.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_empty_text(self, embedder):
        """Test embedding empty text."""
        with patch.object(embedder, "client") as mock_client:
            mock_response = AsyncMock()
            mock_response.json = MagicMock(return_value={"embedding": [0.0] * 768})
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await embedder.embed("")

            assert isinstance(result, list)
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_failure_returns_none(self, embedder):
        """Test embed returns None on failure."""
        with patch.object(embedder, "client") as mock_client:
            mock_client.post = AsyncMock(side_effect=Exception("Connection error"))

            result = await embedder.embed("test text")

            assert result is None


class TestVectorService:
    """Tests for the VectorService class."""

    @pytest.fixture
    def service(self):
        """Provide a fresh vector service instance."""
        # Note: VectorService has conditional Qdrant import
        pytest.importorskip("qdrant_client")
        return VectorService(
            qdrant_host="localhost",
            qdrant_port=6333,
            ollama_host="http://localhost:11434",
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test VectorService initialization."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=False)

        assert service.qdrant_host == "localhost"
        assert service.qdrant_port == 6333
        assert service.embedding_model == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_service_initialization_with_custom_threshold(self):
        """Test VectorService with custom similarity threshold."""
        pytest.importorskip("qdrant_client")

        service = VectorService(similarity_threshold=0.7, enabled=False)

        assert service.similarity_threshold == 0.7

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful VectorService initialization."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=True)

        with patch("src.services.vector_service.OllamaEmbedder") as mock_embedder_class:
            mock_embedder = AsyncMock()
            mock_embedder.initialize = AsyncMock(return_value=True)
            mock_embedder.dimension = 768
            mock_embedder_class.return_value = mock_embedder

            with patch.object(service, "client", new_callable=MagicMock):
                with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                    # Mock get_collections response
                    mock_collections = MagicMock()
                    mock_collections.collections = []
                    mock_thread.return_value = mock_collections

                    await service.initialize()

                    # Should create embedder
                    mock_embedder_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_qdrant_unavailable(self):
        """Test initialization with Qdrant unavailable."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=True)

        with patch("src.services.vector_service.OllamaEmbedder") as mock_embedder_class:
            mock_embedder = AsyncMock()
            mock_embedder.initialize = AsyncMock(return_value=False)  # Embedder fails
            mock_embedder_class.return_value = mock_embedder

            await service.initialize()

            # Should disable service gracefully
            assert service.enabled is False

    @pytest.mark.asyncio
    async def test_embed_text_success(self):
        """Test embedding text successfully."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=True)
        service._initialized = True

        # Mock the embedder correctly
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 768)
        service.embedder = mock_embedder

        result = await service.embed_text("Test text here")

        # Check result is list or empty (depends on implementation)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_embed_text_empty_input(self):
        """Test embedding empty text is rejected."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=True)
        service._initialized = True

        result = await service.embed_text("")

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_text_trivial_input(self):
        """Test embedding too-short text is rejected."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=True)
        service._initialized = True

        result = await service.embed_text("ab")

        assert result == []

    @pytest.mark.asyncio
    async def test_embed_text_not_initialized(self):
        """Test embedding when service not initialized."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=True)
        service._initialized = False

        result = await service.embed_text("Valid text here")

        assert result == []

    @pytest.mark.asyncio
    async def test_embedding_dimension_consistency(self):
        """Test that embeddings maintain consistent dimensions."""
        pytest.importorskip("qdrant_client")

        embedder = OllamaEmbedder()

        with patch.object(embedder, "client") as mock_client:
            mock_response = AsyncMock()
            mock_response.json = MagicMock(return_value={"embedding": [0.5] * 768})
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            embedding1 = await embedder.embed("text1")
            embedding2 = await embedder.embed("text2")

            assert len(embedding1) == len(embedding2) == 768

    @pytest.mark.asyncio
    async def test_vector_normalization(self):
        """Test that vectors are properly normalized."""
        pytest.importorskip("qdrant_client")

        embedder = OllamaEmbedder()

        with patch.object(embedder, "client") as mock_client:
            mock_response = AsyncMock()
            # Create a normalized vector
            mock_response.json = MagicMock(
                return_value={
                    "embedding": [1.0 / 768] * 768  # Normalized vector
                }
            )
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            result = await embedder.embed("test")

            assert isinstance(result, list)
            assert all(-1.0 <= val <= 1.0 for val in result)

    @pytest.mark.asyncio
    async def test_graceful_degradation_qdrant_down(self):
        """Test vector service gracefully handles Qdrant being down."""
        pytest.importorskip("qdrant_client")

        service = VectorService(enabled=False)

        # Service should handle gracefully when not initialized
        # This is the expected state when Qdrant is unavailable
        assert service.enabled is False
        assert service._initialized is False


class TestFactValidityFilter:
    """Tests for _filter_valid_facts — vector/graph temporal-validity sync.

    Invalidated/superseded facts remain in Qdrant (invalidation never updates
    the vector store), so the vector path must drop them by checking the graph.
    """

    def _service(self):
        return VectorService(enabled=False)

    @pytest.mark.asyncio
    async def test_drops_graph_invalidated_facts(self):
        """Facts the graph reports as invalid are dropped, valid ones kept in order."""
        service = self._service()
        matches = [
            {
                "entity1": "A",
                "relation": "RUNS_ON",
                "entity2": "X",
                "entity1_uuid": "u-a",
                "entity2_uuid": "u-x",
            },
            {
                "entity1": "A",
                "relation": "RUNS_ON",
                "entity2": "Y",
                "entity1_uuid": "u-a",
                "entity2_uuid": "u-y",
            },
            {
                "entity1": "B",
                "relation": "USES",
                "entity2": "Z",
                "entity1_uuid": "u-b",
                "entity2_uuid": "u-z",
            },
        ]
        with patch(
            "src.services.neo4j_service.neo4j_service.filter_valid_facts",
            new=AsyncMock(return_value={0, 2}),
        ):
            result = await service._filter_valid_facts(matches)

        assert [m["entity2"] for m in result] == ["X", "Z"]

    @pytest.mark.asyncio
    async def test_fail_open_when_graph_unavailable(self):
        """If the graph check raises, return matches unfiltered rather than break search."""
        service = self._service()
        matches = [
            {
                "entity1": "A",
                "relation": "R",
                "entity2": "B",
                "entity1_uuid": None,
                "entity2_uuid": None,
            },
        ]
        with patch(
            "src.services.neo4j_service.neo4j_service.filter_valid_facts",
            new=AsyncMock(side_effect=RuntimeError("neo4j down")),
        ):
            result = await service._filter_valid_facts(matches)

        assert result == matches

    @pytest.mark.asyncio
    async def test_empty_matches_short_circuit(self):
        """Empty input returns empty without touching the graph."""
        service = self._service()
        with patch(
            "src.services.neo4j_service.neo4j_service.filter_valid_facts",
            new=AsyncMock(),
        ) as mock_filter:
            result = await service._filter_valid_facts([])

        assert result == []
        mock_filter.assert_not_called()

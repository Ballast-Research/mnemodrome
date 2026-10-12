"""Unit tests for entity factory service (src/services/entity_factory.py).

Tests the EntityFactory class which handles UUID-based entity creation and resolution.
Uses mocks for Neo4j and Redis dependencies.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.entity_factory import EntityFactory


@pytest.fixture
def entity_factory():
    """Provide a fresh EntityFactory instance for each test."""
    return EntityFactory()


@pytest.mark.asyncio
class TestEntityFactoryCreateOrResolve:
    """Tests for create_or_resolve method."""

    async def test_create_or_resolve_valid_type(self, entity_factory):
        """Test creating entity with valid type."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock(return_value=None)
                mock_neo4j.query = AsyncMock(return_value=[{"uuid": "test-uuid-123"}])

                result = await entity_factory.create_or_resolve(
                    "Alice", "Person", "A developer"
                )

                assert result == "test-uuid-123"
                mock_neo4j.query.assert_called_once()
                mock_cache.set.assert_called_once()

    async def test_create_or_resolve_cache_hit(self, entity_factory):
        """Test that cached UUID is returned without Neo4j query."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_cache.get = AsyncMock(return_value="cached-uuid-456")

                result = await entity_factory.create_or_resolve(
                    "Alice", "Person", "A developer"
                )

                assert result == "cached-uuid-456"
                mock_neo4j.query.assert_not_called()

    async def test_create_or_resolve_invalid_type_fallback(self, entity_factory):
        """Test that invalid type falls back to 'Entity'."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock(return_value=None)
                mock_neo4j.query = AsyncMock(return_value=[{"uuid": "fallback-uuid"}])

                result = await entity_factory.create_or_resolve(
                    "Test", "InvalidType", "Test entity"
                )

                assert result == "fallback-uuid"
                # Check that query was called with Entity type
                call_args = mock_neo4j.query.call_args
                assert "Entity" in call_args[0][0]

    async def test_create_or_resolve_empty_summary(self, entity_factory):
        """Test creating entity with empty summary."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock(return_value=None)
                mock_neo4j.query = AsyncMock(return_value=[{"uuid": "empty-uuid"}])

                result = await entity_factory.create_or_resolve("NoSummary", "Concept")

                assert result == "empty-uuid"

    async def test_create_or_resolve_neo4j_failure(self, entity_factory):
        """Test handling of Neo4j query failure."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_cache.get = AsyncMock(return_value=None)
                mock_neo4j.query = AsyncMock(side_effect=Exception("Neo4j error"))

                with pytest.raises(Exception):
                    await entity_factory.create_or_resolve("Test", "Person")


@pytest.mark.asyncio
class TestEntityFactoryBatchCreateOrResolve:
    """Tests for batch_create_or_resolve method."""

    async def test_batch_create_empty_list(self, entity_factory):
        """Test batch creation with empty list."""
        result = await entity_factory.batch_create_or_resolve([])
        assert result == {}

    async def test_batch_create_with_cache_hits(self, entity_factory):
        """Test batch creation with some cache hits."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                # First entity cached, second needs Neo4j
                mock_cache.batch_get = AsyncMock(
                    return_value={"Alice:Person": "cached-uuid-1"}
                )
                # Ensure get is async too for cache misses
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock(return_value=None)
                mock_neo4j.query = AsyncMock(return_value=[{"uuid": "new-uuid-2"}])

                entities = [
                    ("Alice", "Person", "Developer"),
                    ("Neo4j", "Technology", "Database"),
                ]

                result = await entity_factory.batch_create_or_resolve(entities)

                assert "Alice:Person" in result
                assert len(result) >= 1

    async def test_batch_create_all_cache_misses(self, entity_factory):
        """Test batch creation with all cache misses."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_cache.batch_get = AsyncMock(return_value={})
                # Ensure get is async too for cache misses
                mock_cache.get = AsyncMock(return_value=None)
                mock_cache.set = AsyncMock(return_value=None)
                mock_neo4j.query = AsyncMock(
                    side_effect=[
                        [{"uuid": "uuid-1"}],
                        [{"uuid": "uuid-2"}],
                    ]
                )

                entities = [
                    ("Entity1", "Person", ""),
                    ("Entity2", "Technology", ""),
                ]

                result = await entity_factory.batch_create_or_resolve(entities)

                assert len(result) == 2


@pytest.mark.asyncio
class TestEntityFactoryRetrieval:
    """Tests for entity retrieval methods."""

    async def test_get_entity_by_uuid_found(self, entity_factory):
        """Test retrieving entity by UUID when found."""
        with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": "test-uuid",
                        "name": "TestEntity",
                        "entity_type": "Person",
                        "summary": "Test summary",
                    }
                ]
            )

            result = await entity_factory.get_entity_by_uuid("test-uuid")

            assert result is not None
            assert result["uuid"] == "test-uuid"
            assert result["name"] == "TestEntity"

    async def test_get_entity_by_uuid_not_found(self, entity_factory):
        """Test retrieving entity by UUID when not found."""
        with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            result = await entity_factory.get_entity_by_uuid("nonexistent-uuid")

            assert result is None

    async def test_get_entity_by_name_with_type(self, entity_factory):
        """Test retrieving entity by name with type filter."""
        with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": "uuid-123",
                        "name": "Alice",
                        "entity_type": "Person",
                        "summary": "Developer",
                    }
                ]
            )

            result = await entity_factory.get_entity_by_name("Alice", "Person")

            assert result is not None
            assert result["name"] == "Alice"
            assert result["entity_type"] == "Person"

    async def test_get_entity_by_name_without_type(self, entity_factory):
        """Test retrieving entity by name without type filter."""
        with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "uuid": "uuid-456",
                        "name": "Test",
                        "entity_type": "Concept",
                        "summary": "",
                    }
                ]
            )

            result = await entity_factory.get_entity_by_name("Test")

            assert result is not None
            assert result["name"] == "Test"


@pytest.mark.asyncio
class TestEntityFactoryEnrichment:
    """Tests for entity enrichment."""

    async def test_enrich_entity_success(self, entity_factory):
        """Test enriching entity with new summary."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
                mock_neo4j.query = AsyncMock(
                    return_value=[
                        {
                            "uuid": "uuid-to-enrich",
                            "name": "Alice",
                            "entity_type": "Person",
                        }
                    ]
                )
                mock_cache.delete = AsyncMock(return_value=1)

                result = await entity_factory.enrich_entity(
                    "uuid-to-enrich", "Updated developer summary"
                )

                assert result is True
                mock_cache.delete.assert_called_once()

    async def test_enrich_entity_not_found(self, entity_factory):
        """Test enriching non-existent entity."""
        with patch("src.services.entity_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            result = await entity_factory.enrich_entity(
                "nonexistent-uuid", "New summary"
            )

            assert result is False


@pytest.mark.asyncio
class TestEntityFactoryCacheInvalidation:
    """Tests for cache invalidation."""

    async def test_invalidate_entity_cache(self, entity_factory):
        """Test invalidating entity cache."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            mock_cache.delete = AsyncMock(return_value=1)

            await entity_factory.invalidate_entity_cache("Alice", "Person")

            mock_cache.delete.assert_called_once_with("Alice", "Person")

    async def test_invalidate_entity_cache_failure(self, entity_factory):
        """Test handling cache invalidation failure."""
        with patch("src.services.entity_factory.entity_cache") as mock_cache:
            mock_cache.delete = AsyncMock(side_effect=Exception("Cache error"))

            # Should not raise, but log error
            await entity_factory.invalidate_entity_cache("Alice", "Person")


class TestEntityFactoryValidation:
    """Tests for entity type validation."""

    def test_validate_entity_type_valid(self, entity_factory):
        """Test validation of valid entity type."""
        result = entity_factory.validate_entity_type("Person")
        assert result is True

    def test_validate_entity_type_invalid(self, entity_factory):
        """Test validation of invalid entity type."""
        result = entity_factory.validate_entity_type("InvalidType")
        assert result is False

    def test_get_valid_types(self, entity_factory):
        """Test getting set of valid entity types."""
        valid_types = entity_factory.get_valid_types()

        assert isinstance(valid_types, set)
        assert "Person" in valid_types
        assert "Project" in valid_types
        assert "Technology" in valid_types
        assert "Entity" in valid_types

    def test_get_valid_types_is_copy(self, entity_factory):
        """Test that get_valid_types returns a copy."""
        types1 = entity_factory.get_valid_types()
        types2 = entity_factory.get_valid_types()

        assert types1 == types2
        # Modifying returned set shouldn't affect internal state
        types1.add("NewType")
        types3 = entity_factory.get_valid_types()
        assert "NewType" not in types3

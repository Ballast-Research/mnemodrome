"""Unit tests for Neo4j Service.

Tests for the Neo4jService class covering:
- Connection and disconnection
- Entity creation and deletion
- Relationship management
- Temporal queries
- Batch operations
- Error handling
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.services.neo4j_service import Neo4jService


class TestNeo4jService:
    """Tests for the Neo4jService class."""

    @pytest.fixture
    def service(self):
        """Provide a fresh Neo4j service instance."""
        return Neo4jService()

    @pytest.fixture
    def mock_driver(self):
        """Provide a mock Neo4j driver."""
        return AsyncMock()

    # ========================
    # Connection Tests
    # ========================

    @pytest.mark.asyncio
    async def test_connect_creates_driver(self, service):
        """Test that connect() creates a Neo4j driver with correct config."""
        with patch(
            "src.services.neo4j_service.AsyncGraphDatabase.driver"
        ) as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver_factory.return_value = mock_driver

            await service.connect()

            assert service.driver is not None
            mock_driver_factory.assert_called_once()
            # Check config parameters
            args, kwargs = mock_driver_factory.call_args
            assert kwargs["max_connection_pool_size"] == 200
            assert kwargs["connection_timeout"] == 30.0
            assert kwargs["max_connection_lifetime"] == 3600

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, service):
        """Test that connect() is idempotent - doesn't recreate driver."""
        with patch(
            "src.services.neo4j_service.AsyncGraphDatabase.driver"
        ) as mock_driver_factory:
            mock_driver = AsyncMock()
            mock_driver_factory.return_value = mock_driver

            await service.connect()
            await service.connect()

            # Should only be called once
            assert mock_driver_factory.call_count == 1

    @pytest.mark.asyncio
    async def test_close_closes_driver(self, service, mock_driver):
        """Test that close() properly closes the driver."""
        service.driver = mock_driver
        await service.close()

        mock_driver.close.assert_called_once()
        assert service.driver is None

    @pytest.mark.asyncio
    async def test_close_handles_missing_driver(self, service):
        """Test that close() handles case where driver is None."""
        service.driver = None
        await service.close()  # Should not raise
        assert service.driver is None

    # ========================
    # Ping Tests
    # ========================

    @pytest.mark.asyncio
    async def test_ping_success(self, service):
        """Test successful ping to Neo4j."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"ping": 1}

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.ping()

        assert result is True
        mock_result.single.assert_called_once()

    @pytest.mark.asyncio
    async def test_ping_failure(self, service):
        """Test ping failure when driver raises exception."""
        mock_driver = AsyncMock()
        mock_driver.session = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            side_effect=Exception("Connection failed")
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        with pytest.raises(Exception, match="Connection failed"):
            await service.ping()

    # ========================
    # Entity Creation Tests
    # ========================

    @pytest.mark.asyncio
    async def test_create_entity_with_uuid(self, service):
        """Test creating an entity with explicit UUID."""
        entity_uuid = str(uuid4())
        entity_data = {
            "uuid": entity_uuid,
            "name": "Test Entity",
            "summary": "A test entity",
        }

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"uuid": entity_uuid}

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.create_entity(entity_data)

        assert result == entity_uuid
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert call_args[0][1]["uuid"] == entity_uuid
        assert call_args[0][1]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_create_entity_generates_uuid(self, service):
        """Test that create_entity generates UUID if not provided."""
        entity_data = {"name": "Test Entity", "summary": "A test entity"}

        mock_session = AsyncMock()
        mock_result = AsyncMock()

        generated_uuid = str(uuid4())
        mock_record = {"uuid": generated_uuid}

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.create_entity(entity_data)

        assert result == generated_uuid
        # Verify a UUID was passed to the query
        call_args = mock_session.run.call_args
        assert len(call_args[0][1]["uuid"]) > 0

    @pytest.mark.asyncio
    async def test_create_entity_failure(self, service):
        """Test entity creation failure when query returns no record."""
        entity_data = {"name": "Test Entity", "summary": "A test entity"}

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=None)  # No record returned

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        with pytest.raises(Exception, match="Failed to create entity"):
            await service.create_entity(entity_data)

    # ========================
    # Relationship Tests
    # ========================

    @pytest.mark.asyncio
    async def test_create_relationship_success(self, service):
        """Test creating a relationship between entities."""
        rel_data = {
            "entity1_name": "Entity A",
            "entity1_summary": "Summary A",
            "relation_type": "USES",
            "entity2_name": "Entity B",
            "entity2_summary": "Summary B",
            "context": "Test context",
        }

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {
            "entity1_uuid": "uuid-a",
            "entity1_name": "Entity A",
            "entity2_uuid": "uuid-b",
            "entity2_name": "Entity B",
            # D1/D8: dual-key return — both 'uuid' and 'relationship_uuid'
            "uuid": "rel-uuid-1",
            "relationship_uuid": "rel-uuid-1",
            "relation_type": "USES",
            "valid_at": "2025-03-12T10:00:00Z",
        }

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.create_relationship(**rel_data)

        assert result["entity1_uuid"] == "uuid-a"
        assert result["entity2_uuid"] == "uuid-b"
        # D8: both keys present
        assert result["uuid"] == "rel-uuid-1"
        assert result["relationship_uuid"] == "rel-uuid-1"
        assert result["relation_type"] == "USES"
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_failure(self, service):
        """Test relationship creation failure."""
        rel_data = {
            "entity1_name": "Entity A",
            "entity1_summary": "Summary A",
            "relation_type": "USES",
            "entity2_name": "Entity B",
            "entity2_summary": "Summary B",
        }

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=None)  # Failed

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        with pytest.raises(Exception, match="Failed to create relationship"):
            await service.create_relationship(**rel_data)

    # ========================
    # Entity Deletion Tests
    # ========================

    @pytest.mark.asyncio
    async def test_delete_entity_success(self, service):
        """Test successful entity deletion."""
        entity_uuid = str(uuid4())

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"deleted": 1}

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.delete_entity(entity_uuid)

        assert result is True
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_not_found(self, service):
        """Test deleting non-existent entity returns False."""
        entity_uuid = str(uuid4())

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_record = {"deleted": 0}

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.delete_entity(entity_uuid)

        assert result is False

    # ========================
    # Query Tests
    # ========================

    @pytest.mark.asyncio
    async def test_query_with_params(self, service):
        """Test query execution with parameters."""
        cypher = "MATCH (n:Entity {name: $name}) RETURN n"
        params = {"name": "Test"}
        expected_records = [{"name": "Test", "uuid": "uuid-1"}]

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.data = AsyncMock(return_value=expected_records)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.query(cypher, params)

        assert result == expected_records
        mock_session.run.assert_called_once_with(cypher, params)

    @pytest.mark.asyncio
    async def test_query_without_params(self, service):
        """Test query execution without parameters."""
        cypher = "RETURN 1 as count"
        expected_records = [{"count": 1}]

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.data = AsyncMock(return_value=expected_records)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.query(cypher)

        assert result == expected_records
        mock_session.run.assert_called_once_with(cypher, {})

    # ========================
    # Temporal Query Tests
    # ========================

    @pytest.mark.asyncio
    async def test_temporal_query_with_keyword(self, service):
        """Test temporal query with keyword filter."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()

        # Create proper async iterator
        async def async_iter():
            return iter([])

        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.__aiter__ = MagicMock(return_value=mock_result)
        mock_result.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        result = await service.temporal_query(keyword="test", days=7, limit=10)

        assert isinstance(result, list)
        mock_session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_update_access_empty_list(self, service):
        """Test batch_update_access with empty list is idempotent."""
        mock_driver = AsyncMock()
        service.driver = mock_driver

        await service.batch_update_access([])

        # Query should not be called
        assert mock_driver.session.call_count == 0

    @pytest.mark.asyncio
    async def test_batch_update_access_multiple(self, service):
        """Test batch_update_access with multiple UUIDs."""
        uuids = [str(uuid4()), str(uuid4()), str(uuid4())]

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_result.data = AsyncMock(return_value=[])

        mock_driver = AsyncMock()
        mock_driver.session = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        service.driver = mock_driver

        await service.batch_update_access(uuids)

        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert call_args[0][1]["uuids"] == uuids

"""Unit tests for temporal validity service (src/services/temporal_validity_service.py).

Tests the TemporalValidityService which manages temporal validity fields for
fact invalidation and temporal queries.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.services.temporal_validity_service import TemporalValidityService


@pytest.fixture
def temporal_service():
    """Provide a TemporalValidityService instance."""
    return TemporalValidityService()


class TestTemporalValiditySetFields:
    """Tests for set_temporal_fields method."""

    def test_set_temporal_fields_natural_invalidation(self, temporal_service):
        """Test natural invalidation sets only invalid_at."""
        result = temporal_service.set_temporal_fields(
            "uuid-123", "natural", "Project completed"
        )

        assert "invalid_at" in result
        assert "expired_at" not in result
        assert isinstance(result["invalid_at"], datetime)

    def test_set_temporal_fields_system_invalidation(self, temporal_service):
        """Test system invalidation sets only expired_at."""
        result = temporal_service.set_temporal_fields(
            "uuid-456", "system", "Duplicate detected"
        )

        assert "expired_at" in result
        assert "invalid_at" not in result
        assert isinstance(result["expired_at"], datetime)

    def test_set_temporal_fields_both_invalidation(self, temporal_service):
        """Test both invalidation sets both fields."""
        result = temporal_service.set_temporal_fields(
            "uuid-789", "both", "Complete invalidation"
        )

        assert "invalid_at" in result
        assert "expired_at" in result
        assert isinstance(result["invalid_at"], datetime)
        assert isinstance(result["expired_at"], datetime)

    def test_set_temporal_fields_default_both(self, temporal_service):
        """Test that default invalidation type is 'both'."""
        result = temporal_service.set_temporal_fields("uuid-000")

        assert "invalid_at" in result
        assert "expired_at" in result

    def test_set_temporal_fields_invalid_type(self, temporal_service):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid invalidation_type"):
            temporal_service.set_temporal_fields("uuid-111", "invalid_type")

    def test_set_temporal_fields_utc_timezone(self, temporal_service):
        """Test that timestamps use UTC timezone."""
        result = temporal_service.set_temporal_fields("uuid-test", "natural")

        timestamp = result["invalid_at"]
        assert timestamp.tzinfo is not None
        assert timestamp.tzinfo == UTC


class TestTemporalValidityBuildQuery:
    """Tests for build_temporal_query method."""

    def test_build_temporal_query_current(self, temporal_service):
        """Test building query for current relationships."""
        query = temporal_service.build_temporal_query("current")

        assert "WHERE" in query
        assert "r.invalid_at IS NULL" in query
        assert "r.expired_at IS NULL" in query

    def test_build_temporal_query_historical(self, temporal_service):
        """Test building query for historical relationships."""
        query = temporal_service.build_temporal_query("historical")

        # Historical should return empty string (no filtering)
        assert query == ""

    def test_build_temporal_query_natural_invalid(self, temporal_service):
        """Test building query for naturally invalid relationships."""
        query = temporal_service.build_temporal_query("natural")

        assert "WHERE" in query
        assert "r.invalid_at IS NOT NULL" in query
        assert "r.expired_at IS NULL" in query

    def test_build_temporal_query_system_invalid(self, temporal_service):
        """Test building query for system-invalidated relationships."""
        query = temporal_service.build_temporal_query("system")

        assert "WHERE" in query
        assert "r.expired_at IS NOT NULL" in query

    def test_build_temporal_query_invalid_scope(self, temporal_service):
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid temporal_scope"):
            temporal_service.build_temporal_query("invalid_scope")

    def test_build_temporal_query_full_integration(self, temporal_service):
        """Test building full query with temporal filtering."""
        base_query = "MATCH (a)-[r]->(b) "
        temporal_filter = temporal_service.build_temporal_query("current")
        full_query = base_query + temporal_filter + " RETURN r"

        assert "MATCH (a)-[r]->(b)" in full_query
        assert "WHERE r.invalid_at IS NULL" in full_query


@pytest.mark.asyncio
class TestTemporalValidityInvalidateRelationship:
    """Tests for invalidate_relationship method."""

    async def test_invalidate_relationship_natural(self, temporal_service):
        """Test invalidating relationship with natural type."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "fact_uuid": "uuid-123",
                    "invalid_at": "2025-03-12T10:00:00Z",
                    "expired_at": None,
                    "reason": "Project completed",
                    "superseded_by": None,
                    "relation_type": "BELONGS_TO",
                }
            ]
        )

        result = await temporal_service.invalidate_relationship(
            mock_neo4j, "uuid-123", "natural", "Project completed"
        )

        assert result["status"] == "success"
        assert result["fact_uuid"] == "uuid-123"
        assert "invalid_at" in result["fields_set"]
        assert "expired_at" not in result["fields_set"]
        assert result["reason"] == "Project completed"

    async def test_invalidate_relationship_system(self, temporal_service):
        """Test invalidating relationship with system type."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "fact_uuid": "uuid-456",
                    "invalid_at": None,
                    "expired_at": "2025-03-12T10:00:00Z",
                    "reason": "Duplicate",
                    "superseded_by": None,
                    "relation_type": "USES",
                }
            ]
        )

        result = await temporal_service.invalidate_relationship(
            mock_neo4j, "uuid-456", "system", "Duplicate"
        )

        assert result["status"] == "success"
        assert "expired_at" in result["fields_set"]
        assert "invalid_at" not in result["fields_set"]

    async def test_invalidate_relationship_with_supersession(self, temporal_service):
        """Test invalidating relationship with superseded_by link."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "fact_uuid": "uuid-old",
                    "invalid_at": "2025-03-12T10:00:00Z",
                    "expired_at": "2025-03-12T10:00:00Z",
                    "reason": "Superseded by newer fact",
                    "superseded_by": "uuid-new",
                    "relation_type": "USES",
                }
            ]
        )

        result = await temporal_service.invalidate_relationship(
            mock_neo4j,
            "uuid-old",
            "both",
            "Superseded by newer fact",
            superseded_by="uuid-new",
        )

        assert result["superseded_by"] == "uuid-new"
        assert "superseded_by" in str(mock_neo4j.query.call_args)

    async def test_invalidate_relationship_not_found(self, temporal_service):
        """Test invalidation fails when relationship not found."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(return_value=[])

        with pytest.raises(Exception, match="Relationship not found"):
            await temporal_service.invalidate_relationship(
                mock_neo4j, "nonexistent-uuid"
            )

    async def test_invalidate_relationship_query_error(self, temporal_service):
        """Test handling of query errors."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(side_effect=Exception("Neo4j error"))

        with pytest.raises(Exception):
            await temporal_service.invalidate_relationship(mock_neo4j, "uuid-123")


@pytest.mark.asyncio
class TestTemporalValidityCheckValidity:
    """Tests for check_relationship_validity method."""

    async def test_check_relationship_validity_valid(self, temporal_service):
        """Test checking valid relationship."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[{"invalid_at": None, "expired_at": None, "reason": None}]
        )

        result = await temporal_service.check_relationship_validity(
            mock_neo4j, "uuid-123"
        )

        assert result["is_valid"] is True
        assert result["invalid_at"] is None
        assert result["expired_at"] is None

    async def test_check_relationship_validity_naturally_invalid(
        self, temporal_service
    ):
        """Test checking naturally invalid relationship."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "invalid_at": "2025-03-12T10:00:00Z",
                    "expired_at": None,
                    "reason": "Project ended",
                }
            ]
        )

        result = await temporal_service.check_relationship_validity(
            mock_neo4j, "uuid-456"
        )

        assert result["is_valid"] is False
        assert result["invalid_at"] == "2025-03-12T10:00:00Z"
        assert result["expired_at"] is None
        assert result["invalidation_reason"] == "Project ended"

    async def test_check_relationship_validity_system_expired(self, temporal_service):
        """Test checking system-expired relationship."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "invalid_at": None,
                    "expired_at": "2025-03-12T10:00:00Z",
                    "reason": "Duplicate",
                }
            ]
        )

        result = await temporal_service.check_relationship_validity(
            mock_neo4j, "uuid-789"
        )

        assert result["is_valid"] is False
        assert result["expired_at"] == "2025-03-12T10:00:00Z"

    async def test_check_relationship_validity_both_invalid(self, temporal_service):
        """Test checking relationship with both fields set."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(
            return_value=[
                {
                    "invalid_at": "2025-03-12T10:00:00Z",
                    "expired_at": "2025-03-12T10:00:00Z",
                    "reason": "Complete invalidation",
                }
            ]
        )

        result = await temporal_service.check_relationship_validity(
            mock_neo4j, "uuid-000"
        )

        assert result["is_valid"] is False
        assert result["invalid_at"] is not None
        assert result["expired_at"] is not None

    async def test_check_relationship_validity_not_found(self, temporal_service):
        """Test checking non-existent relationship."""
        mock_neo4j = AsyncMock()
        mock_neo4j.query = AsyncMock(return_value=[])

        with pytest.raises(Exception, match="Relationship not found"):
            await temporal_service.check_relationship_validity(
                mock_neo4j, "nonexistent-uuid"
            )


class TestTemporalValidityConstants:
    """Tests for service constants."""

    def test_invalidation_type_constants(self, temporal_service):
        """Test invalidation type constants."""
        assert temporal_service.NATURAL == "natural"
        assert temporal_service.SYSTEM == "system"
        assert temporal_service.BOTH == "both"

    def test_temporal_scope_constants(self, temporal_service):
        """Test temporal scope constants."""
        assert temporal_service.CURRENT == "current"
        assert temporal_service.HISTORICAL == "historical"
        assert temporal_service.NATURAL_INVALID == "natural"
        assert temporal_service.SYSTEM_INVALID == "system"

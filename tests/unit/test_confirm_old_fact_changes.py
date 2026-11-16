"""Tests for the `confirm` parameter on invalidate_fact / update_fact (C1).

Facts older than 30 days score above the complexity threshold, so the tools
used to return validation_required with no way to approve. `confirm=True` is
that way.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_server import invalidate_fact, update_fact


def _old_relationship(days: int = 90) -> dict:
    return {
        "uuid": "rel-old-1",
        "entity1_name": "Mnemodrome",
        "entity1_uuid": "ent-1",
        "entity2_name": "server-02",
        "entity2_uuid": "ent-2",
        "relation_type": "RUNS_ON",
        "created_at": datetime.now(UTC) - timedelta(days=days),
    }


@pytest.fixture
def neo4j_mock():
    with patch("src.mcp_server.neo4j_service") as mock:
        mock.find_relationship = AsyncMock(return_value=_old_relationship())
        mock.invalidate_relationship = AsyncMock(
            return_value={
                "fields_set": ["invalid_at", "expired_at"],
                "invalidated_at": "2026-11-16T00:00:00+00:00",
            }
        )
        mock.execute_update_transaction = AsyncMock(
            return_value={
                "old_fact_uuid": "rel-old-1",
                "new_fact_uuid": "rel-new-1",
                "invalidated_at": "2026-11-16T00:00:00+00:00",
                "supersession_chain": ["rel-old-1", "rel-new-1"],
            }
        )
        with (
            patch("src.mcp_server._resolve_entity_uuid", AsyncMock(return_value="u")),
            patch("src.mcp_server.file_service"),
            patch("src.mcp_server._fire_and_forget", lambda coro, **kw: coro.close()),
        ):
            yield mock


@pytest.mark.asyncio
class TestInvalidateFactConfirm:
    async def test_old_fact_without_confirm_requires_validation(self, neo4j_mock):
        result = await invalidate_fact(
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            reason="migrated",
        )
        assert result["status"] == "validation_required"
        assert "confirm=True" in result["how_to_proceed"]
        neo4j_mock.invalidate_relationship.assert_not_called()

    async def test_old_fact_with_confirm_executes(self, neo4j_mock):
        result = await invalidate_fact(
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2="server-02",
            reason="migrated",
            confirm=True,
        )
        assert result["status"] == "success"
        assert result["confirmed"] is True
        neo4j_mock.invalidate_relationship.assert_awaited_once()


@pytest.mark.asyncio
class TestUpdateFactConfirm:
    async def test_old_fact_without_confirm_requires_validation(self, neo4j_mock):
        result = await update_fact(
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2_old="server-02",
            entity2_new="server-01",
        )
        assert result["status"] == "validation_required"
        assert "confirm=True" in result["how_to_proceed"]
        neo4j_mock.execute_update_transaction.assert_not_called()

    async def test_old_fact_with_confirm_executes(self, neo4j_mock):
        result = await update_fact(
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2_old="server-02",
            entity2_new="server-01",
            confirm=True,
        )
        assert result["status"] == "success"
        assert result["confirmed"] is True
        neo4j_mock.execute_update_transaction.assert_awaited_once()

    async def test_auto_mode_old_fact_confirm_executes(self, neo4j_mock):
        result = await update_fact(
            entity1="Mnemodrome",
            relation="RUNS_ON",
            entity2_old="server-02",
            entity2_new="server-01",
            conflict_resolution="auto",
            confirm=True,
        )
        assert result["status"] == "success"

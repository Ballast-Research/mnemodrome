"""Unit tests for relationship factory service (src/services/relationship_factory.py).

Tests the RelationshipFactory class which handles temporal relationship creation
and management between entities.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.relationship_factory import RelationshipFactory


@pytest.fixture
def relationship_factory():
    """Provide a fresh RelationshipFactory instance for each test."""
    return RelationshipFactory()


def empty_graph_query_mock():
    """AsyncMock for neo4j_service.query that simulates an empty graph.

    create() issues up to three queries: _find_existing (dedup check, has
    "LIMIT 1"), _find_conflicting (auto-supersede check), then the CREATE
    query. This mock returns NO existing duplicate and NO conflict, so the
    CREATE path always runs, and it builds the returned row from the CREATE
    params -- letting assertions verify the real created data instead of a
    dedup short-circuit.

    Keyed on the cypher text (not call order) so it stays correct under
    batch_create's parallel asyncio.gather, where two creates' queries
    interleave non-deterministically.
    """

    def _query(cypher, params=None):
        if "CREATE (source)-[r:RELATES_TO" in cypher:
            return [
                {
                    "source_uuid": params["source_uuid"],
                    "source_name": "Source",
                    "source_type": "Person",
                    "target_uuid": params["target_uuid"],
                    "target_name": "Target",
                    "target_type": "Technology",
                    "relation_type": params["relation_type"],
                    "valid_at": "2025-03-12T10:00:00Z",
                    "context": params.get("context", ""),
                }
            ]
        # _find_existing + _find_conflicting -> empty graph (no dup, no conflict)
        return []

    return AsyncMock(side_effect=_query)


def created_via_create_path(rel: dict) -> bool:
    """A relationship dict came from the CREATE path (not the dedup
    short-circuit) iff create() did not stamp it with a _dedup marker."""
    return "_dedup" not in rel


def _query_mock_with_conflict(
    conflict_target_uuid="old-target", conflict_target_name="Old Target"
):
    """neo4j_service.query mock where a conflicting prior edge EXISTS.

    No exact duplicate (_find_existing -> []), but _find_conflicting finds one
    different-target edge. invalidate() succeeds; CREATE returns the new row.
    Keyed on distinctive cypher substrings so it is order-independent.
    """

    def _query(cypher, params=None):
        if "CREATE (source)-[r:RELATES_TO" in cypher:
            return [
                {
                    "source_uuid": params["source_uuid"],
                    "source_name": "Subject",
                    "source_type": "Person",
                    "target_uuid": params["target_uuid"],
                    "target_name": "New Target",
                    "target_type": "Thing",
                    "uuid": params.get("relationship_uuid"),
                    "relation_type": params["relation_type"],
                    "valid_at": "2025-03-12T10:00:00Z",
                    "context": params.get("context", ""),
                }
            ]
        if "target.uuid <> $target_uuid" in cypher:  # _find_conflicting
            return [
                {
                    "source_uuid": params["source_uuid"],
                    "source_name": "Subject",
                    "relation_type": params["relation_type"],
                    "target_uuid": conflict_target_uuid,
                    "target_name": conflict_target_name,
                }
            ]
        if "SET r.invalid_at = datetime()" in cypher:  # invalidate (single-edge guard)
            return [{"invalidated_uuid": "old-rel-uuid"}]
        return []  # _find_existing (dedup) -> no exact duplicate

    return AsyncMock(side_effect=_query)


def _conflict_check_issued(mock) -> bool:
    return any(
        "target.uuid <> $target_uuid" in c.args[0] for c in mock.query.call_args_list
    )


def _invalidate_issued(mock) -> bool:
    return any(
        "SET r.invalid_at = datetime()" in c.args[0] for c in mock.query.call_args_list
    )


@pytest.mark.asyncio
class TestRelationshipFactorySupersession:
    """Supersession must respect predicate cardinality (transfer_0002 regression).

    Functional (single-valued) predicates supersede a prior different-target
    edge; cumulative (many-valued) predicates must NOT — independent facts that
    share subject+predicate coexist.
    """

    async def test_cumulative_predicate_does_not_supersede(self, relationship_factory):
        """REGRESSION: Carlos -ORDERED-> B must NOT invalidate Carlos -ORDERED-> A.

        ORDERED is cumulative, so the conflict check must be skipped entirely and
        the prior fact left valid.
        """
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = _query_mock_with_conflict(
                conflict_target_name="RTX PRO 4500"
            )

            result = await relationship_factory.create(
                source_uuid="carlos",
                relation_type="ORDERED",
                target_uuid="elitedesks",
                context="ordered 3x HP EliteDesk 800 G6",
            )

            # The cumulative predicate skips supersession altogether.
            assert not _conflict_check_issued(mock_neo4j), (
                "must not even query for conflicts"
            )
            assert not _invalidate_issued(mock_neo4j), (
                "must not invalidate the prior ORDERED fact"
            )
            assert "_superseded_targets" not in result
            assert result["relation_type"] == "ORDERED"

    async def test_functional_predicate_supersedes(self, relationship_factory):
        """Functional predicate (LIVES_IN) still supersedes a prior target."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = _query_mock_with_conflict(conflict_target_name="Austin")

            result = await relationship_factory.create(
                source_uuid="carlos",
                relation_type="LIVES_IN",
                target_uuid="denver",
                context="moved to Denver",
            )

            assert _conflict_check_issued(mock_neo4j), (
                "functional predicate must check conflicts"
            )
            assert _invalidate_issued(mock_neo4j), (
                "functional predicate must invalidate prior target"
            )
            assert result.get("_superseded_targets") == ["Austin"]


@pytest.mark.asyncio
class TestRelationshipFactoryCreate:
    """Tests for create method."""

    async def test_create_relationship_success(self, relationship_factory):
        """Test successful relationship creation runs the CREATE query.

        No pre-existing duplicate, so create() takes the CREATE path and
        returns the freshly-created relationship data.
        """
        created_row = {
            "source_uuid": "source-123",
            "source_name": "Carlos",
            "source_type": "Person",
            "relation_type": "USES",
            "target_uuid": "target-456",
            "target_name": "Neo4j",
            "target_type": "Technology",
            "valid_at": "2025-03-12T10:00:00Z",
            "context": "Graph database for Mnemodrome",
        }
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:

            def _query(cypher, params=None):
                if "CREATE (source)-[r:RELATES_TO" in cypher:
                    return [created_row]
                return []  # no existing duplicate, no conflict

            mock_neo4j.query = AsyncMock(side_effect=_query)

            result = await relationship_factory.create(
                source_uuid="source-123",
                relation_type="USES",
                target_uuid="target-456",
                context="Graph database for Mnemodrome",
            )

            assert created_via_create_path(result)  # CREATE path, not dedup
            assert result["source_uuid"] == "source-123"
            assert result["source_name"] == "Carlos"
            assert result["relation_type"] == "USES"
            assert result["target_uuid"] == "target-456"
            assert result["target_name"] == "Neo4j"
            # The CREATE query was actually issued.
            assert any(
                "CREATE (source)-[r:RELATES_TO" in c.args[0]
                for c in mock_neo4j.query.call_args_list
            )

    async def test_create_relationship_without_context(self, relationship_factory):
        """Test relationship creation with no context defaults context to ''."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = empty_graph_query_mock()

            result = await relationship_factory.create(
                source_uuid="s1", relation_type="KNOWS", target_uuid="t1"
            )

            assert result is not None
            assert created_via_create_path(result)
            assert result["relation_type"] == "KNOWS"
            assert result["context"] == ""

    async def test_create_relationship_entity_not_found(self, relationship_factory):
        """Test creation fails when entity not found."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            with pytest.raises(Exception, match="Failed to create relationship"):
                await relationship_factory.create(
                    source_uuid="nonexistent",
                    relation_type="USES",
                    target_uuid="nonexistent",
                )

    async def test_create_relationship_query_error(self, relationship_factory):
        """Test creation fails on query error."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(side_effect=Exception("Neo4j error"))

            with pytest.raises(Exception):
                await relationship_factory.create(
                    source_uuid="s1", relation_type="USES", target_uuid="t1"
                )


@pytest.mark.asyncio
class TestRelationshipFactoryInvalidation:
    """Tests for invalidate method."""

    async def test_invalidate_relationship_success(self, relationship_factory):
        """Test successful relationship invalidation (D3: cardinality-guarded)."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            # D3: first call returns the invalidated edge uuid (single-statement guard succeeded)
            mock_neo4j.query = AsyncMock(
                return_value=[{"invalidated_uuid": "rel-uuid-123"}]
            )

            result = await relationship_factory.invalidate(
                source_uuid="source-uuid",
                relation_type="USES",
                target_uuid="target-uuid",
            )

            assert result is True
            mock_neo4j.query.assert_called_once()

    async def test_invalidate_relationship_not_found(self, relationship_factory):
        """Test invalidation when relationship not found (D3: 0 rows → not_found)."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            # D3: first call returns [] (guard returned 0 rows — not found or dup)
            # second call returns count=0 (not found)
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [],  # primary D3 guard query
                    [{"cnt": 0}],  # count fallback → not found
                ]
            )

            result = await relationship_factory.invalidate(
                source_uuid="nonexistent",
                relation_type="USES",
                target_uuid="nonexistent",
            )

            assert result is False

    async def test_invalidate_relationship_already_invalid(self, relationship_factory):
        """Test invalidation of already-invalid relationship (D3: 0 rows → not_found)."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            # Already invalid means guard finds no active edge
            mock_neo4j.query = AsyncMock(
                side_effect=[
                    [],  # primary D3 guard query (no active native edge)
                    [{"cnt": 0}],  # count fallback → not found
                ]
            )

            result = await relationship_factory.invalidate(
                source_uuid="s1", relation_type="USES", target_uuid="t1"
            )

            assert result is False


@pytest.mark.asyncio
class TestRelationshipFactoryExpiration:
    """Tests for expire method."""

    async def test_expire_relationship_success(self, relationship_factory):
        """Test successful relationship expiration."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[{"expired": 1}])

            result = await relationship_factory.expire(
                source_uuid="source-uuid",
                relation_type="USES",
                target_uuid="target-uuid",
            )

            assert result is True

    async def test_expire_relationship_not_found(self, relationship_factory):
        """Test expiration when relationship not found."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[{"expired": 0}])

            result = await relationship_factory.expire(
                source_uuid="nonexistent",
                relation_type="USES",
                target_uuid="nonexistent",
            )

            assert result is False

    async def test_expire_relationship_already_expired(self, relationship_factory):
        """Test expiration of already-expired relationship."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[{"expired": 0}])

            result = await relationship_factory.expire(
                source_uuid="s1", relation_type="USES", target_uuid="t1"
            )

            assert result is False


@pytest.mark.asyncio
class TestRelationshipFactoryBatchCreate:
    """Tests for batch_create method."""

    async def test_batch_create_empty_list(self, relationship_factory):
        """Test batch creation with empty list."""
        result = await relationship_factory.batch_create([])
        assert result == []

    async def test_batch_create_single_relationship(self, relationship_factory):
        """Test batch creation actually creates a single relationship.

        Guards the create path end-to-end: no pre-existing duplicate, so
        create() runs the CREATE query and returns freshly-created data.
        """
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = empty_graph_query_mock()

            relationships = [
                {"source_uuid": "s1", "relation_type": "USES", "target_uuid": "t1"}
            ]

            result = await relationship_factory.batch_create(
                relationships, context="Test context"
            )

            assert len(result) == 1
            created = result[0]
            # Came from the CREATE path, NOT the dedup short-circuit.
            assert created_via_create_path(created)
            assert created["relation_type"] == "USES"
            assert created["source_uuid"] == "s1"
            assert created["target_uuid"] == "t1"
            assert created["context"] == "Test context"

            # Exactly one CREATE query was issued.
            create_calls = [
                c
                for c in mock_neo4j.query.call_args_list
                if "CREATE (source)-[r:RELATES_TO" in c.args[0]
            ]
            assert len(create_calls) == 1

    async def test_batch_create_multiple_relationships(self, relationship_factory):
        """Test batch creation creates multiple distinct relationships."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = empty_graph_query_mock()

            relationships = [
                {"source_uuid": "s1", "relation_type": "USES", "target_uuid": "t1"},
                {
                    "source_uuid": "s1",
                    "relation_type": "IMPLEMENTS",
                    "target_uuid": "t2",
                },
            ]

            result = await relationship_factory.batch_create(relationships)

            assert len(result) == 2
            # Both went through the CREATE path, not dedup.
            assert all(created_via_create_path(r) for r in result)
            assert {r["relation_type"] for r in result} == {"USES", "IMPLEMENTS"}
            assert {r["target_uuid"] for r in result} == {"t1", "t2"}

            # Two distinct CREATE queries were issued.
            create_calls = [
                c
                for c in mock_neo4j.query.call_args_list
                if "CREATE (source)-[r:RELATES_TO" in c.args[0]
            ]
            assert len(create_calls) == 2

    async def test_batch_create_with_shared_context(self, relationship_factory):
        """Test the shared context is threaded into the CREATE query.

        With no pre-existing duplicate, create() runs the CREATE query; this
        verifies the shared `context` lands in BOTH the query params and the
        returned row -- the behavior the name promises.
        """
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = empty_graph_query_mock()

            relationships = [
                {"source_uuid": "s1", "relation_type": "USES", "target_uuid": "t1"}
            ]

            result = await relationship_factory.batch_create(
                relationships, context="Phase 4.5"
            )

            assert len(result) == 1
            assert created_via_create_path(result[0])
            # Context made it into the returned (created) data.
            assert result[0]["context"] == "Phase 4.5"

            # Context made it into the actual CREATE query params.
            create_calls = [
                c
                for c in mock_neo4j.query.call_args_list
                if "CREATE (source)-[r:RELATES_TO" in c.args[0]
            ]
            assert len(create_calls) == 1
            assert create_calls[0].args[1]["context"] == "Phase 4.5"

    async def test_batch_create_skips_duplicates(self, relationship_factory):
        """Dedup behavior, tested on purpose: when an active relationship
        already exists, create() skips the CREATE and returns the existing
        row marked _dedup='skipped'.
        """
        existing_row = {
            "source_uuid": "s1",
            "source_name": "Source",
            "source_type": "Person",
            "target_uuid": "t1",
            "target_name": "Target",
            "target_type": "Technology",
            "relation_type": "USES",
            "valid_at": "2025-03-12T10:00:00Z",
            "context": "original context",
        }

        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:

            def _query(cypher, params=None):
                # _find_existing carries "LIMIT 1" -> report a duplicate.
                if "LIMIT 1" in cypher:
                    return [existing_row]
                return []

            mock_neo4j.query = AsyncMock(side_effect=_query)

            result = await relationship_factory.batch_create(
                [{"source_uuid": "s1", "relation_type": "USES", "target_uuid": "t1"}],
                context="new context",
            )

            assert len(result) == 1
            # Returned the existing row, flagged as a skipped duplicate.
            assert result[0]["_dedup"] == "skipped"
            assert result[0]["context"] == "original context"

            # Critically: NO CREATE query ran -- the duplicate was not recreated.
            assert not any(
                "CREATE (source)-[r:RELATES_TO" in c.args[0]
                for c in mock_neo4j.query.call_args_list
            )

    async def test_batch_create_failure_propagates(self, relationship_factory):
        """Test that failure in batch create propagates."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(side_effect=Exception("Query error"))

            relationships = [
                {"source_uuid": "s1", "relation_type": "USES", "target_uuid": "t1"}
            ]

            with pytest.raises(Exception):
                await relationship_factory.batch_create(relationships)


@pytest.mark.asyncio
class TestRelationshipFactoryGetActive:
    """Tests for get_active_relationships method."""

    async def test_get_active_relationships_outgoing(self, relationship_factory):
        """Test retrieving outgoing relationships."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {
                        "source": {"uuid": "s1", "name": "Source"},
                        "r": {"type": "USES"},
                        "target": {"uuid": "t1", "name": "Target"},
                    }
                ]
            )

            result = await relationship_factory.get_active_relationships(
                "s1", direction="outgoing"
            )

            assert isinstance(result, list)
            mock_neo4j.query.assert_called_once()

    async def test_get_active_relationships_incoming(self, relationship_factory):
        """Test retrieving incoming relationships."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            result = await relationship_factory.get_active_relationships(
                "t1", direction="incoming"
            )

            assert isinstance(result, list)

    async def test_get_active_relationships_both(self, relationship_factory):
        """Test retrieving relationships in both directions."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(
                return_value=[{"outgoing": [], "incoming": []}]
            )

            result = await relationship_factory.get_active_relationships(
                "e1", direction="both"
            )

            assert isinstance(result, list)

    async def test_get_active_relationships_default_both(self, relationship_factory):
        """Test that default direction is 'both'."""
        with patch("src.services.relationship_factory.neo4j_service") as mock_neo4j:
            mock_neo4j.query = AsyncMock(return_value=[])

            await relationship_factory.get_active_relationships("e1")

            # Check that the query uses both directions
            call_args = mock_neo4j.query.call_args[0][0]
            assert "both" in call_args or "OPTIONAL MATCH" in call_args

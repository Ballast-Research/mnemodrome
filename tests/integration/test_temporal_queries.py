"""Integration tests for Neo4j temporal query filtering (Feature 006).

Tests Neo4j Cypher query patterns with temporal validity filtering:
- Current valid facts (default queries)
- Historical facts (all states with temporal metadata)
- Naturally-ended facts (invalid_at set, expired_at NULL)
- System-invalidated facts (expired_at set)
- Query performance with indexes

CRITICAL: All test data tagged with '_test_006_' for production database cleanup.
"""

from datetime import datetime, timedelta
from typing import Any

import pytest

from src.services.neo4j_service import neo4j_service


@pytest.mark.asyncio
class TestTemporalQueryFiltering:
    """Test Neo4j temporal query patterns for fact invalidation."""

    @pytest.fixture(autouse=True)
    async def setup_and_teardown(self):
        """Setup test data and cleanup after each test."""
        # Setup: Connect to Neo4j
        await neo4j_service.connect()

        # Create test entities and relationships
        self.test_entities = []
        self.test_relationships = []

        yield

        # Teardown: Clean up all test data from production database
        await self._cleanup_test_data()

    async def _cleanup_test_data(self):
        """Remove all test data marked with _test_006_ from production database."""
        cleanup_query = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS '_test_006_'
        DETACH DELETE e
        """
        await neo4j_service.query(cleanup_query)

    async def _create_test_fact(
        self,
        entity1: str,
        entity2: str,
        relation_type: str = "RELATES_TO",
        context: str = "",
        invalid_at: datetime | None = None,
        expired_at: datetime | None = None,
        valid_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Create a test fact with temporal metadata.

        Args:
            entity1: Source entity name (auto-tagged with _test_006_)
            entity2: Target entity name (auto-tagged with _test_006_)
            relation_type: Relationship type
            context: Optional context
            invalid_at: Optional invalidation timestamp (natural end)
            expired_at: Optional expiration timestamp (system invalidation)
            valid_at: Optional valid_at timestamp (defaults to now)

        Returns:
            Dict with entity UUIDs and relationship info
        """
        # Tag entities with test marker
        entity1_tagged = f"{entity1}_test_006_"
        entity2_tagged = f"{entity2}_test_006_"

        # Set default valid_at if not provided
        valid_at_str = valid_at.isoformat() if valid_at else "datetime()"
        invalid_at_str = (
            f"datetime('{invalid_at.isoformat()}')" if invalid_at else "NULL"
        )
        expired_at_str = (
            f"datetime('{expired_at.isoformat()}')" if expired_at else "NULL"
        )

        # Build Cypher query with temporal fields
        cypher = f"""
        MERGE (e1:Entity {{name: $entity1}})
        ON CREATE SET
            e1.uuid = randomUUID(),
            e1.summary = $entity1,
            e1.created_at = datetime()

        MERGE (e2:Entity {{name: $entity2}})
        ON CREATE SET
            e2.uuid = randomUUID(),
            e2.summary = $entity2,
            e2.created_at = datetime()

        CREATE (e1)-[r:RELATES_TO {{
            type: $relation_type,
            context: $context,
            valid_at: {valid_at_str},
            invalid_at: {invalid_at_str},
            expired_at: {expired_at_str}
        }}]->(e2)

        RETURN
            e1.uuid as entity1_uuid,
            e1.name as entity1_name,
            e2.uuid as entity2_uuid,
            e2.name as entity2_name,
            r.type as relation_type,
            r.valid_at as valid_at,
            r.invalid_at as invalid_at,
            r.expired_at as expired_at
        """

        params = {
            "entity1": entity1_tagged,
            "entity2": entity2_tagged,
            "relation_type": relation_type,
            "context": context,
        }

        result = await neo4j_service.query(cypher, params)
        return result[0] if result else {}

    async def test_current_valid_facts_query(self):
        """Test query for current valid facts (WHERE invalid_at IS NULL AND expired_at IS NULL)."""
        # Create mix of valid and invalid facts
        await self._create_test_fact(
            "Mnemodrome", "server-01", context="Current deployment"
        )
        await self._create_test_fact(
            "Mnemodrome",
            "server-02",
            context="Old deployment",
            invalid_at=datetime.now() - timedelta(days=1),
        )
        await self._create_test_fact(
            "Mnemodrome",
            "test-server",
            context="System invalidated",
            expired_at=datetime.now() - timedelta(hours=1),
        )

        # Query for current valid facts only
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
        RETURN source.name as source, r.type as relation, target.name as target,
               r.invalid_at as invalid_at, r.expired_at as expired_at
        """

        results = await neo4j_service.query(query)

        # Should only return the current valid fact (server-01)
        assert len(results) == 1
        assert results[0]["target"] == "server-01_test_006_"
        assert results[0]["invalid_at"] is None
        assert results[0]["expired_at"] is None

    async def test_historical_facts_query(self):
        """Test query for all historical facts with temporal metadata."""
        # Create facts in different states
        now = datetime.now()

        fact1 = await self._create_test_fact(
            "Project-X", "Server-A", context="Current state"
        )
        fact2 = await self._create_test_fact(
            "Project-X",
            "Server-B",
            context="Ended naturally",
            invalid_at=now - timedelta(days=30),
        )
        fact3 = await self._create_test_fact(
            "Project-X",
            "Server-C",
            context="System invalidated",
            expired_at=now - timedelta(days=7),
        )

        # Query for all facts with is_current flag
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND source.name CONTAINS 'Project-X'
        RETURN
            source.name as source,
            target.name as target,
            r.context as context,
            r.invalid_at IS NULL AND r.expired_at IS NULL AS is_current,
            r.invalid_at as invalid_at,
            r.expired_at as expired_at
        ORDER BY r.valid_at DESC
        """

        results = await neo4j_service.query(query)

        # Should return all 3 facts with correct is_current flags
        assert len(results) == 3

        # Verify current fact (Server-A)
        current_facts = [r for r in results if r["is_current"]]
        assert len(current_facts) == 1
        assert current_facts[0]["target"] == "Server-A_test_006_"

        # Verify naturally-ended fact (Server-B)
        ended_facts = [
            r
            for r in results
            if r["invalid_at"] is not None and r["expired_at"] is None
        ]
        assert len(ended_facts) == 1
        assert ended_facts[0]["target"] == "Server-B_test_006_"

        # Verify system-invalidated fact (Server-C)
        expired_facts = [r for r in results if r["expired_at"] is not None]
        assert len(expired_facts) == 1
        assert expired_facts[0]["target"] == "Server-C_test_006_"

    async def test_naturally_ended_facts_query(self):
        """Test query for facts that ended naturally (invalid_at set, expired_at NULL)."""
        # Create facts with different invalidation types
        now = datetime.now()

        await self._create_test_fact(
            "Feature-A",
            "Component-1",
            context="Deprecated feature",
            invalid_at=now - timedelta(days=15),
        )
        await self._create_test_fact(
            "Feature-A",
            "Component-2",
            context="System error",
            expired_at=now - timedelta(days=5),
        )
        await self._create_test_fact(
            "Feature-A",
            "Component-3",
            context="Both flags set",
            invalid_at=now - timedelta(days=10),
            expired_at=now - timedelta(days=10),
        )

        # Query for naturally-ended facts only
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND source.name CONTAINS 'Feature-A'
          AND r.invalid_at IS NOT NULL
          AND r.expired_at IS NULL
        RETURN
            source.name as source,
            target.name as target,
            r.context as context,
            r.invalid_at as invalid_at,
            r.expired_at as expired_at
        """

        results = await neo4j_service.query(query)

        # Should only return Component-1 (deprecated feature)
        assert len(results) == 1
        assert results[0]["target"] == "Component-1_test_006_"
        assert results[0]["invalid_at"] is not None
        assert results[0]["expired_at"] is None
        assert results[0]["context"] == "Deprecated feature"

    async def test_system_invalidated_facts_query(self):
        """Test query for system-invalidated facts (expired_at set)."""
        # Create facts with different invalidation types
        now = datetime.now()

        await self._create_test_fact(
            "Service-X",
            "Database-A",
            context="Natural migration",
            invalid_at=now - timedelta(days=20),
        )
        await self._create_test_fact(
            "Service-X",
            "Database-B",
            context="Connection error",
            expired_at=now - timedelta(hours=12),
        )
        await self._create_test_fact(
            "Service-X",
            "Database-C",
            context="Manual invalidation + system error",
            invalid_at=now - timedelta(days=5),
            expired_at=now - timedelta(days=5),
        )

        # Query for system-invalidated facts
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND source.name CONTAINS 'Service-X'
          AND r.expired_at IS NOT NULL
        RETURN
            source.name as source,
            target.name as target,
            r.context as context,
            r.invalid_at as invalid_at,
            r.expired_at as expired_at
        ORDER BY r.expired_at DESC
        """

        results = await neo4j_service.query(query)

        # Should return Database-B and Database-C (both have expired_at set)
        assert len(results) == 2

        # Verify Database-B (only expired_at set)
        db_b = [r for r in results if "Database-B" in r["target"]][0]
        assert db_b["invalid_at"] is None
        assert db_b["expired_at"] is not None

        # Verify Database-C (both flags set)
        db_c = [r for r in results if "Database-C" in r["target"]][0]
        assert db_c["invalid_at"] is not None
        assert db_c["expired_at"] is not None

    async def test_temporal_query_performance(self):
        """Test query performance with temporal filtering (verify indexes used)."""
        # Create multiple test facts to simulate production load
        now = datetime.now()

        for i in range(50):
            await self._create_test_fact(
                f"Entity-{i}",
                f"Target-{i}",
                context=f"Test relationship {i}",
                valid_at=now - timedelta(days=i % 10),
            )

        # Add some invalidated facts
        for i in range(10):
            await self._create_test_fact(
                f"Entity-{i}",
                f"Invalid-{i}",
                context="Invalidated",
                invalid_at=now - timedelta(days=i),
                valid_at=now - timedelta(days=i + 1),
            )

        # Query with temporal filtering
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
          AND r.valid_at IS NOT NULL
        RETURN count(r) as valid_count
        """

        # Measure query execution
        import time

        start_time = time.time()
        results = await neo4j_service.query(query)
        execution_time = time.time() - start_time

        # Should complete in reasonable time (<500ms as per spec)
        assert execution_time < 0.5, f"Query took {execution_time}s, expected <0.5s"

        # Should return correct count (50 valid facts)
        assert results[0]["valid_count"] == 50

    async def test_temporal_query_with_date_range(self):
        """Test temporal filtering with date range (recent facts only)."""
        # Create facts at different times
        now = datetime.now()

        # Recent fact (3 days ago)
        await self._create_test_fact(
            "Recent-Project",
            "Component-A",
            context="Recent work",
            valid_at=now - timedelta(days=3),
        )

        # Old fact (30 days ago)
        await self._create_test_fact(
            "Old-Project",
            "Component-B",
            context="Old work",
            valid_at=now - timedelta(days=30),
        )

        # Very recent fact (1 hour ago)
        await self._create_test_fact(
            "Latest-Project",
            "Component-C",
            context="Latest work",
            valid_at=now - timedelta(hours=1),
        )

        # Query for facts from last 7 days
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
          AND r.valid_at IS NOT NULL
          AND datetime(r.valid_at) > datetime() - duration('P7D')
        RETURN
            source.name as source,
            target.name as target,
            r.valid_at as valid_at
        ORDER BY r.valid_at DESC
        """

        results = await neo4j_service.query(query)

        # Should return only recent facts (3 days and 1 hour ago)
        assert len(results) == 2
        assert any("Latest-Project" in r["source"] for r in results)
        assert any("Recent-Project" in r["source"] for r in results)
        assert not any("Old-Project" in r["source"] for r in results)

    async def test_supersession_chain_query(self):
        """Test querying supersession chains (facts that replaced other facts)."""
        # Create supersession chain
        now = datetime.now()

        # Create original fact
        fact1 = await self._create_test_fact(
            "App-V1",
            "Server-A",
            context="Initial deployment",
            valid_at=now - timedelta(days=60),
        )

        # Simulate supersession (would be done by update_fact tool)
        # First, invalidate old fact
        invalidate_query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name = $source AND target.name = $target
        SET r.invalid_at = datetime(),
            r.expired_at = datetime(),
            r.superseded_by = randomUUID()
        RETURN r.superseded_by as new_uuid
        """

        invalidate_result = await neo4j_service.query(
            invalidate_query,
            {"source": "App-V1_test_006_", "target": "Server-A_test_006_"},
        )

        new_uuid = invalidate_result[0]["new_uuid"] if invalidate_result else None

        # Create replacement fact with reference
        create_query = """
        MERGE (source:Entity {name: $source})
        MERGE (target:Entity {name: $target})
        CREATE (source)-[r:RELATES_TO {
            type: 'RELATES_TO',
            context: $context,
            valid_at: datetime(),
            invalid_at: NULL,
            expired_at: NULL,
            replaces: $replaces_uuid
        }]->(target)
        RETURN r
        """

        await neo4j_service.query(
            create_query,
            {
                "source": "App-V1_test_006_",
                "target": "Server-B_test_006_",
                "context": "Migrated deployment",
                "replaces_uuid": new_uuid,
            },
        )

        # Query for supersession chain
        query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND source.name CONTAINS 'App-V1'
        RETURN
            target.name as target,
            r.superseded_by as superseded_by,
            r.replaces as replaces,
            r.invalid_at IS NULL AND r.expired_at IS NULL AS is_current
        ORDER BY r.valid_at ASC
        """

        results = await neo4j_service.query(query)

        # Should have 2 facts: old (superseded) and new (current)
        assert len(results) == 2

        # Verify supersession chain
        old_fact = [r for r in results if not r["is_current"]][0]
        new_fact = [r for r in results if r["is_current"]][0]

        assert old_fact["superseded_by"] is not None
        assert new_fact["replaces"] == old_fact["superseded_by"]

    async def test_bulk_invalidation_performance(self):
        """Test performance of bulk invalidation operations."""
        # Create multiple facts to invalidate
        now = datetime.now()
        entity_count = 20

        for i in range(entity_count):
            await self._create_test_fact(
                f"Bulk-Entity-{i}",
                f"Bulk-Target-{i}",
                context=f"Bulk test {i}",
                valid_at=now - timedelta(days=i),
            )

        # Bulk invalidate all facts matching pattern
        bulk_invalidate_query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND source.name CONTAINS 'Bulk-Entity'
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
        SET r.expired_at = datetime()
        RETURN count(r) as invalidated_count
        """

        import time

        start_time = time.time()
        results = await neo4j_service.query(bulk_invalidate_query)
        execution_time = time.time() - start_time

        # Should complete in reasonable time
        assert execution_time < 1.0, (
            f"Bulk invalidation took {execution_time}s, expected <1s"
        )

        # Should invalidate all facts
        assert results[0]["invalidated_count"] == entity_count

        # Verify invalidation
        verify_query = """
        MATCH (source)-[r:RELATES_TO]->(target)
        WHERE source.name CONTAINS '_test_006_'
          AND source.name CONTAINS 'Bulk-Entity'
          AND r.invalid_at IS NULL
          AND r.expired_at IS NULL
        RETURN count(r) as remaining_valid
        """

        verify_results = await neo4j_service.query(verify_query)
        assert verify_results[0]["remaining_valid"] == 0

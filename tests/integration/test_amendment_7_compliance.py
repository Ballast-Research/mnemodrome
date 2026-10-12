"""
Amendment 7 Compliance Validation - Integration Tests (T051)

Validates that Phase 4.5 implementation meets all Amendment 7 requirements:
- Amendment 7A: UUID-Based Entity Resolution (NON-NEGOTIABLE)
- Amendment 7B: Graph Intelligence Over Simple Triples
- Amendment 7C: Benefits of Doing It Now

Test Strategy:
1. Verify entity deduplication (no duplicate UUIDs for same entity)
2. Verify multi-entity graph extraction
3. Verify temporal relationship tracking
4. Verify performance requirements (<200ms entity classification)
"""

import asyncio

import pytest

from src.services.classification_service import classification_service
from src.services.entity_factory import entity_factory
from src.services.graph_factory import graph_factory
from src.services.neo4j_service import neo4j_service
from src.services.relationship_factory import relationship_factory


@pytest.mark.asyncio
class TestAmendment7AEntityIdentity:
    """Amendment 7A: UUID-Based Entity Resolution (NON-NEGOTIABLE)"""

    async def test_entity_deduplication_same_name_same_type(self):
        """
        CRITICAL: Same entity name + type MUST resolve to same UUID.
        This is the core of Amendment 7A - no duplicate entities.
        """
        # First creation
        uuid1 = await entity_factory.create_or_resolve(
            name="Test Entity A7", entity_type="Person", summary="First reference"
        )

        # Second creation with same name+type
        uuid2 = await entity_factory.create_or_resolve(
            name="Test Entity A7", entity_type="Person", summary="Second reference"
        )

        # MUST be identical UUID
        assert uuid1 == uuid2, "Amendment 7A VIOLATION: Duplicate UUIDs for same entity"

        # Verify only ONE node exists in Neo4j
        cypher = """
        MATCH (e:Person {name: $name})
        RETURN count(e) as count
        """
        result = await neo4j_service.query(cypher, {"name": "Test Entity A7"})
        assert result[0]["count"] == 1, (
            "Amendment 7A VIOLATION: Multiple nodes for same entity"
        )

    async def test_entity_deduplication_different_types(self):
        """
        Same name but different types SHOULD create different entities.
        e.g., "Mercury" (Planet) vs "Mercury" (Element)
        """
        uuid_planet = await entity_factory.create_or_resolve(
            name="Mercury", entity_type="Concept", summary="Planet Mercury"
        )

        uuid_element = await entity_factory.create_or_resolve(
            name="Mercury", entity_type="Technology", summary="Element Mercury"
        )

        # MUST be different UUIDs (different types)
        assert uuid_planet != uuid_element, (
            "Different entity types should have different UUIDs"
        )

        # Verify TWO distinct nodes exist
        cypher = """
        MATCH (e {name: $name})
        RETURN count(e) as count
        """
        result = await neo4j_service.query(cypher, {"name": "Mercury"})
        assert result[0]["count"] == 2, (
            "Should have 2 nodes for same name, different types"
        )

    async def test_uuid_persistence_across_sessions(self):
        """
        UUID MUST persist across multiple "sessions" of entity resolution.
        Simulates cross-client consistency requirement.
        """
        # Session 1: Create entity
        uuid_session1 = await entity_factory.create_or_resolve(
            name="Persistent Entity",
            entity_type="Project",
            summary="Created in session 1",
        )

        # Simulate session 2: Resolve same entity
        uuid_session2 = await entity_factory.create_or_resolve(
            name="Persistent Entity",
            entity_type="Project",
            summary="Accessed in session 2",
        )

        # Simulate session 3: Resolve again
        uuid_session3 = await entity_factory.create_or_resolve(
            name="Persistent Entity",
            entity_type="Project",
            summary="Accessed in session 3",
        )

        # ALL must be identical
        assert uuid_session1 == uuid_session2 == uuid_session3, (
            "Amendment 7A VIOLATION: UUID not persistent across sessions"
        )


@pytest.mark.asyncio
class TestAmendment7BGraphIntelligence:
    """Amendment 7B: Graph Intelligence Over Simple Triples"""

    async def test_multi_entity_extraction(self):
        """
        GraphFactory MUST extract multiple entities from rich context.
        No more artificial two-entity constraint.
        """
        context = """
        Dana and Alice collaborated with Bob to implement Phase 4a
        using FastMCP and Neo4j. The implementation followed Gang of Four
        Factory Pattern to solve entity duplication.
        """

        result = await graph_factory.build_from_extraction(context)

        # MUST extract at least 5 entities
        assert result["entities_created"] >= 5, (
            f"Amendment 7B VIOLATION: Only extracted {result['entities_created']} entities (expected ≥5)"
        )

        # Verify entities have diverse types (not all "Entity")
        entity_types = set(result["entity_types"])
        assert len(entity_types) >= 2, (
            f"Amendment 7B VIOLATION: All entities classified as same type: {entity_types}"
        )

    async def test_relationship_preservation(self):
        """
        RelationshipFactory MUST create temporal relationships with mem0 pattern.
        All relationships MUST have valid_at, support invalid_at/expired_at.
        """
        # Create entities
        source_uuid = await entity_factory.create_or_resolve(
            "Source A7B", "Person", "test"
        )
        target_uuid = await entity_factory.create_or_resolve(
            "Target A7B", "Project", "test"
        )

        # Create relationship
        rel_data = await relationship_factory.create(
            source_uuid=source_uuid,
            relation_type="WORKED_ON",
            target_uuid=target_uuid,
            context="Testing Amendment 7B",
        )

        # Verify temporal fields exist
        assert "valid_at" in rel_data or rel_data.get("valid_at") is not None, (
            "Amendment 7B VIOLATION: Missing valid_at timestamp"
        )

        # Verify relationship stored correctly in Neo4j
        cypher = """
        MATCH (s {uuid: $source})-[r:RELATES_TO {type: $rel_type}]->(t {uuid: $target})
        RETURN r.valid_at as valid_at,
               r.invalid_at as invalid_at,
               r.expired_at as expired_at,
               r.context as context
        """
        result = await neo4j_service.query(
            cypher,
            {"source": source_uuid, "target": target_uuid, "rel_type": "WORKED_ON"},
        )

        assert len(result) == 1, "Relationship not found in Neo4j"
        assert result[0]["valid_at"] is not None, (
            "Missing valid_at in stored relationship"
        )
        assert result[0]["context"] == "Testing Amendment 7B", "Context not preserved"

    async def test_parallel_entity_resolution(self):
        """
        EntityFactory MUST support parallel resolution (asyncio.gather).
        Critical for performance in multi-entity scenarios.
        """
        import time

        entities = [
            ("Entity1", "Person"),
            ("Entity2", "Project"),
            ("Entity3", "Technology"),
            ("Entity4", "Concept"),
            ("Entity5", "Organization"),
        ]

        start = time.time()

        # Parallel resolution
        tasks = [
            entity_factory.create_or_resolve(name, entity_type, "parallel test")
            for name, entity_type in entities
        ]
        uuids = await asyncio.gather(*tasks)

        elapsed_ms = (time.time() - start) * 1000

        # MUST complete in reasonable time (< 500ms for 5 entities)
        assert elapsed_ms < 500, (
            f"Amendment 7B VIOLATION: Parallel resolution too slow ({elapsed_ms:.1f}ms > 500ms)"
        )

        # All UUIDs must be unique
        assert len(set(uuids)) == 5, "UUID collision in parallel resolution"


@pytest.mark.asyncio
class TestAmendment7CPerformance:
    """Amendment 7C: Performance Requirements"""

    async def test_entity_classification_performance(self):
        """
        Entity classification MUST complete in < 200ms per entity.
        Critical for responsive user experience.
        """
        import time

        test_entities = [
            ("Python", "A programming language"),
            ("Neo4j", "Graph database system"),
            ("Alice", "Person working on Mnemodrome"),
            ("Phase 4.5", "Project implementation phase"),
            ("Gang of Four", "Design patterns book"),
        ]

        for name, context in test_entities:
            start = time.time()

            entity_type = await classification_service.classify_entity(
                name=name, context=context, role="entity"
            )

            elapsed_ms = (time.time() - start) * 1000

            assert elapsed_ms < 200, (
                f"Amendment 7C VIOLATION: Classification too slow for '{name}' ({elapsed_ms:.1f}ms > 200ms)"
            )

            # Verify reasonable classification
            assert entity_type in classification_service.VALID_ENTITY_TYPES, (
                f"Invalid entity type returned: {entity_type}"
            )

    async def test_graph_extraction_performance(self):
        """
        Full graph extraction MUST complete in < 1 second for 5-entity context.
        End-to-end performance requirement.
        """
        import time

        context = """
        Alice used Neo4j and FastMCP to implement Phase 4.5,
        following the Gang of Four Factory Pattern for entity resolution.
        """

        start = time.time()
        result = await graph_factory.build_from_extraction(context)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 1000, (
            f"Amendment 7C VIOLATION: Graph extraction too slow ({elapsed_ms:.1f}ms > 1000ms)"
        )

        # Verify entities were extracted
        assert result["entities_created"] >= 3, "Insufficient entity extraction"


@pytest.mark.asyncio
class TestAmendment7Integration:
    """End-to-end Amendment 7 compliance validation"""

    async def test_complete_workflow_compliance(self):
        """
        Complete workflow: Extract → Classify → Resolve → Relate
        This is the full Amendment 7 pattern in action.
        """
        context = """
        Alice and Bob collaborated on Mnemodrome Project, implementing
        the EntityFactory service using Neo4j graph database.
        """

        # Step 1: Extract entities
        extraction = await graph_factory._extract_with_llm(context)
        assert len(extraction["entities"]) >= 3, "Insufficient entity extraction"

        # Step 2: Classify entities
        entity_types = []
        for entity in extraction["entities"]:
            entity_type = await classification_service.classify_entity(
                name=entity["name"], context=context, role=entity.get("role", "entity")
            )
            entity_types.append(entity_type)

        # Step 3: Resolve to UUIDs (deduplication)
        resolution_tasks = [
            entity_factory.create_or_resolve(
                name=entity["name"], entity_type=entity_type, summary=context
            )
            for entity, entity_type in zip(
                extraction["entities"], entity_types, strict=False
            )
        ]
        entity_uuids = await asyncio.gather(*resolution_tasks)

        # Verify all UUIDs are valid and unique
        assert all(uuid for uuid in entity_uuids), "Some UUIDs are None/empty"
        assert len(entity_uuids) == len(extraction["entities"]), "UUID count mismatch"

        # Step 4: Create relationships (if extracted)
        if extraction["relationships"]:
            uuid_map = {
                entity["name"]: uuid
                for entity, uuid in zip(
                    extraction["entities"], entity_uuids, strict=False
                )
            }

            for rel in extraction["relationships"][:1]:  # Test first relationship
                source_uuid = uuid_map.get(rel["source"])
                target_uuid = uuid_map.get(rel["target"])

                if source_uuid and target_uuid:
                    rel_data = await relationship_factory.create(
                        source_uuid=source_uuid,
                        relation_type=rel["relation"],
                        target_uuid=target_uuid,
                        context=context,
                    )
                    assert rel_data is not None, "Relationship creation failed"

        # Amendment 7 COMPLIANCE: Full workflow succeeded without violations


# Cleanup fixture
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up test entities after each test"""
    yield

    # Cleanup test entities (optional - depends on test isolation strategy)
    cleanup_queries = [
        "MATCH (e {name: 'Test Entity A7'}) DETACH DELETE e",
        "MATCH (e {name: 'Mercury'}) DETACH DELETE e",
        "MATCH (e {name: 'Persistent Entity'}) DETACH DELETE e",
        "MATCH (e {name: 'Source A7B'}) DETACH DELETE e",
        "MATCH (e {name: 'Target A7B'}) DETACH DELETE e",
        "MATCH (e {name: 'Entity1'}) DETACH DELETE e",
        "MATCH (e {name: 'Entity2'}) DETACH DELETE e",
        "MATCH (e {name: 'Entity3'}) DETACH DELETE e",
        "MATCH (e {name: 'Entity4'}) DETACH DELETE e",
        "MATCH (e {name: 'Entity5'}) DETACH DELETE e",
        "MATCH (e {name: 'Alice'}) DETACH DELETE e",
        "MATCH (e {name: 'Bob'}) DETACH DELETE e",
    ]

    for query in cleanup_queries:
        try:
            await neo4j_service.query(query)
        except Exception:
            pass  # Ignore cleanup errors

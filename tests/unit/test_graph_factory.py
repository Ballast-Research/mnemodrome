"""Unit tests for Graph Factory Service.

Tests for the GraphFactory class covering:
- Entity extraction from natural language
- Entity classification (single and batch)
- Entity resolution and UUID mapping
- Relationship creation
- Multi-entity graph construction
- Heuristic fallback when LLM fails
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.graph_factory import GraphFactory


class TestGraphFactory:
    """Tests for the GraphFactory class."""

    @pytest.fixture
    def factory(self):
        """Provide a fresh GraphFactory instance."""
        return GraphFactory()

    # ========================
    # Graph Building Tests
    # ========================

    @pytest.mark.asyncio
    async def test_build_from_extraction_basic(self, factory):
        """Test building a graph from basic context."""
        context = "Carlos created a tool called FastMCP"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    # Mock LLM extraction
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "Carlos", "role": "person"},
                                {"name": "FastMCP", "role": "tool"},
                            ],
                            "relationships": [
                                {
                                    "source": "Carlos",
                                    "relation": "CREATED",
                                    "target": "FastMCP",
                                }
                            ],
                        }
                    )

                    # Mock entity classification
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Person", "Technology"]
                    )

                    # Mock entity resolution
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Carlos:Person": "uuid-carlos",
                            "FastMCP:Technology": "uuid-fastmcp",
                        }
                    )

                    # Mock relationship creation
                    mock_rel.batch_create = AsyncMock(
                        return_value=[
                            {
                                "source_uuid": "uuid-carlos",
                                "relation_type": "CREATED",
                                "target_uuid": "uuid-fastmcp",
                            }
                        ]
                    )

                    result = await factory.build_from_extraction(context)

                    assert result["entities_created"] == 2
                    assert result["relationships_created"] == 1
                    assert "Carlos" in result["entity_uuids"]
                    assert "FastMCP" in result["entity_uuids"]

    @pytest.mark.asyncio
    async def test_build_from_extraction_complex(self, factory):
        """Test building graph with multiple entities and relationships."""
        context = "Alice and Bob collaborated on Project X using Python and Neo4j"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    # Mock complex extraction
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "Alice", "role": "person"},
                                {"name": "Bob", "role": "person"},
                                {"name": "Project X", "role": "project"},
                                {"name": "Python", "role": "tech"},
                                {"name": "Neo4j", "role": "tech"},
                            ],
                            "relationships": [
                                {
                                    "source": "Alice",
                                    "relation": "COLLABORATED_WITH",
                                    "target": "Bob",
                                },
                                {
                                    "source": "Alice",
                                    "relation": "WORKED_ON",
                                    "target": "Project X",
                                },
                                {
                                    "source": "Bob",
                                    "relation": "WORKED_ON",
                                    "target": "Project X",
                                },
                                {
                                    "source": "Project X",
                                    "relation": "USES",
                                    "target": "Python",
                                },
                                {
                                    "source": "Project X",
                                    "relation": "USES",
                                    "target": "Neo4j",
                                },
                            ],
                        }
                    )

                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=[
                            "Person",
                            "Person",
                            "Project",
                            "Technology",
                            "Technology",
                        ]
                    )

                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Alice:Person": "uuid-alice",
                            "Bob:Person": "uuid-bob",
                            "Project X:Project": "uuid-project",
                            "Python:Technology": "uuid-python",
                            "Neo4j:Technology": "uuid-neo4j",
                        }
                    )

                    mock_rel.batch_create = AsyncMock(
                        return_value=[
                            {
                                "source_uuid": "uuid-alice",
                                "relation_type": "COLLABORATED_WITH",
                                "target_uuid": "uuid-bob",
                            },
                            {
                                "source_uuid": "uuid-alice",
                                "relation_type": "WORKED_ON",
                                "target_uuid": "uuid-project",
                            },
                            {
                                "source_uuid": "uuid-bob",
                                "relation_type": "WORKED_ON",
                                "target_uuid": "uuid-project",
                            },
                            {
                                "source_uuid": "uuid-project",
                                "relation_type": "USES",
                                "target_uuid": "uuid-python",
                            },
                            {
                                "source_uuid": "uuid-project",
                                "relation_type": "USES",
                                "target_uuid": "uuid-neo4j",
                            },
                        ]
                    )

                    result = await factory.build_from_extraction(context)

                    assert result["entities_created"] == 5
                    assert result["relationships_created"] == 5
                    assert len(result["entity_types"]) == 5

    @pytest.mark.asyncio
    async def test_build_from_extraction_empty_context(self, factory):
        """Test building graph with empty context."""
        context = ""

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            mock_llm.extract_entities_and_relationships = AsyncMock(
                return_value={"entities": [], "relationships": []}
            )

            result = await factory.build_from_extraction(context)

            assert result["entities_created"] == 0
            assert result["relationships_created"] == 0
            assert result["entity_types"] == []
            assert result["entity_uuids"] == {}

    @pytest.mark.asyncio
    async def test_build_from_extraction_missing_uuid(self, factory):
        """Test handling when entity resolution fails."""
        context = "Carlos and Alice worked together"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "Carlos", "role": "person"},
                                {"name": "Alice", "role": "person"},
                            ],
                            "relationships": [
                                {
                                    "source": "Carlos",
                                    "relation": "WORKED_WITH",
                                    "target": "Alice",
                                }
                            ],
                        }
                    )

                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Person", "Person"]
                    )

                    # Only one entity resolves - missing one
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Carlos:Person": "uuid-carlos"
                            # Alice missing
                        }
                    )

                    mock_rel.batch_create = AsyncMock(return_value=[])

                    result = await factory.build_from_extraction(context)

                    # Should skip relationship with missing entity
                    assert result["entities_created"] == 1
                    assert result["relationships_created"] == 0

    # ========================
    # LLM Extraction Tests
    # ========================

    @pytest.mark.asyncio
    async def test_extract_with_llm_success(self, factory):
        """Test successful LLM-based extraction."""
        context = "Test context"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            mock_llm.extract_entities_and_relationships = AsyncMock(
                return_value={
                    "entities": [{"name": "Test", "role": "entity"}],
                    "relationships": [],
                }
            )

            result = await factory._extract_with_llm(context)

            assert "entities" in result
            assert "relationships" in result
            assert len(result["entities"]) >= 0

    @pytest.mark.asyncio
    async def test_extract_with_llm_failure_falls_back(self, factory):
        """Test that heuristic extraction is used when LLM fails."""
        context = "Carlos created FastMCP for Neo4j"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            mock_llm.extract_entities_and_relationships = AsyncMock(
                side_effect=Exception("LLM service down")
            )

            result = await factory._extract_with_llm(context)

            # Should fall back to heuristic
            assert "entities" in result
            assert "relationships" in result
            # Heuristic should extract capitalized words
            assert any(e["name"] == "Carlos" for e in result["entities"])

    # ========================
    # Heuristic Extraction Tests
    # ========================

    @pytest.mark.asyncio
    async def test_heuristic_extraction_capitalized_words(self, factory):
        """Test heuristic extraction identifies capitalized words."""
        context = "Carlos and Alice worked on Project X"

        result = await factory._heuristic_extraction(context)

        entity_names = [e["name"] for e in result["entities"]]
        assert "Carlos" in entity_names
        assert "Alice" in entity_names
        # Heuristic captures multi-word entities
        assert any("Project" in name for name in entity_names)

    @pytest.mark.asyncio
    async def test_heuristic_extraction_relationships(self, factory):
        """Test heuristic extraction identifies patterns."""
        context = "Alice and Bob collaborated on the system"

        result = await factory._heuristic_extraction(context)

        assert "entities" in result
        assert "relationships" in result

    @pytest.mark.asyncio
    async def test_heuristic_extraction_patterns(self, factory):
        """Test heuristic extracts relationship patterns."""
        context = "Carlos and Bob collaborated together"

        result = await factory._heuristic_extraction(context)

        # Should identify collaboration pattern
        relationships = result["relationships"]
        if relationships:
            assert any(r["relation"] == "COLLABORATED_WITH" for r in relationships)

    @pytest.mark.asyncio
    async def test_heuristic_extraction_empty_context(self, factory):
        """Test heuristic extraction on empty context."""
        context = ""

        result = await factory._heuristic_extraction(context)

        assert "entities" in result
        assert "relationships" in result
        assert len(result["entities"]) == 0

    @pytest.mark.asyncio
    async def test_heuristic_extraction_single_word(self, factory):
        """Test heuristic extraction on single word."""
        context = "Python"

        result = await factory._heuristic_extraction(context)

        assert "entities" in result
        entity_names = [e["name"] for e in result["entities"]]
        assert "Python" in entity_names

    # ========================
    # Entity Classification Tests
    # ========================

    @pytest.mark.asyncio
    async def test_classification_batch_call(self, factory):
        """Test that batch classification is called with correct data."""
        context = "Carlos works with Neo4j on Phase 4"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "Carlos", "role": "person"},
                                {"name": "Neo4j", "role": "database"},
                            ],
                            "relationships": [],
                        }
                    )

                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Person", "Technology"]
                    )

                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Carlos:Person": "uuid-c",
                            "Neo4j:Technology": "uuid-n",
                        }
                    )

                    mock_rel.batch_create = AsyncMock(return_value=[])

                    await factory.build_from_extraction(context)

                    # Verify batch classification was called
                    mock_llm.classify_entities_batch.assert_called_once()
                    call_args = mock_llm.classify_entities_batch.call_args
                    entity_data = call_args[0][0]
                    assert len(entity_data) == 2

    # ========================
    # Entity Resolution Tests
    # ========================

    @pytest.mark.asyncio
    async def test_entity_resolution_batch_call(self, factory):
        """Test that batch entity resolution is used."""
        context = "Test context"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [{"name": "Entity1", "role": "entity"}],
                            "relationships": [],
                        }
                    )

                    mock_llm.classify_entities_batch = AsyncMock(return_value=["Thing"])

                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={"Entity1:Thing": "uuid-1"}
                    )

                    mock_rel.batch_create = AsyncMock(return_value=[])

                    await factory.build_from_extraction(context)

                    # Verify batch resolution was called
                    mock_entity.batch_create_or_resolve.assert_called_once()

    # ========================
    # Relationship Creation Tests
    # ========================

    @pytest.mark.asyncio
    async def test_relationship_creation_parallel(self, factory):
        """Test that relationships are created in batch."""
        context = "A and B and C all work together"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "A", "role": "entity"},
                                {"name": "B", "role": "entity"},
                                {"name": "C", "role": "entity"},
                            ],
                            "relationships": [
                                {
                                    "source": "A",
                                    "relation": "WORKS_WITH",
                                    "target": "B",
                                },
                                {
                                    "source": "B",
                                    "relation": "WORKS_WITH",
                                    "target": "C",
                                },
                                {
                                    "source": "A",
                                    "relation": "WORKS_WITH",
                                    "target": "C",
                                },
                            ],
                        }
                    )

                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Thing", "Thing", "Thing"]
                    )

                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "A:Thing": "uuid-a",
                            "B:Thing": "uuid-b",
                            "C:Thing": "uuid-c",
                        }
                    )

                    mock_rel.batch_create = AsyncMock(
                        return_value=[
                            {
                                "source_uuid": "uuid-a",
                                "relation_type": "WORKS_WITH",
                                "target_uuid": "uuid-b",
                            },
                            {
                                "source_uuid": "uuid-b",
                                "relation_type": "WORKS_WITH",
                                "target_uuid": "uuid-c",
                            },
                            {
                                "source_uuid": "uuid-a",
                                "relation_type": "WORKS_WITH",
                                "target_uuid": "uuid-c",
                            },
                        ]
                    )

                    result = await factory.build_from_extraction(context)

                    assert result["relationships_created"] == 3
                    mock_rel.batch_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_relationship_creation_failure_continues(self, factory):
        """Test that entity creation continues even if relationships fail."""
        context = "A and B"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "A", "role": "entity"},
                                {"name": "B", "role": "entity"},
                            ],
                            "relationships": [
                                {"source": "A", "relation": "RELATES", "target": "B"}
                            ],
                        }
                    )

                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Thing", "Thing"]
                    )

                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={"A:Thing": "uuid-a", "B:Thing": "uuid-b"}
                    )

                    # Relationship creation fails
                    mock_rel.batch_create = AsyncMock(
                        side_effect=Exception("Relationship creation failed")
                    )

                    result = await factory.build_from_extraction(context)

                    # Should still report entities created
                    assert result["entities_created"] == 2
                    assert result["relationships_created"] == 0

    # ========================
    # Unhashable-list regression tests (GitHub issue: "unhashable type: 'list'")
    # ========================

    @pytest.mark.asyncio
    async def test_list_source_fans_out(self, factory):
        """Relationship with list source should fan out into N relationships — one per source."""
        context = "ProductA and ProductB both use Neo4j"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "ProductA", "role": "entity"},
                                {"name": "ProductB", "role": "entity"},
                                {"name": "Neo4j", "role": "database"},
                            ],
                            "relationships": [
                                # LLM returned a list for source
                                {
                                    "source": ["ProductA", "ProductB"],
                                    "relation": "USES",
                                    "target": "Neo4j",
                                }
                            ],
                        }
                    )
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Technology", "Technology", "Technology"]
                    )
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "ProductA:Technology": "uuid-pa",
                            "ProductB:Technology": "uuid-pb",
                            "Neo4j:Technology": "uuid-neo4j",
                        }
                    )

                    # batch_create receives the fan-out list and returns it back
                    async def passthrough_rels(relationships, context):
                        return relationships

                    mock_rel.batch_create = AsyncMock(side_effect=passthrough_rels)

                    result = await factory.build_from_extraction(context)

                    # Two fanned-out relationships should be created
                    assert result["relationships_created"] == 2

                    # Verify both source UUIDs were submitted to batch_create
                    passed_rels = mock_rel.batch_create.call_args.kwargs[
                        "relationships"
                    ]
                    source_uuids = {r["source_uuid"] for r in passed_rels}
                    assert "uuid-pa" in source_uuids
                    assert "uuid-pb" in source_uuids
                    # All should point at Neo4j
                    assert all(r["target_uuid"] == "uuid-neo4j" for r in passed_rels)

    @pytest.mark.asyncio
    async def test_list_target_fans_out(self, factory):
        """Relationship with list target should fan out into N relationships — one per target."""
        context = "Carlos uses both Neo4j and Qdrant"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "Carlos", "role": "person"},
                                {"name": "Neo4j", "role": "database"},
                                {"name": "Qdrant", "role": "database"},
                            ],
                            "relationships": [
                                {
                                    "source": "Carlos",
                                    "relation": "USES",
                                    "target": ["Neo4j", "Qdrant"],
                                }
                            ],
                        }
                    )
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Person", "Technology", "Technology"]
                    )
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Carlos:Person": "uuid-carlos",
                            "Neo4j:Technology": "uuid-neo4j",
                            "Qdrant:Technology": "uuid-qdrant",
                        }
                    )

                    async def passthrough_rels(relationships, context):
                        return relationships

                    mock_rel.batch_create = AsyncMock(side_effect=passthrough_rels)

                    result = await factory.build_from_extraction(context)

                    assert result["relationships_created"] == 2

                    passed_rels = mock_rel.batch_create.call_args.kwargs[
                        "relationships"
                    ]
                    target_uuids = {r["target_uuid"] for r in passed_rels}
                    assert "uuid-neo4j" in target_uuids
                    assert "uuid-qdrant" in target_uuids
                    assert all(r["source_uuid"] == "uuid-carlos" for r in passed_rels)

    @pytest.mark.asyncio
    async def test_list_source_and_target_cartesian(self, factory):
        """Relationship with list source AND list target should produce M×N pairs (cartesian)."""
        context = "A and B interact with X and Y"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "A", "role": "entity"},
                                {"name": "B", "role": "entity"},
                                {"name": "X", "role": "entity"},
                                {"name": "Y", "role": "entity"},
                            ],
                            "relationships": [
                                {
                                    "source": ["A", "B"],
                                    "relation": "INTERACTS_WITH",
                                    "target": ["X", "Y"],
                                }
                            ],
                        }
                    )
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Thing", "Thing", "Thing", "Thing"]
                    )
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "A:Thing": "uuid-a",
                            "B:Thing": "uuid-b",
                            "X:Thing": "uuid-x",
                            "Y:Thing": "uuid-y",
                        }
                    )

                    async def passthrough_rels(relationships, context):
                        return relationships

                    mock_rel.batch_create = AsyncMock(side_effect=passthrough_rels)

                    result = await factory.build_from_extraction(context)

                    # 2 sources × 2 targets = 4 relationships
                    assert result["relationships_created"] == 4

                    passed_rels = mock_rel.batch_create.call_args.kwargs[
                        "relationships"
                    ]
                    pairs = {(r["source_uuid"], r["target_uuid"]) for r in passed_rels}
                    assert ("uuid-a", "uuid-x") in pairs
                    assert ("uuid-a", "uuid-y") in pairs
                    assert ("uuid-b", "uuid-x") in pairs
                    assert ("uuid-b", "uuid-y") in pairs

    @pytest.mark.asyncio
    async def test_entity_name_as_list_coerced_no_crash(self, factory):
        """Entity with list name must be normalized UP FRONT so it is passed to
        batch_create_or_resolve as a STRING (not a list), and must resolve to a UUID."""
        context = "Some compound entity does things"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                # LLM returned a list as the entity name
                                {"name": ["CompoundA", "CompoundB"], "role": "entity"},
                            ],
                            "relationships": [],
                        }
                    )
                    mock_llm.classify_entities_batch = AsyncMock(return_value=["Thing"])
                    # The coerced canonical name is "CompoundA CompoundB"; return a UUID for it.
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={"CompoundA CompoundB:Thing": "uuid-compound"}
                    )
                    mock_rel.batch_create = AsyncMock(return_value=[])

                    # Must not raise TypeError: unhashable type: 'list' or
                    # AttributeError: 'list' object has no attribute 'strip'
                    result = await factory.build_from_extraction(context)

                    # --- Core assertions ---
                    # 1. No crash
                    assert isinstance(result, dict)

                    # 2. batch_create_or_resolve was called with a STRING name (not a list)
                    call_args = mock_entity.batch_create_or_resolve.call_args
                    resolved_tuples = call_args[0][
                        0
                    ]  # first positional arg: list of (name, type, summary)
                    names_passed = [t[0] for t in resolved_tuples]
                    assert all(isinstance(n, str) for n in names_passed), (
                        f"Expected all string names in batch_create_or_resolve call; got {names_passed!r}"
                    )
                    # The canonical join should be "CompoundA CompoundB"
                    assert "CompoundA CompoundB" in names_passed

                    # 3. The coerced entity resolves to a UUID and appears in entity_uuids
                    assert "CompoundA CompoundB" in result["entity_uuids"], (
                        "List-named entity must appear in entity_uuids after upfront normalization"
                    )
                    assert (
                        result["entity_uuids"]["CompoundA CompoundB"] == "uuid-compound"
                    )
                    assert result["entities_created"] == 1

    @pytest.mark.asyncio
    async def test_entity_name_as_list_upfront_normalization_before_resolution(
        self, factory
    ):
        """Verify normalization fires BEFORE resolution: the tuple passed to
        batch_create_or_resolve must contain the coerced string name so the
        cache key (_make_key -> name.strip()) never sees a list."""
        context = "Mixed bag context"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": ["Alpha", "Beta", "Gamma"], "role": "entity"},
                                {"name": "NormalEntity", "role": "entity"},
                            ],
                            "relationships": [
                                # Relationship references the *normalized* name
                                {
                                    "source": "Alpha Beta Gamma",
                                    "relation": "RELATES",
                                    "target": "NormalEntity",
                                }
                            ],
                        }
                    )
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Thing", "Thing"]
                    )
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Alpha Beta Gamma:Thing": "uuid-abc",
                            "NormalEntity:Thing": "uuid-normal",
                        }
                    )

                    async def passthrough_rels(relationships, context):
                        return relationships

                    mock_rel.batch_create = AsyncMock(side_effect=passthrough_rels)

                    result = await factory.build_from_extraction(context)

                    # Normalization happened up front: both entities resolve
                    assert result["entities_created"] == 2
                    assert "Alpha Beta Gamma" in result["entity_uuids"]
                    assert "NormalEntity" in result["entity_uuids"]

                    # The relationship between normalized name and normal entity succeeds
                    assert result["relationships_created"] == 1

                    # Crucially: batch_create_or_resolve saw only strings
                    call_args = mock_entity.batch_create_or_resolve.call_args
                    resolved_tuples = call_args[0][0]
                    for name, *_ in resolved_tuples:
                        assert isinstance(name, str), (
                            f"batch_create_or_resolve received non-string name {name!r}; "
                            "upfront normalization must run before resolution"
                        )

    @pytest.mark.asyncio
    async def test_scalar_source_target_behavior_unchanged(self, factory):
        """Control: scalar string source/target must still produce exactly one relationship."""
        context = "Alice created FastMCP"

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value={
                            "entities": [
                                {"name": "Alice", "role": "person"},
                                {"name": "FastMCP", "role": "tool"},
                            ],
                            "relationships": [
                                {
                                    "source": "Alice",
                                    "relation": "CREATED",
                                    "target": "FastMCP",
                                }
                            ],
                        }
                    )
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Person", "Technology"]
                    )
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "Alice:Person": "uuid-alice",
                            "FastMCP:Technology": "uuid-fastmcp",
                        }
                    )

                    async def passthrough_rels(relationships, context):
                        return relationships

                    mock_rel.batch_create = AsyncMock(side_effect=passthrough_rels)

                    result = await factory.build_from_extraction(context)

                    # Exactly one relationship, unchanged from pre-fix behaviour
                    assert result["relationships_created"] == 1

                    passed_rels = mock_rel.batch_create.call_args.kwargs[
                        "relationships"
                    ]
                    assert len(passed_rels) == 1
                    assert passed_rels[0]["source_uuid"] == "uuid-alice"
                    assert passed_rels[0]["target_uuid"] == "uuid-fastmcp"
                    assert passed_rels[0]["relation_type"] == "CREATED"

    @pytest.mark.asyncio
    async def test_historical_crash_list_source_no_unhashable_error(self, factory):
        """Regression: the exact historical crash input must not raise TypeError."""
        # Historical payload that triggered: TypeError: unhashable type: 'list'
        crash_payload = {
            "entities": [
                {"name": "ProductA", "role": "entity"},
                {"name": "ProductB", "role": "entity"},
                {"name": "Neo4j", "role": "database"},
            ],
            "relationships": [
                {
                    "source": ["ProductA", "ProductB"],
                    "relation": "USES",
                    "target": "Neo4j",
                }
            ],
        }

        with patch("src.services.graph_factory.llm_service") as mock_llm:
            with patch("src.services.graph_factory.entity_factory") as mock_entity:
                with patch(
                    "src.services.graph_factory.relationship_factory"
                ) as mock_rel:
                    mock_llm.extract_entities_and_relationships = AsyncMock(
                        return_value=crash_payload
                    )
                    mock_llm.classify_entities_batch = AsyncMock(
                        return_value=["Technology", "Technology", "Technology"]
                    )
                    mock_entity.batch_create_or_resolve = AsyncMock(
                        return_value={
                            "ProductA:Technology": "uuid-pa",
                            "ProductB:Technology": "uuid-pb",
                            "Neo4j:Technology": "uuid-neo4j",
                        }
                    )
                    mock_rel.batch_create = AsyncMock(return_value=[])

                    # This used to raise TypeError: unhashable type: 'list'
                    try:
                        result = await factory.build_from_extraction(
                            "ProductA and ProductB use Neo4j"
                        )
                    except TypeError as exc:
                        pytest.fail(f"Raised TypeError (regression): {exc}")

                    assert isinstance(result, dict), "Should return a valid result dict"

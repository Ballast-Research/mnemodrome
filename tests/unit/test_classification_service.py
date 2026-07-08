"""Unit tests for classification service (src/services/classification_service.py).

Tests the ClassificationService which uses LLM for entity type classification
with fallback to heuristic-based classification.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.classification_service import ClassificationService


@pytest.fixture
def classification_service():
    """Provide a ClassificationService instance."""
    return ClassificationService()


@pytest.mark.asyncio
class TestClassifyEntityLLM:
    """Tests for classify_entity method using LLM."""

    async def test_classify_entity_llm_success(self, classification_service):
        """Test successful entity classification using LLM."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            mock_llm.classify_entity = AsyncMock(return_value="Person")

            result = await classification_service.classify_entity(
                "Carlos", "Carlos uses Claude Code", "source"
            )

            assert result == "Person"
            mock_llm.classify_entity.assert_called_once_with(
                "Carlos", "Carlos uses Claude Code", "source"
            )

    async def test_classify_entity_llm_fallback_on_error(self, classification_service):
        """Test fallback to heuristic when LLM fails."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            mock_llm.classify_entity = AsyncMock(
                side_effect=Exception("LLM service unavailable")
            )

            result = await classification_service.classify_entity("Alice", "Developer")

            # Should fall back to heuristic
            assert result == "Person"

    async def test_classify_entity_various_types(self, classification_service):
        """Test classification of various entity types."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            test_cases = [
                ("Alice", "Project Manager", "Person"),
                ("Neo4j", "Graph database", "Technology"),
                ("Mnemodrome", "AI infrastructure", "Project"),
            ]

            for name, context, expected_type in test_cases:
                mock_llm.classify_entity = AsyncMock(return_value=expected_type)

                result = await classification_service.classify_entity(name, context)

                assert result == expected_type


@pytest.mark.asyncio
class TestHeuristicClassification:
    """Tests for heuristic-based classification fallback."""

    async def test_heuristic_classify_person(self, classification_service):
        """Test heuristic classification of Person."""
        result = await classification_service._heuristic_classify("Bob", "developer")
        assert result == "Person"

    async def test_heuristic_classify_technology(self, classification_service):
        """Test heuristic classification of Technology."""
        result = await classification_service._heuristic_classify("Neo4j", "")
        assert result == "Technology"

    async def test_heuristic_classify_project(self, classification_service):
        """Test heuristic classification of Project."""
        result = await classification_service._heuristic_classify("Phase4a", "")
        assert result == "Project"

    async def test_heuristic_classify_organization(self, classification_service):
        """Test heuristic classification of Organization."""
        result = await classification_service._heuristic_classify("Anthropic", "")
        assert result == "Organization"

    async def test_heuristic_classify_concept(self, classification_service):
        """Test heuristic classification of Concept."""
        result = await classification_service._heuristic_classify("Design Pattern", "")
        assert result == "Concept"

    async def test_heuristic_classify_event(self, classification_service):
        """Test heuristic classification of Event."""
        result = await classification_service._heuristic_classify("Deployment", "")
        assert result == "Event"

    async def test_heuristic_classify_document(self, classification_service):
        """Test heuristic classification of Document."""
        result = await classification_service._heuristic_classify("README.md", "")
        assert result == "Document"

    async def test_heuristic_classify_location(self, classification_service):
        """Test heuristic classification of Location."""
        result = await classification_service._heuristic_classify("vault-storage", "")
        assert result == "Location"

    async def test_heuristic_classify_default(self, classification_service):
        """Test default classification when no keywords match."""
        result = await classification_service._heuristic_classify("xyz123", "")
        assert result == "Entity"

    async def test_heuristic_classify_context_based(self, classification_service):
        """Test context-based heuristic classification."""
        result = await classification_service._heuristic_classify(
            "something", "stores data processes information"
        )
        assert result == "Technology"

    async def test_heuristic_classify_case_insensitive(self, classification_service):
        """Test that heuristic classification is case-insensitive."""
        result = await classification_service._heuristic_classify("BOB", "DEVELOPER")
        assert result == "Person"


@pytest.mark.asyncio
class TestClassifyBatch:
    """Tests for batch_classify method."""

    async def test_classify_batch_empty(self, classification_service):
        """Test batch classification with empty list."""
        result = await classification_service.classify_batch([])
        assert result == []

    async def test_classify_batch_llm_success(self, classification_service):
        """Test successful batch classification via LLM."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            mock_llm.classify_entities_batch = AsyncMock(
                return_value=["Person", "Technology", "Project"]
            )

            entities = [
                {"name": "Carlos", "role": "subject"},
                {"name": "Neo4j", "role": "object"},
                {"name": "Mnemodrome", "role": "entity"},
            ]

            result = await classification_service.classify_batch(
                entities, context="Carlos uses Neo4j for Mnemodrome"
            )

            assert len(result) == 3
            assert result[0] == "Person"
            assert result[1] == "Technology"
            assert result[2] == "Project"

    async def test_classify_batch_llm_failure_fallback(self, classification_service):
        """Test fallback to sequential classification when batch fails."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            mock_llm.classify_entities_batch = AsyncMock(
                side_effect=Exception("LLM error")
            )
            mock_llm.classify_entity = AsyncMock(side_effect=Exception("LLM error"))

            entities = [
                {"name": "Carlos", "role": "subject", "context": "developer"},
                {"name": "Neo4j", "role": "object", "context": "database"},
            ]

            result = await classification_service.classify_batch(entities)

            # Should fall back to heuristic sequential classification
            assert len(result) == 2

    async def test_classify_batch_builds_context(self, classification_service):
        """Test that batch builds context from individual contexts."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            mock_llm.classify_entities_batch = AsyncMock(
                return_value=["Person", "Technology"]
            )

            entities = [
                {"name": "Carlos", "context": "developer", "role": "subject"},
                {"name": "Neo4j", "context": "database", "role": "object"},
            ]

            await classification_service.classify_batch(entities)

            # Verify context was built and passed
            call_args = mock_llm.classify_entities_batch.call_args
            context_arg = call_args[0][1]
            assert "developer" in context_arg or "database" in context_arg

    async def test_classify_batch_default_context(self, classification_service):
        """Test batch classification with default context."""
        with patch("src.services.classification_service.llm_service") as mock_llm:
            mock_llm.classify_entities_batch = AsyncMock(
                return_value=["Entity", "Entity"]
            )

            entities = [
                {"name": "Entity1", "role": "subject"},
                {"name": "Entity2", "role": "object"},
            ]

            result = await classification_service.classify_batch(entities)

            assert len(result) == 2


class TestValidateType:
    """Tests for validate_type method."""

    def test_validate_type_valid_types(self, classification_service):
        """Test validation of all valid entity types."""
        valid_types = [
            "Person",
            "Project",
            "Organization",
            "Concept",
            "Technology",
            "Event",
            "Location",
            "Document",
            "Entity",
        ]

        for entity_type in valid_types:
            result = classification_service.validate_type(entity_type)
            assert result == entity_type

    def test_validate_type_invalid_type(self, classification_service):
        """Test validation of invalid type returns default."""
        result = classification_service.validate_type("InvalidType")
        assert result == "Entity"

    def test_validate_type_empty_string(self, classification_service):
        """Test validation of empty string."""
        result = classification_service.validate_type("")
        assert result == "Entity"

    def test_validate_type_case_sensitive(self, classification_service):
        """Test that validation is case-sensitive."""
        result = classification_service.validate_type("person")
        assert result == "Entity"  # lowercase not valid


class TestClassificationServiceInitialization:
    """Tests for ClassificationService initialization."""

    def test_initialization_loads_valid_types(self):
        """Test that service initializes with valid types."""
        service = ClassificationService()

        assert hasattr(service, "valid_types")
        assert "Person" in service.valid_types
        assert "Technology" in service.valid_types

    def test_valid_entity_types_constant(self, classification_service):
        """Test that VALID_ENTITY_TYPES constant exists."""
        assert hasattr(ClassificationService, "VALID_ENTITY_TYPES")
        assert "Person" in ClassificationService.VALID_ENTITY_TYPES
        assert "Project" in ClassificationService.VALID_ENTITY_TYPES


@pytest.mark.asyncio
class TestClassificationEdgeCases:
    """Edge case tests for classification."""

    async def test_classify_empty_name(self, classification_service):
        """Test classification with empty name."""
        result = await classification_service._heuristic_classify("", "")
        assert result == "Entity"

    async def test_classify_special_characters(self, classification_service):
        """Test classification with special characters in name."""
        result = await classification_service._heuristic_classify("test@#$%", "")
        assert result == "Entity"

    async def test_classify_very_long_name(self, classification_service):
        """Test classification with very long entity name."""
        long_name = "a" * 1000
        result = await classification_service._heuristic_classify(long_name, "")
        assert result in ClassificationService.VALID_ENTITY_TYPES

    async def test_classify_with_numbers(self, classification_service):
        """Test classification of entities with numbers."""
        result = await classification_service._heuristic_classify("Phase123", "")
        # Should classify based on 'phase' keyword
        assert result == "Project"

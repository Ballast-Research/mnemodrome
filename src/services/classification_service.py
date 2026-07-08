"""
Classification Service - LLM-Based Entity Type Determination

Implements Amendment 7A: LLM-Based Entity Classification
Uses fast LLM inference to determine entity types from context.
"""

from loguru import logger

from src.services.entity_factory import EntityFactory
from src.services.llm_service import llm_service


class ClassificationService:
    """
    LLM-based entity type classification service.

    Uses LLM inference to determine entity type from name, context, and role.
    Optimized for speed (<100ms per entity) using cheap/fast models.

    Constitutional Authority: Amendment 7A (LLM Classification)
    """

    # Valid entity types (used by LLM service)
    VALID_ENTITY_TYPES = [
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

    def __init__(self):
        """Initialize classification service."""
        self.valid_types = EntityFactory.get_valid_types()

    async def classify_entity(
        self, name: str, context: str = "", role: str = "entity"
    ) -> str:
        """
        Classify entity type using LLM analysis.

        T053: LLM Classification Implementation

        Performance Target: < 200ms per entity (Amendment 7C)

        Args:
            name: Entity name (e.g., "Alice", "Mnemodrome", "Neo4j")
            context: Surrounding context (e.g., "Alice uses Claude Code")
            role: Entity role in relationship ("source", "target", "entity")

        Returns:
            Entity type string (Person, Project, Technology, etc.)

        Example:
            # Infer from name + context
            type1 = await classify("Alice", "Alice uses Claude Code", "source")
            # Returns: "Person"

            type2 = await classify("Neo4j", "Mnemodrome stores data in Neo4j", "target")
            # Returns: "Technology"

            type3 = await classify("Phase 4a", "We completed Phase 4a yesterday", "entity")
            # Returns: "Project"
        """
        logger.debug(f"Classifying entity: {name} (role: {role})")

        try:
            # Use LLM service for classification
            entity_type = await llm_service.classify_entity(name, context, role)

            logger.debug(f"LLM classified: {name} → {entity_type}")
            return entity_type

        except Exception as e:
            logger.error(
                f"LLM classification failed for '{name}', falling back to heuristic: {e}"
            )

            # Fallback to heuristic
            entity_type = await self._heuristic_classify(name, context, role)

            logger.debug(f"Heuristic classified: {name} → {entity_type}")
            return entity_type

    async def _heuristic_classify(
        self, name: str, context: str = "", role: str = "entity"
    ) -> str:
        """
        Heuristic-based classification (temporary placeholder for LLM).

        This is a stopgap until LLM classification is implemented.
        Uses pattern matching and keyword detection.

        Args:
            name: Entity name
            context: Context string
            role: Entity role

        Returns:
            Best-guess entity type
        """
        name_lower = name.lower()
        context_lower = context.lower()

        # Person indicators
        person_keywords = {
            "alice",
            "bob",
            "developer",
            "engineer",
            "user",
            "architect",
            "designer",
            "manager",
            "claude",
        }
        if any(keyword in name_lower for keyword in person_keywords):
            return "Person"

        # Technology indicators
        tech_keywords = {
            "neo4j",
            "python",
            "fastmcp",
            "docker",
            "ssh",
            "api",
            "database",
            "server",
            "framework",
            "library",
            "driver",
            "mcp",
            "llm",
            "claude code",
            "claude desktop",
        }
        if any(keyword in name_lower for keyword in tech_keywords):
            return "Technology"

        # Project indicators
        project_keywords = {
            "phase",
            "mnemodrome",
            "amendment",
            "feature",
            "sprint",
            "milestone",
            "project",
            "initiative",
            "branch",
        }
        if any(keyword in name_lower for keyword in project_keywords):
            return "Project"

        # Organization indicators
        org_keywords = {
            "anthropic",
            "company",
            "organization",
            "team",
            "group",
            "corporation",
            "agency",
            "foundation",
        }
        if any(keyword in name_lower for keyword in org_keywords):
            return "Organization"

        # Concept indicators
        concept_keywords = {
            "pattern",
            "principle",
            "architecture",
            "design",
            "strategy",
            "methodology",
            "approach",
            "framework",
            "paradigm",
            "theory",
            "constraint",
            "protocol",
            "hygiene",
        }
        if any(keyword in name_lower for keyword in concept_keywords):
            return "Concept"

        # Event indicators
        event_keywords = {
            "meeting",
            "session",
            "merge",
            "commit",
            "deployment",
            "release",
            "launch",
            "discovery",
            "testing",
        }
        if any(keyword in name_lower for keyword in event_keywords):
            return "Event"

        # Document indicators
        doc_keywords = {
            "spec",
            "plan",
            "readme",
            "documentation",
            "guide",
            "manual",
            "book",
            "paper",
            "report",
            ".md",
            ".pdf",
        }
        if any(keyword in name_lower for keyword in doc_keywords):
            return "Document"

        # Location indicators
        location_keywords = {
            "server",
            "directory",
            "path",
            "repository",
            "folder",
            "vault",
            "storage",
        }
        if any(keyword in name_lower for keyword in location_keywords):
            # Distinguish between Technology (server software) and Location (server machine)
            if "server" in name_lower and context_lower:
                if any(
                    word in context_lower for word in ["running", "deployed", "hosted"]
                ):
                    return "Location"
                return "Technology"
            return "Location"

        # Context-based refinement
        if context_lower:
            # Check for action verbs that indicate entity type
            if any(
                verb in context_lower
                for verb in ["wrote", "said", "created", "developed"]
            ):
                # Subject is likely a person
                if name_lower in context_lower[: context_lower.find(name_lower) + 50]:
                    return "Person"

            if any(
                verb in context_lower
                for verb in ["stores", "processes", "executes", "runs"]
            ):
                # Subject is likely technology
                return "Technology"

        # Default fallback
        logger.debug(f"No classification match for '{name}', defaulting to 'Entity'")
        return "Entity"

    async def classify_batch(
        self, entities: list[dict], context: str = ""
    ) -> list[str]:
        """
        Classify multiple entities in a single LLM call (T057 - batch optimization).

        Performance: O(1) LLM call vs O(n) sequential calls.
        Reduces 40s (sequential) to ~8s (batch) for large entity sets.

        Args:
            entities: List of dicts with 'name', 'context', 'role' keys
            context: Optional full context string (used for batch classification)

        Returns:
            List of entity type strings (same order as input)

        Example:
            entities = [
                {"name": "Alice", "role": "subject"},
                {"name": "Neo4j", "role": "object"}
            ]
            types = await classification_service.classify_batch(
                entities,
                context="Alice uses Neo4j for Mnemodrome"
            )
            # Returns: ["Person", "Technology"]
        """
        if not entities:
            return []

        logger.debug(f"Batch classifying {len(entities)} entities using LLM (T057)")

        try:
            # Use LLM batch classification service (single call for all entities)
            entity_data = [
                {"name": e.get("name", ""), "role": e.get("role", "entity")}
                for e in entities
            ]

            # Build full context from individual contexts if not provided
            if not context:
                individual_contexts = [e.get("context", "") for e in entities]
                context = " ".join(filter(None, individual_contexts))
                if not context:
                    context = "Entity classification"

            # Call batch classification (single LLM call)
            types = await llm_service.classify_entities_batch(entity_data, context)

            logger.debug(
                f"Batch classification complete: {len(entities)} entities classified "
                f"in single LLM call"
            )
            return types

        except Exception as e:
            logger.error(
                f"Batch classification failed: {e}, falling back to sequential classification"
            )

            # Fallback to sequential classification if batch fails
            types = []
            for entity_data in entities:
                entity_type = await self.classify_entity(
                    name=entity_data.get("name", ""),
                    context=entity_data.get("context", ""),
                    role=entity_data.get("role", "entity"),
                )
                types.append(entity_type)

            logger.debug(
                f"Sequential fallback complete: {len(types)} entities classified"
            )
            return types

    def validate_type(self, entity_type: str) -> str:
        """
        Validate and normalize entity type.

        Args:
            entity_type: Type string to validate

        Returns:
            Valid type string (falls back to "Entity" if invalid)
        """
        if entity_type in self.valid_types:
            return entity_type

        logger.warning(f"Invalid entity type '{entity_type}', using 'Entity'")
        return "Entity"


# Global service instance
classification_service = ClassificationService()

"""
Graph Factory Service - Multi-Entity Graph Extraction

Implements Amendment 7B: Graph Intelligence Over Simple Triples
Orchestrates LLM extraction, parallel entity resolution, and relationship creation.
"""

from loguru import logger

from src.services.entity_factory import entity_factory
from src.services.llm_service import llm_service
from src.services.relationship_factory import relationship_factory


def _as_name_list(value) -> list:
    """Coerce a source/target field to a list of string names.

    The extraction LLM occasionally returns a list for source/target when a
    sentence has multiple subjects or objects (e.g. {"source": ["A","B"],
    "relation": "USES", "target": "Neo4j"}).  Passing such a list directly to
    ``dict.get()`` raises ``TypeError: unhashable type: 'list'``.  This helper
    normalises the field so the relationship loop can always iterate.

    Args:
        value: Raw source/target value from the LLM extraction dict.

    Returns:
        A non-empty list of stripped, non-empty strings, or an empty list when
        the value is None / empty / whitespace-only.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    return [str(value).strip()] if str(value).strip() else []


class GraphFactory:
    """
    Orchestrator for multi-entity graph extraction from natural language.

    Implements the "entities-first transaction pattern":
    1. Extract all entities and relationships from context (LLM)
    2. Resolve all entities in parallel (EntityFactory + asyncio.gather)
    3. Create all relationships sequentially (RelationshipFactory)

    Constitutional Authority: Amendment 7B (Graph Intelligence)
    """

    async def build_from_extraction(self, context: str) -> dict:
        """
        Extract and create complete subgraph from rich natural language context.

        Orchestration Flow:
        1. LLM extracts entities and relationships
        2. Classify all entities (parallel)
        3. Resolve all entities to UUIDs (parallel - asyncio.gather)
        4. Create all relationships using UUIDs (sequential for integrity)

        Args:
            context: Natural language context with multiple entities/relationships
                    Example: "Alice and Bob collaborated with Charlie to implement
                             Phase 4a using FastMCP and Neo4j"

        Returns:
            Summary dict:
            {
                "entities_created": int,
                "relationships_created": int,
                "entity_types": list[str],
                "entity_uuids": dict[name -> uuid],
                "relationships": list[dict]
            }

        Performance Target: < 1 second for 5-entity context

        Example:
            result = await graph_factory.build_from_extraction(
                "Alice discovered a bug in Phase 4a's capture_fact tool "
                "that only created one entity instead of both"
            )

            # Creates:
            # - Alice (Person)
            # - Phase 4a (Project)
            # - capture_fact (Technology)
            # - entity creation bug (Event)
            # Plus relationships:
            # - Alice -DISCOVERED-> entity creation bug
            # - bug -FOUND_IN-> capture_fact
            # - capture_fact -PART_OF-> Phase 4a
        """
        logger.info(f"Building graph from context: {context[:100]}...")

        # Step 1: Extract entities and relationships using LLM
        extraction = await self._extract_with_llm(context)

        if not extraction["entities"]:
            logger.warning("No entities extracted from context")
            return {
                "entities_created": 0,
                "relationships_created": 0,
                "entity_types": [],
                "entity_uuids": {},
                "relationships": [],
            }

        # Up-front normalization: coerce any list-valued entity name to a canonical
        # string BEFORE any downstream use (classification, resolution, uuid mapping).
        # The LLM occasionally returns a list for multi-subject entities; a raw list
        # passed into batch_create_or_resolve causes AttributeError in _make_key
        # (``name.strip()``) before the late coercion block is ever reached.
        for _entity in extraction["entities"]:
            _raw = _entity["name"]
            if isinstance(_raw, list):
                _normalized = " ".join(_as_name_list(_raw)) or "Unknown"
                logger.warning(
                    f"Entity name is a list {_raw!r}; normalizing to {_normalized!r}"
                )
                _entity["name"] = _normalized

        # Step 2: Batch classify all entities (T057 - batch optimization)
        logger.debug(f"Batch classifying {len(extraction['entities'])} entities...")
        entity_data = [
            {"name": e["name"], "role": e.get("role", "entity")}
            for e in extraction["entities"]
        ]
        entity_types = await llm_service.classify_entities_batch(entity_data, context)

        # Update extraction with types
        for entity, entity_type in zip(
            extraction["entities"], entity_types, strict=False
        ):
            entity["type"] = entity_type

        # Step 3: Resolve all entities to UUIDs in parallel with batch optimization (Optimization 2)
        logger.debug(
            f"Resolving {len(extraction['entities'])} entities with batch cache..."
        )

        # Prepare entities for batch resolution (name, type, summary tuples)
        entities_to_resolve = [
            (entity["name"], entity["type"], context[:500])
            for entity in extraction["entities"]
        ]

        # Use batch_create_or_resolve for efficient cache lookup and parallel Neo4j resolution
        uuid_map_result = await entity_factory.batch_create_or_resolve(
            entities_to_resolve
        )

        # Map entity names to UUIDs for relationship creation.
        # Result dict has format "name:type" -> uuid; extract just name -> uuid.
        # Entity names are guaranteed to be strings by the up-front normalization
        # pass above, so no list-coercion is needed here.
        uuid_map = {}
        for entity in extraction["entities"]:
            entity_key = f"{entity['name']}:{entity['type']}"
            if entity_key in uuid_map_result:
                uuid_map[entity["name"]] = uuid_map_result[entity_key]

        logger.debug(f"Entity resolution complete: {len(uuid_map)} entities")

        # Step 4: Create all relationships using resolved UUIDs in parallel (Optimization 2)
        logger.debug(f"Creating {len(extraction['relationships'])} relationships...")

        # Build list of relationships with resolved UUIDs.
        # Fan-out: when the LLM returns a list for source/target (multi-subject
        # sentences) we expand each combination into its own relationship rather
        # than crashing on an unhashable-list key.
        relationships_to_create = []
        for rel in extraction["relationships"]:
            sources = _as_name_list(rel.get("source"))
            targets = _as_name_list(rel.get("target"))

            if not sources or not targets:
                logger.warning(
                    f"Skipping relationship with empty source or target: {rel!r}"
                )
                continue

            for source_name in sources:
                for target_name in targets:
                    # Get UUIDs from map
                    source_uuid = uuid_map.get(source_name)
                    target_uuid = uuid_map.get(target_name)

                    if not source_uuid or not target_uuid:
                        logger.warning(
                            f"Skipping relationship {source_name} -{rel['relation']}-> {target_name}: "
                            f"Entity not found in extraction"
                        )
                        continue

                    relationships_to_create.append(
                        {
                            "source_uuid": source_uuid,
                            "relation_type": rel["relation"],
                            "target_uuid": target_uuid,
                        }
                    )

        # Create all relationships in parallel using batch_create
        relationships_created = []
        if relationships_to_create:
            try:
                relationships_created = await relationship_factory.batch_create(
                    relationships=relationships_to_create, context=context
                )
                logger.debug(
                    f"Batch relationship creation complete: {len(relationships_created)} created"
                )
            except Exception as e:
                logger.error(f"Batch relationship creation failed: {e}")
                # Continue - relationships are optional enrichment

        # Return summary
        result = {
            "entities_created": len(uuid_map),
            "relationships_created": len(relationships_created),
            "entity_types": entity_types,
            "entity_uuids": uuid_map,
            "relationships": relationships_created,
        }

        logger.info(
            f"Graph created: {result['entities_created']} entities, "
            f"{result['relationships_created']} relationships"
        )

        return result

    async def _extract_with_llm(self, context: str) -> dict:
        """
        Extract structured graph from natural language using LLM.

        T053: LLM Extraction Implementation (replaces heuristic placeholder)

        Args:
            context: Natural language text

        Returns:
            Dict with extracted structure:
            {
                "entities": [
                    {"name": str, "role": str},
                    ...
                ],
                "relationships": [
                    {"source": str, "relation": str, "target": str},
                    ...
                ]
            }

        Example Output:
            {
                "entities": [
                    {"name": "Alice", "role": "subject"},
                    {"name": "Phase 4a", "role": "object"},
                    {"name": "entity creation bug", "role": "object"}
                ],
                "relationships": [
                    {"source": "Alice", "relation": "DISCOVERED", "target": "entity creation bug"},
                    {"source": "entity creation bug", "relation": "FOUND_IN", "target": "Phase 4a"}
                ]
            }
        """
        try:
            # Use LLM service for extraction
            logger.debug(
                f"Extracting graph structure using LLM ({llm_service.provider}/{llm_service.model})"
            )

            result = await llm_service.extract_entities_and_relationships(context)

            logger.debug(
                f"LLM extracted {len(result['entities'])} entities, "
                f"{len(result['relationships'])} relationships"
            )

            return result

        except Exception as e:
            logger.error(f"LLM extraction failed, falling back to heuristic: {e}")

            # Fallback to heuristic extraction
            return await self._heuristic_extraction(context)

    async def _heuristic_extraction(self, context: str) -> dict:
        """
        Fallback heuristic extraction when LLM fails.

        Preserved from original implementation for reliability.
        """
        logger.debug("Using heuristic extraction (LLM fallback)")

        entities = []
        relationships = []

        # Simple heuristic: extract capitalized words and common entities
        import re

        # Find capitalized words and multi-word entities
        words = context.split()
        potential_entities = set()

        # Find capitalized sequences
        i = 0
        while i < len(words):
            if words[i] and words[i][0].isupper():
                entity_words = [words[i]]
                j = i + 1
                # Collect consecutive capitalized words
                while j < len(words) and words[j] and words[j][0].isupper():
                    entity_words.append(words[j])
                    j += 1
                entity_name = " ".join(entity_words)
                # Clean punctuation
                entity_name = re.sub(r"[.,!?;:]", "", entity_name)
                if len(entity_name) > 1:  # Skip single letters
                    potential_entities.add(entity_name)
                i = j
            else:
                i += 1

        # Add entities
        for entity_name in potential_entities:
            entities.append({"name": entity_name, "role": "entity"})

        # Simple relationship extraction (subject-verb-object patterns)
        relation_patterns = [
            (r"(\w+)\s+and\s+(\w+)\s+collaborated", "COLLABORATED_WITH"),
            (r"(\w+)\s+uses\s+(\w+)", "USES"),
            (r"(\w+)\s+implements\s+(\w+)", "IMPLEMENTS"),
            (r"(\w+)\s+includes\s+(\w+)", "INCLUDES"),
            (r"(\w+)\s+discovered\s+(\w+)", "DISCOVERED"),
            (r"(\w+)\s+created\s+(\w+)", "CREATED"),
            (r"(\w+)\s+worked on\s+(\w+)", "WORKED_ON"),
            (r"(\w+)\s+followed\s+(\w+)", "FOLLOWED"),
            (r"(\w+)\s+build\s+(\w+)", "BUILT"),
            (r"(\w+)\s+solve\s+(\w+)", "SOLVES"),
        ]

        context_lower = context.lower()
        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, context_lower)
            for match in matches:
                source = match.group(1).capitalize()
                target = match.group(2).capitalize()

                # Add to relationships if entities exist
                if any(e["name"].lower() == source.lower() for e in entities) and any(
                    e["name"].lower() == target.lower() for e in entities
                ):
                    relationships.append(
                        {"source": source, "relation": relation_type, "target": target}
                    )

        logger.debug(
            f"Heuristic extracted {len(entities)} entities, {len(relationships)} relationships"
        )

        return {"entities": entities, "relationships": relationships}


# Global factory instance
graph_factory = GraphFactory()

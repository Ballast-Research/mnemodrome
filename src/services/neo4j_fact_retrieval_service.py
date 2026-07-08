"""
Neo4j fact retrieval service for bidirectional conflict detection (Feature 025).

Queries Neo4j for existing facts related to entities, enabling conflict validation
against stored knowledge graph.

Constitutional Principle: All validation must reference actual stored knowledge.
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.models.conflict_detection_models import StoredFact

if TYPE_CHECKING:
    from src.services.neo4j_service import Neo4jService


class Neo4jFactRetrievalService:
    """
    Service for retrieving facts from Neo4j for conflict validation.

    Implements:
    - Entity-based fact retrieval
    - Relationship filtering
    - Temporal validity handling
    - Pagination support

    Attributes:
        neo4j_service: Neo4jService instance for graph queries
    """

    def __init__(self, neo4j_service: "Neo4jService"):
        """
        Initialize Neo4jFactRetrievalService.

        Args:
            neo4j_service: Neo4jService instance

        Raises:
            ValueError: If neo4j_service is None
        """
        if neo4j_service is None:
            raise ValueError("neo4j_service is required")

        self.neo4j_service = neo4j_service

        logger.info("Neo4jFactRetrievalService initialized")

    async def get_facts_for_entities(
        self,
        entity_names: list[str],
        relations: list[str] | None = None,
    ) -> list[StoredFact]:
        """
        Retrieve facts from Neo4j for given entity names.

        Finds all relationships where entity appears as source or target.
        Optionally filters by specific relationship types.

        Args:
            entity_names: List of entity names to find facts for
            relations: Optional list of relationship types to filter (default: all)

        Returns:
            List of StoredFact objects

        Raises:
            ValueError: If entity_names is empty
            Exception: If Neo4j query fails
        """
        if not entity_names:
            raise ValueError("entity_names cannot be empty")

        logger.debug(f"Retrieving facts for {len(entity_names)} entities")

        # Build Cypher query to find all facts for these entities
        entity_params = [name.lower().strip() for name in entity_names]

        # Query: Find all relationships where these entities appear
        cypher_query = (
            """
        MATCH (e1:Entity)-[rel:FACT]->(e2:Entity)
        WHERE toLower(e1.name) IN $entity_names OR toLower(e2.name) IN $entity_names
        """
            + ("AND rel.relation_type IN $relations" if relations else "")
            + """
        RETURN
            e1.uuid as entity1_uuid,
            e1.name as entity1_name,
            rel.relation_type as relation,
            e2.uuid as entity2_uuid,
            e2.name as entity2_name,
            rel.created_at as created_at,
            rel.context as context,
            rel.uuid as fact_uuid,
            rel.invalid_at is not null as is_temporal,
            rel.expired_at as expired_at
        ORDER BY rel.created_at DESC
        """
        )

        params = {
            "entity_names": entity_params,
        }
        if relations:
            params["relations"] = [r.upper() for r in relations]

        try:
            results = await self.neo4j_service.query(cypher_query, params)

            facts = []
            for result in results:
                try:
                    # Parse created_at timestamp (may be datetime, str, or Neo4j DateTime)
                    created_at_raw = result.get("created_at")
                    if isinstance(created_at_raw, datetime):
                        created_at = created_at_raw
                    elif isinstance(created_at_raw, str):
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_raw.replace("Z", "+00:00")
                            )
                        except ValueError:
                            created_at = datetime.now(UTC)
                    else:
                        created_at = datetime.now(UTC)

                    fact = StoredFact(
                        entity1_uuid=result.get("entity1_uuid", ""),
                        entity1_name=result.get("entity1_name", ""),
                        relation=result.get("relation", ""),
                        entity2_uuid=result.get("entity2_uuid", ""),
                        entity2_name=result.get("entity2_name", ""),
                        created_at=created_at,
                        context=result.get("context", ""),
                        fact_uuid=result.get("fact_uuid", ""),
                        is_temporal=result.get("is_temporal", False),
                    )

                    facts.append(fact)

                except Exception as e:
                    logger.warning(f"Failed to parse fact result: {result}, error: {e}")
                    continue

            logger.info(
                f"Retrieved {len(facts)} facts for {len(entity_names)} entities"
            )
            return facts

        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            raise

    async def get_facts_by_relationship(
        self,
        entity1_name: str,
        relation: str,
    ) -> list[StoredFact]:
        """
        Retrieve specific facts by entity1 and relationship type.

        Args:
            entity1_name: Source entity name
            relation: Relationship type (e.g., "RUNS_ON")

        Returns:
            List of StoredFact objects

        Raises:
            ValueError: If entity1_name or relation is empty
            Exception: If Neo4j query fails
        """
        if not entity1_name or not entity1_name.strip():
            raise ValueError("entity1_name cannot be empty")
        if not relation or not relation.strip():
            raise ValueError("relation cannot be empty")

        logger.debug(f"Retrieving facts for {entity1_name} {relation}")

        cypher_query = """
        MATCH (e1:Entity)-[rel:FACT]->(e2:Entity)
        WHERE toLower(e1.name) = toLower($entity1_name)
        AND rel.relation_type = $relation
        RETURN
            e1.uuid as entity1_uuid,
            e1.name as entity1_name,
            rel.relation_type as relation,
            e2.uuid as entity2_uuid,
            e2.name as entity2_name,
            rel.created_at as created_at,
            rel.context as context,
            rel.uuid as fact_uuid,
            rel.invalid_at is not null as is_temporal,
            rel.expired_at as expired_at
        ORDER BY rel.created_at DESC
        """

        params = {
            "entity1_name": entity1_name.lower().strip(),
            "relation": relation.upper(),
        }

        try:
            results = await self.neo4j_service.query(cypher_query, params)

            facts = []
            for result in results:
                try:
                    # Parse created_at timestamp
                    created_at_str = result.get("created_at")
                    if isinstance(created_at_str, str):
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                        except ValueError:
                            created_at = datetime.now()
                    else:
                        created_at = datetime.now()

                    fact = StoredFact(
                        entity1_uuid=result.get("entity1_uuid", ""),
                        entity1_name=result.get("entity1_name", ""),
                        relation=result.get("relation", ""),
                        entity2_uuid=result.get("entity2_uuid", ""),
                        entity2_name=result.get("entity2_name", ""),
                        created_at=created_at,
                        context=result.get("context", ""),
                        fact_uuid=result.get("fact_uuid", ""),
                        is_temporal=result.get("is_temporal", False),
                    )

                    facts.append(fact)

                except Exception as e:
                    logger.warning(f"Failed to parse fact result: {result}, error: {e}")
                    continue

            logger.debug(
                f"Retrieved {len(facts)} facts for relationship: {entity1_name} {relation}"
            )
            return facts

        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            raise

    async def get_facts_by_relationship_pair(
        self,
        entity1_name: str,
        relation: str,
        entity2_name: str,
    ) -> StoredFact | None:
        """
        Retrieve a specific fact by exact entity1, relation, entity2 triple.

        Args:
            entity1_name: Source entity name
            relation: Relationship type
            entity2_name: Target entity name

        Returns:
            StoredFact if found, None otherwise

        Raises:
            ValueError: If any parameter is empty
            Exception: If Neo4j query fails
        """
        if not entity1_name or not entity1_name.strip():
            raise ValueError("entity1_name cannot be empty")
        if not relation or not relation.strip():
            raise ValueError("relation cannot be empty")
        if not entity2_name or not entity2_name.strip():
            raise ValueError("entity2_name cannot be empty")

        logger.debug(
            f"Retrieving fact for triple: {entity1_name} {relation} {entity2_name}"
        )

        cypher_query = """
        MATCH (e1:Entity)-[rel:FACT]->(e2:Entity)
        WHERE toLower(e1.name) = toLower($entity1_name)
        AND rel.relation_type = $relation
        AND toLower(e2.name) = toLower($entity2_name)
        RETURN
            e1.uuid as entity1_uuid,
            e1.name as entity1_name,
            rel.relation_type as relation,
            e2.uuid as entity2_uuid,
            e2.name as entity2_name,
            rel.created_at as created_at,
            rel.context as context,
            rel.uuid as fact_uuid,
            rel.invalid_at is not null as is_temporal,
            rel.expired_at as expired_at
        LIMIT 1
        """

        params = {
            "entity1_name": entity1_name.lower().strip(),
            "relation": relation.upper(),
            "entity2_name": entity2_name.lower().strip(),
        }

        try:
            results = await self.neo4j_service.query(cypher_query, params)

            if not results:
                logger.debug(
                    f"No fact found for triple: {entity1_name} {relation} {entity2_name}"
                )
                return None

            result = results[0]

            # Parse created_at timestamp
            created_at_str = result.get("created_at")
            if isinstance(created_at_str, str):
                try:
                    created_at = datetime.fromisoformat(
                        created_at_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    created_at = datetime.now()
            else:
                created_at = datetime.now()

            fact = StoredFact(
                entity1_uuid=result.get("entity1_uuid", ""),
                entity1_name=result.get("entity1_name", ""),
                relation=result.get("relation", ""),
                entity2_uuid=result.get("entity2_uuid", ""),
                entity2_name=result.get("entity2_name", ""),
                created_at=created_at,
                context=result.get("context", ""),
                fact_uuid=result.get("fact_uuid", ""),
                is_temporal=result.get("is_temporal", False),
            )

            logger.debug(f"Found fact: {entity1_name} {relation} {entity2_name}")
            return fact

        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            raise

    async def check_fact_exists(
        self,
        entity1_name: str,
        relation: str,
        entity2_name: str,
    ) -> bool:
        """
        Quick check if a specific fact exists in Neo4j.

        Args:
            entity1_name: Source entity name
            relation: Relationship type
            entity2_name: Target entity name

        Returns:
            True if fact exists, False otherwise
        """
        fact = await self.get_facts_by_relationship_pair(
            entity1_name, relation, entity2_name
        )
        return fact is not None

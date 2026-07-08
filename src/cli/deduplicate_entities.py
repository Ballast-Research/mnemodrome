#!/usr/bin/env python3
"""
Entity Deduplication Script - Phase 4.5 (T050)

Finds and merges duplicate entities in Neo4j graph using EntityFactory UUID resolution.
Addresses legacy data created before Amendment 7 entity deduplication was implemented.

Problem:
- Phase 4a created entities without deduplication (9+ Alice nodes)
- Each name-based creation resulted in new node with different UUID
- Relationships scattered across duplicate nodes

Solution:
- Identify duplicate entities (same name, same type)
- Choose canonical UUID (oldest or most connected)
- Migrate all relationships to canonical node
- Delete duplicate nodes

Usage:
    python -m src.cli.deduplicate_entities --dry-run         # Preview changes
    python -m src.cli.deduplicate_entities --entity "Alice" # Deduplicate specific entity
    python -m src.cli.deduplicate_entities                   # Deduplicate all entities
"""

import argparse
import asyncio
from uuid import uuid4

from loguru import logger

from src.services.neo4j_service import neo4j_service


class EntityDeduplicator:
    """Deduplicates entities in Neo4j graph using EntityFactory pattern."""

    async def find_duplicates(
        self, entity_name: str | None = None
    ) -> dict[str, list[dict]]:
        """
        Find all duplicate entities in the graph.

        Args:
            entity_name: Optional specific entity name to check (default: scan all)

        Returns:
            Dict mapping (name, type) tuples to list of duplicate node records:
            {
                "Alice|Person": [
                    {"uuid": "abc-123", "name": "Alice", "entity_type": "Person",
                     "created_at": datetime, "relationship_count": 5},
                    {"uuid": "def-456", "name": "Alice", "entity_type": "Person",
                     "created_at": datetime, "relationship_count": 3},
                    ...
                ],
                ...
            }
        """
        await neo4j_service.connect()

        # Build name filter if provided
        name_filter = ""
        params = {}
        if entity_name:
            name_filter = "WHERE n.name = $entity_name"
            params["entity_name"] = entity_name

        cypher = f"""
        MATCH (n)
        {name_filter}
        WITH n.name as name, n.entity_type as entity_type, collect(n) as nodes
        WHERE size(nodes) > 1
        UNWIND nodes as node
        OPTIONAL MATCH (node)-[r:RELATES_TO]-() WHERE r.type IS NOT NULL
        RETURN
            name,
            entity_type,
            node.uuid as uuid,
            node.created_at as created_at,
            count(DISTINCT r) as relationship_count
        ORDER BY name, entity_type, created_at ASC
        """

        result = await neo4j_service.query(cypher, params)

        # Group by (name, type)
        duplicates = {}
        for record in result:
            key = f"{record['name']}|{record['entity_type']}"
            if key not in duplicates:
                duplicates[key] = []
            duplicates[key].append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "entity_type": record["entity_type"],
                    "created_at": record["created_at"],
                    "relationship_count": record["relationship_count"],
                }
            )

        logger.info(f"Found {len(duplicates)} duplicate entity groups")
        return duplicates

    async def deduplicate_entity(
        self, entity_name: str, entity_type: str, dry_run: bool = True
    ) -> dict:
        """
        Deduplicate a specific entity by merging all nodes to canonical UUID.

        Strategy:
        1. Find all duplicate nodes for (name, type)
        2. Choose canonical node (most relationships, then oldest)
        3. Migrate all relationships to canonical node
        4. Delete duplicate nodes

        Args:
            entity_name: Entity name to deduplicate
            entity_type: Entity type (Person, Project, etc.)
            dry_run: If True, preview changes without executing (default: True)

        Returns:
            Summary dict with migration stats
        """
        await neo4j_service.connect()

        # Find all nodes for this entity
        # D5 (FR-012): count only native edges (r.type IS NOT NULL) so foreign legacy
        # edges never inflate a dup node's count and cause it to be chosen canonical.
        cypher_find = """
        MATCH (n {name: $name})
        WHERE n.entity_type = $entity_type OR labels(n)[0] = $entity_type
        OPTIONAL MATCH (n)-[r:RELATES_TO]-() WHERE r.type IS NOT NULL
        RETURN
            n.uuid as uuid,
            n.created_at as created_at,
            count(DISTINCT r) as relationship_count,
            collect(DISTINCT type(r)) as relationship_types
        ORDER BY relationship_count DESC, created_at ASC
        """

        nodes = await neo4j_service.query(
            cypher_find, {"name": entity_name, "entity_type": entity_type}
        )

        if len(nodes) <= 1:
            logger.info(f"No duplicates found for {entity_name} ({entity_type})")
            return {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "duplicates_found": len(nodes),
                "canonical_uuid": nodes[0]["uuid"] if nodes else None,
                "merged_count": 0,
                "relationships_migrated": 0,
            }

        # Choose canonical node (most relationships, then oldest)
        canonical = nodes[
            0
        ]  # Already sorted by relationship_count DESC, created_at ASC
        duplicates = nodes[1:]

        logger.info(
            f"Deduplicating {entity_name} ({entity_type}): "
            f"{len(duplicates)} duplicates → canonical UUID {canonical['uuid']}"
        )

        if dry_run:
            logger.info(
                f"[DRY RUN] Would merge {len(duplicates)} nodes to {canonical['uuid']}"
            )
            total_relationships = sum(d["relationship_count"] for d in duplicates)
            return {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "duplicates_found": len(nodes),
                "canonical_uuid": canonical["uuid"],
                "merged_count": len(duplicates),
                "relationships_migrated": total_relationships,
                "dry_run": True,
            }

        # D5 (FR-007, FR-012): native-scoped, fresh-uuid, non-lossy, foreign-safe migration.
        # Key invariants:
        #   - Only migrate edges where r.type IS NOT NULL (native RELATES_TO only)
        #   - Assign a fresh uuid per moved edge (never copy the original uuid)
        #   - NON-LOSSY: do NOT skip if canonical already has (type, target) — keep both
        #   - Foreign-safe DELETE: only delete dup node if it has NO remaining edges
        relationships_migrated = 0
        for duplicate in duplicates:
            # Fetch native outgoing edges on the dup node.
            # D5: use elementId(r) (Neo4j 5 stable-within-tx id) instead of r.uuid so:
            #   - null-uuid edges are not skipped (uuid match on null misses them)
            #   - duplicate-uuid edges each get their OWN fresh uuid (not shared)
            native_out = await neo4j_service.query(
                """
                MATCH (dup {uuid: $dup_uuid})-[r:RELATES_TO]->(target)
                WHERE r.type IS NOT NULL
                RETURN elementId(r) AS element_id, r.uuid AS old_uuid, target.uuid AS target_uuid
                """,
                {"dup_uuid": duplicate["uuid"]},
            )

            for edge in native_out:
                fresh_uuid = str(uuid4())
                await neo4j_service.query(
                    """
                    MATCH (dup {uuid: $dup_uuid})-[r:RELATES_TO]->(target)
                    WHERE elementId(r) = $element_id AND r.type IS NOT NULL
                    MATCH (canonical {uuid: $canonical_uuid})
                    CREATE (canonical)-[new:RELATES_TO]->(target)
                    SET new = properties(r), new.uuid = $fresh_uuid
                    DELETE r
                    """,
                    {
                        "dup_uuid": duplicate["uuid"],
                        "element_id": edge["element_id"],
                        "canonical_uuid": canonical["uuid"],
                        "fresh_uuid": fresh_uuid,
                    },
                )
                relationships_migrated += 1
                logger.debug(
                    f"D5: migrated outgoing edge (elementId={edge['element_id']}, "
                    f"old_uuid={edge['old_uuid']}) → fresh {fresh_uuid}"
                )

            # Fetch native incoming edges on the dup node.
            # Same elementId pattern for safety (null-uuid + duplicate-uuid coverage).
            native_in = await neo4j_service.query(
                """
                MATCH (source)-[r:RELATES_TO]->(dup {uuid: $dup_uuid})
                WHERE r.type IS NOT NULL
                RETURN elementId(r) AS element_id, r.uuid AS old_uuid, source.uuid AS source_uuid
                """,
                {"dup_uuid": duplicate["uuid"]},
            )

            for edge in native_in:
                fresh_uuid = str(uuid4())
                await neo4j_service.query(
                    """
                    MATCH (source)-[r:RELATES_TO]->(dup {uuid: $dup_uuid})
                    WHERE elementId(r) = $element_id AND r.type IS NOT NULL
                    MATCH (canonical {uuid: $canonical_uuid})
                    CREATE (source)-[new:RELATES_TO]->(canonical)
                    SET new = properties(r), new.uuid = $fresh_uuid
                    DELETE r
                    """,
                    {
                        "dup_uuid": duplicate["uuid"],
                        "element_id": edge["element_id"],
                        "canonical_uuid": canonical["uuid"],
                        "fresh_uuid": fresh_uuid,
                    },
                )
                relationships_migrated += 1
                logger.debug(
                    f"D5: migrated incoming edge (elementId={edge['element_id']}, "
                    f"old_uuid={edge['old_uuid']}) → fresh {fresh_uuid}"
                )

            # D5 foreign-safe delete: only delete dup if NO remaining edges (foreign or otherwise)
            remaining = await neo4j_service.query(
                "MATCH (dup {uuid: $dup_uuid})-[r]-() RETURN count(r) AS cnt",
                {"dup_uuid": duplicate["uuid"]},
            )
            remaining_cnt = remaining[0]["cnt"] if remaining else 0

            if remaining_cnt == 0:
                await neo4j_service.query(
                    "MATCH (dup {uuid: $dup_uuid}) DELETE dup",
                    {"dup_uuid": duplicate["uuid"]},
                )
                logger.info(
                    f"D5: deleted dup node {duplicate['uuid']} (no remaining edges)"
                )
            else:
                logger.info(
                    f"D5: dup node {duplicate['uuid']} preserved "
                    f"({remaining_cnt} remaining edges — likely foreign legacy data)"
                )

            logger.info(
                f"Merged {duplicate['uuid']} → {canonical['uuid']} "
                f"({relationships_migrated} native relationships migrated)"
            )

        return {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "duplicates_found": len(nodes),
            "canonical_uuid": canonical["uuid"],
            "merged_count": len(duplicates),
            "relationships_migrated": relationships_migrated,
            "dry_run": False,
        }

    async def deduplicate_all(self, dry_run: bool = True) -> dict:
        """
        Deduplicate all entities in the graph.

        Args:
            dry_run: If True, preview changes without executing (default: True)

        Returns:
            Summary dict with total migration stats
        """
        duplicates = await self.find_duplicates()

        total_merged = 0
        total_relationships = 0
        results = []

        for key, nodes in duplicates.items():
            name, entity_type = key.split("|", 1)
            result = await self.deduplicate_entity(name, entity_type, dry_run=dry_run)
            results.append(result)
            total_merged += result["merged_count"]
            total_relationships += result["relationships_migrated"]

        summary = {
            "total_duplicate_groups": len(duplicates),
            "total_nodes_merged": total_merged,
            "total_relationships_migrated": total_relationships,
            "dry_run": dry_run,
            "details": results,
        }

        logger.info(
            f"Deduplication complete: {total_merged} nodes merged, "
            f"{total_relationships} relationships migrated"
        )

        return summary


async def main():
    """CLI entry point for entity deduplication."""
    parser = argparse.ArgumentParser(
        description="Deduplicate entities in Mnemodrome Neo4j graph"
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="Specific entity name to deduplicate (default: all entities)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing (default: False)",
    )
    parser.add_argument(
        "--type",
        type=str,
        help="Entity type when using --entity (e.g., Person, Project)",
    )

    args = parser.parse_args()

    deduplicator = EntityDeduplicator()

    try:
        if args.entity:
            if not args.type:
                # Try to infer type or use default
                logger.warning(
                    "No --type specified, scanning all types for this entity"
                )
                duplicates = await deduplicator.find_duplicates(entity_name=args.entity)

                if not duplicates:
                    logger.info(f"No duplicates found for entity: {args.entity}")
                    return

                # Deduplicate all type variants
                for key in duplicates.keys():
                    name, entity_type = key.split("|", 1)
                    result = await deduplicator.deduplicate_entity(
                        name, entity_type, dry_run=args.dry_run
                    )
                    print(f"\n{name} ({entity_type}):")
                    print(f"  Canonical UUID: {result['canonical_uuid']}")
                    print(f"  Nodes merged: {result['merged_count']}")
                    print(
                        f"  Relationships migrated: {result['relationships_migrated']}"
                    )
            else:
                result = await deduplicator.deduplicate_entity(
                    args.entity, args.type, dry_run=args.dry_run
                )
                print(f"\n{args.entity} ({args.type}):")
                print(f"  Canonical UUID: {result['canonical_uuid']}")
                print(f"  Nodes merged: {result['merged_count']}")
                print(f"  Relationships migrated: {result['relationships_migrated']}")
        else:
            # Deduplicate all entities
            result = await deduplicator.deduplicate_all(dry_run=args.dry_run)
            print("\nDeduplication Summary:")
            print(f"  Duplicate groups: {result['total_duplicate_groups']}")
            print(f"  Nodes merged: {result['total_nodes_merged']}")
            print(f"  Relationships migrated: {result['total_relationships_migrated']}")

            if args.dry_run:
                print(
                    "\n⚠️  DRY RUN - No changes made. Run without --dry-run to execute."
                )

    except Exception as e:
        logger.error(f"Deduplication failed: {e}")
        raise

    finally:
        await neo4j_service.close()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Mnemodrome Conflict Audit Tool

Scans Neo4j for conflicting facts and generates markdown reports for human resolution.
Designed to run via cron job for ongoing graph hygiene.

Features:
- Detects same entity1+relation with multiple entity2 values
- Distinguishes complementary facts from true conflicts
- Processes previous reports and implements user selections
- Generates new report with radio button UI
- Saves reports to the configured output directory (CONFLICT_REPORT_DIR)

Usage:
    python -m src.tools.conflict_audit

Cron Example:
    0 9 * * * cd /path/to/Mnemodrome && .venv/bin/python -m src.tools.conflict_audit
"""

import asyncio
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.services.neo4j_service import Neo4jService

# Report output directory (override via CONFLICT_REPORT_DIR env var)
REPORT_DIR = Path(os.getenv("CONFLICT_REPORT_DIR", "./data/conflict-reports"))
PREVIOUS_REPORT_PATTERN = "conflict-audit-*.md"

# Limit conflicts per report to keep report files manageable
MAX_CONFLICTS_PER_REPORT = 100


class ConflictAudit:
    """Manages conflict detection and resolution workflow"""

    def __init__(self):
        self.neo4j = Neo4jService()
        self.conflicts: list[dict] = []
        self.total_conflicts: int = 0

    async def initialize(self):
        """Initialize Neo4j connection"""
        await self.neo4j.connect()
        logger.info("Initialized Neo4j connection for conflict audit")

    async def close(self):
        """Close Neo4j connection"""
        await self.neo4j.close()

    async def find_conflicts(self) -> list[dict]:
        """
        Query Neo4j for potential conflicts.

        A conflict is: Same entity1 + relation with different entity2 values
        where both facts are currently valid (invalid_at IS NULL AND expired_at IS NULL)

        Returns:
            List of conflict dictionaries with entity details and timestamps (limited to MAX_CONFLICTS_PER_REPORT)
        """
        # D7 (FR-012): scope to native edges (r.type IS NOT NULL).
        # Also fix r.name → r.type (r.name was always wrong; r.type is the semantic type field).
        count_query = """
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        WHERE r.invalid_at IS NULL
          AND r.expired_at IS NULL
          AND r.type IS NOT NULL
        WITH e1, r.type as relation, collect({entity2: e2}) as facts
        WHERE size(facts) > 1
        RETURN count(*) as total_conflicts
        """

        query = """
        // Find all currently valid native relationships grouped by entity1+relation
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        WHERE r.invalid_at IS NULL
          AND r.expired_at IS NULL
          AND r.type IS NOT NULL
        WITH e1, r.type as relation, collect({
            entity2: e2,
            relationship: r,
            created_at: r.created_at,
            valid_at: r.valid_at,
            uuid: r.uuid,
            context: r.context
        }) as facts
        WHERE size(facts) > 1  // Only where multiple facts exist

        // Return grouped conflicts (limited)
        RETURN e1.name as entity1,
               e1.uuid as entity1_uuid,
               relation,
               facts,
               size(facts) as fact_count
        ORDER BY fact_count DESC, entity1
        LIMIT $limit
        """

        async with self.neo4j.driver.session(database=self.neo4j.database) as session:
            # Get total count
            count_result = await session.run(count_query)
            count_data = await count_result.data()
            self.total_conflicts = count_data[0]["total_conflicts"] if count_data else 0

            # Get limited conflicts
            result = await session.run(query, {"limit": MAX_CONFLICTS_PER_REPORT})
            records = await result.data()

        logger.info(
            f"Found {self.total_conflicts} total conflicts, returning {len(records)} for review"
        )

        conflicts = []
        for record in records:
            conflict = {
                "entity1": record["entity1"],
                "entity1_uuid": record["entity1_uuid"],
                "relation": record["relation"],
                "facts": [
                    {
                        "entity2": fact["entity2"]["name"],
                        "entity2_uuid": fact["entity2"]["uuid"],
                        "created_at": fact["created_at"],
                        "valid_at": fact["valid_at"],
                        "uuid": fact["uuid"],
                        "context": fact.get("context", ""),
                    }
                    for fact in record["facts"]
                ],
                "fact_count": record["fact_count"],
            }
            conflicts.append(conflict)

        self.conflicts = conflicts
        return conflicts

    def process_previous_report(self, report_path: Path) -> list[dict]:
        """
        Parse previous report and extract user selections.

        Looks for patterns like:
        - [x] Resolve
        - [x] Ignore
        - [x] Invalidate

        Returns:
            List of actions to perform: [{"action": "resolve", "uuid": "...", ...}]
        """
        if not report_path.exists():
            logger.info(f"No previous report found at {report_path}")
            return []

        logger.info(f"Processing previous report: {report_path}")

        content = report_path.read_text()
        actions = []

        # Pattern to match conflict sections and selected actions
        # New format: ## Conflict 1: Mnemodrome has 2 different RUNS_ON relationships
        # Each fact has its checkbox inline:
        #   1. `entity1` **RELATION** `entity2`
        #      - [x] **Invalidate this fact**

        conflict_pattern = r"## Conflict \d+: (.+?) has \d+ different (.+?) relationships\n(.*?)(?=\n## |$)"
        conflicts = re.finditer(conflict_pattern, content, re.DOTALL)

        for conflict_match in conflicts:
            entity1 = conflict_match.group(1).strip()
            relation = conflict_match.group(2).strip()
            section = conflict_match.group(3)

            # Check global ignore first
            ignore_match = re.search(r"- \[x\] Ignore", section)
            if ignore_match:
                actions.append(
                    {"action": "ignore", "entity1": entity1, "relation": relation}
                )
                logger.info(
                    f"Action: Ignore {entity1} {relation} (complementary facts)"
                )
                continue

            # Find individual fact invalidations
            # Pattern: 1. `entity1` **RELATION** `entity2`
            #          - Created: date
            #          - [x] **Invalidate this fact**
            fact_pattern = (
                r"\d+\. `"
                + re.escape(entity1)
                + r"` \*\*"
                + re.escape(relation)
                + r"\*\* `(.+?)`\n.*?- \[x\] \*\*Invalidate this fact\*\*"
            )
            invalidate_matches = re.finditer(fact_pattern, section, re.DOTALL)

            for inv_match in invalidate_matches:
                target = inv_match.group(1).strip()
                actions.append(
                    {
                        "action": "invalidate",
                        "entity1": entity1,
                        "relation": relation,
                        "target": target,
                    }
                )
                logger.info(f"Action: Invalidate {entity1} {relation} {target}")

        logger.success(f"Extracted {len(actions)} actions from previous report")
        return actions

    async def implement_actions(self, actions: list[dict]):
        """
        Implement user-selected conflict resolutions.

        Uses MCP tools via direct Neo4j queries (since this is internal tooling).
        """
        if not actions:
            logger.info("No actions to implement")
            return

        logger.info(f"Implementing {len(actions)} conflict resolutions...")

        async with self.neo4j.driver.session(database=self.neo4j.database) as session:
            for action in actions:
                try:
                    if action["action"] == "resolve":
                        # D7 (FR-012): native-scoped; fix r.name → r.type
                        query = """
                        MATCH (e1:Entity {name: $entity1})-[r:RELATES_TO]->(e2:Entity {name: $invalidate})
                        WHERE r.type = $relation
                          AND r.type IS NOT NULL
                          AND r.invalid_at IS NULL
                          AND r.expired_at IS NULL
                        SET r.invalid_at = datetime(),
                            r.expired_at = datetime()
                        RETURN r.uuid as uuid
                        """
                        result = await session.run(
                            query,
                            {
                                "entity1": action["entity1"],
                                "relation": action["relation"],
                                "invalidate": action["invalidate"],
                            },
                        )
                        records = await result.data()
                        if records:
                            logger.success(
                                f"✓ Resolved: {action['entity1']} {action['relation']} - invalidated {action['invalidate']}"
                            )
                        else:
                            logger.warning(
                                f"  No matching fact found for {action['entity1']} {action['relation']} {action['invalidate']}"
                            )

                    elif action["action"] == "ignore":
                        # Just log - these are complementary facts
                        logger.info(
                            f"✓ Ignored: {action['entity1']} {action['relation']} (complementary facts)"
                        )

                    elif action["action"] == "invalidate":
                        # D7 (FR-012): native-scoped; fix r.name → r.type
                        query = """
                        MATCH (e1:Entity {name: $entity1})-[r:RELATES_TO]->(e2:Entity {name: $target})
                        WHERE r.type = $relation
                          AND r.type IS NOT NULL
                          AND r.invalid_at IS NULL
                          AND r.expired_at IS NULL
                        SET r.invalid_at = datetime(),
                            r.expired_at = datetime()
                        RETURN r.uuid as uuid
                        """
                        result = await session.run(
                            query,
                            {
                                "entity1": action["entity1"],
                                "relation": action["relation"],
                                "target": action["target"],
                            },
                        )
                        records = await result.data()
                        if records:
                            logger.success(
                                f"✓ Invalidated: {action['entity1']} {action['relation']} {action['target']}"
                            )
                        else:
                            logger.warning(
                                f"  No matching fact found for {action['entity1']} {action['relation']} {action['target']}"
                            )

                except Exception as e:
                    logger.error(f"Error implementing action {action}: {e}")

        logger.success("Finished implementing conflict resolutions")

    def generate_report(self, output_path: Path):
        """
        Generate markdown conflict report with checkbox UI.

        Note: Markdown doesn't support radio buttons, so we use checkboxes
        with instructions to select only one.
        """
        if not self.conflicts:
            logger.info("No conflicts found - no report generated")
            return

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

        # Build header with total vs. displayed counts
        header = f"""# Mnemodrome Conflict Audit Report

**Generated**: {timestamp}
**Total Conflicts**: {self.total_conflicts}
**Showing**: {len(self.conflicts)} (limit: {MAX_CONFLICTS_PER_REPORT} per report)
"""

        # Add note if there are more conflicts beyond this report
        if self.total_conflicts > len(self.conflicts):
            remaining = self.total_conflicts - len(self.conflicts)
            header += f"\n> **Note**: {remaining} additional conflicts remain after resolving this batch.\n"
            header += "> The next audit run will show the next batch.\n"

        header += "\n---"

        report = (
            header
            + """

## Instructions

For each conflict below, select ONE action by marking it with `[x]`:

- **Resolve**: Keep one fact, invalidate the other (supersession)
- **Ignore**: Both facts are valid (complementary, not conflicting)
- **Invalidate**: Mark a specific fact as no longer true

After marking your selections, the next audit run will implement them automatically.

---

"""
        )

        for idx, conflict in enumerate(self.conflicts, 1):
            fact_count = len(conflict["facts"])

            # Human-readable conflict title and explanation
            report += f"## Conflict {idx}: {conflict['entity1']} has {fact_count} different {conflict['relation']} relationships\n\n"

            # Plain English explanation
            report += f"**What's happening:** The graph says `{conflict['entity1']}` {conflict['relation']} {fact_count} different things. "
            if fact_count == 2:
                report += "Usually only one should be true (supersession), or both might be valid (complementary facts).\n\n"
            else:
                report += "This might be okay (plural relationships) or might be duplicates/errors.\n\n"

            # List facts with invalidate checkbox right next to each one
            report += "**Current facts in the graph:**\n\n"
            for fact_idx, fact in enumerate(conflict["facts"], 1):
                created_date = (
                    fact["created_at"].strftime("%Y-%m-%d")
                    if hasattr(fact["created_at"], "strftime")
                    else str(fact["created_at"])[:10]
                )
                report += f"{fact_idx}. `{conflict['entity1']}` **{conflict['relation']}** `{fact['entity2']}`\n"
                report += f"   - Created: {created_date}\n"
                if fact["context"]:
                    report += f"   - Context: {fact['context']}\n"
                # Invalidate checkbox right after the fact
                report += "   - [ ] **Invalidate this fact**\n"
                report += "\n"

            # Global ignore option at the end
            report += "**Or, if all facts are valid:**\n\n"
            report += "- [ ] Ignore - all these relationships are correct (complementary, not conflicting)\n\n"

            # Technical details in collapsible section
            report += "<details>\n<summary>🔍 Technical Details (UUIDs)</summary>\n\n"
            for fact_idx, fact in enumerate(conflict["facts"], 1):
                report += f"**Fact {fact_idx}:** `{fact['uuid']}`\n\n"
            report += "</details>\n\n"

            report += "---\n\n"

        # Write report
        output_path.write_text(report)
        logger.success(f"Generated conflict report: {output_path}")
        print(f"\n📋 Conflict Report: {output_path}")
        print(f"   Total conflicts: {self.total_conflicts}")
        print(f"   Showing in this report: {len(self.conflicts)}")
        if self.total_conflicts > len(self.conflicts):
            remaining = self.total_conflicts - len(self.conflicts)
            print(f"   Remaining after this batch: {remaining}")
        print("   Review and mark your selections with [x]")


async def main():
    """Main audit workflow"""
    logger.info("Starting Mnemodrome conflict audit...")

    audit = ConflictAudit()
    await audit.initialize()

    try:
        # Step 1: Check for and process previous report
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        previous_reports = sorted(REPORT_DIR.glob(PREVIOUS_REPORT_PATTERN))

        if previous_reports:
            latest_previous = previous_reports[-1]
            logger.info(f"Found previous report: {latest_previous.name}")

            # Extract and implement user selections
            actions = audit.process_previous_report(latest_previous)
            if actions:
                await audit.implement_actions(actions)

                # Archive processed report
                archive_path = latest_previous.with_suffix(".processed.md")
                latest_previous.rename(archive_path)
                logger.info(f"Archived processed report: {archive_path.name}")

        # Step 2: Scan for current conflicts
        logger.info("Scanning for conflicts...")
        conflicts = await audit.find_conflicts()

        # Step 3: Generate new report
        if conflicts:
            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            report_path = REPORT_DIR / f"conflict-audit-{timestamp}.md"
            audit.generate_report(report_path)
        else:
            logger.success("✅ No conflicts found - graph is clean!")

    finally:
        await audit.close()


if __name__ == "__main__":
    asyncio.run(main())

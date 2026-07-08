Feature 006 Live MCP Test Plan

  Here's your structured test plan for after Claude Code restart:

  Pre-Test Verification

  # Confirm server is running
  systemctl is-active mnemodrome-mcp.service
  # Should show: active

  # Check recent logs
  sudo journalctl -u mnemodrome-mcp.service -n 20
  # Should show: ✅ All services initialized

  Test Scenario 1: Simple Conflict Auto-Resolution

  Goal: Verify that simple conflicts (recent facts, clear supersession) auto-resolve

  # Step 1: Create initial fact
  Use: mcp__mnemodrome__capture_fact
  entity1: "TestApp-Feature006"
  relation: "RUNS_ON"
  entity2: "host-primary"
  context: "Initial deployment for Feature 006 testing"

  # Step 2: Create conflicting fact WITHOUT auto-resolve (should detect conflict)
  Use: mcp__mnemodrome__capture_fact
  entity1: "TestApp-Feature006"
  relation: "RUNS_ON"
  entity2: "host-secondary"
  context: "Attempting migration without auto-resolve"
  auto_resolve_conflicts: false

  Expected: Should return conflict_detected status with complexity score

  # Step 3: Create conflicting fact WITH auto-resolve (should succeed)
  Use: mcp__mnemodrome__capture_fact
  entity1: "TestApp-Feature006"
  relation: "RUNS_ON"
  entity2: "host-secondary"
  context: "Migration with auto-resolution enabled"
  auto_resolve_conflicts: true

  Expected: Should return conflict_auto_resolved: true, with invalidated_facts list

  Test Scenario 2: Update Fact Tool

  Goal: Verify atomic fact replacement with supersession chain

  Use: mcp__mnemodrome__update_fact
  old_fact: {
    "entity1": "TestApp-Feature006",
    "relation": "RUNS_ON",
    "entity2": "host-secondary"
  }
  new_fact: {
    "entity1": "TestApp-Feature006",
    "relation": "RUNS_ON",
    "entity2": "Mac mini"
  }
  reason: "Final migration to Mac mini for local testing"

  Expected: Old fact invalidated, new fact created, supersession chain maintained

  Test Scenario 3: Manual Fact Invalidation

  Goal: Test explicit fact invalidation

  # First, find a fact UUID to invalidate
  Use: mcp__mnemodrome__load_context
  prompt: "TestApp-Feature006 RUNS_ON"
  days: 1
  limit: 5

  # Then invalidate it
  Use: mcp__mnemodrome__invalidate_fact
  fact_uuid: "<uuid from previous result>"
  reason: "Testing manual invalidation workflow"
  invalidation_type: "both"

  Expected: Fact marked with invalid_at and expired_at timestamps

  Test Scenario 4: Temporal Queries

  Goal: Verify we can query both current and historical facts

  # Query current facts only
  Use: mcp__mnemodrome__load_context
  prompt: "TestApp-Feature006"
  days: 1

  Expected: Should show only currently valid facts (invalid_at IS NULL)

  # For historical facts, we'd need a custom Neo4j query
  # (Could add a temporal query tool in future if needed)

  Success Criteria

  - ✅ Conflict detection triggers when expected
  - ✅ Auto-resolution works for simple conflicts (complexity < 30)
  - ✅ Manual validation required for complex conflicts
  - ✅ Supersession chains created correctly
  - ✅ Temporal fields (invalid_at, expired_at) set properly
  - ✅ No errors in server logs during testing

  Cleanup After Testing

  # Optional: Remove test entities from Neo4j
  # Would require direct Neo4j query or custom cleanup tool
  # For now, test entities are prefixed with "TestApp-" for easy identification

  Reference Documentation

  - Full examples:
  ./specs/006-fact-invalidation-and-updating/quickstart.md
  - Tool parameters: Check MCP tool descriptions in the Claude Code tool list

  Ready when you restart Claude Code!
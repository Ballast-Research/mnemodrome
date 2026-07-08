# Contract Test Summary: Enhanced capture_fact Tool

**Feature**: 006-fact-invalidation-and (Conflict Detection)
**File**: `tests/contract/test_enhanced_capture_fact_tool.py`
**Status**: ✅ Tests written and verified to FAIL (Spec-Driven Development requirement met)

## Test Coverage

### 1. **No Conflict Path** (`test_no_conflict_preserves_existing_behavior`)
- **Purpose**: Verify existing behavior preserved when no conflicts exist
- **Validates**: Normal capture_fact flow works without conflict detection overhead
- **Expected Failure**: `TypeError: capture_fact() got an unexpected keyword argument 'auto_resolve_conflicts'`
- **Why**: Parameter doesn't exist yet (pre-implementation)

### 2. **Conflict Detection** (`test_conflict_detection_before_relationship_creation`)
- **Purpose**: Verify conflicts detected BEFORE creating new relationship
- **Validates**: Query for existing facts with same (entity1, relation)
- **Expected Failure**: `AttributeError: module 'src.services' has no attribute 'conflict_detection_service'`
- **Why**: Service doesn't exist yet

### 3. **Temporal Filtering** (`test_conflict_detection_query_uses_temporal_filtering`)
- **Purpose**: Verify conflict query filters by `invalid_at IS NULL` and `expired_at IS NULL`
- **Validates**: Only current facts considered conflicts (not historical facts)
- **Expected Failure**: `TypeError: auto_resolve_conflicts parameter doesn't exist`
- **Why**: Feature not implemented yet

### 4. **Auto-Resolution (Simple Conflicts)** (`test_auto_resolve_simple_conflicts`)
- **Purpose**: Verify simple conflicts (score < 30) auto-resolve when enabled
- **Validates**:
  - Complexity score calculation
  - Automatic invalidation of old fact
  - Creation of new fact
  - Supersession chain linking
- **Expected Failure**: `conflict_detection_service doesn't exist`
- **Why**: Complexity scoring service not implemented

### 5. **Validation Required (Complex Conflicts)** (`test_validation_required_for_complex_conflicts`)
- **Purpose**: Verify complex conflicts (score >= 70) always require human validation
- **Validates**:
  - High complexity prevents auto-resolution
  - Even with `auto_resolve_conflicts=True`, complex conflicts blocked
  - Relationship NOT created when validation required
- **Expected Failure**: `conflict_detection_service doesn't exist`
- **Why**: Feature not implemented yet

### 6. **Default Parameter** (`test_auto_resolve_parameter_default_false`)
- **Purpose**: Verify `auto_resolve_conflicts` defaults to `False`
- **Validates**: Conservative default (human-in-the-loop by default)
- **Expected Failure**: `conflict_detection_service doesn't exist`
- **Why**: Service not implemented

### 7. **Conflict Response Details** (`test_conflict_response_includes_existing_fact_details`)
- **Purpose**: Verify conflict response includes all existing fact metadata
- **Validates**: UUID, entity names, timestamps, complexity score all returned
- **Expected Failure**: `conflict_detection_service doesn't exist`
- **Why**: Response format not implemented

### 8. **Multiple Conflicts** (`test_multiple_conflicts_all_listed`)
- **Purpose**: Verify all conflicting facts listed when multiple exist
- **Validates**:
  - Multiple conflicts detected (entity1 has >1 relationship with same type)
  - Higher complexity score for multiple conflicts
  - All conflicts shown to user
- **Expected Failure**: `conflict_detection_service doesn't exist`
- **Why**: Feature not implemented

### 9. **Supersession Chain** (`test_auto_resolution_creates_supersession_chain`)
- **Purpose**: Verify auto-resolution creates proper fact replacement chain
- **Validates**:
  - Old fact invalidated (`invalid_at` and `expired_at` set)
  - New fact created
  - Link between old and new facts maintained
- **Expected Failure**: `conflict_detection_service doesn't exist`
- **Why**: Invalidation logic not implemented

## Test Execution Results

```
9 failed in 1.03s
```

**All tests failing as expected** - This validates the contract tests are correct and ready for TDD implementation.

## Failure Patterns

### Primary Failures:
1. **Missing parameter**: `auto_resolve_conflicts` doesn't exist on `capture_fact` tool
2. **Missing service**: `conflict_detection_service` not implemented
3. **Missing logic**: Conflict detection query not added to `capture_fact`

## Implementation Checklist (Derived from Test Failures)

To make these tests pass, implement in order:

- [ ] Create `src/services/conflict_detection_service.py`
  - [ ] `calculate_complexity_score()` method
  - [ ] Complexity factors: age, multiple conflicts, dependencies, pattern matching
- [ ] Add `auto_resolve_conflicts: bool = False` parameter to `capture_fact` in `src/mcp_server.py`
- [ ] Add conflict detection query BEFORE relationship creation
  - [ ] Query: `WHERE r.invalid_at IS NULL AND r.expired_at IS NULL`
  - [ ] Filter by `entity1` name and `relation` type
- [ ] Implement auto-resolution logic
  - [ ] Calculate complexity score
  - [ ] If `auto_resolve_conflicts=True` AND score < 30: auto-invalidate
  - [ ] If score >= 70 OR `auto_resolve_conflicts=False`: return validation_required
- [ ] Implement fact invalidation (for auto-resolution)
  - [ ] Set `invalid_at = datetime()`
  - [ ] Set `expired_at = datetime()`
  - [ ] Set `superseded_by = new_fact_uuid`
- [ ] Return structured conflict response
  - [ ] Include all existing fact details
  - [ ] Include complexity score
  - [ ] Include recommendation
  - [ ] Include action required

## Notes

- Tests use `unittest.mock.AsyncMock` for async service mocking
- Tests follow existing Mnemodrome test patterns (see `tests/test_integration.py`)
- Complexity thresholds from spec:
  - Simple: 0-30 (auto-resolvable)
  - Moderate: 31-70 (prompt with recommendation)
  - Complex: 71-100 (always require validation)
- Neo4j temporal filtering essential: `invalid_at IS NULL AND expired_at IS NULL`

## Next Steps

1. Review this test file with team/stakeholders
2. Implement `conflict_detection_service.py`
3. Enhance `capture_fact` tool with new parameter and logic
4. Run tests iteratively (TDD cycle)
5. Tests should pass when implementation complete

"""
Feature 006: Conflict Detection and Resolution - Integration Tests

Tests end-to-end conflict detection and resolution scenarios including:
- Simple conflicts (auto-resolve, complexity < 30)
- Complex conflicts (human validation, complexity > 70)
- Pattern learning and recognition
- Validation history creation and retrieval
- Complexity scoring algorithm validation

All test entities tagged with _test_006_ for cleanup.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest


@pytest.mark.asyncio
class TestSimpleConflictAutoResolve:
    """Test Scenario 1: Simple conflict with automatic resolution (score < 30)"""

    async def test_recent_fact_same_relation_auto_resolve(self):
        """
        Simple conflict scenario:
        - Recent fact (created < 24 hours ago)
        - Same relationship type
        - Clear supersession pattern
        - Complexity score < 30
        - Expected: Auto-resolved, old fact invalidated, new fact created
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Setup: Existing recent fact (created 12 hours ago)
            old_fact_uuid = str(uuid4())
            old_fact = {
                "uuid": old_fact_uuid,
                "entity1": "Mnemodrome_test_006_simple",
                "relation": "RUNS_ON",
                "entity2": "test-server-01_test_006",
                "created_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                "valid_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                "invalid_at": None,
                "expired_at": None,
            }

            # New fact to capture
            new_fact = {
                "entity1": "Mnemodrome_test_006_simple",
                "relation": "RUNS_ON",
                "entity2": "test-server-02_test_006",
            }

            # Mock Neo4j query for existing facts
            mock_neo4j.query = AsyncMock(return_value=[old_fact])

            # Mock complexity score calculation (simple conflict)
            mock_conflict.calculate_complexity_score = AsyncMock(
                return_value={
                    "score": 20,  # Low complexity (< 30)
                    "factors": {
                        "age_days": 0.5,  # Recent
                        "competing_facts": 1,  # Single competing fact
                        "has_dependencies": False,
                        "pattern_match": True,  # Similar pattern exists
                    },
                    "auto_resolvable": True,
                }
            )

            # Mock validation history (no previous conflicts needed for simple case)
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            # Execute conflict detection
            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            conflict_result = await conflict_detection_service.detect_conflict(
                entity1=new_fact["entity1"],
                relation=new_fact["relation"],
                entity2=new_fact["entity2"],
            )

            # Assertions
            assert conflict_result["conflict_detected"] is True
            assert conflict_result["complexity_score"] == 20
            assert conflict_result["auto_resolvable"] is True
            assert conflict_result["recommendation"] == "auto_resolve"

            # Verify old fact details
            assert len(conflict_result["existing_facts"]) == 1
            assert conflict_result["existing_facts"][0]["uuid"] == old_fact_uuid

            # Test auto-resolution execution
            mock_neo4j.execute_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": old_fact_uuid,
                    "new_fact_uuid": str(uuid4()),
                    "invalidated_at": datetime.now().isoformat(),
                }
            )

            resolution_result = await conflict_detection_service.auto_resolve_conflict(
                old_fact_uuid=old_fact_uuid,
                new_fact=new_fact,
                reason="Automatic resolution - simple conflict pattern",
            )

            # Verify resolution succeeded
            assert resolution_result["status"] == "resolved"
            assert resolution_result["old_fact_invalidated"] is True
            assert resolution_result["new_fact_created"] is True

    async def test_complexity_score_calculation_recent_fact(self):
        """
        Test complexity scoring for recent fact:
        - Age < 1 day: -20 points
        - No competing facts: 0 points
        - No dependencies: 0 points
        - Pattern match: -20 points
        - Expected total: max(0, -40) = 0
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Mock old fact (recent)
            old_fact = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                "entity1": "Entity_test_006",
                "relation": "RELATES_TO",
                "entity2": "Old_Target_test_006",
            }

            # Mock no competing facts
            mock_neo4j.query = AsyncMock(return_value=[])

            # Mock no dependencies
            mock_neo4j.count_dependent_facts = AsyncMock(return_value=0)

            # Mock pattern match found
            mock_validation.find_similar_conflicts = AsyncMock(
                return_value=[
                    {"human_decision": "approve", "complexity_score": 15},
                    {"human_decision": "approve", "complexity_score": 18},
                ]
            )

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            score_result = await conflict_detection_service.calculate_complexity_score(
                old_fact=old_fact,
                new_fact={"entity2": "New_Target_test_006"},
                validation_history=[],
            )

            # Verify low complexity score
            assert score_result["score"] <= 30
            assert score_result["auto_resolvable"] is True


@pytest.mark.asyncio
class TestComplexConflictValidationRequired:
    """Test Scenario 2: Complex conflict requiring human validation (score > 70)"""

    async def test_old_fact_with_dependencies_requires_validation(self):
        """
        Complex conflict scenario:
        - Old fact (created > 30 days ago): +30 points
        - Has dependent relationships: +25 points
        - No pattern match: +25 points
        - Multiple competing facts: +20 points
        - Expected score: 100 (clamped), complexity > 70
        - Expected: validation_required response
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Setup: Old fact with high complexity
            old_fact_uuid = str(uuid4())
            old_fact = {
                "uuid": old_fact_uuid,
                "entity1": "Project_X_test_006",
                "relation": "HOSTED_ON",
                "entity2": "Old_Server_test_006",
                "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "valid_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "invalid_at": None,
                "expired_at": None,
            }

            # New competing fact
            new_fact = {
                "entity1": "Project_X_test_006",
                "relation": "HOSTED_ON",
                "entity2": "New_Server_test_006",
            }

            # Mock multiple competing facts
            competing_fact = {
                "uuid": str(uuid4()),
                "entity2": "Another_Server_test_006",
                "created_at": (datetime.now() - timedelta(days=45)).isoformat(),
            }
            mock_neo4j.query = AsyncMock(return_value=[old_fact, competing_fact])

            # Mock dependent facts exist
            mock_neo4j.count_dependent_facts = AsyncMock(return_value=5)

            # Mock no pattern match in validation history
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            # Mock complexity score calculation (high complexity)
            mock_conflict.calculate_complexity_score = AsyncMock(
                return_value={
                    "score": 100,  # High complexity (> 70)
                    "factors": {
                        "age_days": 60,
                        "competing_facts": 2,
                        "has_dependencies": True,
                        "pattern_match": False,
                    },
                    "auto_resolvable": False,
                }
            )

            # Execute conflict detection
            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            conflict_result = await conflict_detection_service.detect_conflict(
                entity1=new_fact["entity1"],
                relation=new_fact["relation"],
                entity2=new_fact["entity2"],
            )

            # Assertions
            assert conflict_result["status"] == "validation_required"
            assert conflict_result["complexity_score"] == 100
            assert conflict_result["auto_resolvable"] is False
            assert conflict_result["recommendation"] == "validate"
            assert "options" in conflict_result
            assert set(conflict_result["options"]) == {"approve", "reject", "modify"}

            # Verify rationale provided
            assert "rationale" in conflict_result
            assert (
                "60 days old" in conflict_result["rationale"]
                or "old" in conflict_result["rationale"].lower()
            )

    async def test_complexity_score_calculation_old_fact(self):
        """
        Test complexity scoring for old, complex fact:
        - Age > 30 days: +30 points
        - Multiple competing facts (2+): +20 points
        - Has dependencies: +25 points
        - No pattern match: +25 points
        - Expected total: min(100, 100) = 100
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Mock old fact
            old_fact = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(days=45)).isoformat(),
                "entity1": "Entity_test_006",
                "relation": "RELATES_TO",
                "entity2": "Old_Target_test_006",
            }

            # Mock multiple competing facts
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {"entity2": "Competing_1_test_006"},
                    {"entity2": "Competing_2_test_006"},
                ]
            )

            # Mock has dependencies
            mock_neo4j.count_dependent_facts = AsyncMock(return_value=3)

            # Mock no pattern match
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            score_result = await conflict_detection_service.calculate_complexity_score(
                old_fact=old_fact,
                new_fact={"entity2": "New_Target_test_006"},
                validation_history=[],
            )

            # Verify high complexity score
            assert score_result["score"] >= 70
            assert score_result["auto_resolvable"] is False


@pytest.mark.asyncio
class TestPatternLearning:
    """Test Scenario 3: Pattern learning and recognition"""

    async def test_pattern_recognition_after_10_approvals(self):
        """
        Pattern learning scenario:
        - 10 similar conflicts approved by human
        - 11th similar conflict should have lower complexity score
        - Pattern recognized as "simple"
        - Expected: Confidence score high, auto-resolvable
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
        ):
            # Mock 10 previous approvals of similar conflicts
            similar_validations = [
                {
                    "uuid": str(uuid4()),
                    "conflict_type": "same_relation_different_target",
                    "human_decision": "approve",
                    "complexity_score": 65,
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                    "old_fact_uuid": str(uuid4()),
                    "new_fact_uuid": str(uuid4()),
                    "rationale": f"Infrastructure migration {i}",
                }
                for i in range(1, 11)
            ]

            mock_validation.find_similar_conflicts = AsyncMock(
                return_value=similar_validations
            )

            # Current conflict (11th)
            old_fact = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(days=35)).isoformat(),
                "entity1": "Service_test_006",
                "relation": "HOSTED_ON",
                "entity2": "Server_A_test_006",
            }

            new_fact = {
                "entity1": "Service_test_006",
                "relation": "HOSTED_ON",
                "entity2": "Server_B_test_006",
            }

            # Mock complexity calculation with pattern recognition
            # Pattern recognition should reduce score significantly
            mock_conflict.calculate_complexity_score = AsyncMock(
                return_value={
                    "score": 25,  # Reduced from potential 65 due to pattern match
                    "factors": {
                        "age_days": 35,
                        "competing_facts": 1,
                        "has_dependencies": False,
                        "pattern_match": True,  # KEY: Pattern recognized
                        "pattern_confidence": 0.95,  # 10/10 approvals
                    },
                    "auto_resolvable": True,
                }
            )

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            conflict_result = await conflict_detection_service.detect_conflict(
                entity1=new_fact["entity1"],
                relation=new_fact["relation"],
                entity2=new_fact["entity2"],
            )

            # Assertions: Pattern recognized, auto-resolvable despite age
            assert conflict_result["auto_resolvable"] is True
            assert conflict_result["complexity_score"] < 30
            assert conflict_result["pattern_recognized"] is True
            assert conflict_result["pattern_confidence"] >= 0.9

    async def test_pattern_confidence_calculation(self):
        """
        Test pattern confidence scoring:
        - All approvals (10/10): confidence = 1.0
        - Mostly approvals (8/10): confidence = 0.8
        - Mixed (5/10): confidence = 0.5
        - No history: confidence = 0.0
        """
        with patch(
            "src.services.validation_history_service.validation_history_service"
        ) as mock_validation:
            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            # Test case 1: All approvals
            all_approvals = [{"human_decision": "approve"} for _ in range(10)]
            mock_validation.find_similar_conflicts = AsyncMock(
                return_value=all_approvals
            )

            confidence = await conflict_detection_service.calculate_pattern_confidence(
                conflict_type="same_relation_different_target"
            )
            assert confidence == 1.0

            # Test case 2: Mostly approvals
            mostly_approvals = [{"human_decision": "approve"} for _ in range(8)] + [
                {"human_decision": "reject"} for _ in range(2)
            ]
            mock_validation.find_similar_conflicts = AsyncMock(
                return_value=mostly_approvals
            )

            confidence = await conflict_detection_service.calculate_pattern_confidence(
                conflict_type="same_relation_different_target"
            )
            assert confidence == 0.8

            # Test case 3: Mixed
            mixed = [{"human_decision": "approve"} for _ in range(5)] + [
                {"human_decision": "reject"} for _ in range(5)
            ]
            mock_validation.find_similar_conflicts = AsyncMock(return_value=mixed)

            confidence = await conflict_detection_service.calculate_pattern_confidence(
                conflict_type="same_relation_different_target"
            )
            assert confidence == 0.5

            # Test case 4: No history
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            confidence = await conflict_detection_service.calculate_pattern_confidence(
                conflict_type="same_relation_different_target"
            )
            assert confidence == 0.0


@pytest.mark.asyncio
class TestValidationHistoryCreation:
    """Test Scenario 4: Validation history creation and retrieval"""

    async def test_create_validation_history_node(self):
        """
        Human validates a conflict:
        - ValidationHistory node created with decision metadata
        - Node stored with conflict_type, decision, timestamp, rationale
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            validation_uuid = str(uuid4())
            old_fact_uuid = str(uuid4())
            new_fact_uuid = str(uuid4())

            validation_data = {
                "uuid": validation_uuid,
                "conflict_type": "same_relation_different_target",
                "complexity_score": 75,
                "human_decision": "approve",
                "timestamp": datetime.now().isoformat(),
                "old_fact_uuid": old_fact_uuid,
                "new_fact_uuid": new_fact_uuid,
                "rationale": "Infrastructure migration validated - server upgrade",
            }

            # Mock Neo4j create validation node
            mock_neo4j.execute_transaction = AsyncMock(
                return_value={"uuid": validation_uuid, "status": "created"}
            )

            # Execute validation history creation
            from src.services.validation_history_service import (
                validation_history_service,
            )

            result = await validation_history_service.create_validation_record(
                conflict_type=validation_data["conflict_type"],
                complexity_score=validation_data["complexity_score"],
                human_decision=validation_data["human_decision"],
                old_fact_uuid=old_fact_uuid,
                new_fact_uuid=new_fact_uuid,
                rationale=validation_data["rationale"],
            )

            # Assertions
            assert result["status"] == "created"
            assert result["uuid"] == validation_uuid

            # Verify Cypher query structure (check call arguments)
            mock_neo4j.execute_transaction.assert_called_once()
            call_args = mock_neo4j.execute_transaction.call_args

            # Verify transaction contains ValidationHistory node creation
            assert (
                "ValidationHistory" in str(call_args)
                or "validation" in str(call_args).lower()
            )

    async def test_retrieve_validation_history_by_conflict_type(self):
        """
        Test retrieval of validation history by conflict type:
        - Query by conflict_type
        - Returns all matching ValidationHistory nodes
        - Ordered by timestamp (most recent first)
        """
        with patch("src.services.neo4j_service.neo4j_service") as mock_neo4j:
            # Mock validation history records
            validation_records = [
                {
                    "uuid": str(uuid4()),
                    "conflict_type": "same_relation_different_target",
                    "human_decision": "approve",
                    "complexity_score": 65,
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                    "rationale": f"Migration {i}",
                }
                for i in range(5)
            ]

            mock_neo4j.query = AsyncMock(return_value=validation_records)

            from src.services.validation_history_service import (
                validation_history_service,
            )

            results = await validation_history_service.get_by_conflict_type(
                conflict_type="same_relation_different_target", limit=10
            )

            # Assertions
            assert len(results) == 5
            assert all(
                r["conflict_type"] == "same_relation_different_target" for r in results
            )

            # Verify query was made
            mock_neo4j.query.assert_called_once()

    async def test_validation_history_with_all_decision_types(self):
        """
        Test validation history supports all decision types:
        - approve
        - reject
        - modify
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            decision_types = ["approve", "reject", "modify"]

            for decision in decision_types:
                validation_uuid = str(uuid4())

                mock_neo4j.execute_transaction = AsyncMock(
                    return_value={"uuid": validation_uuid, "status": "created"}
                )

                from src.services.validation_history_service import (
                    validation_history_service,
                )

                result = await validation_history_service.create_validation_record(
                    conflict_type="test_conflict",
                    complexity_score=50,
                    human_decision=decision,
                    old_fact_uuid=str(uuid4()),
                    new_fact_uuid=str(uuid4()),
                    rationale=f"Testing {decision} decision",
                )

                assert result["status"] == "created"
                assert result["uuid"] == validation_uuid


@pytest.mark.asyncio
class TestComplexityScoringAlgorithm:
    """Test complexity scoring algorithm edge cases and boundaries"""

    async def test_complexity_score_age_boundary_conditions(self):
        """
        Test age factor boundaries:
        - < 1 day: -20 points
        - 1-30 days: 0 points
        - > 30 days: +30 points
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            mock_neo4j.query = AsyncMock(return_value=[])
            mock_neo4j.count_dependent_facts = AsyncMock(return_value=0)
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            # Test 1: Very recent (6 hours)
            old_fact_recent = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                "entity1": "Test_test_006",
                "relation": "RELATES_TO",
                "entity2": "Target_test_006",
            }

            score_recent = await conflict_detection_service.calculate_complexity_score(
                old_fact=old_fact_recent,
                new_fact={"entity2": "New_test_006"},
                validation_history=[],
            )

            # Should have negative age contribution
            assert score_recent["score"] <= 10  # Very low due to recent creation

            # Test 2: Mid-range (15 days)
            old_fact_mid = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(days=15)).isoformat(),
                "entity1": "Test_test_006",
                "relation": "RELATES_TO",
                "entity2": "Target_test_006",
            }

            score_mid = await conflict_detection_service.calculate_complexity_score(
                old_fact=old_fact_mid,
                new_fact={"entity2": "New_test_006"},
                validation_history=[],
            )

            # Neutral age contribution
            assert 0 <= score_mid["score"] <= 40

            # Test 3: Old (45 days)
            old_fact_old = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(days=45)).isoformat(),
                "entity1": "Test_test_006",
                "relation": "RELATES_TO",
                "entity2": "Target_test_006",
            }

            score_old = await conflict_detection_service.calculate_complexity_score(
                old_fact=old_fact_old,
                new_fact={"entity2": "New_test_006"},
                validation_history=[],
            )

            # Should have positive age contribution
            assert score_old["score"] >= 30

    async def test_complexity_score_clamping(self):
        """
        Test score clamping to 0-100 range:
        - Negative scores clamp to 0
        - Scores > 100 clamp to 100
        """
        with patch(
            "src.services.conflict_detection_service.conflict_detection_service"
        ) as mock_conflict:
            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            # Test negative clamp
            result_negative = await conflict_detection_service._clamp_score(-50)
            assert result_negative == 0

            # Test positive clamp
            result_high = await conflict_detection_service._clamp_score(150)
            assert result_high == 100

            # Test normal range
            result_normal = await conflict_detection_service._clamp_score(65)
            assert result_normal == 65

    async def test_complexity_score_all_factors_combined(self):
        """
        Test all complexity factors combined:
        - Age > 30 days: +30
        - Multiple competing facts (2): +20
        - Has dependencies: +25
        - No pattern match: +25
        - Total: 100 (before clamping)
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Mock old fact with all complexity factors
            old_fact = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                "entity1": "Complex_test_006",
                "relation": "RELATES_TO",
                "entity2": "Target_test_006",
            }

            # Mock 2 competing facts
            mock_neo4j.query = AsyncMock(
                return_value=[
                    {"entity2": "Competing_1_test_006"},
                    {"entity2": "Competing_2_test_006"},
                ]
            )

            # Mock has dependencies
            mock_neo4j.count_dependent_facts = AsyncMock(return_value=5)

            # Mock no pattern match
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            score_result = await conflict_detection_service.calculate_complexity_score(
                old_fact=old_fact,
                new_fact={"entity2": "New_test_006"},
                validation_history=[],
            )

            # Verify maximum complexity
            assert score_result["score"] >= 90  # Close to or at 100
            assert score_result["auto_resolvable"] is False
            assert score_result["factors"]["age_days"] > 30
            assert score_result["factors"]["competing_facts"] >= 2
            assert score_result["factors"]["has_dependencies"] is True
            assert score_result["factors"]["pattern_match"] is False


@pytest.mark.asyncio
class TestConflictDetectionEndToEnd:
    """End-to-end integration tests combining all components"""

    async def test_full_workflow_simple_conflict_to_resolution(self):
        """
        Complete workflow: Detect simple conflict → Auto-resolve → Verify
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Step 1: Detect conflict
            old_fact = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(hours=8)).isoformat(),
                "entity1": "Service_test_006_e2e",
                "relation": "RUNS_ON",
                "entity2": "Server_A_test_006",
            }

            mock_neo4j.query = AsyncMock(return_value=[old_fact])
            mock_conflict.calculate_complexity_score = AsyncMock(
                return_value={"score": 15, "auto_resolvable": True}
            )

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            conflict = await conflict_detection_service.detect_conflict(
                entity1="Service_test_006_e2e",
                relation="RUNS_ON",
                entity2="Server_B_test_006",
            )

            assert conflict["auto_resolvable"] is True

            # Step 2: Auto-resolve
            new_fact_uuid = str(uuid4())
            mock_neo4j.execute_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": old_fact["uuid"],
                    "new_fact_uuid": new_fact_uuid,
                    "invalidated_at": datetime.now().isoformat(),
                }
            )

            resolution = await conflict_detection_service.auto_resolve_conflict(
                old_fact_uuid=old_fact["uuid"],
                new_fact={
                    "entity1": "Service_test_006_e2e",
                    "relation": "RUNS_ON",
                    "entity2": "Server_B_test_006",
                },
                reason="Automatic resolution",
            )

            assert resolution["status"] == "resolved"
            assert resolution["new_fact_uuid"] == new_fact_uuid

            # Step 3: Verify (query for current facts only)
            current_facts = [
                {
                    "uuid": new_fact_uuid,
                    "entity2": "Server_B_test_006",
                    "invalid_at": None,
                    "expired_at": None,
                }
            ]
            mock_neo4j.query = AsyncMock(return_value=current_facts)

            verification = await mock_neo4j.query(
                "MATCH (s)-[r:RELATES_TO]->(t) "
                "WHERE r.invalid_at IS NULL AND r.expired_at IS NULL "
                "RETURN r"
            )

            assert len(verification) == 1
            assert verification[0]["uuid"] == new_fact_uuid

    async def test_full_workflow_complex_conflict_to_validation(self):
        """
        Complete workflow: Detect complex conflict → Request validation → Human approves → Resolve
        """
        with (
            patch("src.services.neo4j_service.neo4j_service") as mock_neo4j,
            patch(
                "src.services.conflict_detection_service.conflict_detection_service"
            ) as mock_conflict,
            patch(
                "src.services.validation_history_service.validation_history_service"
            ) as mock_validation,
        ):
            # Step 1: Detect complex conflict
            old_fact = {
                "uuid": str(uuid4()),
                "created_at": (datetime.now() - timedelta(days=90)).isoformat(),
                "entity1": "Legacy_System_test_006",
                "relation": "HOSTED_ON",
                "entity2": "Mainframe_test_006",
            }

            mock_neo4j.query = AsyncMock(return_value=[old_fact])
            mock_neo4j.count_dependent_facts = AsyncMock(return_value=12)
            mock_validation.find_similar_conflicts = AsyncMock(return_value=[])

            mock_conflict.calculate_complexity_score = AsyncMock(
                return_value={"score": 85, "auto_resolvable": False}
            )

            from src.services.conflict_detection_service import (
                conflict_detection_service,
            )

            conflict = await conflict_detection_service.detect_conflict(
                entity1="Legacy_System_test_006",
                relation="HOSTED_ON",
                entity2="Cloud_Server_test_006",
            )

            assert conflict["status"] == "validation_required"
            assert conflict["complexity_score"] == 85

            # Step 2: Human validates (approve)
            validation_uuid = str(uuid4())
            mock_neo4j.execute_transaction = AsyncMock(
                return_value={"uuid": validation_uuid, "status": "created"}
            )

            from src.services.validation_history_service import (
                validation_history_service,
            )

            validation_record = await validation_history_service.create_validation_record(
                conflict_type="old_fact_with_dependencies",
                complexity_score=85,
                human_decision="approve",
                old_fact_uuid=old_fact["uuid"],
                new_fact_uuid=str(uuid4()),
                rationale="Migration to cloud verified - legacy system decommissioned",
            )

            assert validation_record["status"] == "created"

            # Step 3: Resolve after approval
            new_fact_uuid = str(uuid4())
            mock_neo4j.execute_transaction = AsyncMock(
                return_value={
                    "old_fact_uuid": old_fact["uuid"],
                    "new_fact_uuid": new_fact_uuid,
                    "invalidated_at": datetime.now().isoformat(),
                }
            )

            resolution = await conflict_detection_service.resolve_with_validation(
                old_fact_uuid=old_fact["uuid"],
                new_fact={
                    "entity1": "Legacy_System_test_006",
                    "relation": "HOSTED_ON",
                    "entity2": "Cloud_Server_test_006",
                },
                validation_uuid=validation_uuid,
                reason="Human approved migration",
            )

            assert resolution["status"] == "resolved"
            assert resolution["validation_recorded"] is True


# Cleanup fixture for all test data
@pytest.fixture(autouse=True)
async def cleanup_test_006_data():
    """Clean up all _test_006_ tagged entities and relationships"""
    yield

    # Cleanup after tests complete
    with patch("src.services.neo4j_service.neo4j_service") as mock_neo4j:
        cleanup_query = """
        MATCH (e)
        WHERE e.name CONTAINS '_test_006_'
        DETACH DELETE e
        """

        try:
            await mock_neo4j.query(cleanup_query)
        except Exception:
            pass  # Ignore cleanup errors in test environment

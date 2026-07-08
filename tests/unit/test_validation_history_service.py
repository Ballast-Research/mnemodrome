"""Unit tests for validation_history_service.py (Feature 006).

Covers:
- create_validation_history (async, Neo4j write)
- find_similar_conflicts (sync, pattern matching)
- calculate_confidence_score (sync, statistics)
- is_simple_conflict (sync, auto-resolution logic)
- get_validation_history_by_type (async, Neo4j query)
- determine_conflict_type (sync, classification)
- _calculate_complexity_for_matching (sync, scoring)
- ValidationHistoryService class (instance wrappers)
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from src.services.validation_history_service import (
    ValidationHistoryService,
    _calculate_complexity_for_matching,
    calculate_confidence_score,
    create_validation_history,
    determine_conflict_type,
    find_similar_conflicts,
    get_validation_history_by_type,
    is_simple_conflict,
    validation_history_service,
)

# ========================================
# Helpers
# ========================================


def _make_validation(
    decision: str,
    conflict_type: str = "infrastructure_relocation",
    complexity_score: int = 20,
    timestamp: str = "2025-06-01T00:00:00+00:00",
):
    """Build a validation dict for tests."""
    return {
        "uuid": "val-uuid",
        "conflict_type": conflict_type,
        "complexity_score": complexity_score,
        "human_decision": decision,
        "timestamp": timestamp,
    }


def _make_validations(count: int, decision: str = "approve", **kwargs):
    """Build a list of N identical validation dicts."""
    return [_make_validation(decision, **kwargs) for _ in range(count)]


@dataclass
class FakeValidation:
    """Dataclass stand-in for testing dual dict/dataclass handling."""

    conflict_type: str = "infrastructure_relocation"
    complexity_score: int = 20
    human_decision: str = "approve"
    timestamp: str = "2025-06-01T00:00:00+00:00"


@dataclass
class FakeFact:
    """Dataclass stand-in for fact objects."""

    relation: str = "RUNS_ON"
    entity2: str = "server-b"
    created_at: datetime | None = None


# ========================================
# determine_conflict_type
# ========================================


class TestDetermineConflictType:
    def test_runs_on_returns_infrastructure(self):
        old = {"relation": "RUNS_ON", "entity2": "server-a"}
        new = {"relation": "RUNS_ON", "entity2": "server-b"}
        assert determine_conflict_type(old, new) == "infrastructure_relocation"

    def test_hosted_on_returns_infrastructure(self):
        old = {"relation": "HOSTED_ON", "entity2": "host-a"}
        new = {"relation": "HOSTED_ON", "entity2": "host-b"}
        assert determine_conflict_type(old, new) == "infrastructure_relocation"

    def test_owned_by_returns_ownership(self):
        old = {"relation": "OWNED_BY", "entity2": "alice"}
        new = {"relation": "OWNED_BY", "entity2": "bob"}
        assert determine_conflict_type(old, new) == "ownership_transfer"

    def test_different_entity2_returns_same_relation_different_target(self):
        old = {"relation": "USES", "entity2": "tool-a"}
        new = {"relation": "USES", "entity2": "tool-b"}
        assert determine_conflict_type(old, new) == "same_relation_different_target"

    def test_same_entity2_returns_temporal_update(self):
        old = {"relation": "USES", "entity2": "tool-a"}
        new = {"relation": "USES", "entity2": "tool-a"}
        assert determine_conflict_type(old, new) == "temporal_update"

    def test_dataclass_inputs(self):
        old = FakeFact(relation="RUNS_ON", entity2="server-a")
        new = FakeFact(relation="RUNS_ON", entity2="server-b")
        assert determine_conflict_type(old, new) == "infrastructure_relocation"

    def test_missing_relation_key(self):
        old = {"entity2": "a"}
        new = {"entity2": "b"}
        assert determine_conflict_type(old, new) == "same_relation_different_target"

    def test_missing_entity2_both_none(self):
        old = {"relation": "USES"}
        new = {"relation": "USES"}
        # Both entity2 default to None, so old_entity2 == new_entity2
        assert determine_conflict_type(old, new) == "temporal_update"


# ========================================
# _calculate_complexity_for_matching
# ========================================


class TestCalculateComplexityForMatching:
    """Note: For dict inputs, created_at is stored in created_at_str but
    the code sets created_at=None, so age_days always falls to 0 (recent baseline=15).
    Only dataclass inputs use the actual created_at value for age calculation."""

    def test_dict_created_at_always_gives_baseline(self):
        """Dict path: created_at is set to None internally, so age_days=0 -> score=15."""
        old = {
            "created_at": (datetime.now(UTC) - timedelta(days=60)).isoformat(),
            "relation": "USES",
        }
        new = {"relation": "USES"}
        score = _calculate_complexity_for_matching(old, new)
        assert score == 15  # baseline (age_days=0 due to dict code path)

    def test_dict_infrastructure_relation_reduces_score(self):
        old = {
            "created_at": (datetime.now(UTC) - timedelta(days=15)).isoformat(),
            "relation": "RUNS_ON",
        }
        new = {"relation": "RUNS_ON"}
        score = _calculate_complexity_for_matching(old, new)
        # 15 (baseline) - 10 (infra) = 5
        assert score == 5

    def test_dict_hosted_on_also_reduces(self):
        old = {
            "created_at": (datetime.now(UTC) - timedelta(days=60)).isoformat(),
            "relation": "HOSTED_ON",
        }
        new = {"relation": "HOSTED_ON"}
        # 15 (baseline) - 10 (infra) = 5
        assert _calculate_complexity_for_matching(old, new) == 5

    def test_score_clamped_to_zero(self):
        old = {
            "created_at": datetime.now(UTC).isoformat(),
            "relation": "RUNS_ON",
        }
        new = {"relation": "RUNS_ON"}
        score = _calculate_complexity_for_matching(old, new)
        assert score >= 0

    def test_missing_created_at_dict(self):
        """No created_at key -> created_at_str='', created_at=None -> age_days=0 -> score=15."""
        old = {"relation": "USES"}
        new = {"relation": "USES"}
        score = _calculate_complexity_for_matching(old, new)
        assert score == 15

    def test_missing_created_at_infra(self):
        """No created_at + infra relation: 15 - 10 = 5."""
        old = {"relation": "RUNS_ON"}
        new = {"relation": "RUNS_ON"}
        score = _calculate_complexity_for_matching(old, new)
        assert score == 5

    def test_dataclass_with_datetime_over_30_days(self):
        old = FakeFact(
            relation="USES",
            entity2="x",
            created_at=datetime.now(UTC) - timedelta(days=45),
        )
        new = FakeFact(relation="USES", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        assert score == 30  # >30 days, non-infra

    def test_dataclass_with_datetime_7_to_30_days(self):
        old = FakeFact(
            relation="USES",
            entity2="x",
            created_at=datetime.now(UTC) - timedelta(days=15),
        )
        new = FakeFact(relation="USES", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        assert score == 15  # 7-30 days

    def test_dataclass_recent_under_7_days(self):
        old = FakeFact(
            relation="USES",
            entity2="x",
            created_at=datetime.now(UTC) - timedelta(days=2),
        )
        new = FakeFact(relation="USES", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        assert score == 15  # baseline for recent

    def test_dataclass_naive_datetime(self):
        """Naive datetime should get tzinfo replaced and still work."""
        old = FakeFact(
            relation="USES", entity2="x", created_at=datetime.now() - timedelta(days=10)
        )
        new = FakeFact(relation="USES", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        assert score == 15  # 7-30 days

    def test_dataclass_infra_over_30_days(self):
        old = FakeFact(
            relation="RUNS_ON",
            entity2="x",
            created_at=datetime.now(UTC) - timedelta(days=45),
        )
        new = FakeFact(relation="RUNS_ON", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        # 30 (age) - 10 (infra) = 20
        assert score == 20

    def test_dataclass_none_created_at(self):
        """Dataclass with created_at=None -> not datetime, not str -> age_days=0 -> score=15."""
        old = FakeFact(relation="USES", entity2="x", created_at=None)
        new = FakeFact(relation="USES", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        assert score == 15

    def test_dataclass_string_created_at(self):
        """Dataclass with string created_at uses the string parsing branch."""
        ts = (datetime.now(UTC) - timedelta(days=45)).isoformat()
        old = FakeFact(relation="USES", entity2="x", created_at=ts)
        new = FakeFact(relation="USES", entity2="y")
        score = _calculate_complexity_for_matching(old, new)
        assert score == 30  # >30 days


# ========================================
# calculate_confidence_score
# ========================================


class TestCalculateConfidenceScore:
    def test_fewer_than_10_returns_zero(self):
        assert calculate_confidence_score(_make_validations(9)) == 0

    def test_empty_list_returns_zero(self):
        assert calculate_confidence_score([]) == 0

    def test_exactly_10_all_approve(self):
        assert calculate_confidence_score(_make_validations(10, "approve")) == 100

    def test_exactly_10_all_reject(self):
        assert calculate_confidence_score(_make_validations(10, "reject")) == 0

    def test_mixed_decisions(self):
        vals = _make_validations(8, "approve") + _make_validations(2, "reject")
        # 8/10 = 80%
        assert calculate_confidence_score(vals) == 80

    def test_50_50_split(self):
        vals = _make_validations(5, "approve") + _make_validations(5, "reject")
        assert calculate_confidence_score(vals) == 50

    def test_large_list(self):
        vals = _make_validations(90, "approve") + _make_validations(10, "reject")
        assert calculate_confidence_score(vals) == 90

    def test_modify_is_not_approve(self):
        vals = _make_validations(10, "modify")
        assert calculate_confidence_score(vals) == 0

    def test_dataclass_inputs(self):
        vals = [FakeValidation(human_decision="approve") for _ in range(10)]
        assert calculate_confidence_score(vals) == 100


# ========================================
# find_similar_conflicts
# ========================================


class TestFindSimilarConflicts:
    def test_matches_same_type_and_close_complexity(self):
        # Dict RUNS_ON: complexity = 15 (baseline) - 10 (infra) = 5
        old_fact = {"relation": "RUNS_ON", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "RUNS_ON", "entity2": "b"}
        history = [
            _make_validation("approve", "infrastructure_relocation", complexity_score=5)
        ]
        result = find_similar_conflicts(history, old_fact, new_fact)
        assert len(result) == 1

    def test_filters_different_type(self):
        old_fact = {"relation": "RUNS_ON", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "RUNS_ON", "entity2": "b"}
        history = [
            _make_validation("approve", "ownership_transfer", complexity_score=5)
        ]
        result = find_similar_conflicts(history, old_fact, new_fact)
        assert len(result) == 0

    def test_filters_distant_complexity(self):
        # Dict RUNS_ON: complexity = 5. History at 50 is 45 away (>10).
        old_fact = {"relation": "RUNS_ON", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "RUNS_ON", "entity2": "b"}
        history = [
            _make_validation(
                "approve", "infrastructure_relocation", complexity_score=50
            )
        ]
        result = find_similar_conflicts(history, old_fact, new_fact)
        assert len(result) == 0

    def test_empty_history(self):
        old_fact = {"relation": "RUNS_ON", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "RUNS_ON", "entity2": "b"}
        assert find_similar_conflicts([], old_fact, new_fact) == []

    def test_dataclass_validation_entries(self):
        old_fact = {"relation": "RUNS_ON", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "RUNS_ON", "entity2": "b"}
        history = [
            FakeValidation(
                conflict_type="infrastructure_relocation", complexity_score=5
            )
        ]
        result = find_similar_conflicts(history, old_fact, new_fact)
        assert len(result) == 1

    def test_boundary_complexity_plus_10(self):
        """Exactly +10 should be included."""
        # Dict USES with no valid created_at: complexity = 15 (baseline, non-infra)
        old_fact = {"relation": "USES", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "USES", "entity2": "b"}
        # 15 + 10 = 25, exactly +10 away
        history = [
            _make_validation(
                "approve", "same_relation_different_target", complexity_score=25
            )
        ]
        result = find_similar_conflicts(history, old_fact, new_fact)
        assert len(result) == 1

    def test_boundary_complexity_plus_11(self):
        """Exactly +11 should be excluded."""
        # Dict USES: complexity = 15. 15 + 11 = 26, which is >10 away.
        old_fact = {"relation": "USES", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "USES", "entity2": "b"}
        history = [
            _make_validation(
                "approve", "same_relation_different_target", complexity_score=26
            )
        ]
        result = find_similar_conflicts(history, old_fact, new_fact)
        assert len(result) == 0


# ========================================
# is_simple_conflict
# ========================================


class TestIsSimpleConflict:
    def test_returns_false_insufficient_data(self):
        assert is_simple_conflict(_make_validations(5)) is False

    def test_returns_true_high_confidence_no_rejections(self):
        vals = _make_validations(10, "approve")
        assert is_simple_conflict(vals) is True

    def test_returns_false_below_threshold(self):
        # 7 approve + 3 reject = 70%, below default 80
        vals = _make_validations(7, "approve") + _make_validations(3, "reject")
        assert is_simple_conflict(vals) is False

    def test_custom_threshold_with_recent_rejections(self):
        # 7/10 = 70%, threshold 70, but recent rejections block
        vals = [
            _make_validation(
                "approve", timestamp=f"2025-01-{i + 10:02d}T00:00:00+00:00"
            )
            for i in range(7)
        ]
        # Rejections are the most recent
        vals += [
            _make_validation("reject", timestamp=f"2025-06-{i + 10:02d}T00:00:00+00:00")
            for i in range(3)
        ]
        assert is_simple_conflict(vals, confidence_threshold=70) is False

    def test_custom_threshold_passes_no_recent_rejections(self):
        # 7/10 = 70%, threshold 70, rejections are old (not in recent 5)
        # Old rejections
        vals = [
            _make_validation("reject", timestamp=f"2025-01-{i + 10:02d}T00:00:00+00:00")
            for i in range(3)
        ]
        # Recent approvals
        vals += [
            _make_validation(
                "approve", timestamp=f"2025-06-{i + 10:02d}T00:00:00+00:00"
            )
            for i in range(7)
        ]
        assert is_simple_conflict(vals, confidence_threshold=70) is True

    def test_recent_rejection_blocks(self):
        """Even with 90% confidence, a recent rejection returns False."""
        # 9 approve + 1 reject, but the reject has latest timestamp
        vals = _make_validations(9, "approve", timestamp="2025-01-01T00:00:00+00:00")
        vals.append(_make_validation("reject", timestamp="2025-12-01T00:00:00+00:00"))
        assert is_simple_conflict(vals) is False

    def test_no_recent_rejection_approves(self):
        """All 10 approve, old rejection is not in recent 5."""
        # 11 approve (latest timestamps) + 1 old reject
        vals = [
            _make_validation(
                "approve", timestamp=f"2025-06-{i + 10:02d}T00:00:00+00:00"
            )
            for i in range(11)
        ]
        vals.append(_make_validation("reject", timestamp="2025-01-01T00:00:00+00:00"))
        # 11/12 = 91%, recent 5 are all approves
        assert is_simple_conflict(vals) is True

    def test_empty_list(self):
        assert is_simple_conflict([]) is False

    def test_dataclass_inputs_with_rejection(self):
        vals = [
            FakeValidation(
                human_decision="approve",
                timestamp=f"2025-06-{i + 10:02d}T00:00:00+00:00",
            )
            for i in range(10)
        ]
        vals.append(
            FakeValidation(
                human_decision="reject", timestamp="2025-12-01T00:00:00+00:00"
            )
        )
        # Recent reject should block
        assert is_simple_conflict(vals) is False


# ========================================
# create_validation_history (async)
# ========================================


class TestCreateValidationHistory:
    @pytest.mark.asyncio
    async def test_creates_record_returns_uuid(
        self, mock_neo4j_service, mock_neo4j_driver
    ):
        # Make session.run return a record with uuid
        record = {"uuid": "new-val-uuid"}
        session = mock_neo4j_driver.session.return_value
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value=record)
        session.run = AsyncMock(return_value=result_mock)

        # Set database attribute
        mock_neo4j_service.database = "neo4j"

        uuid = await create_validation_history(
            neo4j_service=mock_neo4j_service,
            conflict_type="infrastructure_relocation",
            complexity_score=25,
            human_decision="approve",
            old_fact_uuid="old-uuid",
            new_fact_uuid="new-uuid",
            rationale="Looks correct",
        )

        assert uuid == "new-val-uuid"
        mock_neo4j_service.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_raises_on_empty_result(self, mock_neo4j_service, mock_neo4j_driver):
        session = mock_neo4j_driver.session.return_value
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value=None)
        session.run = AsyncMock(return_value=result_mock)
        mock_neo4j_service.database = "neo4j"

        with pytest.raises(Exception, match="Failed to create ValidationHistory"):
            await create_validation_history(
                neo4j_service=mock_neo4j_service,
                conflict_type="infrastructure_relocation",
                complexity_score=25,
                human_decision="approve",
                old_fact_uuid="old-uuid",
            )

    @pytest.mark.asyncio
    async def test_none_rationale_stored_as_empty_string(
        self, mock_neo4j_service, mock_neo4j_driver
    ):
        record = {"uuid": "val-uuid"}
        session = mock_neo4j_driver.session.return_value
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value=record)
        session.run = AsyncMock(return_value=result_mock)
        mock_neo4j_service.database = "neo4j"

        await create_validation_history(
            neo4j_service=mock_neo4j_service,
            conflict_type="ownership_transfer",
            complexity_score=50,
            human_decision="reject",
            old_fact_uuid="old-uuid",
            rationale=None,
        )

        # Verify the params passed to session.run
        call_args = session.run.call_args
        params = (
            call_args[0][1]
            if len(call_args[0]) > 1
            else call_args[1].get("params", call_args.kwargs)
        )
        # The rationale should be "" when None is passed
        assert params["rationale"] == ""

    @pytest.mark.asyncio
    async def test_none_new_fact_uuid_allowed(
        self, mock_neo4j_service, mock_neo4j_driver
    ):
        record = {"uuid": "val-uuid"}
        session = mock_neo4j_driver.session.return_value
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value=record)
        session.run = AsyncMock(return_value=result_mock)
        mock_neo4j_service.database = "neo4j"

        uuid = await create_validation_history(
            neo4j_service=mock_neo4j_service,
            conflict_type="infrastructure_relocation",
            complexity_score=10,
            human_decision="reject",
            old_fact_uuid="old-uuid",
            new_fact_uuid=None,
        )

        assert uuid == "val-uuid"
        call_args = session.run.call_args
        params = call_args[0][1]
        assert params["new_fact_uuid"] is None


# ========================================
# get_validation_history_by_type (async)
# ========================================


class TestGetValidationHistoryByType:
    @pytest.mark.asyncio
    async def test_returns_results(self, mock_neo4j_service):
        expected = [
            {
                "uuid": "v1",
                "conflict_type": "infrastructure_relocation",
                "human_decision": "approve",
            },
            {
                "uuid": "v2",
                "conflict_type": "infrastructure_relocation",
                "human_decision": "reject",
            },
        ]
        mock_neo4j_service.query = AsyncMock(return_value=expected)

        results = await get_validation_history_by_type(
            neo4j_service=mock_neo4j_service,
            conflict_type="infrastructure_relocation",
        )

        assert len(results) == 2
        assert results[0]["uuid"] == "v1"
        mock_neo4j_service.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_results(self, mock_neo4j_service):
        mock_neo4j_service.query = AsyncMock(return_value=[])

        results = await get_validation_history_by_type(
            neo4j_service=mock_neo4j_service,
            conflict_type="nonexistent_type",
        )

        assert results == []


# ========================================
# ValidationHistoryService (class wrappers)
# ========================================


class TestValidationHistoryService:
    def test_global_instance_exists(self):
        assert isinstance(validation_history_service, ValidationHistoryService)

    @pytest.mark.asyncio
    async def test_create_validation_record(
        self, mock_neo4j_service, mock_neo4j_driver
    ):
        record = {"uuid": "svc-uuid"}
        session = mock_neo4j_driver.session.return_value
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value=record)
        session.run = AsyncMock(return_value=result_mock)
        mock_neo4j_service.database = "neo4j"

        svc = ValidationHistoryService()
        result = await svc.create_validation_record(
            neo4j_service=mock_neo4j_service,
            conflict_type="infrastructure_relocation",
            complexity_score=20,
            human_decision="approve",
            old_fact_uuid="old-uuid",
        )

        assert result["uuid"] == "svc-uuid"
        assert result["status"] == "created"

    @pytest.mark.asyncio
    async def test_get_by_conflict_type_with_limit(self, mock_neo4j_service):
        mock_neo4j_service.query = AsyncMock(
            return_value=[{"uuid": f"v{i}"} for i in range(10)]
        )

        svc = ValidationHistoryService()
        results = await svc.get_by_conflict_type(
            neo4j_service=mock_neo4j_service,
            conflict_type="infrastructure_relocation",
            limit=3,
        )

        assert len(results) == 3

    def test_find_similar_delegates(self):
        svc = ValidationHistoryService()
        old_fact = {"relation": "RUNS_ON", "entity2": "a", "created_at": ""}
        new_fact = {"relation": "RUNS_ON", "entity2": "b"}
        history = [
            _make_validation("approve", "infrastructure_relocation", complexity_score=5)
        ]

        result = svc.find_similar(history, old_fact, new_fact)
        assert len(result) == 1

    def test_calculate_confidence_delegates(self):
        svc = ValidationHistoryService()
        vals = _make_validations(10, "approve")
        assert svc.calculate_confidence(vals) == 100

    def test_is_simple_delegates(self):
        svc = ValidationHistoryService()
        vals = _make_validations(10, "approve")
        assert svc.is_simple(vals) is True

    def test_is_simple_with_threshold(self):
        svc = ValidationHistoryService()
        vals = _make_validations(10, "approve")
        assert svc.is_simple(vals, confidence_threshold=100) is True

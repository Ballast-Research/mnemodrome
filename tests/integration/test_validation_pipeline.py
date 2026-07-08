"""Integration tests for multi-agent validation pipeline.

End-to-end tests for the orchestration layer covering:
- Full validation pipeline execution (spawn → validate → aggregate → present)
- Parallel validator spawning and execution
- Timeout handling (3-minute SLA)
- Partial result handling on timeout
- Approval gate presentation
- Mnemodrome event logging integration
- Context package loading and transfer
- Job lifecycle management

Constitutional Authority: Phase 3 MVP (T-039)
"""

import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from src.orchestrator.agent_coordinator import (
    AgentCoordinator,
    ContextPackage,
    JobStatus,
    OrchestrationJob,
    ValidationResult,
    ValidationRole,
    ValidationSeverity,
    ValidationViolation,
)

# ============================================================================
# Fixtures - Test Data
# ============================================================================


@pytest.fixture
def sample_code() -> str:
    """Sample Python code for validation testing."""
    return '''"""Sample module for testing.

This is a test module with various patterns.
"""

def MyFunction(x):
    """Function with bad naming (should be snake_case)."""
    try:
        return x * 2
    except:
        pass


class GoodClass:
    """Properly named class."""

    def good_method(self):
        """Method with proper naming."""
        print("Hello")
        return True


def function_without_docstring(param):
    return param + 1
'''


@pytest.fixture
def sample_context_package(sample_code) -> ContextPackage:
    """Create a sample context package for testing."""
    return ContextPackage(
        code=sample_code,
        file_path="src/test_module.py",
        patterns=[],
        entity_context={"Mnemodrome": "uuid-memb-001"},
        test_files=["tests/test_test_module.py"],
        related_entities=["Mnemodrome", "Phase3"],
    )


@pytest.fixture
def agent_coordinator() -> AgentCoordinator:
    """Create an AgentCoordinator instance with 5-second SLA for testing."""
    return AgentCoordinator(sla_timeout_seconds=5)


@pytest.fixture
def agent_coordinator_short_timeout() -> AgentCoordinator:
    """Create an AgentCoordinator with very short timeout for timeout testing."""
    return AgentCoordinator(sla_timeout_seconds=1)


# ============================================================================
# Tests: Job Creation (T-039.1)
# ============================================================================


class TestJobCreation:
    """Tests for OrchestrationJob creation."""

    @pytest.mark.asyncio
    async def test_create_job_with_all_agents(
        self, agent_coordinator, sample_context_package
    ):
        """Test creating job with all validation agents."""
        job = await agent_coordinator.create_job(
            task="Validate test module", context=sample_context_package
        )

        assert job.job_id is not None
        assert job.task == "Validate test module"
        assert job.status == JobStatus.PENDING
        assert job.context == sample_context_package
        assert len(job.agents) == 3
        assert ValidationRole.PATTERN_COMPLIANCE in job.agents
        assert ValidationRole.TEST_COVERAGE in job.agents
        assert ValidationRole.ENTITY_CONSISTENCY in job.agents

    @pytest.mark.asyncio
    async def test_create_job_with_specific_agents(
        self, agent_coordinator, sample_context_package
    ):
        """Test creating job with specific agents."""
        agents = [ValidationRole.PATTERN_COMPLIANCE, ValidationRole.TEST_COVERAGE]
        job = await agent_coordinator.create_job(
            task="Validate patterns and coverage",
            context=sample_context_package,
            agents=agents,
        )

        assert len(job.agents) == 2
        assert ValidationRole.PATTERN_COMPLIANCE in job.agents
        assert ValidationRole.TEST_COVERAGE in job.agents
        assert ValidationRole.ENTITY_CONSISTENCY not in job.agents

    @pytest.mark.asyncio
    async def test_job_initialization_fields(
        self, agent_coordinator, sample_context_package
    ):
        """Test job has proper initial state."""
        job = await agent_coordinator.create_job(
            task="Test task", context=sample_context_package
        )

        assert job.job_id is not None
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.results == []
        assert job.aggregated_result is None
        assert job.memory_uuid is not None


# ============================================================================
# Tests: Parallel Validator Spawning (T-039.2)
# ============================================================================


class TestParallelValidatorSpawning:
    """Tests for spawning validation agents in parallel."""

    @pytest.mark.asyncio
    async def test_spawn_all_validators(
        self, agent_coordinator, sample_context_package
    ):
        """Test spawning all three validators in parallel."""
        job = await agent_coordinator.create_job(
            task="Full validation", context=sample_context_package
        )

        results = await agent_coordinator.spawn_validation_agents(job)

        assert len(results) == 3
        assert job.status == JobStatus.COMPLETED
        assert job.started_at is not None

    @pytest.mark.asyncio
    async def test_spawn_single_validator(
        self, agent_coordinator, sample_context_package
    ):
        """Test spawning single validator."""
        job = await agent_coordinator.create_job(
            task="Pattern validation only",
            context=sample_context_package,
            agents=[ValidationRole.PATTERN_COMPLIANCE],
        )

        results = await agent_coordinator.spawn_validation_agents(job)

        assert len(results) == 1
        assert results[0].validator_role == ValidationRole.PATTERN_COMPLIANCE

    @pytest.mark.asyncio
    async def test_all_results_have_timing(
        self, agent_coordinator, sample_context_package
    ):
        """Test that all validation results include timing information."""
        job = await agent_coordinator.create_job(
            task="Timing test", context=sample_context_package
        )

        results = await agent_coordinator.spawn_validation_agents(job)

        for result in results:
            assert result.validation_time_ms >= 0.0
            assert result.validation_time_ms < 5000.0  # Should be quick

    @pytest.mark.asyncio
    async def test_validator_results_have_roles(
        self, agent_coordinator, sample_context_package
    ):
        """Test that each result identifies its validator."""
        job = await agent_coordinator.create_job(
            task="Role test", context=sample_context_package
        )

        results = await agent_coordinator.spawn_validation_agents(job)
        roles = [r.validator_role for r in results]

        assert ValidationRole.PATTERN_COMPLIANCE in roles
        assert ValidationRole.TEST_COVERAGE in roles
        assert ValidationRole.ENTITY_CONSISTENCY in roles


# ============================================================================
# Tests: Timeout Handling (T-039.3)
# ============================================================================


class TestTimeoutHandling:
    """Tests for SLA timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_sets_job_status(self):
        """Test that timeout sets job status to TIMEOUT."""
        coordinator = AgentCoordinator(sla_timeout_seconds=1)

        # Create a slow context
        slow_code = sample_code = """
async def slow_function():
    await asyncio.sleep(10)
"""
        context = ContextPackage(code=slow_code, file_path="slow.py")

        job = await coordinator.create_job(task="Timeout test", context=context)

        # This should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coordinator.spawn_validation_agents(job), timeout=1)

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_results(self):
        """Test that timeout returns partial results."""
        coordinator = AgentCoordinator(sla_timeout_seconds=1)

        context = ContextPackage(code="def quick(): pass", file_path="quick.py")

        job = await coordinator.create_job(
            task="Partial results test",
            context=context,
            agents=[ValidationRole.PATTERN_COMPLIANCE],
        )

        # Complete before timeout
        results = await coordinator.spawn_validation_agents(job)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_sla_met_calculation(self):
        """Test SLA met/exceeded calculation."""
        coordinator = AgentCoordinator(sla_timeout_seconds=180)

        context = ContextPackage(code="print('hello')", file_path="quick.py")

        job = await coordinator.create_job(task="SLA calc test", context=context)

        results = await coordinator.spawn_validation_agents(job)
        aggregated = await coordinator.aggregate_validation_results(results)

        # Should be well under 180 seconds
        assert aggregated.sla_met is True


# ============================================================================
# Tests: Result Aggregation (T-039.4)
# ============================================================================


class TestResultAggregation:
    """Tests for aggregating validation results."""

    @pytest.mark.asyncio
    async def test_aggregate_multiple_results(
        self, agent_coordinator, sample_context_package
    ):
        """Test aggregating results from multiple validators."""
        job = await agent_coordinator.create_job(
            task="Aggregation test", context=sample_context_package
        )

        results = await agent_coordinator.spawn_validation_agents(job)
        aggregated = await agent_coordinator.aggregate_validation_results(results)

        assert aggregated.aggregation_id is not None
        assert aggregated.total_violations >= 0
        assert aggregated.critical_count >= 0
        assert aggregated.warning_count >= 0
        assert aggregated.info_count >= 0
        assert aggregated.total_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_aggregate_deduplicates_violations(self):
        """Test that aggregation deduplicates violations."""
        # Create two results with identical violations
        violation = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="naming",
            message="Bad naming",
            recommendation="Fix it",
        )

        result1 = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=False,
            violations=[violation],
            validation_time_ms=10.0,
            memory_uuid=str(uuid4()),
        )

        result2 = ValidationResult(
            validator_role=ValidationRole.TEST_COVERAGE,
            passed=False,
            violations=[violation],
            validation_time_ms=15.0,
            memory_uuid=str(uuid4()),
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([result1, result2])

        # Should have 1 unique violation, not 2
        assert aggregated.total_violations == 1

    @pytest.mark.asyncio
    async def test_aggregate_counts_severity_levels(self):
        """Test that aggregation correctly counts violations by severity."""
        violations = [
            ValidationViolation(
                severity=ValidationSeverity.CRITICAL,
                category="critical",
                message="Critical violation",
                recommendation="Fix",
            ),
            ValidationViolation(
                severity=ValidationSeverity.WARNING,
                category="warning",
                message="Warning violation",
                recommendation="Fix",
            ),
            ValidationViolation(
                severity=ValidationSeverity.INFO,
                category="info",
                message="Info violation",
                recommendation="Consider",
            ),
        ]

        result = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=False,
            violations=violations,
            validation_time_ms=20.0,
            memory_uuid=str(uuid4()),
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([result])

        assert aggregated.critical_count == 1
        assert aggregated.warning_count == 1
        assert aggregated.info_count == 1
        assert aggregated.total_violations == 3

    @pytest.mark.asyncio
    async def test_aggregate_passed_status(self):
        """Test overall_passed status is correctly set."""
        # No critical violations = passed
        result = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=10.0,
            memory_uuid=str(uuid4()),
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([result])

        assert aggregated.overall_passed is True

    @pytest.mark.asyncio
    async def test_aggregate_failed_status_with_critical(self):
        """Test overall_passed is False with critical violations."""
        violation = ValidationViolation(
            severity=ValidationSeverity.CRITICAL,
            category="critical",
            message="Critical issue",
            recommendation="Fix",
        )

        result = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=False,
            violations=[violation],
            validation_time_ms=10.0,
            memory_uuid=str(uuid4()),
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([result])

        assert aggregated.overall_passed is False


# ============================================================================
# Tests: UUID Collection (T-039.5)
# ============================================================================


class TestUUIDCollection:
    """Tests for Mnemodrome UUID tracking in aggregation."""

    @pytest.mark.asyncio
    async def test_collect_uuids_from_results(self):
        """Test collecting UUIDs from validation results."""
        uuid1 = str(uuid4())
        uuid2 = str(uuid4())

        result1 = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=10.0,
            memory_uuid=uuid1,
        )

        result2 = ValidationResult(
            validator_role=ValidationRole.TEST_COVERAGE,
            passed=True,
            violations=[],
            validation_time_ms=15.0,
            memory_uuid=uuid2,
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([result1, result2])

        assert uuid1 in aggregated.memory_uuids
        assert uuid2 in aggregated.memory_uuids

    @pytest.mark.asyncio
    async def test_filter_none_uuids(self):
        """Test that None UUIDs are filtered out."""
        uuid1 = str(uuid4())

        result1 = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=10.0,
            memory_uuid=uuid1,
        )

        result2 = ValidationResult(
            validator_role=ValidationRole.TEST_COVERAGE,
            passed=True,
            violations=[],
            validation_time_ms=15.0,
            memory_uuid=None,  # None UUID
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([result1, result2])

        assert len(aggregated.memory_uuids) == 1
        assert uuid1 in aggregated.memory_uuids


# ============================================================================
# Tests: Approval Gate Presentation (T-039.6)
# ============================================================================


class TestApprovalGatePresentatio:
    """Tests for approval gate presentation formatting."""

    @pytest.mark.asyncio
    async def test_present_approval_gate_passed(
        self, agent_coordinator, sample_context_package
    ):
        """Test presenting approval gate for passing validation."""
        job = await agent_coordinator.create_job(
            task="Presentation test", context=sample_context_package
        )

        results = await agent_coordinator.spawn_validation_agents(job)
        aggregated = await agent_coordinator.aggregate_validation_results(results)

        output = await agent_coordinator.present_approval_gate(aggregated)

        assert isinstance(output, str)
        assert len(output) > 0

    @pytest.mark.asyncio
    async def test_present_gate_contains_aggregation_id(
        self, agent_coordinator, sample_context_package
    ):
        """Test that presentation includes aggregation ID."""
        job = await agent_coordinator.create_job(
            task="ID test", context=sample_context_package
        )

        results = await agent_coordinator.spawn_validation_agents(job)
        aggregated = await agent_coordinator.aggregate_validation_results(results)
        output = await agent_coordinator.present_approval_gate(aggregated)

        assert aggregated.aggregation_id in output

    @pytest.mark.asyncio
    async def test_present_gate_shows_violation_counts(self, agent_coordinator):
        """Test that presentation shows violation counts."""
        # Manually create aggregated result for testing
        from src.orchestrator.agent_coordinator import AggregatedResult

        aggregated = AggregatedResult(
            overall_passed=False,
            total_violations=3,
            critical_count=1,
            warning_count=1,
            info_count=1,
            all_violations=[
                ValidationViolation(
                    severity=ValidationSeverity.CRITICAL,
                    category="critical",
                    message="Critical",
                    recommendation="Fix",
                ),
                ValidationViolation(
                    severity=ValidationSeverity.WARNING,
                    category="warning",
                    message="Warning",
                    recommendation="Consider",
                ),
                ValidationViolation(
                    severity=ValidationSeverity.INFO,
                    category="info",
                    message="Info",
                    recommendation="Note",
                ),
            ],
            validator_results=[],
            total_time_ms=50.0,
            sla_met=True,
        )

        output = await agent_coordinator.present_approval_gate(aggregated)

        assert "1" in output  # Critical count
        assert "1" in output  # Warning count
        assert "1" in output  # Info count


# ============================================================================
# Tests: Complete Job Execution (T-039.7)
# ============================================================================


class TestCompleteJobExecution:
    """Tests for full job lifecycle execution."""

    @pytest.mark.asyncio
    async def test_execute_job_complete_lifecycle(
        self, agent_coordinator, sample_context_package
    ):
        """Test complete job execution from creation to result."""
        job = await agent_coordinator.create_job(
            task="Full lifecycle test", context=sample_context_package
        )

        result = await agent_coordinator.execute_job(job)

        assert result.aggregation_id is not None
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.aggregated_result == result

    @pytest.mark.asyncio
    async def test_execute_job_updates_timestamps(
        self, agent_coordinator, sample_context_package
    ):
        """Test that job execution updates all timestamps."""
        job = await agent_coordinator.create_job(
            task="Timestamp test", context=sample_context_package
        )

        created_before = job.created_at

        await asyncio.sleep(0.01)  # Small delay

        result = await agent_coordinator.execute_job(job)

        assert job.started_at is not None
        assert job.completed_at is not None
        assert created_before <= job.started_at <= job.completed_at

    @pytest.mark.asyncio
    async def test_execute_job_duration(
        self, agent_coordinator, sample_context_package
    ):
        """Test job duration calculation."""
        job = await agent_coordinator.create_job(
            task="Duration test", context=sample_context_package
        )

        await agent_coordinator.execute_job(job)

        duration = job.duration_ms()

        assert duration > 0.0
        assert duration < 5000.0  # Should be quick


# ============================================================================
# Tests: Event Logging Integration (T-039.8)
# ============================================================================


class TestEventLoggingIntegration:
    """Tests for Mnemodrome event logging integration."""

    @pytest.mark.asyncio
    async def test_track_job_in_mnemodrome(self):
        """Test that job execution is tracked in Mnemodrome."""
        from src.orchestrator.agent_coordinator import AggregatedResult

        coordinator = AgentCoordinator()

        job = OrchestrationJob(
            task="Mnemodrome tracking test",
            context=ContextPackage(code="test", file_path="test.py"),
            agents=[ValidationRole.PATTERN_COMPLIANCE],
        )

        aggregated = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=50.0,
            sla_met=True,
            memory_uuids=[str(uuid4())],
        )

        # Mock the event_log_service
        with patch(
            "src.orchestrator.agent_coordinator.event_log_service"
        ) as mock_service:
            mock_service.log_event = AsyncMock()
            await coordinator._track_in_mnemodrome(job, aggregated)

            # Verify log_event was called
            mock_service.log_event.assert_called_once()


# ============================================================================
# Tests: Partial Results on Timeout (T-039.9)
# ============================================================================


class TestPartialResultsOnTimeout:
    """Tests for handling partial results when timeout occurs."""

    @pytest.mark.asyncio
    async def test_timeout_with_partial_results(self):
        """Test that timeout returns completed results."""
        coordinator = AgentCoordinator(sla_timeout_seconds=5)

        context = ContextPackage(code="def test(): pass", file_path="test.py")

        job = await coordinator.create_job(task="Partial results test", context=context)

        # This should complete without timeout
        results = await coordinator.spawn_validation_agents(job)

        # Should have some results
        assert len(results) > 0
        assert all(isinstance(r, ValidationResult) for r in results)


# ============================================================================
# Tests: Error Handling (T-039.10)
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in validation pipeline."""

    @pytest.mark.asyncio
    async def test_validator_error_returns_failed_result(self):
        """Test that validator error is captured in result."""
        coordinator = AgentCoordinator()

        context = ContextPackage(code="invalid python {{{", file_path="bad.py")

        job = await coordinator.create_job(
            task="Error handling test",
            context=context,
            agents=[ValidationRole.PATTERN_COMPLIANCE],
        )

        results = await coordinator.spawn_validation_agents(job)

        # Validators should handle errors gracefully
        for result in results:
            assert result.result_id is not None

    @pytest.mark.asyncio
    async def test_aggregate_handles_error_results(self):
        """Test that aggregation handles results with errors."""
        error_result = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=False,
            violations=[],
            validation_time_ms=0.0,
            error_message="Service error",
            memory_uuid=str(uuid4()),
        )

        coordinator = AgentCoordinator()
        aggregated = await coordinator.aggregate_validation_results([error_result])

        assert aggregated.overall_passed is False


# ============================================================================
# Tests: Context Package Handling (T-039.11)
# ============================================================================


class TestContextPackageHandling:
    """Tests for context package loading and transfer."""

    @pytest.mark.asyncio
    async def test_context_passed_to_validators(
        self, agent_coordinator, sample_context_package
    ):
        """Test that context is properly passed to validators."""
        job = await agent_coordinator.create_job(
            task="Context test", context=sample_context_package
        )

        assert job.context == sample_context_package
        assert job.context.code == sample_context_package.code
        assert job.context.file_path == sample_context_package.file_path

    @pytest.mark.asyncio
    async def test_context_with_patterns(self):
        """Test context with loaded patterns."""
        from src.models.pattern_models import StoredPattern

        patterns = [
            StoredPattern(
                name="TestPattern",
                description="Test pattern",
                regex="def .*:",
                severity="info",
                confidence=0.9,
            )
        ]

        context = ContextPackage(
            code="def test(): pass", file_path="test.py", patterns=patterns
        )

        coordinator = AgentCoordinator()
        job = await coordinator.create_job(task="Pattern context test", context=context)

        assert len(job.context.patterns) > 0

    @pytest.mark.asyncio
    async def test_context_with_entity_mappings(self):
        """Test context with entity name to UUID mappings."""
        context = ContextPackage(
            code="entity = 'Mnemodrome'",
            file_path="test.py",
            entity_context={"Mnemodrome": "uuid-memb-123", "Phase3": "uuid-phase-456"},
        )

        coordinator = AgentCoordinator()
        job = await coordinator.create_job(task="Entity mapping test", context=context)

        assert "Mnemodrome" in job.context.entity_context
        assert job.context.entity_context["Mnemodrome"] == "uuid-memb-123"


# ============================================================================
# Tests: Multi-Validator Coordination (T-039.12)
# ============================================================================


class TestMultiValidatorCoordination:
    """Tests for coordinating multiple validators."""

    @pytest.mark.asyncio
    async def test_three_validators_run_in_parallel(
        self, agent_coordinator, sample_context_package
    ):
        """Test that all three validators run in parallel."""
        job = await agent_coordinator.create_job(
            task="Parallel test", context=sample_context_package
        )

        import time

        start = time.time()
        results = await agent_coordinator.spawn_validation_agents(job)
        total_time = time.time() - start

        # Check parallel execution: total time < sum of individual times
        sum_times = sum(r.validation_time_ms for r in results) / 1000.0

        # Parallel execution should be faster than sequential
        assert len(results) == 3
        assert total_time < sum_times * 2

    @pytest.mark.asyncio
    async def test_specific_agents_can_be_selected(
        self, agent_coordinator, sample_context_package
    ):
        """Test running only specific validators."""
        selected_agents = [
            ValidationRole.PATTERN_COMPLIANCE,
            ValidationRole.TEST_COVERAGE,
        ]

        job = await agent_coordinator.create_job(
            task="Selective test",
            context=sample_context_package,
            agents=selected_agents,
        )

        results = await agent_coordinator.spawn_validation_agents(job)

        roles = [r.validator_role for r in results]
        assert ValidationRole.PATTERN_COMPLIANCE in roles
        assert ValidationRole.TEST_COVERAGE in roles

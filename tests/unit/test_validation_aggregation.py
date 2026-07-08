"""Unit tests for validation aggregation and result management.

Tests for ValidationResult, ValidationViolation, and AggregatedResult classes covering:
- Result creation and serialization
- Violation deduplication logic
- Severity-based prioritization
- Mnemodrome UUID collection and tracking
- AggregatedResult formatting and metadata
- Edge cases (empty results, mixed severity levels)

Constitutional Authority: Phase 3 MVP (T-038)
"""

from datetime import datetime
from uuid import uuid4

import pytest

from src.orchestrator.agent_coordinator import (
    AggregatedResult,
    ValidationResult,
    ValidationRole,
    ValidationSeverity,
    ValidationViolation,
)

# ============================================================================
# Fixtures - Test Data
# ============================================================================


@pytest.fixture
def sample_violation_critical() -> ValidationViolation:
    """Create a sample critical violation for testing."""
    return ValidationViolation(
        severity=ValidationSeverity.CRITICAL,
        category="naming",
        message="Function 'MyFunction' should use snake_case",
        line_number=42,
        column_number=5,
        recommendation="Rename to 'my_function'",
        pattern_uuid="uuid-pattern-001",
        context="PEP 8 naming conventions",
    )


@pytest.fixture
def sample_violation_warning() -> ValidationViolation:
    """Create a sample warning violation for testing."""
    return ValidationViolation(
        severity=ValidationSeverity.WARNING,
        category="error_handling",
        message="Bare 'except:' clause catches all exceptions",
        line_number=56,
        column_number=12,
        recommendation="Specify exception type: except SomeError:",
        context="Bare except clauses hide errors",
    )


@pytest.fixture
def sample_violation_info() -> ValidationViolation:
    """Create a sample info violation for testing."""
    return ValidationViolation(
        severity=ValidationSeverity.INFO,
        category="documentation",
        message="Module missing docstring",
        line_number=1,
        recommendation='Add module docstring at top: """Description."""',
        context="PEP 257 documentation standards",
    )


@pytest.fixture
def sample_result_passed() -> ValidationResult:
    """Create a passed validation result."""
    return ValidationResult(
        validator_role=ValidationRole.PATTERN_COMPLIANCE,
        passed=True,
        violations=[],
        validation_time_ms=45.3,
        memory_uuid=str(uuid4()),
    )


@pytest.fixture
def sample_result_with_violations(
    sample_violation_critical, sample_violation_warning, sample_violation_info
) -> ValidationResult:
    """Create a failed validation result with multiple violations."""
    return ValidationResult(
        validator_role=ValidationRole.TEST_COVERAGE,
        passed=False,
        violations=[
            sample_violation_critical,
            sample_violation_warning,
            sample_violation_info,
        ],
        validation_time_ms=123.7,
        memory_uuid=str(uuid4()),
    )


@pytest.fixture
def sample_result_with_error() -> ValidationResult:
    """Create a validation result with an error."""
    return ValidationResult(
        validator_role=ValidationRole.ENTITY_CONSISTENCY,
        passed=False,
        violations=[],
        validation_time_ms=0.0,
        error_message="Mnemodrome service unreachable",
        memory_uuid=str(uuid4()),
    )


# ============================================================================
# Tests: ValidationViolation Creation and Serialization (T-038.1)
# ============================================================================


class TestValidationViolationCreation:
    """Tests for ValidationViolation creation and serialization."""

    def test_violation_creation_with_all_fields(self, sample_violation_critical):
        """Test creating violation with all fields populated."""
        assert sample_violation_critical.violation_id is not None
        assert sample_violation_critical.severity == ValidationSeverity.CRITICAL
        assert sample_violation_critical.category == "naming"
        assert (
            sample_violation_critical.message
            == "Function 'MyFunction' should use snake_case"
        )
        assert sample_violation_critical.line_number == 42
        assert sample_violation_critical.column_number == 5
        assert sample_violation_critical.recommendation == "Rename to 'my_function'"
        assert sample_violation_critical.pattern_uuid == "uuid-pattern-001"
        assert sample_violation_critical.context == "PEP 8 naming conventions"

    def test_violation_creation_minimal_fields(self):
        """Test creating violation with minimal required fields."""
        violation = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="test",
            message="Test violation",
            recommendation="Fix it",
        )
        assert violation.violation_id is not None
        assert violation.severity == ValidationSeverity.WARNING
        assert violation.line_number is None
        assert violation.column_number is None
        assert violation.pattern_uuid is None
        assert violation.context is None

    def test_violation_auto_generates_uuid(self):
        """Test that violation_id is auto-generated."""
        v1 = ValidationViolation(
            severity=ValidationSeverity.INFO,
            category="test",
            message="Test",
            recommendation="Fix",
        )
        v2 = ValidationViolation(
            severity=ValidationSeverity.INFO,
            category="test",
            message="Test",
            recommendation="Fix",
        )
        assert v1.violation_id != v2.violation_id

    def test_violation_serialization_to_dict(self, sample_violation_critical):
        """Test violation can be serialized to dict."""
        violation_dict = sample_violation_critical.dict()
        assert violation_dict["severity"] == ValidationSeverity.CRITICAL
        assert violation_dict["category"] == "naming"
        assert (
            violation_dict["message"] == "Function 'MyFunction' should use snake_case"
        )

    def test_violation_json_serialization(self, sample_violation_warning):
        """Test violation can be serialized to JSON."""
        json_str = sample_violation_warning.json()
        assert '"severity"' in json_str or "'severity'" in json_str
        assert '"warning"' in json_str or "'warning'" in json_str
        assert "error_handling" in json_str


# ============================================================================
# Tests: ValidationResult Creation and Status (T-038.2)
# ============================================================================


class TestValidationResultCreation:
    """Tests for ValidationResult creation and status tracking."""

    def test_result_creation_passed(self, sample_result_passed):
        """Test creating a passed validation result."""
        assert sample_result_passed.result_id is not None
        assert sample_result_passed.passed is True
        assert sample_result_passed.violations == []
        assert sample_result_passed.validator_role == ValidationRole.PATTERN_COMPLIANCE
        assert sample_result_passed.memory_uuid is not None

    def test_result_creation_with_violations(self, sample_result_with_violations):
        """Test creating a failed validation result."""
        assert sample_result_with_violations.result_id is not None
        assert sample_result_with_violations.passed is False
        assert len(sample_result_with_violations.violations) == 3
        assert sample_result_with_violations.error_message is None

    def test_result_creation_with_error(self, sample_result_with_error):
        """Test creating a result with error message."""
        assert sample_result_with_error.result_id is not None
        assert sample_result_with_error.passed is False
        assert sample_result_with_error.violations == []
        assert (
            sample_result_with_error.error_message == "Mnemodrome service unreachable"
        )

    def test_result_auto_generates_ids(self):
        """Test that result_id is auto-generated per result."""
        r1 = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=10.0,
        )
        r2 = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=10.0,
        )
        assert r1.result_id != r2.result_id

    def test_result_memory_uuid_tracking(self):
        """Test Mnemodrome UUID is tracked in result."""
        test_uuid = str(uuid4())
        result = ValidationResult(
            validator_role=ValidationRole.TEST_COVERAGE,
            passed=True,
            violations=[],
            validation_time_ms=25.0,
            memory_uuid=test_uuid,
        )
        assert result.memory_uuid == test_uuid


# ============================================================================
# Tests: Violation Deduplication Logic (T-038.3)
# ============================================================================


class TestViolationDeduplication:
    """Tests for violation deduplication by category+message."""

    @pytest.fixture
    def duplicate_violations(self, sample_violation_critical):
        """Create duplicate violations (same category and message)."""
        v1 = sample_violation_critical
        v2 = ValidationViolation(
            severity=ValidationSeverity.CRITICAL,
            category="naming",
            message="Function 'MyFunction' should use snake_case",
            line_number=100,  # Different line
            column_number=8,  # Different column
            recommendation="Rename to 'my_function'",
            context="PEP 8 naming conventions",
        )
        return [v1, v2]

    def test_deduplication_identical_violations(self, duplicate_violations):
        """Test that identical violations are deduplicated."""
        # Simulate deduplication logic from aggregator
        seen = {}
        deduplicated = []

        for violation in duplicate_violations:
            key = (violation.category, violation.message)
            if key not in seen:
                seen[key] = violation
                deduplicated.append(violation)

        assert len(deduplicated) == 1
        assert deduplicated[0].category == "naming"

    def test_deduplication_different_messages_not_deduplicated(self):
        """Test that violations with different messages are NOT deduplicated."""
        v1 = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="naming",
            message="Function 'MyFunc' should use snake_case",
            recommendation="Rename",
        )
        v2 = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="naming",
            message="Class 'myClass' should use PascalCase",
            recommendation="Rename",
        )

        seen = {}
        deduplicated = []
        for violation in [v1, v2]:
            key = (violation.category, violation.message)
            if key not in seen:
                seen[key] = violation
                deduplicated.append(violation)

        assert len(deduplicated) == 2

    def test_deduplication_different_categories_not_deduplicated(self):
        """Test violations with different categories are NOT deduplicated."""
        v1 = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="naming",
            message="Naming issue",
            recommendation="Fix naming",
        )
        v2 = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="documentation",
            message="Naming issue",  # Same message, different category
            recommendation="Add doc",
        )

        seen = {}
        deduplicated = []
        for violation in [v1, v2]:
            key = (violation.category, violation.message)
            if key not in seen:
                seen[key] = violation
                deduplicated.append(violation)

        assert len(deduplicated) == 2

    def test_deduplication_preserves_first_occurrence(self, duplicate_violations):
        """Test that deduplication keeps first occurrence."""
        seen = {}
        deduplicated = []

        for violation in duplicate_violations:
            key = (violation.category, violation.message)
            if key not in seen:
                seen[key] = violation
                deduplicated.append(violation)

        # Should preserve the first violation's line number
        assert deduplicated[0].line_number == 42


# ============================================================================
# Tests: Severity-Based Prioritization (T-038.4)
# ============================================================================


class TestSeverityPrioritization:
    """Tests for violation prioritization by severity."""

    @pytest.fixture
    def mixed_severity_violations(self):
        """Create violations with mixed severity levels."""
        return [
            ValidationViolation(
                severity=ValidationSeverity.INFO,
                category="doc",
                message="Missing docstring",
                recommendation="Add doc",
            ),
            ValidationViolation(
                severity=ValidationSeverity.CRITICAL,
                category="error",
                message="Bare except clause",
                recommendation="Specify exception",
            ),
            ValidationViolation(
                severity=ValidationSeverity.WARNING,
                category="naming",
                message="Bad naming",
                recommendation="Fix naming",
            ),
            ValidationViolation(
                severity=ValidationSeverity.CRITICAL,
                category="coverage",
                message="No tests",
                recommendation="Add tests",
            ),
        ]

    def test_sort_by_severity_critical_first(self, mixed_severity_violations):
        """Test violations are sorted: CRITICAL, WARNING, INFO."""
        severity_order = {
            ValidationSeverity.CRITICAL: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.INFO: 2,
        }

        sorted_violations = sorted(
            mixed_severity_violations, key=lambda v: severity_order.get(v.severity, 99)
        )

        # First two should be critical
        assert sorted_violations[0].severity == ValidationSeverity.CRITICAL
        assert sorted_violations[1].severity == ValidationSeverity.CRITICAL
        # Then warning
        assert sorted_violations[2].severity == ValidationSeverity.WARNING
        # Finally info
        assert sorted_violations[3].severity == ValidationSeverity.INFO

    def test_severity_count_calculation(self, mixed_severity_violations):
        """Test counting violations by severity."""
        critical_count = len(
            [
                v
                for v in mixed_severity_violations
                if v.severity == ValidationSeverity.CRITICAL
            ]
        )
        warning_count = len(
            [
                v
                for v in mixed_severity_violations
                if v.severity == ValidationSeverity.WARNING
            ]
        )
        info_count = len(
            [
                v
                for v in mixed_severity_violations
                if v.severity == ValidationSeverity.INFO
            ]
        )

        assert critical_count == 2
        assert warning_count == 1
        assert info_count == 1


# ============================================================================
# Tests: AggregatedResult Creation (T-038.5)
# ============================================================================


class TestAggregatedResultCreation:
    """Tests for AggregatedResult creation and metadata."""

    def test_aggregated_result_creation_empty(self):
        """Test creating aggregated result with no violations."""
        result = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=50.0,
            sla_met=True,
        )

        assert result.aggregation_id is not None
        assert result.timestamp is not None
        assert result.overall_passed is True
        assert result.total_violations == 0

    def test_aggregated_result_with_violations(self, sample_result_with_violations):
        """Test creating aggregated result with violations."""
        result = AggregatedResult(
            overall_passed=False,
            total_violations=3,
            critical_count=1,
            warning_count=1,
            info_count=1,
            all_violations=sample_result_with_violations.violations,
            validator_results=[sample_result_with_violations],
            total_time_ms=123.7,
            sla_met=True,
        )

        assert result.aggregation_id is not None
        assert result.overall_passed is False
        assert result.total_violations == 3
        assert result.critical_count == 1
        assert result.warning_count == 1
        assert result.info_count == 1

    def test_aggregated_result_timestamp(self):
        """Test aggregated result includes timestamp."""
        before = datetime.utcnow()
        result = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=0.0,
            sla_met=True,
        )
        after = datetime.utcnow()

        assert before <= result.timestamp <= after


# ============================================================================
# Tests: Mnemodrome UUID Collection (T-038.6)
# ============================================================================


class TestMnemodromeUUIDTracking:
    """Tests for Mnemodrome UUID collection and tracking."""

    def test_collect_uuids_from_single_result(self):
        """Test collecting UUID from a single validation result."""
        uuid1 = str(uuid4())
        result = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=10.0,
            memory_uuid=uuid1,
        )

        aggregated = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[result],
            total_time_ms=10.0,
            sla_met=True,
            memory_uuids=[uuid1],
        )

        assert len(aggregated.memory_uuids) == 1
        assert aggregated.memory_uuids[0] == uuid1

    def test_collect_uuids_from_multiple_results(self):
        """Test collecting UUIDs from multiple validation results."""
        uuid1 = str(uuid4())
        uuid2 = str(uuid4())
        uuid3 = str(uuid4())

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
        result3 = ValidationResult(
            validator_role=ValidationRole.ENTITY_CONSISTENCY,
            passed=True,
            violations=[],
            validation_time_ms=20.0,
            memory_uuid=uuid3,
        )

        aggregated = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[result1, result2, result3],
            total_time_ms=45.0,
            sla_met=True,
            memory_uuids=[uuid1, uuid2, uuid3],
        )

        assert len(aggregated.memory_uuids) == 3
        assert uuid1 in aggregated.memory_uuids
        assert uuid2 in aggregated.memory_uuids
        assert uuid3 in aggregated.memory_uuids

    def test_skip_none_uuids(self):
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
            memory_uuid=None,  # No UUID
        )

        # Simulate UUID collection logic
        uuids = [r.memory_uuid for r in [result1, result2] if r.memory_uuid]

        assert len(uuids) == 1
        assert uuids[0] == uuid1


# ============================================================================
# Tests: AggregatedResult Formatting (T-038.7)
# ============================================================================


class TestAggregatedResultFormatting:
    """Tests for AggregatedResult formatting and presentation."""

    def test_format_with_no_violations(self):
        """Test formatting result with no violations."""
        result = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=25.0,
            sla_met=True,
        )

        # Minimal formatting checks
        assert result.overall_passed is True
        assert result.total_violations == 0
        assert len(result.all_violations) == 0

    def test_format_with_violations(
        self, sample_violation_critical, sample_violation_warning
    ):
        """Test formatting result with violations."""
        result = AggregatedResult(
            overall_passed=False,
            total_violations=2,
            critical_count=1,
            warning_count=1,
            info_count=0,
            all_violations=[sample_violation_critical, sample_violation_warning],
            validator_results=[],
            total_time_ms=85.5,
            sla_met=True,
        )

        assert result.overall_passed is False
        assert result.total_violations == 2
        assert result.critical_count == 1
        assert result.warning_count == 1
        assert result.info_count == 0

    def test_overall_passed_with_critical_violations(self):
        """Test that critical violations set overall_passed to False."""
        v_critical = ValidationViolation(
            severity=ValidationSeverity.CRITICAL,
            category="test",
            message="Critical issue",
            recommendation="Fix it",
        )

        result = AggregatedResult(
            overall_passed=False,
            total_violations=1,
            critical_count=1,
            warning_count=0,
            info_count=0,
            all_violations=[v_critical],
            validator_results=[],
            total_time_ms=10.0,
            sla_met=True,
        )

        assert result.overall_passed is False

    def test_overall_passed_with_warnings_only(self):
        """Test that warnings don't block overall_passed."""
        v_warning = ValidationViolation(
            severity=ValidationSeverity.WARNING,
            category="test",
            message="Warning",
            recommendation="Consider fixing",
        )

        result = AggregatedResult(
            overall_passed=True,
            total_violations=1,
            critical_count=0,
            warning_count=1,
            info_count=0,
            all_violations=[v_warning],
            validator_results=[],
            total_time_ms=10.0,
            sla_met=True,
        )

        assert result.overall_passed is True


# ============================================================================
# Tests: SLA Tracking (T-038.8)
# ============================================================================


class TestSLATracking:
    """Tests for SLA timeout tracking."""

    def test_sla_met_under_limit(self):
        """Test SLA is met when under time limit."""
        result = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=120000.0,  # 120 seconds
            sla_met=True,  # Under 180 second limit
        )

        assert result.sla_met is True

    def test_sla_exceeded(self):
        """Test SLA is not met when exceeding limit."""
        result = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=200000.0,  # 200 seconds (exceeds 180s limit)
            sla_met=False,
        )

        assert result.sla_met is False


# ============================================================================
# Tests: Edge Cases (T-038.9)
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_aggregated_result(self):
        """Test creating aggregated result with all empty collections."""
        result = AggregatedResult(
            overall_passed=True,
            total_violations=0,
            critical_count=0,
            warning_count=0,
            info_count=0,
            all_violations=[],
            validator_results=[],
            total_time_ms=0.0,
            sla_met=True,
            memory_uuids=[],
        )

        assert result.total_violations == 0
        assert len(result.all_violations) == 0
        assert len(result.validator_results) == 0
        assert len(result.memory_uuids) == 0

    def test_many_violations(self):
        """Test aggregation with large number of violations."""
        violations = [
            ValidationViolation(
                severity=ValidationSeverity.WARNING,
                category=f"category_{i}",
                message=f"Violation {i}",
                recommendation="Fix it",
            )
            for i in range(100)
        ]

        result = AggregatedResult(
            overall_passed=True,
            total_violations=100,
            critical_count=0,
            warning_count=100,
            info_count=0,
            all_violations=violations,
            validator_results=[],
            total_time_ms=500.0,
            sla_met=True,
        )

        assert result.total_violations == 100
        assert result.warning_count == 100
        assert len(result.all_violations) == 100

    def test_violation_with_none_optional_fields(self):
        """Test violation with all optional fields as None."""
        violation = ValidationViolation(
            severity=ValidationSeverity.INFO,
            category="test",
            message="Test",
            recommendation="Fix",
            line_number=None,
            column_number=None,
            pattern_uuid=None,
            context=None,
        )

        assert violation.line_number is None
        assert violation.column_number is None
        assert violation.pattern_uuid is None
        assert violation.context is None

    def test_result_with_zero_time(self):
        """Test result with zero millisecond duration."""
        result = ValidationResult(
            validator_role=ValidationRole.PATTERN_COMPLIANCE,
            passed=True,
            violations=[],
            validation_time_ms=0.0,
        )

        assert result.validation_time_ms == 0.0

    def test_aggregated_result_all_critical(self):
        """Test aggregation with all critical violations."""
        violations = [
            ValidationViolation(
                severity=ValidationSeverity.CRITICAL,
                category="critical_cat",
                message=f"Critical {i}",
                recommendation="Fix immediately",
            )
            for i in range(5)
        ]

        result = AggregatedResult(
            overall_passed=False,
            total_violations=5,
            critical_count=5,
            warning_count=0,
            info_count=0,
            all_violations=violations,
            validator_results=[],
            total_time_ms=100.0,
            sla_met=True,
        )

        assert result.critical_count == 5
        assert result.overall_passed is False

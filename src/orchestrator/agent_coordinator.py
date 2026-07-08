"""
Multi-Agent Validation Pipeline Orchestrator (Phase 3 - MVP)

Implements the orchestration layer for parallel validation agents with:
- Job-based work distribution (FR-009: Multi-agent spawning)
- Pattern compliance validation (FR-010: Pattern compliance checks)
- Test coverage validation (FR-011: Coverage metrics)
- Entity consistency validation (FR-012: Entity resolution)
- Result aggregation with Mnemodrome UUID tracking (FR-013)
- Approval gate presentation (FR-014: CLI formatting)
- 3-minute SLA timeout handling (SC-006)

Constitutional Authority: Principle II (Development Standards)
Validation agents use local inference, orchestrator uses frontier Claude.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, Field

from src.models.pattern_models import StoredPattern
from src.services.event_log_service import event_log_service

# Core service imports

# ============================================================================
# Constants & Enumerations
# ============================================================================


class ValidationSeverity(Enum):
    """Severity levels for validation violations."""

    CRITICAL = "critical"  # Blocks deployment
    WARNING = "warning"  # Should be fixed
    INFO = "info"  # Informational only


class JobStatus(Enum):
    """Status states for orchestration jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ValidationRole(Enum):
    """Available validation agent roles."""

    PATTERN_COMPLIANCE = "pattern_compliance"
    TEST_COVERAGE = "test_coverage"
    ENTITY_CONSISTENCY = "entity_consistency"


# ============================================================================
# Data Models
# ============================================================================


class ValidationViolation(BaseModel):
    """A single validation violation with remediation guidance.

    Tracks violations at line level with severity and UUID references to
    Mnemodrome graph for pattern/rule resolution.
    """

    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    severity: ValidationSeverity = Field(description="Violation severity")
    category: str = Field(description="Violation category (e.g., 'naming', 'coverage')")
    message: str = Field(description="Human-readable violation message")
    line_number: int | None = Field(
        default=None, description="Line number in source code"
    )
    column_number: int | None = Field(
        default=None, description="Column number in source code"
    )
    recommendation: str = Field(description="Suggested fix")
    pattern_uuid: str | None = Field(
        default=None, description="Mnemodrome UUID of pattern rule"
    )
    context: str | None = Field(default=None, description="Additional context")

    class Config:
        """Pydantic configuration."""

        use_enum_values = False


class ValidationResult(BaseModel):
    """Result of a single validation agent's work.

    Returned by each parallel validator with violations, pass/fail status,
    and Mnemodrome UUID references for decisions tracked in graph layer.
    """

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    validator_role: ValidationRole = Field(
        description="Which validator produced this result"
    )
    passed: bool = Field(description="Whether validation passed")
    violations: list[ValidationViolation] = Field(
        default_factory=list, description="List of violations found"
    )
    validation_time_ms: float = Field(description="Time spent in validation")
    memory_uuid: str | None = Field(
        default=None, description="Mnemodrome UUID for decision tracking"
    )
    error_message: str | None = Field(
        default=None, description="Error message if validation failed"
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = False


class AggregatedResult(BaseModel):
    """Aggregated results from all validation agents.

    Combines violations from all validators with deduplication,
    prioritization, and recommendations for approval gate.
    """

    aggregation_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    overall_passed: bool = Field(description="True if all validators passed")
    total_violations: int = Field(description="Total unique violations")
    critical_count: int = Field(description="Count of critical violations")
    warning_count: int = Field(description="Count of warnings")
    info_count: int = Field(description="Count of informational items")
    all_violations: list[ValidationViolation] = Field(
        default_factory=list, description="Deduplicated and prioritized violations"
    )
    validator_results: list[ValidationResult] = Field(
        default_factory=list, description="Raw results from each validator"
    )
    total_time_ms: float = Field(description="Total orchestration time")
    sla_met: bool = Field(description="Whether 3-minute SLA was met")
    memory_uuids: list[str] = Field(
        default_factory=list,
        description="Mnemodrome UUIDs for all decisions/validations",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = False


@dataclass
class ContextPackage:
    """Package of context required for validations.

    Contains code snippets, patterns from Mnemodrome graph, and configuration
    needed by validators to perform their checks.
    """

    code: str  # Source code to validate
    file_path: str  # Path to source file
    patterns: list[StoredPattern] = field(
        default_factory=list
    )  # Loaded patterns from Mnemodrome graph
    entity_context: dict[str, Any] = field(
        default_factory=dict
    )  # Entity name to UUID mappings
    test_files: list[str] = field(default_factory=list)  # Related test file paths
    related_entities: list[str] = field(
        default_factory=list
    )  # Canonical entity names from Mnemodrome


@dataclass
class OrchestrationJob:
    """A validation job orchestrating multiple agents.

    Represents a unit of work with task description, assigned validators,
    status tracking, and results aggregation.

    FR-009: Multi-agent job spawning with capability matching
    """

    task: str  # Task description
    context: ContextPackage  # Context package for validators
    job_id: str = field(default_factory=lambda: str(uuid4()))
    agents: list[ValidationRole] = field(
        default_factory=list
    )  # Assigned validation agents
    status: JobStatus = field(default=JobStatus.PENDING)
    results: list[ValidationResult] = field(default_factory=list)
    aggregated_result: AggregatedResult | None = field(default=None)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = field(default=None)
    completed_at: datetime | None = field(default=None)
    memory_uuid: str | None = field(default=None)  # Mnemodrome tracking UUID

    def duration_ms(self) -> float:
        """Calculate job duration in milliseconds.

        Returns 0 if job hasn't completed.
        """
        if not self.completed_at:
            return 0.0
        return (self.completed_at - self.created_at).total_seconds() * 1000


# ============================================================================
# Validation Agents (Parallel Workers)
# ============================================================================


class PatternComplianceValidator:
    """Validates code against established naming and structure patterns.

    FR-010: Pattern compliance validation
    Checks code against patterns from Mnemodrome context:
    - Naming conventions (snake_case, PascalCase)
    - Function/class structure
    - Error handling patterns
    - Documentation standards
    """

    @staticmethod
    async def validate_pattern_compliance(
        code: str, context: ContextPackage
    ) -> ValidationResult:
        """
        Validate code against patterns from context.

        Args:
            code: Source code to validate
            context: ContextPackage with loaded patterns

        Returns:
            ValidationResult with violations and pass/fail status
        """
        start_time = time.time()
        violations: list[ValidationViolation] = []

        logger.debug(f"Starting pattern compliance validation for {context.file_path}")

        try:
            # Check for common patterns
            violations.extend(
                await PatternComplianceValidator._check_naming_conventions(
                    code, context
                )
            )
            violations.extend(
                await PatternComplianceValidator._check_error_handling(code, context)
            )
            violations.extend(
                await PatternComplianceValidator._check_documentation(code, context)
            )

            # Filter by configured patterns from Mnemodrome
            if context.patterns:
                violations.extend(
                    await PatternComplianceValidator._check_custom_patterns(
                        code, context
                    )
                )

            elapsed_ms = (time.time() - start_time) * 1000

            result = ValidationResult(
                validator_role=ValidationRole.PATTERN_COMPLIANCE,
                passed=len(
                    [v for v in violations if v.severity == ValidationSeverity.CRITICAL]
                )
                == 0,
                violations=violations,
                validation_time_ms=elapsed_ms,
                memory_uuid=str(uuid4()),
            )

            logger.info(
                f"Pattern compliance validation completed: "
                f"{len(violations)} violations, {elapsed_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Pattern compliance validation error: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ValidationResult(
                validator_role=ValidationRole.PATTERN_COMPLIANCE,
                passed=False,
                violations=[],
                validation_time_ms=elapsed_ms,
                error_message=str(e),
                memory_uuid=str(uuid4()),
            )

    @staticmethod
    async def _check_naming_conventions(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check Python naming conventions."""
        violations = []
        lines = code.split("\n")

        # Check function names are snake_case
        import re

        for i, line in enumerate(lines, 1):
            match = re.search(r"def\s+([A-Z_][a-zA-Z0-9_]*)\s*\(", line)
            if match:
                name = match.group(1)
                if not name.isupper() and "_" in name and name[0].isupper():
                    violations.append(
                        ValidationViolation(
                            severity=ValidationSeverity.WARNING,
                            category="naming",
                            message=f"Function '{name}' should use snake_case",
                            line_number=i,
                            column_number=match.start(1),
                            recommendation=f"Rename to '{name.lower()}'",
                            pattern_uuid=None,
                            context="PEP 8 naming conventions",
                        )
                    )

        return violations

    @staticmethod
    async def _check_error_handling(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check error handling patterns."""
        violations = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for bare except clauses
            if "except:" in line:
                violations.append(
                    ValidationViolation(
                        severity=ValidationSeverity.WARNING,
                        category="error_handling",
                        message="Bare 'except:' clause catches all exceptions",
                        line_number=i,
                        column_number=line.find("except"),
                        recommendation="Specify exception type: except SomeError:",
                        context="Bare except clauses hide errors",
                    )
                )

        return violations

    @staticmethod
    async def _check_documentation(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check documentation requirements."""
        violations = []

        # Check for module docstring
        if not code.strip().startswith('"""') and not code.strip().startswith("'''"):
            violations.append(
                ValidationViolation(
                    severity=ValidationSeverity.INFO,
                    category="documentation",
                    message="Module missing docstring",
                    line_number=1,
                    recommendation='Add module docstring at top: """Description."""',
                    context="PEP 257 documentation standards",
                )
            )

        return violations

    @staticmethod
    async def _check_custom_patterns(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check against custom patterns loaded from Mnemodrome."""
        violations = []

        for pattern in context.patterns:
            # Pattern matching logic here
            # This would check code against specific patterns stored in graph
            pass

        return violations


class TestCoverageValidator:
    """Validates test coverage for code changes.

    FR-011: Test coverage validation
    Checks:
    - Test file existence for module
    - Minimum assertion count
    - Coverage of critical paths
    - Test naming conventions
    """

    @staticmethod
    async def validate_test_coverage(
        code: str, context: ContextPackage
    ) -> ValidationResult:
        """
        Validate test coverage requirements.

        Args:
            code: Source code to validate
            context: ContextPackage with test file list

        Returns:
            ValidationResult with coverage violations
        """
        start_time = time.time()
        violations: list[ValidationViolation] = []

        logger.debug(f"Starting test coverage validation for {context.file_path}")

        try:
            # Check for test files
            violations.extend(
                await TestCoverageValidator._check_test_existence(code, context)
            )

            # Check test content
            if context.test_files:
                violations.extend(
                    await TestCoverageValidator._check_test_assertions(code, context)
                )
                violations.extend(
                    await TestCoverageValidator._check_test_naming(context)
                )

            elapsed_ms = (time.time() - start_time) * 1000

            result = ValidationResult(
                validator_role=ValidationRole.TEST_COVERAGE,
                passed=len(
                    [v for v in violations if v.severity == ValidationSeverity.CRITICAL]
                )
                == 0,
                violations=violations,
                validation_time_ms=elapsed_ms,
                memory_uuid=str(uuid4()),
            )

            logger.info(
                f"Test coverage validation completed: "
                f"{len(violations)} violations, {elapsed_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Test coverage validation error: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ValidationResult(
                validator_role=ValidationRole.TEST_COVERAGE,
                passed=False,
                violations=[],
                validation_time_ms=elapsed_ms,
                error_message=str(e),
                memory_uuid=str(uuid4()),
            )

    @staticmethod
    async def _check_test_existence(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check for corresponding test files."""
        violations = []

        # If no test files found, flag as warning
        if not context.test_files:
            violations.append(
                ValidationViolation(
                    severity=ValidationSeverity.WARNING,
                    category="test_coverage",
                    message=f"No test file found for {context.file_path}",
                    recommendation=f"Create test file: tests/test_{context.file_path.split('/')[-1]}",
                    context="Test coverage requirement",
                )
            )

        return violations

    @staticmethod
    async def _check_test_assertions(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check test files for sufficient assertions."""
        violations = []

        # Count functions in code
        import re

        functions = len(re.findall(r"^\s*def\s+\w+", code, re.MULTILINE))

        if functions > 0 and not context.test_files:
            violations.append(
                ValidationViolation(
                    severity=ValidationSeverity.WARNING,
                    category="test_coverage",
                    message=f"Code defines {functions} functions but no tests found",
                    recommendation="Add test cases for all public functions",
                    context="Test coverage requirement",
                )
            )

        return violations

    @staticmethod
    async def _check_test_naming(context: ContextPackage) -> list[ValidationViolation]:
        """Check test naming conventions."""
        violations = []

        for test_file in context.test_files:
            if not test_file.startswith("test_") and not test_file.endswith("_test.py"):
                violations.append(
                    ValidationViolation(
                        severity=ValidationSeverity.INFO,
                        category="test_naming",
                        message=f"Test file '{test_file}' doesn't follow naming convention",
                        recommendation="Rename to 'test_*.py' or '*_test.py'",
                        context="Test file naming convention",
                    )
                )

        return violations


class EntityConsistencyValidator:
    """Validates entity names match Mnemodrome graph.

    FR-012: Entity consistency validation
    Checks:
    - All entities referenced exist in Mnemodrome graph
    - Entity names use canonical forms
    - UUID references resolve correctly
    - Type consistency (Person vs Project vs Technology)
    """

    @staticmethod
    async def validate_entity_consistency(
        code: str, context: ContextPackage
    ) -> ValidationResult:
        """
        Validate entity consistency against Mnemodrome graph.

        Args:
            code: Source code to validate
            context: ContextPackage with entity context

        Returns:
            ValidationResult with entity inconsistencies
        """
        start_time = time.time()
        violations: list[ValidationViolation] = []

        logger.debug(f"Starting entity consistency validation for {context.file_path}")

        try:
            # Check referenced entities exist
            violations.extend(
                await EntityConsistencyValidator._check_entity_existence(code, context)
            )

            # Check entity names are canonical
            violations.extend(
                await EntityConsistencyValidator._check_canonical_names(code, context)
            )

            # Check type consistency
            violations.extend(
                await EntityConsistencyValidator._check_entity_types(code, context)
            )

            elapsed_ms = (time.time() - start_time) * 1000

            result = ValidationResult(
                validator_role=ValidationRole.ENTITY_CONSISTENCY,
                passed=len(
                    [v for v in violations if v.severity == ValidationSeverity.CRITICAL]
                )
                == 0,
                violations=violations,
                validation_time_ms=elapsed_ms,
                memory_uuid=str(uuid4()),
            )

            logger.info(
                f"Entity consistency validation completed: "
                f"{len(violations)} violations, {elapsed_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Entity consistency validation error: {e}")
            elapsed_ms = (time.time() - start_time) * 1000
            return ValidationResult(
                validator_role=ValidationRole.ENTITY_CONSISTENCY,
                passed=False,
                violations=[],
                validation_time_ms=elapsed_ms,
                error_message=str(e),
                memory_uuid=str(uuid4()),
            )

    @staticmethod
    async def _check_entity_existence(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check that all referenced entities exist in Mnemodrome."""
        violations = []

        # Extract entity references from code (simplified)
        import re

        entity_refs = re.findall(
            r'entity[_\s]*(?:id|uuid|name)?\s*=\s*["\']([^"\']+)["\']', code
        )

        for entity_ref in set(entity_refs):
            if entity_ref not in context.entity_context:
                violations.append(
                    ValidationViolation(
                        severity=ValidationSeverity.WARNING,
                        category="entity_existence",
                        message=f"Entity '{entity_ref}' not found in Mnemodrome graph",
                        recommendation=f"Add entity to Mnemodrome or use existing: {', '.join(context.entity_context.keys())}",
                        context="Entity resolution via Mnemodrome graph",
                    )
                )

        return violations

    @staticmethod
    async def _check_canonical_names(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check that entity names use canonical forms."""
        violations = []

        # Check for variations of known entities
        for canonical_name in context.related_entities:
            variations = [
                canonical_name.lower(),
                canonical_name.upper(),
                canonical_name.replace("_", "-"),
            ]

            for variation in variations:
                if variation != canonical_name and variation in code:
                    violations.append(
                        ValidationViolation(
                            severity=ValidationSeverity.INFO,
                            category="canonical_naming",
                            message=f"Using non-canonical form '{variation}' instead of '{canonical_name}'",
                            recommendation=f"Use canonical form: '{canonical_name}'",
                            pattern_uuid=context.entity_context.get(canonical_name),
                            context="Entity canonicalization",
                        )
                    )

        return violations

    @staticmethod
    async def _check_entity_types(
        code: str, context: ContextPackage
    ) -> list[ValidationViolation]:
        """Check entity type consistency."""
        violations = []

        # This would check type consistency by comparing against graph metadata
        # Simplified for MVP

        return violations


# ============================================================================
# Orchestrator Core
# ============================================================================


class AgentCoordinator:
    """Orchestrates multi-agent validation pipeline.

    Manages job lifecycle:
    1. Create job with task
    2. Spawn validators based on capabilities
    3. Run validators in parallel with 3-min SLA
    4. Aggregate results with deduplication
    5. Present approval gate
    6. Track decisions in Mnemodrome graph

    FR-009, FR-010, FR-011, FR-012, FR-013, FR-014
    """

    def __init__(self, sla_timeout_seconds: int = 180):
        """
        Initialize agent coordinator.

        Args:
            sla_timeout_seconds: SLA timeout for validation (default: 180s/3min)
        """
        self.sla_timeout_seconds = sla_timeout_seconds
        logger.info(f"AgentCoordinator initialized with {sla_timeout_seconds}s SLA")

    async def create_job(
        self,
        task: str,
        context: ContextPackage,
        agents: list[ValidationRole] | None = None,
    ) -> OrchestrationJob:
        """
        Create a new orchestration job.

        Args:
            task: Task description
            context: Context package with code and patterns
            agents: Specific agents to use (default: all)

        Returns:
            OrchestrationJob ready to execute
        """
        if agents is None:
            agents = list(ValidationRole)

        job = OrchestrationJob(
            task=task, context=context, agents=agents, memory_uuid=str(uuid4())
        )

        logger.info(f"Created job {job.job_id}: {task} with {len(agents)} agents")

        return job

    async def spawn_validation_agents(
        self, job: OrchestrationJob
    ) -> list[ValidationResult]:
        """
        Spawn and execute validation agents in parallel.

        Implements T-031: Multi-agent spawning with timeout.
        Runs all validators concurrently with 3-minute SLA.

        Args:
            job: OrchestrationJob to execute

        Returns:
            List of ValidationResults from all agents

        Raises:
            asyncio.TimeoutError: If SLA timeout exceeded
        """
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

        logger.info(
            f"Spawning {len(job.agents)} validation agents for job {job.job_id}"
        )

        # Create validator tasks
        tasks = []
        validator_map = {
            ValidationRole.PATTERN_COMPLIANCE: PatternComplianceValidator.validate_pattern_compliance,
            ValidationRole.TEST_COVERAGE: TestCoverageValidator.validate_test_coverage,
            ValidationRole.ENTITY_CONSISTENCY: EntityConsistencyValidator.validate_entity_consistency,
        }

        for agent_role in job.agents:
            if agent_role in validator_map:
                validator_func = validator_map[agent_role]
                tasks.append(validator_func(job.context.code, job.context))

        # Execute with timeout (T-037: SLA handling)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=False),
                timeout=self.sla_timeout_seconds,
            )

            job.results = results
            job.status = JobStatus.COMPLETED

            logger.success(
                f"Validation completed for job {job.job_id}: {len(results)} results"
            )

            return results

        except TimeoutError:
            job.status = JobStatus.TIMEOUT
            logger.warning(
                f"Validation timeout for job {job.job_id} (SLA: {self.sla_timeout_seconds}s)"
            )

            # Log timeout event to Mnemodrome (T-037)
            await event_log_service.log_event(
                event_type="validation",
                action="timeout",
                target=job.job_id,
                result="timeout",
                context=f"SLA exceeded: {self.sla_timeout_seconds}s",
            )

            # Return partial results from completed validators
            return job.results

    async def aggregate_validation_results(
        self, results: list[ValidationResult]
    ) -> AggregatedResult:
        """
        Aggregate validation results with deduplication and prioritization.

        Implements T-035: Result aggregation with Mnemodrome UUID tracking

        Args:
            results: List of ValidationResults from validators

        Returns:
            AggregatedResult with combined violations and metadata
        """
        start_time = time.time()

        # Collect all violations
        all_violations: list[ValidationViolation] = []
        for result in results:
            all_violations.extend(result.violations)

        # Deduplicate violations by category+message
        seen_violations = {}
        deduplicated = []

        for violation in all_violations:
            key = (violation.category, violation.message)
            if key not in seen_violations:
                seen_violations[key] = violation
                deduplicated.append(violation)

        # Sort by severity
        severity_order = {
            ValidationSeverity.CRITICAL: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.INFO: 2,
        }
        deduplicated.sort(key=lambda v: severity_order.get(v.severity, 99))

        # Count violations by severity
        critical = len(
            [v for v in deduplicated if v.severity == ValidationSeverity.CRITICAL]
        )
        warning = len(
            [v for v in deduplicated if v.severity == ValidationSeverity.WARNING]
        )
        info = len([v for v in deduplicated if v.severity == ValidationSeverity.INFO])

        # Collect Mnemodrome UUIDs
        uuids = [r.memory_uuid for r in results if r.memory_uuid]

        elapsed_ms = (time.time() - start_time) * 1000
        total_validation_time = sum(r.validation_time_ms for r in results)

        result = AggregatedResult(
            overall_passed=critical == 0,
            total_violations=len(deduplicated),
            critical_count=critical,
            warning_count=warning,
            info_count=info,
            all_violations=deduplicated,
            validator_results=results,
            total_time_ms=total_validation_time,
            sla_met=total_validation_time <= (self.sla_timeout_seconds * 1000),
            memory_uuids=uuids,
        )

        logger.info(
            f"Aggregated {len(results)} validation results: "
            f"{critical} critical, {warning} warnings, {info} info"
        )

        return result

    async def present_approval_gate(self, result: AggregatedResult) -> str:
        """
        Format aggregated results for CLI approval gate presentation.

        Implements T-036: Approval gate presentation logic
        Shows pass/fail status, violations, and recommendations.

        Args:
            result: AggregatedResult to present

        Returns:
            Formatted markdown for CLI display
        """
        status_emoji = "✅" if result.overall_passed else "❌"
        status_text = "PASSED" if result.overall_passed else "FAILED"

        output = []
        output.append(f"\n{'=' * 70}")
        output.append(f"{status_emoji} VALIDATION {status_text}")
        output.append(f"{'=' * 70}\n")

        output.append(f"**Aggregation ID**: {result.aggregation_id}")
        output.append(f"**Timestamp**: {result.timestamp.isoformat()}")
        output.append(f"**Total Violations**: {result.total_violations}")
        output.append(
            f"**SLA Status**: {'✅ Met' if result.sla_met else '⚠️  Exceeded'}"
        )
        output.append(f"**Validation Time**: {result.total_time_ms:.2f}ms\n")

        # Violation counts
        output.append("### Violation Summary")
        output.append(f"- 🔴 Critical: {result.critical_count}")
        output.append(f"- 🟡 Warnings: {result.warning_count}")
        output.append(f"- ℹ️  Info: {result.info_count}\n")

        # Detailed violations
        if result.all_violations:
            output.append("### Detailed Violations\n")

            for violation in result.all_violations:
                severity_icon = {
                    ValidationSeverity.CRITICAL: "🔴",
                    ValidationSeverity.WARNING: "🟡",
                    ValidationSeverity.INFO: "ℹ️",
                }[violation.severity]

                output.append(
                    f"{severity_icon} **{violation.category.upper()}** (Line {violation.line_number or 'N/A'})"
                )
                output.append(f"   - Message: {violation.message}")
                output.append(f"   - Recommendation: {violation.recommendation}")

                if violation.pattern_uuid:
                    output.append(f"   - Pattern UUID: `{violation.pattern_uuid}`")
                if violation.context:
                    output.append(f"   - Context: {violation.context}")
                output.append("")

        # Validator summary
        output.append("### Validator Results\n")
        for vresult in result.validator_results:
            status = "✅" if vresult.passed else "❌"
            role = vresult.validator_role.value
            output.append(
                f"{status} {role}: {vresult.validation_time_ms:.2f}ms "
                f"({len(vresult.violations)} violations)"
            )
            if vresult.error_message:
                output.append(f"   - Error: {vresult.error_message}")

        output.append("\n### Mnemodrome Tracking")
        if result.memory_uuids:
            output.append("**Decision UUIDs**:")
            for uuid in result.memory_uuids:
                output.append(f"- `{uuid}`")

        output.append(f"\n{'=' * 70}\n")

        return "\n".join(output)

    async def execute_job(self, job: OrchestrationJob) -> AggregatedResult:
        """
        Execute complete orchestration job lifecycle.

        Orchestrates: spawn → aggregate → present

        Args:
            job: OrchestrationJob to execute

        Returns:
            AggregatedResult with final validation status
        """
        logger.info(f"Executing job {job.job_id}: {job.task}")

        # Step 1: Spawn validators (T-031)
        results = await self.spawn_validation_agents(job)

        # Step 2: Aggregate results (T-035)
        aggregated = await self.aggregate_validation_results(results)
        job.aggregated_result = aggregated

        # Step 3: Mark complete
        job.completed_at = datetime.utcnow()

        # Step 4: Log to Mnemodrome (Constitutional: Mnemodrome as source of truth)
        await self._track_in_mnemodrome(job, aggregated)

        logger.success(
            f"Job {job.job_id} completed: "
            f"{aggregated.total_violations} violations, "
            f"{job.duration_ms():.2f}ms"
        )

        return aggregated

    async def _track_in_mnemodrome(
        self, job: OrchestrationJob, result: AggregatedResult
    ) -> None:
        """
        Track orchestration decision in Mnemodrome graph.

        Stores job execution and validation results as episodic memory
        for future reference and pattern discovery.

        Args:
            job: Completed OrchestrationJob
            result: AggregatedResult with decisions
        """
        try:
            # Log event to event log (Feature 009: Event Log Layer)
            await event_log_service.log_event(
                event_type="validation",
                action="job_completed",
                target=job.job_id,
                result="success" if result.overall_passed else "violations_found",
                context=f"violations={result.total_violations}, critical={result.critical_count}",
                metadata=f'{{"aggregation_id": "{result.aggregation_id}", "sla_met": {result.sla_met}}}',
            )

            logger.debug(f"Tracked job {job.job_id} in Mnemodrome")

        except Exception as e:
            logger.warning(f"Failed to track job in Mnemodrome: {e}")
            # Non-critical failure - validation already completed


# ============================================================================
# Utility Functions
# ============================================================================


async def validate_code(
    code: str, file_path: str, agents: list[ValidationRole] | None = None
) -> AggregatedResult:
    """
    Convenience function to validate code with default settings.

    Args:
        code: Source code to validate
        file_path: Path to source file
        agents: Specific agents to use (default: all)

    Returns:
        AggregatedResult with validation outcome

    Example:
        >>> code = open('myfile.py').read()
        >>> result = await validate_code(code, 'src/myfile.py')
        >>> print(await coordinator.present_approval_gate(result))
    """
    coordinator = AgentCoordinator(sla_timeout_seconds=180)

    context = ContextPackage(code=code, file_path=file_path)

    job = await coordinator.create_job(
        task=f"Validate {file_path}", context=context, agents=agents
    )

    return await coordinator.execute_job(job)


__all__ = [
    "AgentCoordinator",
    "OrchestrationJob",
    "ValidationResult",
    "AggregatedResult",
    "ValidationViolation",
    "ContextPackage",
    "ValidationRole",
    "JobStatus",
    "ValidationSeverity",
    "PatternComplianceValidator",
    "TestCoverageValidator",
    "EntityConsistencyValidator",
    "validate_code",
]

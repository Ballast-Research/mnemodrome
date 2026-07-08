"""
Unit tests for Procedure and ProcedureMatch Pydantic models (Feature 031).

Tests model validation, field constraints, serialization, and type enforcement.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.procedure_models import (
    Procedure,
    ProcedureMatch,
)


class TestValidProcedure:
    """Test valid procedure creation and field defaults."""

    def test_valid_procedure(self):
        """Create procedure with all required fields."""
        procedure = Procedure(
            procedure_uuid="proc-001",
            problem_type="infrastructure",
            symptom="Service deployment timeout after 5 minutes",
            diagnostic_steps=[
                "Check Kubernetes pod status",
                "Review pod logs for errors",
                "Verify resource limits",
            ],
            root_cause="Memory limit too low for application startup",
            fix="Increase pod memory limit from 256Mi to 512Mi in deployment.yaml",
            verification="Deploy updated manifest and verify pod reaches Ready state within 30 seconds",
        )

        assert procedure.procedure_uuid == "proc-001"
        assert procedure.problem_type == "infrastructure"
        assert procedure.symptom == "Service deployment timeout after 5 minutes"
        assert len(procedure.diagnostic_steps) == 3
        assert procedure.root_cause == "Memory limit too low for application startup"
        assert (
            procedure.fix
            == "Increase pod memory limit from 256Mi to 512Mi in deployment.yaml"
        )
        assert (
            procedure.verification
            == "Deploy updated manifest and verify pod reaches Ready state within 30 seconds"
        )

    def test_valid_procedure_defaults(self):
        """Verify defaults for optional and defaulted fields."""
        procedure = Procedure(
            problem_type="code",
            symptom="Function returns None instead of expected list",
            diagnostic_steps=["Add debug print statements", "Check return statement"],
            root_cause="Missing return statement in error path",
            fix="Add return empty_list in except block",
            verification="Run unit tests to confirm behavior",
        )

        assert procedure.procedure_uuid == ""
        assert procedure.outcome == "SUCCESS"
        assert procedure.technologies == []
        assert procedure.pattern_uuid is None
        assert procedure.session_uuid is None
        assert isinstance(procedure.created_at, datetime)

    def test_valid_procedure_with_all_fields(self):
        """Create procedure with all fields including optional ones."""
        now = datetime.utcnow()
        procedure = Procedure(
            procedure_uuid="proc-002",
            problem_type="deployment",
            symptom="Database migration fails on production deployment",
            diagnostic_steps=[
                "Check migration script syntax",
                "Verify database connection",
                "Review table structure",
                "Check for data conflicts",
            ],
            root_cause="Migration script missing rollback logic",
            fix="Add rollback transaction block and test both directions",
            verification="Run migration and rollback in staging environment",
            outcome="PARTIAL",
            technologies=["Python", "PostgreSQL", "Alembic"],
            pattern_uuid="pat-123",
            session_uuid="sess-456",
            created_at=now,
        )

        assert procedure.procedure_uuid == "proc-002"
        assert procedure.outcome == "PARTIAL"
        assert procedure.technologies == ["Python", "PostgreSQL", "Alembic"]
        assert procedure.pattern_uuid == "pat-123"
        assert procedure.session_uuid == "sess-456"
        assert procedure.created_at == now


class TestProcedureValidation:
    """Test Procedure field validation."""

    def test_invalid_problem_type(self):
        """Reject invalid problem type value."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="invalid_type",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
            )

        assert "problem_type" in str(exc_info.value)

    def test_empty_diagnostic_steps(self):
        """Reject empty diagnostic steps list."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=[],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
            )

        assert "diagnostic_steps" in str(exc_info.value)

    def test_symptom_too_short(self):
        """Reject symptom shorter than 10 characters."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="short",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
            )

        assert "symptom" in str(exc_info.value)

    def test_symptom_too_long(self):
        """Reject symptom longer than 500 characters."""
        long_symptom = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom=long_symptom,
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
            )

        assert "symptom" in str(exc_info.value)

    def test_root_cause_too_short(self):
        """Reject root cause shorter than 10 characters."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="short",
                fix="This is a valid fix description",
            )

        assert "root_cause" in str(exc_info.value)

    def test_root_cause_too_long(self):
        """Reject root cause longer than 500 characters."""
        long_cause = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause=long_cause,
                fix="This is a valid fix description",
            )

        assert "root_cause" in str(exc_info.value)

    def test_fix_too_short(self):
        """Reject fix shorter than 10 characters."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="short",
            )

        assert "fix" in str(exc_info.value)

    def test_fix_too_long(self):
        """Reject fix longer than 500 characters."""
        long_fix = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix=long_fix,
            )

        assert "fix" in str(exc_info.value)

    def test_verification_too_short(self):
        """Reject verification shorter than 5 characters."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification="hi",
            )

        assert "verification" in str(exc_info.value)

    def test_verification_too_long(self):
        """Reject verification longer than 500 characters."""
        long_verification = "x" * 501
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification=long_verification,
            )

        assert "verification" in str(exc_info.value)

    def test_invalid_outcome(self):
        """Reject invalid outcome value."""
        with pytest.raises(ValidationError) as exc_info:
            Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                outcome="UNKNOWN",
            )

        assert "outcome" in str(exc_info.value)

    def test_all_valid_outcomes(self):
        """Accept all valid outcome values."""
        for outcome in ["SUCCESS", "FAILED", "PARTIAL"]:
            procedure = Procedure(
                problem_type="code",
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification="Verify with unit tests",
                outcome=outcome,
            )
            assert procedure.outcome == outcome

    def test_all_valid_problem_types(self):
        """Accept all valid problem type values."""
        for problem_type in [
            "infrastructure",
            "code",
            "architecture",
            "deployment",
            "data",
            "integration",
            "testing",
            "other",
        ]:
            procedure = Procedure(
                problem_type=problem_type,
                symptom="This is a valid symptom with enough characters",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification="Verify with appropriate testing",
            )
            assert procedure.problem_type == problem_type


class TestProcedureSerialization:
    """Test JSON serialization and deserialization."""

    def test_json_serialization(self):
        """Serialize procedure to JSON."""
        procedure = Procedure(
            procedure_uuid="proc-001",
            problem_type="infrastructure",
            symptom="Service deployment timeout after 5 minutes",
            diagnostic_steps=["Check pod status", "Review logs"],
            root_cause="Memory limit too low",
            fix="Increase pod memory limit",
            verification="Verify pod starts quickly",
            technologies=["Kubernetes", "Docker"],
        )

        json_str = procedure.model_dump_json()
        assert isinstance(json_str, str)
        assert "proc-001" in json_str
        assert "infrastructure" in json_str

    def test_json_round_trip(self):
        """Serialize and deserialize procedure."""
        original = Procedure(
            procedure_uuid="proc-002",
            problem_type="code",
            symptom="Function returns None instead of list",
            diagnostic_steps=["Check return statement", "Review error handling"],
            root_cause="Missing return in except block",
            fix="Add return empty_list in except block",
            verification="Run unit tests",
            outcome="FAILED",
            technologies=["Python", "pytest"],
            pattern_uuid="pat-123",
            session_uuid="sess-456",
        )

        json_str = original.model_dump_json()
        restored = Procedure.model_validate_json(json_str)

        assert restored.procedure_uuid == original.procedure_uuid
        assert restored.problem_type == original.problem_type
        assert restored.symptom == original.symptom
        assert restored.diagnostic_steps == original.diagnostic_steps
        assert restored.root_cause == original.root_cause
        assert restored.fix == original.fix
        assert restored.verification == original.verification
        assert restored.outcome == original.outcome
        assert restored.technologies == original.technologies
        assert restored.pattern_uuid == original.pattern_uuid
        assert restored.session_uuid == original.session_uuid


class TestProcedureMatchValidation:
    """Test ProcedureMatch model validation."""

    def test_procedure_match_valid(self):
        """Create valid ProcedureMatch."""
        match = ProcedureMatch(
            procedure_uuid="proc-001",
            problem_type="infrastructure",
            symptom="Service deployment timeout",
            diagnostic_steps=["Check pod status"],
            root_cause="Memory limit insufficient",
            fix="Increase memory limit",
            verification="Verify pod readiness",
            outcome="SUCCESS",
            similarity_score=0.92,
            created_at=datetime.utcnow(),
            linked_pattern_uuid="pat-123",
            linked_pattern_name="Kubernetes Resource Constraints",
        )

        assert match.procedure_uuid == "proc-001"
        assert match.similarity_score == 0.92
        assert match.linked_pattern_uuid == "pat-123"

    def test_procedure_match_defaults(self):
        """Verify ProcedureMatch defaults."""
        match = ProcedureMatch(
            procedure_uuid="proc-001",
            problem_type="code",
            symptom="This is a valid symptom description here",
            diagnostic_steps=["Step one"],
            root_cause="Valid root cause with sufficient length",
            fix="This is a valid fix description",
            verification="Verify the fix",
            outcome="SUCCESS",
            similarity_score=0.75,
            created_at=datetime.utcnow(),
        )

        assert match.technologies == []
        assert match.linked_pattern_uuid is None
        assert match.linked_pattern_name is None

    def test_procedure_match_similarity_score_valid_bounds(self):
        """Accept similarity scores within bounds."""
        for score in [0.0, 0.5, 0.99, 1.0]:
            match = ProcedureMatch(
                procedure_uuid="proc-001",
                problem_type="code",
                symptom="This is a valid symptom description",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification="Verify the fix",
                outcome="SUCCESS",
                similarity_score=score,
                created_at=datetime.utcnow(),
            )
            assert match.similarity_score == score

    def test_procedure_match_similarity_score_above_one(self):
        """Reject similarity score greater than 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ProcedureMatch(
                procedure_uuid="proc-001",
                problem_type="code",
                symptom="This is a valid symptom description",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification="Verify the fix",
                outcome="SUCCESS",
                similarity_score=1.1,
                created_at=datetime.utcnow(),
            )

        assert "similarity_score" in str(exc_info.value)

    def test_procedure_match_similarity_score_below_zero(self):
        """Reject similarity score less than 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ProcedureMatch(
                procedure_uuid="proc-001",
                problem_type="code",
                symptom="This is a valid symptom description",
                diagnostic_steps=["Step one"],
                root_cause="Valid root cause with sufficient length",
                fix="This is a valid fix description",
                verification="Verify the fix",
                outcome="SUCCESS",
                similarity_score=-0.1,
                created_at=datetime.utcnow(),
            )

        assert "similarity_score" in str(exc_info.value)

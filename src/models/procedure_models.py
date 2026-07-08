"""Pydantic models for procedural memory layer (Feature 031)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Type aliases
ProcedureProblemType = Literal[
    "infrastructure",
    "code",
    "architecture",
    "deployment",
    "data",
    "integration",
    "testing",
    "other",
]

ProcedureOutcome = Literal["SUCCESS", "FAILED", "PARTIAL"]


class Procedure(BaseModel):
    """
    Procedure stored in Neo4j for diagnostic and recovery workflows.

    Represents a documented solution to a specific problem symptom,
    with diagnostic steps, root cause analysis, and fix implementation.
    """

    procedure_uuid: str = Field(
        default="", description="Generated UUID (empty on creation, filled by service)"
    )

    problem_type: ProcedureProblemType = Field(
        ...,
        description="Category of problem (infrastructure, code, architecture, deployment, data, integration, testing, other)",
    )

    symptom: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Observable symptom or error description",
    )

    diagnostic_steps: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered steps to diagnose the root cause (at least one required)",
    )

    root_cause: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Identified root cause of the problem",
    )

    fix: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Steps or implementation to fix the issue",
    )

    verification: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Steps to verify the fix was successful",
    )

    outcome: ProcedureOutcome = Field(
        default="SUCCESS",
        description="Result of applying this procedure (SUCCESS, FAILED, PARTIAL)",
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="Technologies involved (Python, Docker, Kubernetes, etc.)",
    )

    pattern_uuid: str | None = Field(
        default=None,
        description="UUID of linked pattern from Feature 026 (if applicable)",
    )

    session_uuid: str | None = Field(
        default=None, description="UUID of session where procedure was created"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When procedure was created"
    )

    @field_validator("diagnostic_steps")
    @classmethod
    def validate_diagnostic_steps(cls, v: list[str]) -> list[str]:
        """Ensure at least one diagnostic step is provided."""
        if not v or len(v) == 0:
            raise ValueError("At least one diagnostic step is required")
        return v


class ProcedureMatch(BaseModel):
    """
    Procedure match result from semantic search.

    Includes similarity score and metadata for ranked results.
    """

    procedure_uuid: str = Field(..., description="UUID of matched procedure")

    problem_type: str = Field(..., description="Problem category")

    symptom: str = Field(..., description="Observable symptom")

    diagnostic_steps: list[str] = Field(..., description="Steps to diagnose root cause")

    root_cause: str = Field(..., description="Root cause of the problem")

    fix: str = Field(..., description="Solution or fix implementation")

    verification: str = Field(..., description="Steps to verify the fix")

    outcome: str = Field(
        ..., description="Result of applying procedure (SUCCESS, FAILED, PARTIAL)"
    )

    technologies: list[str] = Field(
        default_factory=list, description="Technologies involved"
    )

    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic similarity to query (0.0-1.0)"
    )

    created_at: datetime = Field(..., description="When procedure was created")

    linked_pattern_uuid: str | None = Field(
        default=None, description="UUID of linked pattern from Feature 026"
    )

    linked_pattern_name: str | None = Field(
        default=None, description="Name of linked pattern (if applicable)"
    )

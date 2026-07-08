"""Pydantic models for cross-session pattern recognition (Feature 026)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExtractedPattern(BaseModel):
    """
    Pattern extracted from session summary using LLM analysis.

    Represents a recurring approach to solving a specific problem type.
    """

    pattern_type: Literal[
        "infrastructure",
        "code",
        "architecture",
        "deployment",
        "data",
        "integration",
        "testing",
        "other",
    ] = Field(..., description="Category of problem being solved")

    problem_description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Description of the problem type",
    )

    solution_approach: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Approach or solution strategy attempted",
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence (0.0-1.0)"
    )

    outcome: Literal["SUCCESS", "FAILED", "PARTIAL", "INCONCLUSIVE"] = Field(
        default="INCONCLUSIVE", description="Result of applying this approach"
    )

    metrics: dict | None = Field(
        default=None,
        description="Optional metrics (e.g., {'startup_improvement': 40, 'unit': 'percent'})",
    )

    session_uuid: str | None = Field(default=None, description="UUID of source session")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class StoredPattern(BaseModel):
    """
    Pattern stored in Neo4j with full metadata.

    Represents an established pattern with occurrence tracking and success rates.
    """

    pattern_uuid: str = Field(..., description="UUID of the pattern in Neo4j")

    name: str = Field(
        ..., description="Human-readable pattern name (first 100 chars of solution)"
    )

    pattern_type: Literal[
        "infrastructure",
        "code",
        "architecture",
        "deployment",
        "data",
        "integration",
        "testing",
        "other",
    ] = Field(..., description="Category of problem")

    problem: str = Field(..., description="Problem description")

    solution: str = Field(..., description="Solution approach description")

    occurrence_count: int = Field(
        ..., ge=1, description="Number of times this pattern has occurred"
    )

    success_count: int = Field(
        default=0, ge=0, description="Number of successful applications"
    )

    failure_count: int = Field(
        default=0, ge=0, description="Number of failed applications"
    )

    success_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Success rate (0.0-1.0)"
    )

    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score of extraction"
    )

    consecutive_failures: int = Field(
        default=0, ge=0, description="Number of consecutive failures"
    )

    first_occurrence_date: datetime = Field(
        ..., description="When pattern first occurred"
    )

    latest_occurrence_date: datetime = Field(..., description="Most recent occurrence")

    created_at: datetime = Field(..., description="When pattern was created in Neo4j")

    updated_at: datetime = Field(..., description="When pattern was last updated")

    supporting_sessions: list[str] = Field(
        default_factory=list, description="UUIDs of sessions that exhibit this pattern"
    )

    archived: bool = Field(
        default=False, description="Whether pattern is archived (soft-delete)"
    )


class PatternMatch(BaseModel):
    """
    Pattern match result from query.

    Includes similarity score and relevance information for ranked results.
    """

    pattern_uuid: str = Field(..., description="UUID of matched pattern")

    name: str = Field(..., description="Pattern name")

    pattern_type: str = Field(..., description="Problem category")

    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Semantic similarity to query (if applicable)"
    )

    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Success rate (0.0-1.0)"
    )

    occurrence_count: int = Field(..., ge=1, description="Times pattern has occurred")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")

    relevance_explanation: str = Field(..., description="Why this pattern was matched")

    consecutive_failures: int = Field(
        default=0, ge=0, description="Consecutive failures (warning indicator)"
    )

    last_occurred: datetime | None = Field(
        default=None, description="Most recent occurrence"
    )

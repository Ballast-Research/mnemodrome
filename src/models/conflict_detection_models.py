"""Pydantic models for bidirectional conflict detection (Feature 025)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ExtractedClaim(BaseModel):
    """
    Claim extracted from user prompt or LLM response.

    Represents a factual assertion that can be validated against stored knowledge.
    """

    claim_type: Literal["assertion", "hypothetical", "example"] = Field(
        ..., description="Type of claim - only assertions are validated"
    )

    entity1: str = Field(..., description="Source entity in the relationship")
    relation: str = Field(..., description="Relationship type")
    entity2: str = Field(..., description="Target entity in the relationship")

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence (0.0-1.0)"
    )

    context: str = Field(default="", description="Surrounding context for the claim")

    sentence: str = Field(..., description="Original sentence containing the claim")

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {v}")
        return v


class StoredFact(BaseModel):
    """
    Fact retrieved from Neo4j for conflict validation.

    Represents existing knowledge that new claims are validated against.
    """

    entity1_uuid: str = Field(..., description="Source entity UUID in Neo4j")
    entity1_name: str = Field(..., description="Source entity name")

    relation: str = Field(..., description="Relationship type")

    entity2_uuid: str = Field(..., description="Target entity UUID in Neo4j")
    entity2_name: str = Field(..., description="Target entity name")

    created_at: datetime = Field(..., description="When the fact was created")
    context: str = Field(default="", description="Context for the stored fact")
    fact_uuid: str = Field(..., description="UUID of the relationship in Neo4j")

    is_temporal: bool = Field(
        default=False, description="Whether this fact has temporal validity fields"
    )


class ConflictResult(BaseModel):
    """
    Result of conflict validation between claim and stored fact.

    Includes severity scoring and recommendations for resolution.
    """

    conflict_detected: bool = Field(..., description="Whether a conflict was found")

    conflict_type: (
        Literal[
            "exact_contradiction", "partial_mismatch", "temporal_conflict", "stale_fact"
        ]
        | None
    ) = Field(default=None, description="Type of conflict detected")

    severity: int = Field(..., ge=0, le=100, description="Conflict severity (0-100)")

    recommendation: str = Field(
        ..., description="Recommended action (auto_correct, warn, block)"
    )

    correction_suggestion: str | None = Field(
        default=None,
        description="Suggested corrected text (if auto-correction applicable)",
    )

    stored_fact: StoredFact | None = Field(
        default=None, description="The conflicting stored fact"
    )

    claim: ExtractedClaim | None = Field(
        default=None, description="The claim that conflicts"
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: int) -> int:
        """Ensure severity is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError(f"Severity must be between 0 and 100, got {v}")
        return v


class CorrectionResult(BaseModel):
    """
    Result of automatic correction applied to text.

    Tracks original text, modified text, and all corrections applied.
    """

    action: Literal["corrected", "warned", "blocked", "skipped"] = Field(
        ..., description="Action taken on the text"
    )

    modified_text: str = Field(..., description="Text after corrections applied")

    original_text: str = Field(..., description="Original text before corrections")

    corrections_applied: list[dict] = Field(
        default_factory=list,
        description="List of corrections with (original, corrected, reason, confidence)",
    )

    warning_message: str | None = Field(
        default=None, description="Warning message for major conflicts"
    )

    conflicts_detected: int = Field(
        default=0, ge=0, description="Number of conflicts detected"
    )

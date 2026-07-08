"""Pydantic models for conflict detection results."""

from datetime import datetime

from pydantic import BaseModel, Field


class ConflictingFact(BaseModel):
    """
    A fact that conflicts with a new fact being added.
    """

    uuid: str = Field(description="UUID of the conflicting relationship")
    entity1: str = Field(description="Source entity name")
    relation: str = Field(description="Relationship type")
    entity2: str = Field(description="Target entity name")
    created_at: datetime = Field(description="When this fact was created")
    valid_at: datetime = Field(description="When this fact became valid")
    context: str | None = Field(
        default=None, description="Context for the relationship"
    )


class ConflictDetectionResult(BaseModel):
    """
    Result of conflict detection analysis.

    Returned when a new fact conflicts with existing facts in the knowledge graph.
    Includes complexity assessment to determine whether human validation is required.
    """

    status: str = Field(
        description="Status: 'no_conflict' | 'conflict_detected' | 'auto_resolved'"
    )
    conflict_detected: bool = Field(description="Whether conflicts were found")
    conflicting_facts: list[ConflictingFact] = Field(
        default_factory=list,
        description="List of existing facts that conflict with the new fact",
    )
    complexity_score: int = Field(
        ge=0,
        le=100,
        description="Conflict complexity score (0-100, higher = more complex)",
    )
    recommendation: str = Field(
        description="Recommended action: 'supersede' | 'co-exist' | 'cancel'"
    )
    rationale: str = Field(
        description="Explanation of why this recommendation was made"
    )
    auto_resolvable: bool = Field(
        description="Whether this conflict can be auto-resolved (score < 30)"
    )
    validation_required: bool = Field(
        description="Whether human validation is required (score >= automation_threshold)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "status": "conflict_detected",
                "conflict_detected": True,
                "conflicting_facts": [
                    {
                        "uuid": "abc123-def456",
                        "entity1": "Mnemodrome",
                        "relation": "RUNS_ON",
                        "entity2": "server-02",
                        "created_at": "2025-01-15T10:00:00Z",
                        "valid_at": "2025-01-15T10:00:00Z",
                        "context": "Production deployment",
                    }
                ],
                "complexity_score": 75,
                "recommendation": "supersede",
                "rationale": "Fact is >30 days old and has dependent relationships. Human validation recommended.",
                "auto_resolvable": False,
                "validation_required": True,
            }
        }


class ComplexityFactors(BaseModel):
    """
    Breakdown of complexity score factors.

    Used for transparency and debugging of complexity calculation.
    """

    age_factor: int = Field(
        ge=0, description="Points added for fact age (>30 days = +30)"
    )
    competing_facts_factor: int = Field(
        ge=0, description="Points added for multiple competing facts (+20)"
    )
    dependency_factor: int = Field(
        ge=0, description="Points added for cross-entity dependencies (+25)"
    )
    pattern_match_factor: int = Field(
        description="Points adjusted for pattern matching (±20)"
    )
    total_score: int = Field(
        ge=0, le=100, description="Total complexity score (clamped 0-100)"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "age_factor": 30,
                "competing_facts_factor": 20,
                "dependency_factor": 25,
                "pattern_match_factor": 0,
                "total_score": 75,
            }
        }

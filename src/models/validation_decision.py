"""Pydantic models for human validation decisions."""

from datetime import datetime

from pydantic import BaseModel, Field


class ValidationDecision(BaseModel):
    """
    Record of a human validation decision for conflict resolution.

    Used to build pattern history for progressive automation.
    Stored as ValidationHistory nodes in Neo4j.
    """

    uuid: str = Field(description="Unique identifier for this validation decision")
    conflict_type: str = Field(
        description="Type of conflict (e.g., 'same_relation_different_target')"
    )
    complexity_score: int = Field(
        ge=0, le=100, description="Complexity score at time of validation"
    )
    human_decision: str = Field(
        description="Human decision: 'approve' | 'reject' | 'modify'"
    )
    timestamp: datetime = Field(description="When the validation decision was made")
    old_fact_uuid: str = Field(
        description="UUID of the fact being potentially invalidated"
    )
    new_fact_uuid: str | None = Field(
        default=None,
        description="UUID of the new fact (if decision was approve/modify)",
    )
    rationale: str | None = Field(
        default=None, description="Human explanation for the decision"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "uuid": "validation-123-456",
                "conflict_type": "same_relation_different_target",
                "complexity_score": 45,
                "human_decision": "approve",
                "timestamp": "2025-10-08T14:23:00Z",
                "old_fact_uuid": "fact-abc-123",
                "new_fact_uuid": "fact-def-456",
                "rationale": "Infrastructure was migrated, supersession is correct",
            }
        }


class ValidationPattern(BaseModel):
    """
    Recognized pattern from validation history analysis.

    Used for progressive automation to identify "simple" conflicts.
    """

    conflict_type: str = Field(description="Type of conflict pattern")
    approval_count: int = Field(
        ge=0, description="Number of times this pattern was approved"
    )
    rejection_count: int = Field(
        ge=0, description="Number of times this pattern was rejected"
    )
    modification_count: int = Field(
        ge=0, description="Number of times this pattern was modified"
    )
    average_complexity: float = Field(
        ge=0, le=100, description="Average complexity score for this pattern"
    )
    confidence_score: int = Field(
        ge=0,
        le=100,
        description="Confidence in auto-resolving (0-100, based on approval ratio)",
    )
    is_simple_pattern: bool = Field(
        description="Whether this is a simple pattern (10+ approvals, no rejections)"
    )

    @property
    def total_validations(self) -> int:
        """Total number of validations for this pattern."""
        return self.approval_count + self.rejection_count + self.modification_count

    @property
    def approval_ratio(self) -> float:
        """Ratio of approvals to total validations."""
        if self.total_validations == 0:
            return 0.0
        return self.approval_count / self.total_validations

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "conflict_type": "infrastructure_relocation",
                "approval_count": 15,
                "rejection_count": 0,
                "modification_count": 2,
                "average_complexity": 28.5,
                "confidence_score": 92,
                "is_simple_pattern": True,
            }
        }

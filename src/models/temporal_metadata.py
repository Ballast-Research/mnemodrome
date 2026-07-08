"""Pydantic models for temporal validity metadata."""

from datetime import datetime

from pydantic import BaseModel, Field


class TemporalMetadata(BaseModel):
    """
    Temporal validity metadata for relationships in the knowledge graph.

    Follows Graphiti's complete three-field temporal model:
    - valid_at: When the fact became true in the real world
    - invalid_at: When the fact stopped being true naturally
    - expired_at: When the system invalidated the fact

    NULL semantics: NULL = currently valid
    """

    valid_at: datetime = Field(
        description="When this fact became valid (real-world start time)"
    )
    invalid_at: datetime | None = Field(
        default=None,
        description="When this fact became invalid naturally (real-world end time)",
    )
    expired_at: datetime | None = Field(
        default=None,
        description="When the system invalidated this fact (administrative invalidation)",
    )
    invalidation_reason: str | None = Field(
        default=None, description="Human-readable reason why this fact was invalidated"
    )
    superseded_by: str | None = Field(
        default=None, description="UUID of the relationship that replaced this one"
    )
    complexity_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Complexity score (0-100) for conflict resolution",
    )
    validation_decision: str | None = Field(
        default=None,
        description="Human validation decision: 'approve' | 'reject' | 'modify'",
    )

    @property
    def is_current(self) -> bool:
        """
        Check if this fact is currently valid.

        A fact is current if both invalid_at and expired_at are NULL.
        """
        return self.invalid_at is None and self.expired_at is None

    @property
    def invalidation_type(self) -> str | None:
        """
        Determine the type of invalidation.

        Returns:
            'natural' if only invalid_at is set
            'system' if only expired_at is set
            'both' if both are set
            None if neither is set (currently valid)
        """
        if self.invalid_at is not None and self.expired_at is not None:
            return "both"
        elif self.invalid_at is not None:
            return "natural"
        elif self.expired_at is not None:
            return "system"
        return None

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "valid_at": "2025-01-15T10:00:00Z",
                "invalid_at": "2025-09-30T17:00:00Z",
                "expired_at": None,
                "invalidation_reason": "Infrastructure migrated to new server",
                "superseded_by": "abc123-def456-...",
                "complexity_score": 25,
                "validation_decision": "approve",
            }
        }

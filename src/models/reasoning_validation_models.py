"""
Pydantic models for reasoning-time knowledge validation (Feature 028).

Defines the contract between ReasoningValidationService and the validate_reasoning MCP tool.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LearningReference(BaseModel):
    """A relevant learning from Mnemodrome's vector store."""

    uuid: str
    title: str
    similarity_score: float
    created_at: datetime | None = None
    excerpt: str = ""


class PatternReference(BaseModel):
    """A relevant pattern from Mnemodrome's pattern store."""

    pattern_uuid: str
    name: str
    pattern_type: str
    similarity_score: float
    success_rate: float
    occurrence_count: int
    relevance_explanation: str


class ConflictSummary(BaseModel):
    """Simplified conflict for the validation result output."""

    conflict_type: str
    severity: int
    claim: str
    stored_fact: str
    recommendation: str
    correction_suggestion: str | None = None


class ReasoningValidationResult(BaseModel):
    """
    Result from validate_reasoning MCP tool.

    Always returned (never None). Degradation flags indicate partial results.
    """

    validation_status: Literal["clean", "reinforced", "conflicts_detected", "degraded"]
    conflicts: list[ConflictSummary] = Field(default_factory=list)
    relevant_learnings: list[LearningReference] = Field(default_factory=list)
    relevant_patterns: list[PatternReference] = Field(default_factory=list)
    claims_analyzed: int = 0
    entities_checked: int = 0
    elapsed_ms: float = 0.0
    degraded_queries: list[str] = Field(default_factory=list)
    summary: str = ""

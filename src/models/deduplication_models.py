"""Pydantic models for knowledge layer deduplication (Feature 023)."""

from typing import Literal

from pydantic import BaseModel, Field


class SimilarityMatch(BaseModel):
    """A similar learning found during validation."""

    uuid: str = Field(description="Neo4j UUID of the similar learning")

    title: str = Field(description="Title of the similar learning")

    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Semantic similarity score [0.0-1.0]"
    )

    created_at: str = Field(description="ISO 8601 timestamp when learning was created")

    file_path: str = Field(description="Absolute path to markdown file")

    category: str | None = Field(
        default=None, description="Learning category (insight, solution, pattern, etc.)"
    )

    excerpt: str | None = Field(
        default=None, description="First 200 characters of content for preview"
    )


class ValidationResult(BaseModel):
    """Result of pre-capture validation for a learning."""

    status: Literal["proceed", "auto_skipped", "validation_required"] = Field(
        description="Validation decision: proceed (unique), auto_skipped (duplicate), validation_required (review needed)"
    )

    # Optional fields populated based on status
    existing_learning_uuid: str | None = Field(
        default=None, description="UUID of existing similar learning (if found)"
    )

    similarity_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic similarity score [0.0-1.0] to existing learning",
    )

    conflict_type: Literal["title_match", "semantic_similarity"] | None = Field(
        default=None,
        description="Type of conflict detected: exact title or semantic similarity",
    )

    existing_items: list[SimilarityMatch] | None = Field(
        default=None,
        description="List of similar learnings found (for validation_required status)",
    )

    recommendation: Literal["append", "skip", "force_create"] | None = Field(
        default=None, description="Recommended action for user"
    )

    rationale: str | None = Field(
        default=None, description="Human-readable explanation of validation decision"
    )

    options: dict | None = Field(
        default=None, description="Available actions with descriptions"
    )

    # Telemetry
    validation_stage: str = Field(
        description="Which stage produced result: title_exact | vector_similarity | fuzzy_fallback"
    )

    validation_time_ms: float = Field(
        description="Time taken for validation in milliseconds"
    )


class AppendResult(BaseModel):
    """Result of appending content to an existing learning."""

    status: Literal["success", "error"] = Field(description="Operation outcome")

    action: str = Field(description="Action taken: appended | error_occurred")

    learning_uuid: str = Field(description="UUID of learning that was updated")

    file_path: str = Field(description="Path to updated markdown file")

    append_count: int = Field(
        ge=0,
        description="Total number of times content has been appended to this learning",
    )

    timestamp: str = Field(description="ISO 8601 timestamp of append operation")

    new_section_title: str | None = Field(
        default=None, description="Title of newly appended section"
    )

    embedding_updated: bool = Field(
        default=False,
        description="Whether Qdrant embedding was regenerated with combined content",
    )

    error: str | None = Field(default=None, description="Error message if status=error")


class BootstrapResult(BaseModel):
    """Result of bootstrap script execution."""

    total_files: int = Field(ge=0, description="Total learning files found")

    successful: int = Field(
        ge=0, description="Number of learnings successfully embedded"
    )

    failed: int = Field(ge=0, description="Number of learnings that failed to embed")

    skipped: int = Field(
        ge=0, description="Number of learnings skipped (already exist in Qdrant)"
    )

    duration_seconds: float = Field(
        ge=0.0, description="Total execution time in seconds"
    )

    dry_run: bool = Field(description="Whether this was a dry run (no actual writes)")

    failed_files: list[str] = Field(
        default_factory=list, description="Paths to files that failed processing"
    )

    error_summary: dict = Field(
        default_factory=dict, description="Error types and counts: {error_type: count}"
    )

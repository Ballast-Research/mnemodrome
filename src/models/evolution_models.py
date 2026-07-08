"""Pydantic models for Evolving Strategies (Feature 032)."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvolutionCandidate(BaseModel):
    """
    Proposed instruction change backed by pattern evidence.

    Represents a specific change to a SKILL.md file that is supported by
    high-confidence cross-session patterns.
    """

    uuid: str = Field(..., description="Unique identifier for this candidate")

    skill_path: str = Field(..., description="Absolute path to target SKILL.md file")

    skill_name: str = Field(
        ..., description="Human-readable skill name (extracted from path)"
    )

    proposed_diff: str = Field(..., description="Unified diff of proposed change")

    rationale: str = Field(
        ..., min_length=10, max_length=1000, description="Why this change is proposed"
    )

    pattern_uuids: list[str] = Field(
        min_length=1,
        description="Evidence: Pattern UUIDs supporting this change (min 1 required)",
    )

    pattern_summary: str = Field(
        ..., description="Human-readable summary of pattern evidence"
    )

    status: Literal["pending", "approved", "rejected", "skipped", "applied"] = Field(
        default="pending", description="Current status of the candidate"
    )

    rejection_reason: str | None = Field(
        default=None, description="If rejected, why the change was not applied"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite confidence from supporting patterns (0.0-1.0)",
    )

    occurrence_count: int = Field(
        ..., ge=1, description="Total occurrences across cited patterns"
    )

    success_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weighted success rate across cited patterns (0.0-1.0)",
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When this candidate was generated"
    )

    reviewed_at: datetime | None = Field(
        default=None, description="When human reviewed this candidate"
    )

    applied_at: datetime | None = Field(
        default=None, description="When the change was applied to the file"
    )

    reviewed_by: str | None = Field(
        default=None, description="Who reviewed this candidate (e.g., 'user')"
    )

    @field_validator("skill_path")
    @classmethod
    def validate_skill_path(cls, v: str) -> str:
        """Validate that path is within ~/.claude/skills/ directory.

        Handles symlinked /home directory (issue #43) by:
        1. Expanding ~ and resolving both paths
        2. Normalizing to resolve symlinks consistently
        """
        # Expand ~ to home directory and resolve symlinks
        path = Path(v).expanduser().resolve()

        # Get skills root directory - could be in user's home or via symlink
        skills_root = Path("~/.claude/skills/").expanduser().resolve()

        # Check path is within skills directory
        try:
            path.relative_to(skills_root)
        except ValueError:
            raise ValueError(
                f"skill_path must be within ~/.claude/skills/. "
                f"Got: {v} (resolved to {path})"
            )

        # Check path doesn't contain forbidden patterns
        forbidden = ["CLAUDE.md", "context/", "hooks/", "src/", "/migrations/"]
        path_str = str(path)
        for pattern in forbidden:
            if pattern in path_str:
                raise ValueError(
                    f"skill_path contains forbidden pattern '{pattern}'. "
                    f"Only SKILL.md files in ~/.claude/skills/* are allowed."
                )

        return str(path)

    @field_validator("pattern_uuids")
    @classmethod
    def validate_pattern_uuids(cls, v: list[str]) -> list[str]:
        """Validate pattern UUIDs are non-empty strings."""
        if not v:
            raise ValueError("At least one pattern_uuid is required")

        for uuid in v:
            if not isinstance(uuid, str) or not uuid.strip():
                raise ValueError("All pattern_uuids must be non-empty strings")

        return v

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence score is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"confidence_score must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("success_rate")
    @classmethod
    def validate_success_rate(cls, v: float) -> float:
        """Ensure success rate is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"success_rate must be between 0.0 and 1.0, got {v}")
        return v


class EvolutionProposal(BaseModel):
    """
    Output from LLM comparison of pattern vs. skill instruction.

    Represents the LLM's assessment of whether a pattern suggests a change
    to current skill instructions.
    """

    gap_detected: bool = Field(
        ..., description="Whether LLM detected a gap between pattern and instruction"
    )

    gap_type: Literal["contradiction", "supplement", "refinement"] | None = Field(
        default=None,
        description="Type of gap (contradiction=conflicting, supplement=missing, refinement=improve)",
    )

    proposed_diff: str | None = Field(
        default=None, description="Unified diff if gap detected"
    )

    rationale: str | None = Field(
        default=None, description="Explanation of the gap and proposed fix"
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence in this proposal if gap detected",
    )


class EvolutionResult(BaseModel):
    """Result of applying an evolution candidate to a skill file."""

    success: bool = Field(
        ..., description="Whether the change was applied successfully"
    )

    candidate_uuid: str = Field(..., description="UUID of the applied/failed candidate")

    backup_path: str | None = Field(
        default=None, description="Path to backup file created before change"
    )

    error_message: str | None = Field(
        default=None, description="Error details if apply failed"
    )


class SkillInfo(BaseModel):
    """Metadata about a discovered SKILL.md file."""

    name: str = Field(
        ..., description="Skill name (directory name or derived from path)"
    )

    path: str = Field(..., description="Absolute path to SKILL.md file")

    size_bytes: int = Field(..., ge=0, description="File size in bytes")

    last_modified: datetime = Field(..., description="Last modification timestamp")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Validate path exists and is readable."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {v}")
        return str(path)

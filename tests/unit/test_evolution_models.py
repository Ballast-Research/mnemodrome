"""Unit tests for evolution models (Feature 032)."""

from datetime import datetime
from pathlib import Path

import pytest

from src.models.evolution_models import (
    EvolutionCandidate,
    EvolutionProposal,
    EvolutionResult,
    SkillInfo,
)


class TestEvolutionCandidate:
    """Test EvolutionCandidate model validation."""

    def test_valid_evolution_candidate(self):
        """Create valid evolution candidate."""

        candidate = EvolutionCandidate(
            uuid="test-uuid-123",
            skill_path=str(
                Path("~/.claude/skills/test-skill/SKILL.md").expanduser().resolve()
            ),
            skill_name="test-skill",
            proposed_diff="--- a/SKILL.md\n+++ b/SKILL.md\n@@ -1,1 +1,1 @@\n-old\n+new",
            rationale="This improves the skill based on pattern evidence",
            pattern_uuids=["pattern-1", "pattern-2"],
            pattern_summary="Test patterns with 100% success",
            confidence_score=0.95,
            occurrence_count=5,
            success_rate=1.0,
        )

        assert candidate.uuid == "test-uuid-123"
        assert candidate.skill_name == "test-skill"
        assert candidate.status == "pending"
        assert len(candidate.pattern_uuids) == 2

    def test_confidence_score_validation(self):
        """Validate confidence_score bounds."""

        with pytest.raises(ValueError):
            EvolutionCandidate(
                uuid="test",
                skill_path=str(
                    Path("~/.claude/skills/test/SKILL.md").expanduser().resolve()
                ),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=["p1"],
                pattern_summary="summary",
                confidence_score=1.5,  # Invalid: > 1.0
                occurrence_count=1,
                success_rate=0.8,
            )

    def test_success_rate_validation(self):
        """Validate success_rate bounds."""

        with pytest.raises(ValueError):
            EvolutionCandidate(
                uuid="test",
                skill_path=str(
                    Path("~/.claude/skills/test/SKILL.md").expanduser().resolve()
                ),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=["p1"],
                pattern_summary="summary",
                confidence_score=0.8,
                occurrence_count=1,
                success_rate=-0.5,  # Invalid: < 0.0
            )

    def test_path_validation_rejects_claude_md(self):
        """Reject CLAUDE.md paths."""

        # Create a path with CLAUDE.md within skills directory
        # The validator checks for forbidden patterns after checking path bounds
        with pytest.raises(ValueError, match="must be within"):
            EvolutionCandidate(
                uuid="test",
                skill_path=str(Path("~/.claude/CLAUDE.md").expanduser().resolve()),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=["p1"],
                pattern_summary="summary",
                confidence_score=0.8,
                occurrence_count=1,
                success_rate=0.8,
            )

    def test_path_validation_rejects_context_files(self):
        """Reject context files."""

        # Context files are outside skills directory
        with pytest.raises(ValueError, match="must be within"):
            EvolutionCandidate(
                uuid="test",
                skill_path=str(
                    Path("~/.claude/context/some-context.md").expanduser().resolve()
                ),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=["p1"],
                pattern_summary="summary",
                confidence_score=0.8,
                occurrence_count=1,
                success_rate=0.8,
            )

    def test_path_validation_rejects_hooks(self):
        """Reject hook files."""

        # Hook files are outside skills directory
        with pytest.raises(ValueError, match="must be within"):
            EvolutionCandidate(
                uuid="test",
                skill_path=str(
                    Path("~/.claude/hooks/something.ts").expanduser().resolve()
                ),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=["p1"],
                pattern_summary="summary",
                confidence_score=0.8,
                occurrence_count=1,
                success_rate=0.8,
            )

    def test_path_validation_accepts_skill_md(self):
        """Accept valid SKILL.md paths."""

        candidate = EvolutionCandidate(
            uuid="test",
            skill_path=str(
                Path("~/.claude/skills/my-skill/SKILL.md").expanduser().resolve()
            ),
            skill_name="my-skill",
            proposed_diff="diff content",
            rationale="Test rationale for changes",
            pattern_uuids=["p1"],
            pattern_summary="summary",
            confidence_score=0.8,
            occurrence_count=1,
            success_rate=0.8,
        )

        assert "my-skill" in candidate.skill_path

    def test_pattern_uuids_min_length_validation(self):
        """Validate pattern_uuids has at least 1 entry."""

        with pytest.raises(ValueError):
            EvolutionCandidate(
                uuid="test",
                skill_path=str(
                    Path("~/.claude/skills/test/SKILL.md").expanduser().resolve()
                ),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=[],  # Empty - invalid
                pattern_summary="summary",
                confidence_score=0.8,
                occurrence_count=1,
                success_rate=0.8,
            )

    def test_status_enum_validation(self):
        """Validate status enum values."""

        for status in ["pending", "approved", "rejected", "skipped", "applied"]:
            candidate = EvolutionCandidate(
                uuid="test",
                skill_path=str(
                    Path("~/.claude/skills/test/SKILL.md").expanduser().resolve()
                ),
                skill_name="test",
                proposed_diff="diff content",
                rationale="Test rationale for changes",
                pattern_uuids=["p1"],
                pattern_summary="summary",
                status=status,
                confidence_score=0.8,
                occurrence_count=1,
                success_rate=0.8,
            )
            assert candidate.status == status


class TestEvolutionProposal:
    """Test EvolutionProposal model."""

    def test_valid_proposal_with_gap(self):
        """Create proposal with gap detected."""
        proposal = EvolutionProposal(
            gap_detected=True,
            gap_type="contradiction",
            proposed_diff="--- a/\n+++ b/",
            rationale="Pattern contradicts current instruction",
            confidence=0.95,
        )

        assert proposal.gap_detected is True
        assert proposal.gap_type == "contradiction"
        assert proposal.confidence == 0.95

    def test_proposal_without_gap(self):
        """Create proposal without gap."""
        proposal = EvolutionProposal(
            gap_detected=False,
        )

        assert proposal.gap_detected is False
        assert proposal.proposed_diff is None
        assert proposal.rationale is None

    def test_gap_type_enum_validation(self):
        """Validate gap_type enum."""
        for gap_type in ["contradiction", "supplement", "refinement"]:
            proposal = EvolutionProposal(
                gap_detected=True,
                gap_type=gap_type,
                proposed_diff="diff",
                rationale="reason",
                confidence=0.8,
            )
            assert proposal.gap_type == gap_type


class TestEvolutionResult:
    """Test EvolutionResult model."""

    def test_successful_result(self):
        """Create successful evolution result."""
        result = EvolutionResult(
            success=True,
            candidate_uuid="test-uuid",
            backup_path="/path/to/backup.bak",
        )

        assert result.success is True
        assert result.candidate_uuid == "test-uuid"
        assert result.error_message is None

    def test_failed_result(self):
        """Create failed evolution result."""
        result = EvolutionResult(
            success=False,
            candidate_uuid="test-uuid",
            error_message="Diff application failed",
        )

        assert result.success is False
        assert result.error_message == "Diff application failed"


class TestSkillInfo:
    """Test SkillInfo model."""

    def test_valid_skill_info(self, tmp_path):
        """Create valid skill info."""
        # Create temporary file
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# Test Skill")

        skill_info = SkillInfo(
            name="test-skill",
            path=str(skill_file),
            size_bytes=100,
            last_modified=datetime.utcnow(),
        )

        assert skill_info.name == "test-skill"
        assert skill_info.size_bytes == 100

    def test_path_validation_rejects_nonexistent(self):
        """Reject non-existent path."""
        with pytest.raises(ValueError, match="does not exist"):
            SkillInfo(
                name="test",
                path="/nonexistent/path/SKILL.md",
                size_bytes=100,
                last_modified=datetime.utcnow(),
            )

    def test_path_validation_rejects_directory(self, tmp_path):
        """Reject directory path."""
        dir_path = tmp_path / "directory"
        dir_path.mkdir()

        with pytest.raises(ValueError, match="not a file"):
            SkillInfo(
                name="test",
                path=str(dir_path),
                size_bytes=100,
                last_modified=datetime.utcnow(),
            )

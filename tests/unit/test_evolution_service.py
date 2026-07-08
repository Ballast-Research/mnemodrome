"""Unit tests for evolution candidate service (Feature 032)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models.evolution_models import (
    EvolutionCandidate,
    EvolutionProposal,
)
from src.services.evolution_candidate_service import EvolutionCandidateService


@pytest.mark.asyncio
class TestEvolutionCandidateService:
    """Test EvolutionCandidateService."""

    @pytest.fixture
    def service(self):
        """Create service instance with mocked dependencies."""
        service = EvolutionCandidateService()

        # Mock dependencies
        service.pattern_query_service = AsyncMock()
        service.skill_file_loader = AsyncMock()
        service.neo4j_service = AsyncMock()
        service.llm_service = AsyncMock()

        return service

    async def test_generate_candidates_feature_disabled(self, service):
        """Return empty when evolution disabled."""
        service.enabled = False

        result = await service.generate_candidates()

        assert result["candidates"] == []
        assert result["candidates_generated"] == 0

    async def test_generate_candidates_no_patterns(self, service):
        """Handle no qualifying patterns."""
        service.enabled = True
        service.pattern_query_service.query_patterns.return_value = []

        result = await service.generate_candidates()

        assert result["candidates"] == []
        assert result["patterns_evaluated"] == 0

    async def test_generate_candidates_no_skills(self, service):
        """Handle no discovered skills."""
        service.enabled = True

        # Mock pattern query to return patterns
        mock_pattern = MagicMock()
        mock_pattern.pattern_uuid = "pattern-1"
        mock_pattern.occurrence_count = 5
        mock_pattern.success_rate = 0.95
        mock_pattern.problem_description = "Test problem"
        mock_pattern.solution_approach = "Test solution"

        service.pattern_query_service.query_patterns.return_value = [mock_pattern]
        service.skill_file_loader.discover_skills.return_value = []

        result = await service.generate_candidates()

        assert result["candidates"] == []
        assert result["skills_analyzed"] == 0

    async def test_generate_candidates_returns_structure(self, service):
        """Verify returned structure includes required fields."""
        service.enabled = True
        service.pattern_query_service.query_patterns.return_value = []

        result = await service.generate_candidates()

        assert "candidates" in result
        assert "skills_analyzed" in result
        assert "patterns_evaluated" in result
        assert "candidates_generated" in result

    async def test_compare_pattern_to_skill_with_gap(self, service):
        """LLM detects gap between pattern and skill."""
        pattern = MagicMock()
        pattern.problem_description = "Test problem"
        pattern.solution_approach = "Test solution"
        pattern.success_rate = 0.95
        pattern.occurrence_count = 5
        skill_content = "# Test Skill\nUse Haiku for analysis"
        skill = MagicMock()
        skill.name = "test"

        # Mock LLM response
        llm_response = """{
            "gap_detected": true,
            "gap_type": "contradiction",
            "proposed_change": "Use Sonnet instead of Haiku",
            "rationale": "Pattern shows Sonnet works better",
            "confidence": 0.95
        }"""

        service.llm_service._call_llm = AsyncMock(return_value=llm_response)

        proposal = await service._compare_pattern_to_skill(
            pattern, skill_content, skill
        )

        assert proposal is not None
        assert proposal.gap_detected is True
        assert proposal.gap_type == "contradiction"
        assert proposal.confidence == 0.95

    async def test_compare_pattern_to_skill_no_gap(self, service):
        """LLM detects no gap."""
        pattern = MagicMock()
        pattern.problem_description = "Test problem"
        pattern.solution_approach = "Test solution"
        pattern.success_rate = 0.95
        pattern.occurrence_count = 5
        skill_content = "# Test Skill"
        skill = MagicMock()

        llm_response = """{
            "gap_detected": false,
            "gap_type": null,
            "proposed_change": null,
            "rationale": null,
            "confidence": null
        }"""

        service.llm_service._call_llm = AsyncMock(return_value=llm_response)

        proposal = await service._compare_pattern_to_skill(
            pattern, skill_content, skill
        )

        assert proposal is not None
        assert proposal.gap_detected is False

    async def test_compare_pattern_invalid_json_response(self, service):
        """Handle invalid JSON from LLM."""
        pattern = MagicMock()
        skill_content = "# Test"
        skill = MagicMock()

        service.llm_service._call_llm.return_value = "Not valid JSON"

        proposal = await service._compare_pattern_to_skill(
            pattern, skill_content, skill
        )

        assert proposal is None

    async def test_compare_pattern_llm_error(self, service):
        """Handle LLM service error."""
        pattern = MagicMock()
        skill_content = "# Test"
        skill = MagicMock()

        service.llm_service._call_llm.side_effect = Exception("LLM error")

        proposal = await service._compare_pattern_to_skill(
            pattern, skill_content, skill
        )

        assert proposal is None

    async def test_create_candidate_from_proposal_valid(self, service):
        """Create candidate from valid proposal."""
        from pathlib import Path

        proposal = EvolutionProposal(
            gap_detected=True,
            gap_type="contradiction",
            proposed_diff="--- a/\n+++ b/",
            rationale="Test rationale for changes",
            confidence=0.95,
        )

        pattern = MagicMock()
        pattern.pattern_uuid = "pattern-1"
        pattern.name = "Test pattern"
        pattern.success_rate = 0.95
        pattern.occurrence_count = 5

        skill = MagicMock()
        skill.name = "test-skill"
        # Use expanduser to create valid path within ~/.claude/skills/
        skill.path = str(
            Path("~/.claude/skills/test-skill/SKILL.md").expanduser().resolve()
        )

        candidate = await service._create_candidate_from_proposal(
            proposal, pattern, skill
        )

        assert candidate is not None
        assert candidate.skill_name == "test-skill"
        assert candidate.proposed_diff == "--- a/\n+++ b/"
        assert len(candidate.pattern_uuids) == 1

    async def test_create_candidate_no_diff(self, service):
        """Skip candidate when no diff in proposal."""
        proposal = EvolutionProposal(
            gap_detected=True,
            gap_type="contradiction",
            proposed_diff=None,  # No diff
            rationale="Test",
            confidence=0.95,
        )

        pattern = MagicMock()
        skill = MagicMock()

        candidate = await service._create_candidate_from_proposal(
            proposal, pattern, skill
        )

        assert candidate is None

    async def test_apply_candidate_success(self, service):
        """Successfully apply an evolution candidate."""
        from pathlib import Path

        candidate = EvolutionCandidate(
            uuid="test-candidate",
            skill_path=str(
                Path("~/.claude/skills/test/SKILL.md").expanduser().resolve()
            ),
            skill_name="test",
            proposed_diff="diff content here",
            rationale="Test rationale for changes",
            pattern_uuids=["p1"],
            pattern_summary="summary",
            confidence_score=0.8,
            occurrence_count=1,
            success_rate=0.8,
        )

        # Mock service methods
        service._load_candidate_from_neo4j = AsyncMock(return_value=candidate)
        service.skill_file_loader.validate_path = MagicMock()
        service.skill_file_loader.create_backup = AsyncMock(
            return_value="/backup/path.bak"
        )
        service.skill_file_loader.apply_diff_to_file = AsyncMock()
        service.skill_file_loader.cleanup_old_backups = AsyncMock(return_value=0)
        service._update_candidate_status = AsyncMock()

        result = await service.apply_candidate("test-candidate")

        assert result.success is True
        assert result.backup_path == "/backup/path.bak"

    async def test_apply_candidate_not_found(self, service):
        """Handle candidate not found."""
        service._load_candidate_from_neo4j = AsyncMock(return_value=None)

        result = await service.apply_candidate("nonexistent")

        assert result.success is False
        assert "not found" in result.error_message.lower()

    async def test_apply_candidate_not_pending(self, service):
        """Reject non-pending candidate."""
        from pathlib import Path

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
            status="applied",  # Not pending
            confidence_score=0.8,
            occurrence_count=1,
            success_rate=0.8,
        )

        service._load_candidate_from_neo4j = AsyncMock(return_value=candidate)

        result = await service.apply_candidate("test")

        assert result.success is False
        assert "not pending" in result.error_message

    async def test_apply_candidate_invalid_path(self, service):
        """Reject candidate with invalid path."""
        from pathlib import Path

        from src.services.skill_file_loader import EvolutionError

        # Create candidate with a path that will fail validation
        # Use a path outside skills directory (invalid)
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
            confidence_score=0.8,
            occurrence_count=1,
            success_rate=0.8,
        )

        service._load_candidate_from_neo4j = AsyncMock(return_value=candidate)

        # Mock validate_path to raise an error (it's synchronous, not async)
        service.skill_file_loader.validate_path = MagicMock(
            side_effect=EvolutionError("Invalid path")
        )

        result = await service.apply_candidate("test")

        assert result.success is False

    async def test_apply_candidate_backup_fails(self, service):
        """Handle backup creation failure."""
        from pathlib import Path

        from src.services.skill_file_loader import EvolutionError

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
            confidence_score=0.8,
            occurrence_count=1,
            success_rate=0.8,
        )

        service._load_candidate_from_neo4j = AsyncMock(return_value=candidate)
        service.skill_file_loader.validate_path = MagicMock()

        service.skill_file_loader.create_backup.side_effect = EvolutionError(
            "Backup failed"
        )

        result = await service.apply_candidate("test")

        assert result.success is False
        assert "backup" in result.error_message.lower()

    async def test_reject_candidate(self, service):
        """Reject a candidate with reason."""
        service._update_candidate_status = AsyncMock()

        result = await service.reject_candidate("test-uuid", "Not applicable")

        assert result.success is True
        service._update_candidate_status.assert_called_once()

    async def test_skip_candidate(self, service):
        """Skip a candidate for later review."""
        service._update_candidate_status = AsyncMock()

        result = await service.skip_candidate("test-uuid")

        assert result.success is True
        service._update_candidate_status.assert_called_once()

    async def test_load_candidate_from_neo4j_found(self, service):
        """Load candidate that exists in Neo4j."""
        from pathlib import Path

        skill_path = str(Path("~/.claude/skills/test/SKILL.md").expanduser().resolve())

        neo4j_data = {
            "ec": {
                "uuid": "test",
                "skill_path": skill_path,
                "skill_name": "test",
                "proposed_diff": "diff content",
                "rationale": "Test rationale for changes",
                "pattern_uuids": ["p1"],
                "pattern_summary": "summary",
                "status": "pending",
                "confidence_score": 0.8,
                "occurrence_count": 1,
                "success_rate": 0.8,
            }
        }

        service.neo4j_service.connect = AsyncMock()
        service.neo4j_service.query = AsyncMock(return_value=[neo4j_data])

        candidate = await service._load_candidate_from_neo4j("test")

        assert candidate is not None
        assert candidate.uuid == "test"

    async def test_load_candidate_from_neo4j_not_found(self, service):
        """Handle candidate not found in Neo4j."""
        service.neo4j_service.connect = AsyncMock()
        service.neo4j_service.query = AsyncMock(return_value=[])

        candidate = await service._load_candidate_from_neo4j("nonexistent")

        assert candidate is None

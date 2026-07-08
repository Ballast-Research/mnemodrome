"""Unit tests for skill file loader (Feature 032)."""

import os
from datetime import datetime
from pathlib import Path

import pytest

from src.services.skill_file_loader import EvolutionError, SkillFileLoader


class TestSkillFileLoader:
    """Test SkillFileLoader service."""

    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return SkillFileLoader()

    @pytest.fixture
    def mock_skills_dir(self, loader, tmp_path, monkeypatch):
        """Create mock skills directory structure."""
        skills_root = tmp_path / ".claude" / "skills"
        skills_root.mkdir(parents=True)

        # Create skill 1
        skill1_dir = skills_root / "skill-one"
        skill1_dir.mkdir()
        (skill1_dir / "SKILL.md").write_text("# Skill One\n\nSome content")

        # Create skill 2
        skill2_dir = skills_root / "skill-two"
        skill2_dir.mkdir()
        (skill2_dir / "SKILL.md").write_text("# Skill Two\n\nMore content")

        # Create hidden skill (should be skipped)
        hidden_dir = skills_root / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "SKILL.md").write_text("Hidden skill")

        # Monkeypatch the instance attribute on the loader instance
        monkeypatch.setattr(loader, "skills_root", skills_root)

        return skills_root

    @pytest.mark.asyncio
    async def test_discover_skills(self, loader, mock_skills_dir):
        """Discover all SKILL.md files."""
        skills = await loader.discover_skills()

        assert len(skills) == 2
        skill_names = {s.name for s in skills}
        assert "skill-one" in skill_names
        assert "skill-two" in skill_names

        # Verify metadata
        for skill in skills:
            assert skill.size_bytes > 0
            assert isinstance(skill.last_modified, datetime)

    @pytest.mark.asyncio
    async def test_discover_skills_excludes_hidden(self, loader, mock_skills_dir):
        """Exclude hidden directories from discovery."""
        skills = await loader.discover_skills()
        skill_names = {s.name for s in skills}

        assert ".hidden" not in skill_names

    @pytest.mark.asyncio
    async def test_discover_skills_empty_directory(self, loader, tmp_path, monkeypatch):
        """Handle empty skills directory."""
        empty_skills = tmp_path / ".claude" / "skills"
        empty_skills.mkdir(parents=True)

        monkeypatch.setattr(loader, "skills_root", empty_skills)

        skills = await loader.discover_skills()
        assert len(skills) == 0

    @pytest.mark.asyncio
    async def test_load_skill_content(self, loader, mock_skills_dir):
        """Load skill file content."""
        skill_path = str(mock_skills_dir / "skill-one" / "SKILL.md")
        content = await loader.load_skill_content(skill_path)

        assert "Skill One" in content
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_load_skill_content_file_not_found(
        self, loader, tmp_path, monkeypatch
    ):
        """Handle file not found error."""
        skills_root = tmp_path / ".claude" / "skills"
        skills_root.mkdir(parents=True)
        monkeypatch.setattr(loader, "skills_root", skills_root)

        # Create a path inside skills directory that doesn't exist
        nonexistent_path = str(skills_root / "nonexistent" / "SKILL.md")
        with pytest.raises(EvolutionError, match="not found"):
            await loader.load_skill_content(nonexistent_path)

    def test_validate_path_accepts_skill_md(self, loader, mock_skills_dir):
        """Accept SKILL.md paths."""
        skill_path = str(mock_skills_dir / "skill-one" / "SKILL.md")
        # Should not raise
        loader.validate_path(skill_path)

    def test_validate_path_rejects_outside_skills(self, loader):
        """Reject paths outside skills directory."""
        with pytest.raises(EvolutionError, match="outside skills directory"):
            loader.validate_path("/home/user/some/other/path.md")

    def test_validate_path_rejects_claude_md(self, loader):
        """Reject CLAUDE.md."""
        with pytest.raises(EvolutionError):
            loader.validate_path("/home/user/.claude/CLAUDE.md")

    def test_validate_path_rejects_context_files(self, loader):
        """Reject context files."""
        with pytest.raises(EvolutionError):
            loader.validate_path("/home/user/.claude/context/file.md")

    def test_validate_path_rejects_hooks(self, loader):
        """Reject hook files."""
        with pytest.raises(EvolutionError):
            loader.validate_path("/home/user/.claude/hooks/hook.ts")

    def test_validate_path_rejects_source_code(self, loader):
        """Reject source code files."""
        with pytest.raises(EvolutionError):
            loader.validate_path("/home/user/.claude/src/module.py")

    @pytest.mark.asyncio
    async def test_create_backup(self, loader, mock_skills_dir):
        """Create timestamped backup."""
        skill_path = str(mock_skills_dir / "skill-one" / "SKILL.md")

        backup_path = await loader.create_backup(skill_path)

        # Verify backup file exists
        backup = Path(backup_path)
        assert backup.exists()
        assert backup.parent == Path(skill_path).parent

        # Verify backup contains original content
        original_content = Path(skill_path).read_text()
        backup_content = backup.read_text()
        assert original_content == backup_content

    @pytest.mark.asyncio
    async def test_create_backup_file_not_found(self, loader, tmp_path, monkeypatch):
        """Handle non-existent file for backup."""
        skills_root = tmp_path / ".claude" / "skills"
        skills_root.mkdir(parents=True)
        monkeypatch.setattr(loader, "skills_root", skills_root)

        # Create a path inside skills directory that doesn't exist
        nonexistent_path = str(skills_root / "nonexistent" / "SKILL.md")
        with pytest.raises(EvolutionError, match="not found"):
            await loader.create_backup(nonexistent_path)

    @pytest.mark.asyncio
    async def test_cleanup_old_backups(self, loader, tmp_path, monkeypatch):
        """Remove old backups beyond retention period."""
        # Create mock skills dir
        skills_root = tmp_path / ".claude" / "skills"
        skills_root.mkdir(parents=True)

        skill_dir = skills_root / "test-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("content")

        monkeypatch.setattr(loader, "skills_root", skills_root)

        # Create some old backups manually
        old_backup = skill_dir / "SKILL.md.bak.2020-01-01T00-00-00.000000"
        old_backup.write_text("old content")
        # Set mtime to a date in the past (Jan 1, 2020)
        old_timestamp = 1577836800  # 2020-01-01 00:00:00 UTC
        os.utime(old_backup, (old_timestamp, old_timestamp))

        # Create new backup
        new_backup = (
            skill_dir
            / f"SKILL.md.bak.{datetime.utcnow().isoformat().replace(':', '-')}"
        )
        new_backup.write_text("new content")

        # Cleanup with 30-day retention period (so new backups are kept)
        deleted = await loader.cleanup_old_backups(str(skill_file), retention_days=30)

        # Old backup should be deleted
        assert not old_backup.exists()
        # New backup should remain
        assert new_backup.exists()
        assert deleted >= 1

    @pytest.mark.asyncio
    async def test_apply_diff_to_file(self, loader, mock_skills_dir):
        """Apply unified diff to file."""
        skill_path = str(mock_skills_dir / "skill-one" / "SKILL.md")

        # Create a simple diff (this is a simplified test)
        # In real usage, the diff would be generated by LLM
        original = Path(skill_path).read_text()

        # For this test, we'd need a proper diff library
        # Skip for now - this requires patching library installation
        pytest.skip("Requires patch library for diff application")

    @pytest.mark.asyncio
    async def test_apply_diff_invalid_path(self, loader):
        """Reject invalid path for diff application."""
        with pytest.raises(EvolutionError):
            await loader.apply_diff_to_file(
                "/home/user/invalid/path.md", "diff content"
            )

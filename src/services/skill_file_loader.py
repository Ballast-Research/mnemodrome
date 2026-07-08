"""
Skill File Loader Service - Discover, load, and manage skill files.

Responsible for:
- Discovering SKILL.md files in ~/.claude/skills/
- Loading skill content with path validation
- Creating and managing backups before file modifications
- Enforcing security constraints (SKILL.md scope only)
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

from loguru import logger

from src.models.evolution_models import SkillInfo


class EvolutionError(Exception):
    """Base exception for evolution-related errors."""

    pass


class SkillFileLoader:
    """
    Load and manage skill files with security validation.

    Enforces strict path validation to prevent modifications outside
    the configured skills directory.
    """

    def __init__(self):
        """Initialize skill file loader."""
        skills_path = os.getenv("MNEMODROME_SKILLS_PATH", "~/.claude/skills/")
        self.skills_root = Path(skills_path).expanduser().resolve()

        # Allowed file patterns within skills directory
        self.allowed_patterns = [
            "SKILL.md",
            "*.md",  # Allow other markdown files in skill directories
        ]

        logger.debug(f"SkillFileLoader initialized: skills_root={self.skills_root}")

    async def discover_skills(self) -> list[SkillInfo]:
        """
        Discover all SKILL.md files in ~/.claude/skills/ directory.

        Returns:
            List of SkillInfo objects with metadata about discovered skills

        Raises:
            EvolutionError: If skills directory not found or not readable
        """
        if not self.skills_root.exists():
            logger.warning(f"Skills directory not found: {self.skills_root}")
            return []

        skills = []

        try:
            # Find all SKILL.md files (excluding hidden directories)
            for skill_md in sorted(self.skills_root.glob("*/SKILL.md")):
                # Skip hidden skill directories (directories that start with a dot)
                skill_name = skill_md.parent.name
                if skill_name.startswith("."):
                    continue

                # Get file metadata
                stat = skill_md.stat()
                last_modified = datetime.fromtimestamp(stat.st_mtime)

                skill_info = SkillInfo(
                    name=skill_name,
                    path=str(skill_md.resolve()),
                    size_bytes=stat.st_size,
                    last_modified=last_modified,
                )

                skills.append(skill_info)
                logger.debug(f"Discovered skill: {skill_name} ({skill_md})")

            logger.info(f"Discovered {len(skills)} skills in {self.skills_root}")
            return skills

        except Exception as e:
            logger.error(f"Failed to discover skills: {e}")
            raise EvolutionError(f"Skill discovery failed: {e}")

    async def load_skill_content(self, skill_path: str) -> str:
        """
        Load skill file content with path validation.

        Args:
            skill_path: Path to skill file (will validate before loading)

        Returns:
            File content as string

        Raises:
            EvolutionError: If path is invalid or file cannot be read
        """
        # Validate path first
        self.validate_path(skill_path)

        path = Path(skill_path)

        try:
            content = path.read_text(encoding="utf-8")
            logger.debug(f"Loaded skill content: {skill_path} ({len(content)} bytes)")
            return content
        except FileNotFoundError:
            raise EvolutionError(f"Skill file not found: {skill_path}")
        except PermissionError:
            raise EvolutionError(f"Permission denied reading skill file: {skill_path}")
        except Exception as e:
            raise EvolutionError(f"Failed to load skill content: {e}")

    def validate_path(self, skill_path: str) -> None:
        """
        Validate that path is within ~/.claude/skills/ and safe to modify.

        Args:
            skill_path: Path to validate

        Raises:
            EvolutionError: If path is invalid or outside allowed scope
        """
        # Expand and resolve path
        path = Path(skill_path).expanduser().resolve()

        # Check path is within skills directory
        try:
            path.relative_to(self.skills_root)
        except ValueError:
            raise EvolutionError(
                f"Path is outside skills directory: {skill_path} "
                f"(must be within {self.skills_root})"
            )

        # Reject forbidden patterns
        forbidden = [
            "CLAUDE.md",
            "/.claude/context/",
            "/.claude/hooks/",
            "/.claude/projects/",
            "/src/",
            "/migrations/",
            "/tests/",
        ]

        path_str = str(path)
        for pattern in forbidden:
            if pattern.lstrip("/") in path_str:
                raise EvolutionError(
                    f"Path contains forbidden pattern. "
                    f"Only SKILL.md files in ~/.claude/skills/* are allowed. "
                    f"Got: {skill_path}"
                )

        logger.debug(f"Path validation passed: {skill_path}")

    async def create_backup(self, skill_path: str) -> str:
        """
        Create timestamped backup of skill file.

        Creates file named SKILL.md.bak.{ISO_TIMESTAMP} in the same directory.

        Args:
            skill_path: Path to file to backup

        Returns:
            Path to backup file

        Raises:
            EvolutionError: If backup creation fails
        """
        # Validate path
        self.validate_path(skill_path)

        path = Path(skill_path)

        # Check file exists
        if not path.exists():
            raise EvolutionError(f"File not found, cannot create backup: {skill_path}")

        try:
            # Read original content
            content = path.read_text(encoding="utf-8")

            # Create backup path with timestamp
            timestamp = datetime.utcnow().isoformat().replace(":", "-")
            backup_path = path.parent / f"{path.name}.bak.{timestamp}"

            # Write backup
            backup_path.write_text(content, encoding="utf-8")

            logger.info(f"Created backup: {backup_path}")
            return str(backup_path.resolve())

        except Exception as e:
            raise EvolutionError(f"Failed to create backup: {e}")

    async def cleanup_old_backups(
        self, skill_path: str, retention_days: int = 30
    ) -> int:
        """
        Remove backup files older than retention period.

        Args:
            skill_path: Path to skill file (backups are in same directory)
            retention_days: Keep backups newer than this many days (default 30)

        Returns:
            Number of backups deleted

        Raises:
            EvolutionError: If cleanup fails
        """
        path = Path(skill_path)
        backup_dir = path.parent
        backup_pattern = f"{path.name}.bak.*"

        try:
            deleted = 0
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)

            for backup_file in sorted(backup_dir.glob(backup_pattern)):
                # Parse timestamp from filename
                # Format: SKILL.md.bak.2026-03-04T12-30-45.123456
                stat = backup_file.stat()
                file_time = datetime.fromtimestamp(stat.st_mtime)

                if file_time < cutoff_time:
                    backup_file.unlink()
                    deleted += 1
                    logger.debug(f"Deleted old backup: {backup_file}")

            if deleted > 0:
                logger.info(
                    f"Cleaned up {deleted} old backups (retention: {retention_days} days)"
                )

            return deleted

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            raise EvolutionError(f"Backup cleanup failed: {e}")

    async def apply_diff_to_file(self, skill_path: str, diff: str) -> None:
        """
        Apply a unified diff to a skill file.

        Args:
            skill_path: Path to file to modify
            diff: Unified diff string to apply

        Raises:
            EvolutionError: If diff cannot be applied
        """
        import re

        # Validate path
        self.validate_path(skill_path)

        path = Path(skill_path)

        try:
            current_content = path.read_text(encoding="utf-8")
            current_lines = current_content.splitlines(keepends=True)

            # Parse unified diff hunks and apply
            patched_lines = list(current_lines)
            offset = 0

            hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

            in_hunk = False
            hunk_start = 0
            removals = []
            additions = []

            for line in diff.splitlines(keepends=True):
                match = hunk_pattern.match(line)
                if match:
                    # Apply previous hunk if any
                    if in_hunk:
                        self._apply_hunk(
                            patched_lines, hunk_start + offset, removals, additions
                        )
                        offset += len(additions) - len(removals)

                    hunk_start = int(match.group(1)) - 1  # 0-indexed
                    removals = []
                    additions = []
                    in_hunk = True
                elif in_hunk:
                    if line.startswith("-"):
                        removals.append(line[1:])
                    elif line.startswith("+"):
                        additions.append(line[1:])
                    elif line.startswith(" "):
                        # Context line — flush pending changes
                        if removals or additions:
                            self._apply_hunk(
                                patched_lines, hunk_start + offset, removals, additions
                            )
                            offset += len(additions) - len(removals)
                            hunk_start += len(removals)
                            removals = []
                            additions = []
                        hunk_start += 1

            # Apply final hunk
            if in_hunk and (removals or additions):
                self._apply_hunk(
                    patched_lines, hunk_start + offset, removals, additions
                )

            patched = "".join(patched_lines)
            path.write_text(patched, encoding="utf-8")
            logger.info(f"Applied diff to {skill_path}")

        except EvolutionError:
            raise
        except Exception as e:
            raise EvolutionError(f"Failed to apply diff: {e}")

    @staticmethod
    def _apply_hunk(lines: list, start: int, removals: list, additions: list) -> None:
        """Apply a single diff hunk to the lines list in-place."""
        del lines[start : start + len(removals)]
        for i, add_line in enumerate(additions):
            lines.insert(start + i, add_line)

"""File storage service for markdown memory operations."""

from pathlib import Path

from loguru import logger

from src.config import MNEMODROME_STORAGE_PATH, MemoryType, get_memory_path


class FileService:
    """Service for file storage operations (markdown memory files)."""

    def __init__(self):
        """
        Initialize file service with memory-type-aware paths.

        Feature 010: Caches all memory type paths and creates directories.
        """
        # Cache memory type paths (must include ALL MemoryType enum values)
        self.memory_paths: dict[MemoryType, Path] = {
            MemoryType.FACT: get_memory_path(MemoryType.FACT),
            MemoryType.LEARNING: get_memory_path(MemoryType.LEARNING),
            MemoryType.SESSION: get_memory_path(MemoryType.SESSION),
            MemoryType.GRAPH: get_memory_path(MemoryType.GRAPH),
            MemoryType.UPDATE: get_memory_path(MemoryType.UPDATE),
            MemoryType.PROCEDURE: get_memory_path(MemoryType.PROCEDURE),
        }

        # Create directories for all memory types
        for memory_type, path in self.memory_paths.items():
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Initialized {memory_type.value} directory: {path}")

        # Backward compatibility: keep storage_path for legacy code
        self.storage_path = (
            MNEMODROME_STORAGE_PATH
            if MNEMODROME_STORAGE_PATH
            else self.memory_paths[MemoryType.FACT]
        )

    def _get_target_path(self, memory_type: MemoryType) -> Path:
        """
        Get the target directory path for a specific memory type.

        Args:
            memory_type: Type of memory (FACT, LEARNING, SESSION, GRAPH)

        Returns:
            Path to the memory type directory
        """
        return self.memory_paths[memory_type]

    def validate_path(self, memory_type: MemoryType = MemoryType.FACT) -> bool:
        """
        Validate that a memory type storage path exists and is accessible.

        Args:
            memory_type: Type of memory to validate (default: FACT)

        Returns:
            True if path is valid and accessible

        Raises:
            IOError if path is not accessible
        """
        target_path = self._get_target_path(memory_type)

        if not target_path.exists():
            raise OSError(
                f"{memory_type.value} storage path does not exist: {target_path}"
            )

        if not target_path.is_dir():
            raise OSError(
                f"{memory_type.value} storage path is not a directory: {target_path}"
            )

        # Test write access
        test_file = target_path / ".mnemodrome_access_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise OSError(
                f"{memory_type.value} storage path is not writable: {target_path}"
            ) from e

        logger.debug(f"{memory_type.value} storage path validated: {target_path}")
        return True

    def write_file(
        self, filename: str, content: str, memory_type: MemoryType = MemoryType.FACT
    ) -> Path:
        """
        Write a markdown file to memory-type-specific storage.

        Feature 010: Routes files to correct directory based on memory type.

        Args:
            filename: File name (will be sanitized)
            content: File content
            memory_type: Type of memory (default: FACT)

        Returns:
            Path to written file

        Raises:
            Exception if write fails (exception-based verification)
        """
        # Ensure .md extension
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # Get target directory for this memory type
        target_dir = self._get_target_path(memory_type)
        filepath = target_dir / filename

        try:
            filepath.write_text(content, encoding="utf-8")
            logger.info(f"Wrote {memory_type.value} file: {filepath}")
            return filepath
        except Exception as e:
            raise Exception(
                f"Failed to write {memory_type.value} file {filepath}: {e}"
            ) from e

    def delete_file(self, filepath: Path) -> bool:
        """
        Delete a file from storage.

        Args:
            filepath: Path to file

        Returns:
            True if deleted
        """
        try:
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Deleted file: {filepath}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to delete file {filepath}: {e}")
            return False


# Global service instance
file_service = FileService()

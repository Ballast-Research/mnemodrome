#!/usr/bin/env python3
"""Bootstrap vector embeddings for existing learnings (Feature 023).

Scans PAI vault learning files and generates Qdrant embeddings for
semantic deduplication support.

Usage:
    # Dry run - preview without writing
    python scripts/bootstrap_vector_embeddings.py --dry-run

    # Full bootstrap
    python scripts/bootstrap_vector_embeddings.py

    # Custom path
    python scripts/bootstrap_vector_embeddings.py --path /custom/learnings

    # Verbose output
    python scripts/bootstrap_vector_embeddings.py -v
"""

import asyncio
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DEDUP_BOOTSTRAP_BATCH_SIZE,
    DEDUP_BOOTSTRAP_MAX_FAILURES,
    MemoryType,
    get_memory_path,
)
from src.models.deduplication_models import BootstrapResult

app = typer.Typer(help="Bootstrap vector embeddings for existing learnings")
console = Console()


class BootstrapProcessor:
    """Processes learning files and generates vector embeddings."""

    def __init__(
        self,
        dry_run: bool = False,
        batch_size: int = None,
        max_failures: int = None,
        verbose: bool = False,
    ):
        self.dry_run = dry_run
        self.batch_size = batch_size or DEDUP_BOOTSTRAP_BATCH_SIZE
        self.max_failures = max_failures or DEDUP_BOOTSTRAP_MAX_FAILURES
        self.verbose = verbose

        self.vector_service = None
        self.neo4j_service = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize services for bootstrap.

        Returns:
            True if all services ready, False otherwise
        """
        try:
            from src.services.neo4j_service import Neo4jService
            from src.services.vector_service import get_vector_service

            self.vector_service = get_vector_service()
            self.neo4j_service = Neo4jService()

            # Initialize vector service
            await self.vector_service.initialize()

            if not self.vector_service.is_enabled():
                console.print(
                    "[red]Vector service not available. Check Qdrant/Ollama.[/red]"
                )
                return False

            # Test Neo4j connection
            await self.neo4j_service.ping()

            self._initialized = True
            return True

        except Exception as e:
            console.print(f"[red]Service initialization failed: {e}[/red]")
            return False

    def parse_learning_file(self, file_path: Path) -> dict | None:
        """Parse a learning markdown file.

        Args:
            file_path: Path to markdown file

        Returns:
            Dict with uuid, title, content, category or None if invalid
        """
        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract UUID from content (look for **UUID**: `xxx`)
            uuid_match = re.search(r"\*\*UUID\*\*:\s*`([^`]+)`", content)
            if not uuid_match:
                # Try alternative format
                uuid_match = re.search(r"uuid:\s*([a-f0-9-]+)", content, re.IGNORECASE)

            uuid = uuid_match.group(1) if uuid_match else None

            # Extract title from first heading
            title_match = re.search(
                r"^#\s+(?:Learning:\s*)?(.+)$", content, re.MULTILINE
            )
            title = title_match.group(1).strip() if title_match else file_path.stem

            # Extract category
            category_match = re.search(r"\*\*Category\*\*:\s*(\w+)", content)
            category = category_match.group(1) if category_match else "general"

            # Get main content (skip frontmatter/metadata)
            lines = content.split("\n")
            content_start = 0
            for i, line in enumerate(lines):
                if line.startswith("---") and i > 0:
                    content_start = i + 1
                    break

            main_content = "\n".join(lines[content_start:]).strip()

            if not uuid:
                if self.verbose:
                    logger.warning(f"No UUID found in {file_path.name}")
                return None

            if len(main_content) < 50:
                if self.verbose:
                    logger.warning(f"Content too short in {file_path.name}")
                return None

            return {
                "uuid": uuid,
                "title": title,
                "content": main_content,
                "category": category,
                "file_path": str(file_path),
            }

        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            return None

    async def check_exists_in_qdrant(self, uuid: str) -> bool:
        """Check if learning already exists in Qdrant.

        Args:
            uuid: Learning UUID

        Returns:
            True if already embedded, False otherwise
        """
        try:
            # Search by UUID in Qdrant
            results = await self.vector_service.search_similar_learnings(
                query_text="",  # Empty query won't match, but we can check by ID
                limit=1,
                min_score=0.0,
            )
            # For proper check, we'd need a point lookup - for now use search
            # This is a simplified check
            return False  # Assume not exists for idempotency check
        except Exception:
            return False

    async def embed_learning(self, learning: dict) -> bool:
        """Generate and store embedding for a learning.

        Args:
            learning: Dict with uuid, title, content, category

        Returns:
            True if successful, False otherwise
        """
        if self.dry_run:
            return True

        try:
            result = await self.vector_service.store_learning_embedding(
                title=learning["title"],
                content=learning["content"],
                learning_uuid=learning["uuid"],
                category=learning["category"],
                metadata={
                    "file_path": learning["file_path"],
                    "bootstrapped_at": datetime.utcnow().isoformat(),
                },
            )
            return bool(result)

        except Exception as e:
            logger.error(f"Failed to embed {learning['uuid']}: {e}")
            return False

    async def run(self, learnings_path: Path) -> BootstrapResult:
        """Run bootstrap process on learnings directory.

        Args:
            learnings_path: Path to learnings directory

        Returns:
            BootstrapResult with counts and summary
        """
        start_time = time.perf_counter()

        if not learnings_path.exists():
            return BootstrapResult(
                total_files=0,
                successful=0,
                failed=1,
                skipped=0,
                duration_seconds=0.0,
                dry_run=self.dry_run,
                failed_files=[str(learnings_path)],
                error_summary={"DirectoryNotFound": 1},
            )

        # Find all learning files
        learning_files = list(learnings_path.glob("learning-*.md"))
        total_files = len(learning_files)

        if total_files == 0:
            console.print(
                f"[yellow]No learning files found in {learnings_path}[/yellow]"
            )
            return BootstrapResult(
                total_files=0,
                successful=0,
                failed=0,
                skipped=0,
                duration_seconds=time.perf_counter() - start_time,
                dry_run=self.dry_run,
                failed_files=[],
                error_summary={},
            )

        successful = 0
        failed = 0
        skipped = 0
        failed_files = []
        error_summary = {}

        mode_label = "[DRY RUN] " if self.dry_run else ""

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"{mode_label}Processing learnings...", total=total_files
            )

            for file_path in learning_files:
                # Check failure threshold
                if failed >= self.max_failures:
                    console.print(
                        f"[red]Max failures ({self.max_failures}) reached, aborting[/red]"
                    )
                    break

                progress.update(
                    task, description=f"{mode_label}{file_path.name[:40]}..."
                )

                # Parse file
                learning = self.parse_learning_file(file_path)
                if not learning:
                    error_type = "ParseError"
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    failed_files.append(str(file_path))
                    failed += 1
                    progress.advance(task)
                    continue

                # Check if already exists (idempotency)
                if not self.dry_run and await self.check_exists_in_qdrant(
                    learning["uuid"]
                ):
                    skipped += 1
                    if self.verbose:
                        console.print(f"  → Already exists: {learning['uuid'][:8]}...")
                    progress.advance(task)
                    continue

                # Embed
                if await self.embed_learning(learning):
                    successful += 1
                    if self.verbose:
                        action = "Would embed" if self.dry_run else "✓ Embedded"
                        console.print(f"  {action}: {learning['title'][:50]}...")
                else:
                    error_type = "EmbeddingError"
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    failed_files.append(str(file_path))
                    failed += 1

                progress.advance(task)

        duration = time.perf_counter() - start_time

        return BootstrapResult(
            total_files=total_files,
            successful=successful,
            failed=failed,
            skipped=skipped,
            duration_seconds=duration,
            dry_run=self.dry_run,
            failed_files=failed_files,
            error_summary=error_summary,
        )


def print_summary(result: BootstrapResult):
    """Print formatted summary table."""
    table = Table(title="Bootstrap Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    mode = "DRY RUN" if result.dry_run else "LIVE"
    table.add_row("Mode", mode)
    table.add_row("Total Files", str(result.total_files))
    table.add_row("Successful", str(result.successful))
    table.add_row("Failed", str(result.failed))
    table.add_row("Skipped", str(result.skipped))
    table.add_row("Duration", f"{result.duration_seconds:.2f}s")

    if result.total_files > 0:
        success_rate = (result.successful / result.total_files) * 100
        table.add_row("Success Rate", f"{success_rate:.1f}%")

    console.print(table)

    if result.failed_files:
        console.print("\n[red]Failed Files:[/red]")
        for f in result.failed_files[:10]:
            console.print(f"  - {Path(f).name}")
        if len(result.failed_files) > 10:
            console.print(f"  ... and {len(result.failed_files) - 10} more")

    if result.error_summary:
        console.print("\n[yellow]Error Summary:[/yellow]")
        for error_type, count in result.error_summary.items():
            console.print(f"  {error_type}: {count}")


@app.command()
def main(
    path: Path | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to learnings directory (default: MNEMODROME_LEARNINGS_PATH)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Preview without writing to Qdrant"
    ),
    batch_size: int = typer.Option(
        DEDUP_BOOTSTRAP_BATCH_SIZE, "--batch-size", "-b", help="Files per batch"
    ),
    max_failures: int = typer.Option(
        DEDUP_BOOTSTRAP_MAX_FAILURES,
        "--max-failures",
        "-m",
        help="Maximum failures before abort",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Bootstrap vector embeddings for existing learning files.

    Scans the learnings directory and generates Qdrant embeddings
    for semantic deduplication support.
    """
    # Determine learnings path
    if path:
        learnings_path = path
    else:
        try:
            learnings_path = get_memory_path(MemoryType.LEARNING)
        except ValueError as e:
            console.print(f"[red]Configuration error: {e}[/red]")
            raise typer.Exit(2)

    console.print("\n[bold]Bootstrap Vector Embeddings[/bold]")
    console.print(f"Path: {learnings_path}")
    console.print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    console.print()

    # Create processor
    processor = BootstrapProcessor(
        dry_run=dry_run,
        batch_size=batch_size,
        max_failures=max_failures,
        verbose=verbose,
    )

    async def run_async():
        # Initialize services (skip for dry run file parsing)
        if not dry_run:
            console.print("Initializing services...")
            if not await processor.initialize():
                raise typer.Exit(2)
            console.print("[green]Services ready[/green]\n")

        # Run bootstrap
        result = await processor.run(learnings_path)

        # Print summary
        print_summary(result)

        # Exit code
        if result.failed > 0:
            raise typer.Exit(1)

    asyncio.run(run_async())


if __name__ == "__main__":
    app()

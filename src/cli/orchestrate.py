#!/usr/bin/env python3
"""
Phase 3 CLI - Orchestration Layer Validation Command

Implements CLI validation trigger for manual code validation with Rich output formatting.
Supports JSON and markdown output formats with approval gate presentation.

Usage:
    python -m src.cli.orchestrate validate ./src/mycode.py
    python -m src.cli.orchestrate validate ./src/mycode.py --format json
    python -m src.cli.orchestrate validate ./src/mycode.py --format markdown
    python -m src.cli.orchestrate validate --samples 10 --output test_results/

Requirements:
    - Rich library for colored approval gate output
    - AgentCoordinator from orchestrator layer
    - ContextPackage setup with code and patterns
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.orchestrator.agent_coordinator import (
    AgentCoordinator,
    ContextPackage,
    ValidationRole,
)


class ValidateCommand:
    """CLI command for orchestration layer validation."""

    def __init__(self):
        """Initialize validate command."""
        self.console = Console()
        self.coordinator = AgentCoordinator(sla_timeout_seconds=180)

    async def validate_code_file(
        self,
        code_path: str,
        output_format: str = "text",
        output_file: str | None = None,
    ) -> dict:
        """
        Validate a code file using orchestration layer.

        Args:
            code_path: Path to code file to validate
            output_format: Output format (text, json, markdown)
            output_file: Optional file to write results to

        Returns:
            Dict with validation results and metadata
        """
        try:
            # Validate file exists
            path = Path(code_path)
            if not path.exists():
                self.console.print(f"[red]Error: File not found: {code_path}[/red]")
                return {"success": False, "error": "File not found"}

            if not path.is_file():
                self.console.print(f"[red]Error: Not a file: {code_path}[/red]")
                return {"success": False, "error": "Not a file"}

            # Read code
            with open(path) as f:
                code = f.read()

            self.console.print(
                f"[cyan]Validating[/cyan] {code_path} "
                f"({len(code)} bytes, {len(code.splitlines())} lines)"
            )

            # Create context package
            context = ContextPackage(
                code=code,
                file_path=str(path.absolute()),
                patterns=[],
                entity_context={},
                test_files=[],
                related_entities=[],
            )

            # Create and execute job
            start_time = time.time()
            job = await self.coordinator.create_job(
                task=f"Validate {path.name}",
                context=context,
                agents=[
                    ValidationRole.PATTERN_COMPLIANCE,
                    ValidationRole.TEST_COVERAGE,
                    ValidationRole.ENTITY_CONSISTENCY,
                ],
            )

            result = await self.coordinator.execute_job(job)
            elapsed = time.time() - start_time

            # Format output
            if output_format == "json":
                return self._format_json_result(result, elapsed, code_path)
            elif output_format == "markdown":
                return self._format_markdown_result(result, elapsed, code_path)
            else:  # text (default)
                return self._format_text_result(result, elapsed, code_path)

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            self.console.print(f"[red]Validation failed: {e}[/red]")
            return {"success": False, "error": str(e)}

    def _format_text_result(self, result, elapsed: float, code_path: str) -> dict:
        """Format result as Rich text output with approval gate."""
        # Status panel
        status_icon = "✅" if result.overall_passed else "❌"
        status_text = "PASSED" if result.overall_passed else "FAILED"
        status_color = "green" if result.overall_passed else "red"

        panel_content = Text(
            f"{status_icon} VALIDATION {status_text}", style=status_color
        )
        panel = Panel(
            panel_content,
            title="Approval Gate",
            border_style=status_color,
            expand=False,
        )
        self.console.print(panel)

        # Summary
        summary_table = Table(title="Validation Summary", show_header=False)
        summary_table.add_row("File", code_path)
        summary_table.add_row("Status", status_text, style=status_color)
        summary_table.add_row("Total Violations", str(result.total_violations))
        summary_table.add_row("Critical", str(result.critical_count), style="bold red")
        summary_table.add_row(
            "Warnings", str(result.warning_count), style="bold yellow"
        )
        summary_table.add_row("Info", str(result.info_count), style="bold blue")
        summary_table.add_row("Time", f"{result.total_time_ms:.2f}ms")
        summary_table.add_row("Elapsed", f"{elapsed:.2f}s")
        summary_table.add_row(
            "SLA Status", "✅ Met" if result.sla_met else "⚠️  Exceeded"
        )

        self.console.print(summary_table)

        # Violations detail
        if result.all_violations:
            self.console.print("\n[bold]Violations[/bold]:")

            violations_table = Table(title="Detailed Violations", show_header=True)
            violations_table.add_column("Severity", style="cyan")
            violations_table.add_column("Category", style="magenta")
            violations_table.add_column("Message", style="white")
            violations_table.add_column("Line", style="yellow")
            violations_table.add_column("Recommendation", style="green")

            for v in result.all_violations:
                severity_style = {
                    "critical": "bold red",
                    "warning": "bold yellow",
                    "info": "bold blue",
                }.get(v.severity.value, "white")

                violations_table.add_row(
                    v.severity.value,
                    v.category,
                    v.message,
                    str(v.line_number or "-"),
                    v.recommendation,
                    style=severity_style if v.severity.value == "critical" else "",
                )

            self.console.print(violations_table)

        # Validators summary
        self.console.print("\n[bold]Validator Results[/bold]:")
        validators_table = Table(show_header=True)
        validators_table.add_column("Validator", style="cyan")
        validators_table.add_column("Status", style="magenta")
        validators_table.add_column("Time (ms)", style="yellow")
        validators_table.add_column("Violations", style="white")

        for vresult in result.validator_results:
            status = "✅" if vresult.passed else "❌"
            validators_table.add_row(
                vresult.validator_role.value,
                status,
                f"{vresult.validation_time_ms:.2f}",
                str(len(vresult.violations)),
            )

        self.console.print(validators_table)

        # Mnemodrome tracking
        if result.memory_uuids:
            self.console.print("\n[bold cyan]Mnemodrome Tracking[/bold cyan]:")
            for uuid in result.memory_uuids:
                self.console.print(f"  • {uuid}")

        return {
            "success": True,
            "overall_passed": result.overall_passed,
            "total_violations": result.total_violations,
            "critical_count": result.critical_count,
            "warning_count": result.warning_count,
            "info_count": result.info_count,
            "elapsed": elapsed,
            "sla_met": result.sla_met,
        }

    def _format_json_result(self, result, elapsed: float, code_path: str) -> dict:
        """Format result as JSON."""
        data = {
            "success": True,
            "code_path": code_path,
            "overall_passed": result.overall_passed,
            "timestamp": result.timestamp.isoformat(),
            "total_violations": result.total_violations,
            "critical_count": result.critical_count,
            "warning_count": result.warning_count,
            "info_count": result.info_count,
            "total_time_ms": result.total_time_ms,
            "elapsed_seconds": elapsed,
            "sla_met": result.sla_met,
            "aggregation_id": result.aggregation_id,
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "severity": v.severity.value,
                    "category": v.category,
                    "message": v.message,
                    "line_number": v.line_number,
                    "column_number": v.column_number,
                    "recommendation": v.recommendation,
                    "pattern_uuid": v.pattern_uuid,
                    "context": v.context,
                }
                for v in result.all_violations
            ],
            "validator_results": [
                {
                    "result_id": vr.result_id,
                    "validator_role": vr.validator_role.value,
                    "passed": vr.passed,
                    "validation_time_ms": vr.validation_time_ms,
                    "violation_count": len(vr.violations),
                    "error_message": vr.error_message,
                    "memory_uuid": vr.memory_uuid,
                }
                for vr in result.validator_results
            ],
            "memory_uuids": result.memory_uuids,
        }

        # Print JSON
        json_str = json.dumps(data, indent=2, default=str)
        self.console.print(json_str)

        return data

    def _format_markdown_result(self, result, elapsed: float, code_path: str) -> dict:
        """Format result as Markdown."""
        status_emoji = "✅" if result.overall_passed else "❌"
        status_text = "PASSED" if result.overall_passed else "FAILED"

        md = []
        md.append(f"# Validation Report: {status_emoji} {status_text}")
        md.append("")
        md.append(f"**File**: `{code_path}`")
        md.append(f"**Timestamp**: {result.timestamp.isoformat()}")
        md.append(f"**Aggregation ID**: `{result.aggregation_id}`")
        md.append("")

        md.append("## Summary")
        md.append(f"- **Total Violations**: {result.total_violations}")
        md.append(f"- **Critical**: {result.critical_count}")
        md.append(f"- **Warnings**: {result.warning_count}")
        md.append(f"- **Info**: {result.info_count}")
        md.append(f"- **Validation Time**: {result.total_time_ms:.2f}ms")
        md.append(f"- **Elapsed**: {elapsed:.2f}s")
        md.append(f"- **SLA Status**: {'✅ Met' if result.sla_met else '⚠️  Exceeded'}")
        md.append("")

        if result.all_violations:
            md.append("## Violations")
            md.append("")
            for v in result.all_violations:
                md.append(f"### {v.severity.value.upper()}: {v.category}")
                md.append(f"- **Message**: {v.message}")
                if v.line_number:
                    md.append(f"- **Line**: {v.line_number}")
                if v.column_number:
                    md.append(f"- **Column**: {v.column_number}")
                md.append(f"- **Recommendation**: {v.recommendation}")
                if v.pattern_uuid:
                    md.append(f"- **Pattern UUID**: `{v.pattern_uuid}`")
                if v.context:
                    md.append(f"- **Context**: {v.context}")
                md.append("")

        md.append("## Validator Results")
        md.append("")
        for vr in result.validator_results:
            status = "✅" if vr.passed else "❌"
            md.append(
                f"- {status} **{vr.validator_role.value}**: "
                f"{vr.validation_time_ms:.2f}ms ({len(vr.violations)} violations)"
            )
            if vr.error_message:
                md.append(f"  - Error: {vr.error_message}")

        md.append("")
        md.append("## Mnemodrome Tracking")
        if result.memory_uuids:
            for uuid in result.memory_uuids:
                md.append(f"- `{uuid}`")
        else:
            md.append("No memory UUIDs tracked")

        md_str = "\n".join(md)
        self.console.print(md_str)

        return {
            "success": True,
            "format": "markdown",
            "overall_passed": result.overall_passed,
            "total_violations": result.total_violations,
        }


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3 Orchestration Layer Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single file
  orchestrate validate ./src/mycode.py

  # Validate with JSON output
  orchestrate validate ./src/mycode.py --format json

  # Validate with markdown output
  orchestrate validate ./src/mycode.py --format markdown

  # Save results to file
  orchestrate validate ./src/mycode.py --output test_results/report.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate code file")
    validate_parser.add_argument(
        "code_path", type=str, help="Path to code file to validate"
    )
    validate_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    validate_parser.add_argument(
        "--output",
        type=str,
        help="Optional file to save results to",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        cmd = ValidateCommand()
        result = asyncio.run(
            cmd.validate_code_file(
                code_path=args.code_path,
                output_format=args.format,
                output_file=args.output,
            )
        )

        # Write to file if requested
        if args.output and result.get("success"):
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if args.format == "json":
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            else:
                with open(output_path, "w") as f:
                    f.write(str(result))

            logger.info(f"Results saved to {output_path}")

        # Exit with status code
        sys.exit(0 if result.get("success") and result.get("overall_passed") else 1)


if __name__ == "__main__":
    main()

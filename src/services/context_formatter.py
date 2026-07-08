"""Context formatting utilities for Mnemodrome memory visualization."""

from datetime import datetime
from typing import Any

# Maximum character length for entity summaries (prevents token bloat)
MAX_SUMMARY_LENGTH = 200


def format_context_markdown(results: list[dict[str, Any]]) -> str:
    """
    Format Neo4j query results as readable markdown for context display.

    Args:
        results: List of records from temporal_query with source, r, target

    Returns:
        Formatted markdown string with relationships and entities
    """
    if not results:
        return "No recent context found matching criteria."

    markdown = "# Recent Context (Temporal Memory)\n\n"
    markdown += f"*Found {len(results)} relationship(s)*\n\n"

    for i, record in enumerate(results, 1):
        source = record.get("source", {})
        rel = record.get("r", {})
        target = record.get("target", {})

        # Extract node properties safely
        source_name = _get_property(source, "name", "Unknown")
        target_name = _get_property(target, "name", "Unknown")
        rel_type = _get_relationship_type(rel)
        valid_at = _get_property(rel, "valid_at")

        markdown += f"## {i}. {source_name} → {target_name}\n\n"
        markdown += f"**Relationship**: {rel_type}\n\n"

        if valid_at:
            markdown += f"**Created**: {_format_datetime(valid_at)}\n\n"

        # Per-relationship context (provenance for THIS specific edge). Rendered
        # before node summaries because a node's summary is shared across every
        # one of its edges (and is clobbered last-write-wins — see issue #98),
        # so it repeats noisily; the edge context is what actually distinguishes
        # one relationship from another. (issue #97)
        rel_context = _get_property(rel, "context")
        if rel_context:
            markdown += f"**Context**: {_truncate_summary(rel_context)}\n\n"

        # Add source entity details
        if source:
            markdown += f"**Source Entity**: {source_name}\n"
            summary = _get_property(source, "summary")
            if summary:
                truncated = _truncate_summary(summary)
                markdown += f"- Summary: {truncated}\n"
            markdown += "\n"

        # Add target entity details
        if target:
            markdown += f"**Target Entity**: {target_name}\n"
            summary = _get_property(target, "summary")
            if summary:
                truncated = _truncate_summary(summary)
                markdown += f"- Summary: {truncated}\n"
            markdown += "\n"

        markdown += "---\n\n"

    return markdown


def _truncate_summary(summary: str) -> str:
    """
    Truncate long summaries to prevent token bloat.

    Args:
        summary: Entity summary text

    Returns:
        Truncated summary with ellipsis if needed
    """
    if not summary:
        return ""

    if len(summary) <= MAX_SUMMARY_LENGTH:
        return summary

    # Truncate at word boundary
    truncated = summary[:MAX_SUMMARY_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    return truncated + "..."


def _get_property(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely extract property from Neo4j node or relationship.

    Neo4j Python driver returns different object types depending on query.
    Handle dict, Node objects, and Relationship objects.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)

    if hasattr(obj, "get"):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key, default)

    return default


def _get_relationship_type(rel: Any) -> str:
    """
    Extract relationship type from Neo4j relationship object.

    Args:
        rel: Neo4j relationship object or dict

    Returns:
        Relationship type string
    """
    # Try 'type' attribute (Relationship objects)
    if hasattr(rel, "type"):
        return rel.type

    # Try dict access
    if isinstance(rel, dict) and "type" in rel:
        return rel["type"]

    return "RELATED_TO"


def _format_datetime(dt: Any) -> str:
    """
    Format Neo4j datetime for display.

    Args:
        dt: Neo4j datetime object or ISO string

    Returns:
        Human-readable datetime string
    """
    if isinstance(dt, str):
        try:
            parsed = datetime.fromisoformat(dt.replace("Z", "+00:00"))
            return parsed.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, AttributeError):
            return dt

    # Neo4j datetime objects have iso_format method
    if hasattr(dt, "iso_format"):
        try:
            iso_str = dt.iso_format()
            if isinstance(iso_str, str):
                return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
        except (ValueError, AttributeError, TypeError):
            pass

    # Fallback for datetime objects
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    return str(dt)

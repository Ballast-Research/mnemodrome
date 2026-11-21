"""
Text templates: the exact text that is embedded for each memory type (#134).

Every writer and the reindex tool build the embedded text here, so a stored
vector can always be rebuilt from its Qdrant payload. Changing any template
output changes stored vectors: bump TEMPLATE_VERSION (part of the
fingerprint) and rebuild.
"""

from __future__ import annotations

from typing import Any

TEMPLATE_VERSION = 1

MEMORY_TYPES = ("prompts", "learnings", "facts", "sessions", "procedures")


def prompt_text(text: str) -> str:
    return text


def learning_text(title: str, content: str) -> str:
    return f"{title}\n\n{content}"


def session_text(summary: str, key_insights: list[str] | None = None) -> str:
    insights_text = ""
    if key_insights:
        insights_text = "\n\nKey Insights:\n" + "\n".join(
            f"- {insight}" for insight in key_insights
        )
    return f"{summary}{insights_text}"


def fact_text(entity1: str, relation: str, entity2: str, context: str = "") -> str:
    text = f"{entity1} {relation} {entity2}"
    if context:
        text += f"\nContext: {context}"
    return text


def procedure_text(symptom: str, root_cause: str) -> str:
    return f"{symptom} {root_cause}"


def text_from_payload(memory_type: str, payload: dict[str, Any]) -> str | None:
    """Rebuild the embedded text from a stored payload (None if impossible)."""
    p = payload or {}
    if memory_type == "prompts":
        text = p.get("text")
        return prompt_text(text) if text else None
    if memory_type == "learnings":
        if p.get("title") is None and p.get("content") is None:
            return None
        return learning_text(p.get("title") or "", p.get("content") or "")
    if memory_type == "sessions":
        summary = p.get("summary")
        if summary is None:
            return None
        return session_text(summary, p.get("key_insights") or [])
    if memory_type == "facts":
        if not (p.get("entity1") and p.get("relation") and p.get("entity2")):
            return None
        return fact_text(
            p["entity1"], p["relation"], p["entity2"], p.get("context") or ""
        )
    if memory_type == "procedures":
        if p.get("symptom") is None and p.get("root_cause") is None:
            return None
        return procedure_text(p.get("symptom") or "", p.get("root_cause") or "")
    raise ValueError(f"unknown memory type: {memory_type!r}")

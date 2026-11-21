"""
Embedding profile and fingerprint (#134).

A stored vector depends on the whole pipeline, not only the model name:
text template -> task prefix -> tokenizer -> max length -> weights and
precision -> pooling -> normalization -> dimension. The profile names every
step that changes the vector; the fingerprint is a hash of it.

Rules:
- One collection = one fingerprint. A collection is registered with the
  fingerprint of the embedder that built it (see `registry.py`).
- Library versions (onnxruntime, numpy) are NOT part of the fingerprint.
  Routine dependency bumps must not force a rebuild; the canary
  (`canary.py`) and the golden tests guard them instead.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

ROLE_DOCUMENT = "document"
ROLE_QUERY = "query"
ROLES = (ROLE_DOCUMENT, ROLE_QUERY)


@dataclass(frozen=True)
class EmbedProfile:
    """Every setting that shapes a stored vector."""

    provider: str
    model: str
    dim: int
    template_version: int
    document_prefix: str = ""
    query_prefix: str = ""
    lowercase: bool = False
    max_tokens: int | None = None
    pooling: str = "provider"
    normalize: bool = False
    precision: str = "provider"
    weights_sha256: str = ""
    vocab_sha256: str = ""

    def prefix(self, role: str) -> str:
        if role not in ROLES:
            raise ValueError(f"unknown embedding role: {role!r}")
        return self.document_prefix if role == ROLE_DOCUMENT else self.query_prefix

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @property
    def short_fingerprint(self) -> str:
        return self.fingerprint[:8]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 for a zero vector)."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)

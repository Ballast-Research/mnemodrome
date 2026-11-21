"""
Behavioural canary for embedding drift (#134).

A fingerprint says what the code believes it runs. The canary measures what
the embedder actually produces: a fixed golden set is embedded when a
collection is registered and again at every startup. If the vectors move,
the collection is marked as drifted (search warns, dedup fails closed).

The golden set covers the failure that started #134 (mixed case and ALL
CAPS, Ollama v0.30.0), plus Unicode and one long text.
"""

from __future__ import annotations

from .profile import cosine

CANARY_TEXTS: tuple[str, ...] = (
    "Alice decided to move the Acme catalog from Neo4j Community to a single Qdrant collection.",
    "REPLACE SUMP PUMP",
    "CREATE A SPREADSHEET",
    "the job manager routes all gpu work; never call the model server directly.",
    "Café naïve résumé: Ångström, Straße, İstanbul, 日本語のテキスト, emoji 🚀",
    " ".join(
        ["Session summary: fixed the SSH config bug, updated the firewall rules,"]
        + ["and recorded where the deployment keeps its TLS certificates."] * 40
    ),
)

# A local, pinned pipeline should reproduce itself almost exactly (fp32 noise
# across CPUs and onnxruntime versions is ~1e-6). Remote engines (GPU,
# batching, F16) get a looser bound that still catches tokenization changes:
# the v0.30.0 lowercase change moved these texts to 0.39-0.97.
LOCAL_TOLERANCE = 0.9999
REMOTE_TOLERANCE = 0.999


def compare(
    reference: list[list[float]], current: list[list[float]], tolerance: float
) -> tuple[bool, float]:
    """Return (ok, min cosine) of current vs reference canary vectors."""
    if not reference or len(reference) != len(current):
        return False, 0.0
    scores = [cosine(r, c) for r, c in zip(reference, current, strict=True)]
    worst = min(scores)
    return worst >= tolerance, worst


__all__ = ["CANARY_TEXTS", "LOCAL_TOLERANCE", "REMOTE_TOLERANCE", "compare", "cosine"]

#!/usr/bin/env python3
"""
P1/P2 embedding parity check for #134 (read-only).

P1 (implementation parity): the in-process ONNX pipeline without task
prefixes vs Ollama `nomic-embed-text` fed the lowercased text. Lowercasing
the Ollama input avoids the case bug of Ollama 0.13.1-0.2x, so both sides
should see the same tokens. Gate: min cosine >= 0.999, median >= 0.9995,
on texts both tokenizers split the same way. Ollama 0.13.1-0.2x has two more
tokenizer differences from the reference BERT tokenizer (which the local
tokenizer matches exactly, see tests/unit/test_wordpiece.py):
- non-ASCII text (accents are kept, CJK is not split);
- the ASCII symbols $ + < = > ^ ` | ~ are not split off as punctuation
  (measured: "use `code` here" 0.785, "price $5.00 (approx)" 0.974).
Texts with either are reported separately, not gated.

P2 (stored vectors, optional with --qdrant-*): ONNX without prefix vs the
vector already stored in Qdrant. All-lowercase texts should match (same
weights); mixed-case texts carry the old bug and are only reported.

Nothing is written anywhere: Qdrant is only scrolled, Ollama only embeds.

Usage:
    uv run --extra onnx python scripts/embedding_parity.py \
        --model-dir ~/.cache/mnemodrome/models --ollama-url http://localhost:11434 \
        [--qdrant-host localhost --qdrant-port 6333] [--sample 500]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.embedding import registry, templates  # noqa: E402
from src.services.embedding.canary import CANARY_TEXTS  # noqa: E402
from src.services.embedding.profile import cosine  # noqa: E402
from src.services.embedding.providers import LocalOnnxEmbedder  # noqa: E402

# ASCII characters that BERT treats as punctuation but Unicode does not (S*).
OLLAMA_020_SYMBOLS = frozenset("$+<=>^`|~")

P1_MIN = 0.999
P1_MEDIAN = 0.9995

BUILTIN_TEXTS = list(CANARY_TEXTS) + [
    "Alice WORKS_AT Acme Corporation\nContext: joined in 2024 as a staff engineer",
    "Use the Edit tool or a Python script for file changes, never sed.",
    "Qdrant stores 768-dimension vectors for facts, learnings and sessions.",
    "Session summary: reviewed the release workflow and fixed the tag check.",
    "how do I rotate an agent key without downtime?",
]


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": round(ordered[0], 6),
        "p01": round(ordered[max(0, int(len(ordered) * 0.01) - 1)], 6),
        "median": round(statistics.median(ordered), 6),
    }


def _sample_from_qdrant(host: str, port: int, per_type: int, seed: int):
    from qdrant_client import QdrantClient  # noqa: PLC0415

    client = QdrantClient(host=host, port=port)
    names = {c.name for c in client.get_collections().collections}
    rows = []
    rng = random.Random(seed)  # noqa: S311 (reproducible sampling, not security)
    for memory_type in templates.MEMORY_TYPES:
        reg = registry.CollectionRegistry(client)
        source = reg.alias_target(registry.live_alias(memory_type))
        if source is None and registry.legacy_name(memory_type) in names:
            source = registry.legacy_name(memory_type)
        if source is None:
            continue
        points, offset, pool = [], None, []
        while len(pool) < per_type * 20:
            points, offset = client.scroll(
                collection_name=source,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            pool.extend(points)
            if offset is None or not points:
                break
        for p in rng.sample(pool, min(per_type, len(pool))):
            text = templates.text_from_payload(memory_type, p.payload or {})
            if text and text.strip():
                rows.append((memory_type, text, p.vector))
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ollama-url", required=True)
    parser.add_argument("--ollama-model", default="nomic-embed-text")
    parser.add_argument("--qdrant-host")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--sample", type=int, default=500, help="total stored texts")
    parser.add_argument("--seed", type=int, default=134)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    embedder = LocalOnnxEmbedder(
        args.model_dir, max_tokens=args.max_tokens, allow_download=False
    )
    if not await embedder.initialize():
        print("error: local model not available", file=sys.stderr)
        return 2

    if args.qdrant_host:
        per_type = max(1, args.sample // len(templates.MEMORY_TYPES))
        rows = _sample_from_qdrant(
            args.qdrant_host, args.qdrant_port, per_type, args.seed
        )
    else:
        rows = [("builtin", t, None) for t in BUILTIN_TEXTS]

    p1: dict[str, list[float]] = {}
    p1_non_ascii: list[float] = []
    p1_symbols: list[float] = []
    p2_lower: list[float] = []
    p2_mixed: list[float] = []
    long_texts = 0
    rejected: list[str] = []
    async with httpx.AsyncClient(timeout=120.0) as http:
        for memory_type, text, stored in rows:
            if len(embedder._tokenizer.encode(text)) > args.max_tokens:
                long_texts += 1
            local = (
                await asyncio.get_running_loop().run_in_executor(
                    None, embedder._run, [text]
                )
            )[0]
            resp = await http.post(
                f"{args.ollama_url.rstrip('/')}/api/embeddings",
                json={"model": args.ollama_model, "prompt": text.lower()},
            )
            if resp.status_code == 400:
                # Ollama rejects inputs longer than its context window.
                rejected.append(resp.text[:120])
                continue
            resp.raise_for_status()
            remote = resp.json()["embedding"]
            score = cosine(local, remote)
            if not text.isascii():
                p1_non_ascii.append(score)
            elif OLLAMA_020_SYMBOLS & set(text):
                p1_symbols.append(score)
            else:
                p1.setdefault(memory_type, []).append(score)
            if stored:
                vec = stored if isinstance(stored, list) else list(stored.values())[0]
                (p2_lower if text == text.lower() else p2_mixed).append(
                    cosine(local, vec)
                )

    all_p1 = [v for vals in p1.values() for v in vals]
    report = {
        "P1_overall": _stats(all_p1),
        "P1_by_type": {k: _stats(v) for k, v in p1.items()},
        "P1_symbol_texts_report_only": _stats(p1_symbols) if p1_symbols else None,
        "P1_non_ascii_report_only": _stats(p1_non_ascii) if p1_non_ascii else None,
        "texts_over_max_tokens": long_texts,
        "ollama_rejected": len(rejected),
        "ollama_rejected_reasons": sorted(set(rejected))[:3],
        "gate": {"min": P1_MIN, "median": P1_MEDIAN},
    }
    if p2_lower:
        report["P2_lowercase_texts"] = _stats(p2_lower)
    if p2_mixed:
        report["P2_mixed_case_texts_report_only"] = _stats(p2_mixed)
    ok = min(all_p1) >= P1_MIN and statistics.median(all_p1) >= P1_MEDIAN
    report["P1_pass"] = ok
    print(json.dumps(report, indent=2))
    await embedder.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

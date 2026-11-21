"""Tests that need the pinned ONNX model, and P1 parity against Ollama (#134).

Skipped unless enabled, so CI skips them cleanly:
- EMBED_TEST_MODEL_DIR: a folder with the pinned `model.onnx` and `vocab.txt`
  (see INSTALLATION.md, "Offline embedding model"). Needs the `onnx` extra.
- PARITY_OLLAMA_URL: an Ollama with `nomic-embed-text` (read-only: only
  /api/embeddings is called). Enables the P1 parity test.

`onnx_golden.json` was produced by this pipeline from the pinned model. It
guards onnxruntime/numpy upgrades: library versions are not part of the
fingerprint, so this test (and the startup canary) must catch any drift.
"""

import json
import os
import statistics
from pathlib import Path

import pytest

from src.services.embedding.profile import cosine

FIXTURES = Path(__file__).parent.parent / "fixtures" / "embedding"
MODEL_DIR = os.getenv("EMBED_TEST_MODEL_DIR", "")
OLLAMA_URL = os.getenv("PARITY_OLLAMA_URL", "")

needs_model = pytest.mark.skipif(
    not MODEL_DIR, reason="set EMBED_TEST_MODEL_DIR to run model tests"
)


@pytest.fixture(scope="module")
def embedder():
    pytest.importorskip("onnxruntime")
    import asyncio

    from src.services.embedding.providers import LocalOnnxEmbedder

    emb = LocalOnnxEmbedder(MODEL_DIR, allow_download=False)
    loop = asyncio.new_event_loop()
    assert loop.run_until_complete(emb.initialize())
    yield emb, loop
    loop.run_until_complete(emb.close())
    loop.close()


@pytest.mark.model
@needs_model
def test_golden_vectors(embedder):
    emb, loop = embedder
    golden = json.loads((FIXTURES / "onnx_golden.json").read_text(encoding="utf-8"))
    assert golden["fingerprint"] == emb.profile.fingerprint
    for case in golden["cases"]:
        vec = loop.run_until_complete(emb.embed(case["text"], case["role"]))
        assert cosine(vec, case["vector"]) >= 0.9999, case["text"][:40]


@pytest.mark.model
@needs_model
def test_case_does_not_change_the_vector(embedder):
    emb, loop = embedder
    a = loop.run_until_complete(emb.embed("REPLACE SUMP PUMP"))
    b = loop.run_until_complete(emb.embed("replace sump pump"))
    c = loop.run_until_complete(emb.embed("CREATE A SPREADSHEET"))
    assert cosine(a, b) >= 0.99999
    assert cosine(a, c) < 0.95


@pytest.mark.model
@needs_model
def test_batched_equals_single(embedder):
    emb, loop = embedder
    texts = [
        "short",
        "word " * 900,
        "a medium sized sentence about vectors",
        "x " * 3000,
    ]
    batched = loop.run_until_complete(emb.embed_batch(texts, "document"))
    for text, vec in zip(texts, batched, strict=True):
        single = loop.run_until_complete(emb.embed(text, "document"))
        assert cosine(vec, single) >= 0.99999


@pytest.mark.model
@needs_model
def test_prefix_is_applied_per_role(embedder):
    emb, loop = embedder
    doc = loop.run_until_complete(emb.embed("vector databases", "document"))
    query = loop.run_until_complete(emb.embed("vector databases", "query"))
    assert cosine(doc, query) < 0.999


@pytest.mark.model
@pytest.mark.ollama
@needs_model
@pytest.mark.skipif(not OLLAMA_URL, reason="set PARITY_OLLAMA_URL to run P1 parity")
def test_p1_parity_with_ollama(embedder):
    """ONNX (no prefix) vs Ollama on lowercased input: min >= 0.999, median >= 0.9995.

    Gate on ASCII text only. Ollama 0.13.1-0.2x tokenizes accents and CJK
    differently from the reference BERT tokenizer (measured: cosine 0.77 on
    the Unicode canary text), so non-ASCII text cannot be compared there.
    """
    import httpx

    from src.services.embedding.canary import CANARY_TEXTS

    emb, _ = embedder
    texts = [t for t in CANARY_TEXTS if t.isascii()] + [
        "Alice WORKS_AT Acme Corporation\nContext: joined in 2024",
        "Use the Edit tool or a Python script for file changes, never sed.",
        "how do I rotate an agent key without downtime?",
    ]
    local = emb._run(texts)
    scores = []
    with httpx.Client(timeout=120.0) as http:
        for text, vec in zip(texts, local, strict=True):
            resp = http.post(
                f"{OLLAMA_URL.rstrip('/')}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text.lower()},
            )
            resp.raise_for_status()
            scores.append(cosine(vec, resp.json()["embedding"]))
    assert min(scores) >= 0.999, scores
    assert statistics.median(scores) >= 0.9995, scores

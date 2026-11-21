"""Reindex tool: build, resume, catch-up, swap, rollback (#134).

In-memory Qdrant and a deterministic fake embedder: no network, no model.
"""

import pytest

pytest.importorskip("qdrant_client")
from embedding_fakes import FakeEmbedder  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from src.cli.reindex import Reindexer, ReindexError  # noqa: E402
from src.services.embedding import registry  # noqa: E402
from src.services.embedding.profile import cosine  # noqa: E402

LEGACY = "mnemodrome_facts"
ALIAS = "mnemodrome_facts_live"


def _fact(i):
    return {"entity1": f"E{i}", "relation": "KNOWS", "entity2": f"F{i}", "context": ""}


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    c.create_collection(
        LEGACY, vectors_config=VectorParams(size=8, distance=Distance.COSINE)
    )
    points = [
        PointStruct(id=i, vector=[0.5] * 8, payload=_fact(i)) for i in range(1, 8)
    ]
    points.append(PointStruct(id=99, vector=[0.5] * 8, payload={"note": "no text"}))
    c.upsert(LEGACY, points=points)
    registry.CollectionRegistry(c).point_alias(ALIAS, LEGACY)
    return c


def _new_embedder():
    return FakeEmbedder(
        provider="local-onnx", model="nomic", is_remote=False, prefixes=("d: ", "q: ")
    )


def _reindexer(client, emb=None, batch=3):
    return Reindexer(
        client, emb or _new_embedder(), batch_size=batch, echo=lambda *_: None
    )


@pytest.mark.asyncio
async def test_dry_run_writes_nothing(client):
    r = _reindexer(client)
    counters = await r.build("facts", dry_run=True)
    assert counters.would_embed == 7
    assert len(counters.skipped_no_text) == 1
    assert not registry.CollectionRegistry(client).physical_exists(
        r.target_name("facts")
    )


@pytest.mark.asyncio
async def test_build_embeds_from_payload_with_document_role(client):
    emb = _new_embedder()
    r = _reindexer(client, emb)
    counters = await r.build("facts")
    target = r.target_name("facts")
    assert counters.embedded == 7
    assert client.count(target).count == 7
    point = client.retrieve(target, ids=[3], with_payload=True, with_vectors=True)[0]
    assert point.payload["embed_fp"] == emb.profile.fingerprint
    assert point.payload["entity1"] == "E3"
    expected = emb.vector("E3 KNOWS F3", "document")
    # Qdrant normalizes for cosine; compare directions.
    assert cosine(point.vector, expected) == pytest.approx(1.0, abs=1e-6)
    record = registry.CollectionRegistry(client).get(target)
    assert record["built_from"] == LEGACY
    # Source untouched, alias unchanged.
    assert client.count(LEGACY).count == 8
    assert registry.CollectionRegistry(client).alias_target(ALIAS) == LEGACY


@pytest.mark.asyncio
async def test_build_is_resumable(client):
    r = _reindexer(client)
    await r.build("facts")
    again = await r.build("facts")
    assert again.embedded == 0
    assert again.skipped_current == 7


@pytest.mark.asyncio
async def test_catch_up_copies_new_writes_and_prunes_before_swap(client):
    r = _reindexer(client)
    await r.build("facts")
    client.upsert(
        LEGACY, points=[PointStruct(id=50, vector=[0.5] * 8, payload=_fact(50))]
    )
    client.delete(LEGACY, points_selector=[1])
    counters = await r.catch_up("facts")
    target = r.target_name("facts")
    ids = {p.id for p in client.scroll(target, limit=100)[0]}
    assert 50 in ids and 1 not in ids
    assert counters.pruned == 1


@pytest.mark.asyncio
async def test_swap_then_restamp_then_rollback(client):
    emb = _new_embedder()
    r = _reindexer(client, emb)
    await r.build("facts")
    await r.swap("facts")
    target = r.target_name("facts")
    reg = registry.CollectionRegistry(client)
    assert reg.alias_target(ALIAS) == target

    # A server still running the old embedder writes through the alias after
    # the swap: the point carries the old fingerprint and catch-up fixes it.
    client.upsert(
        ALIAS,
        points=[
            PointStruct(
                id=60, vector=[0.1] * 8, payload={**_fact(60), "embed_fp": "old"}
            )
        ],
    )
    counters = await r.catch_up("facts")
    assert counters.restamped == 1
    assert counters.pruned == 0  # never prune after the swap
    point = client.retrieve(target, ids=[60], with_payload=True)[0]
    assert point.payload["embed_fp"] == emb.profile.fingerprint

    r.rollback("facts")
    assert reg.alias_target(ALIAS) == LEGACY
    assert client.count(LEGACY).count == 8


@pytest.mark.asyncio
async def test_swap_refuses_when_points_failed(client):
    class Flaky(FakeEmbedder):
        async def embed_batch(self, texts, role="query"):
            out = await super().embed_batch(texts, role)
            return [
                None if t.startswith("E2 ") else v
                for t, v in zip(texts, out, strict=True)
            ]

    emb = Flaky(provider="local-onnx", model="nomic", is_remote=False)
    r = _reindexer(client, emb)
    await r.build("facts")
    with pytest.raises(ReindexError, match="Not swapping"):
        await r.swap("facts")
    assert registry.CollectionRegistry(client).alias_target(ALIAS) == LEGACY


@pytest.mark.asyncio
async def test_missing_alias_is_an_error():
    c = QdrantClient(":memory:")
    with pytest.raises(ReindexError, match="start the server once"):
        await _reindexer(c).build("facts")


@pytest.mark.asyncio
async def test_status(client):
    r = _reindexer(client)
    info = r.status("facts")
    assert info["live"] == LEGACY and info["live_fingerprint"] == "untagged"
    assert info["live_points"] == 8 and info["target_exists"] is False

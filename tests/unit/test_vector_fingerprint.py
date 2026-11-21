"""Collection fingerprints, canary, aliases and fail-closed behaviour (#134).

Runs VectorService against qdrant-client's in-memory mode with a
deterministic fake embedder: no network, no model.
"""

from unittest.mock import MagicMock

import pytest

qdrant_client = pytest.importorskip("qdrant_client")
from embedding_fakes import FakeEmbedder  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402

from src.services.deduplication_service import DeduplicationService  # noqa: E402
from src.services.embedding import registry  # noqa: E402
from src.services.vector_service import VectorService  # noqa: E402


async def _service(client, embedder, provider="ollama") -> VectorService:
    svc = VectorService(enabled=True, embed_provider=provider)
    svc.client = client
    svc.embedder = embedder
    await svc._ensure_collections(embedder.dimension)
    svc._initialized = True
    return svc


def _legacy(client, name="mnemodrome_facts", n=3):
    client.create_collection(
        name, vectors_config=VectorParams(size=8, distance=Distance.COSINE)
    )
    client.upsert(
        name,
        points=[
            PointStruct(
                id=i + 1,
                vector=[0.1 * (i + 1)] * 8,
                payload={"entity1": f"A{i}", "relation": "R", "entity2": "B"},
            )
            for i in range(n)
        ],
    )


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.mark.asyncio
async def test_fresh_install_creates_fingerprinted_collections(client):
    emb = FakeEmbedder()
    svc = await _service(client, emb)
    fp8 = emb.profile.short_fingerprint
    reg = registry.CollectionRegistry(client)
    for t in ("prompts", "learnings", "facts", "sessions", "procedures"):
        assert reg.alias_target(f"mnemodrome_{t}_live") == f"mnemodrome_{t}__{fp8}"
        assert svc.collection_status[t]["status"] == registry.STATUS_OK
        assert (
            reg.get(f"mnemodrome_{t}__{fp8}")["fingerprint"] == emb.profile.fingerprint
        )


@pytest.mark.asyncio
async def test_legacy_collection_is_aliased_and_adopted_by_ollama(client):
    _legacy(client)
    svc = await _service(client, FakeEmbedder())
    reg = registry.CollectionRegistry(client)
    assert reg.alias_target("mnemodrome_facts_live") == "mnemodrome_facts"
    assert svc.collection_status["facts"]["status"] == registry.STATUS_ADOPTED
    assert client.count("mnemodrome_facts").count == 3  # data untouched
    # Second start: same fingerprint, canary matches -> still trusted.
    svc2 = await _service(client, FakeEmbedder())
    assert svc2.collection_status["facts"]["status"] == registry.STATUS_ADOPTED
    assert svc2.is_trusted("facts")


@pytest.mark.asyncio
async def test_other_provider_does_not_adopt_legacy_data(client):
    _legacy(client)
    emb = FakeEmbedder(provider="local-onnx", is_remote=False)
    svc = await _service(client, emb, provider="local-onnx")
    assert svc.collection_status["facts"]["status"] == registry.STATUS_UNTAGGED
    assert not svc.is_trusted("facts")
    assert "facts (untagged)" in svc.trust_warning(["facts"])
    # Empty legacy-free types are fine.
    assert svc.is_trusted("learnings")


@pytest.mark.asyncio
async def test_fingerprint_mismatch_detected(client):
    await _service(client, FakeEmbedder(model="model-a"))
    svc = await _service(client, FakeEmbedder(model="model-b"))
    assert svc.collection_status["facts"]["status"] == registry.STATUS_MISMATCH
    health = await svc.health_check()
    assert set(health["embed_mismatch"]) == {
        "prompts",
        "learnings",
        "facts",
        "sessions",
        "procedures",
    }


@pytest.mark.asyncio
async def test_canary_catches_engine_drift_with_same_declared_profile(client):
    # The #134 incident: same model name, the engine changed its output.
    await _service(client, FakeEmbedder())
    svc = await _service(client, FakeEmbedder(variant="ollama-0.30"))
    assert svc.collection_status["facts"]["status"] == registry.STATUS_CANARY_DRIFT
    assert not svc.is_trusted("facts")


@pytest.mark.asyncio
async def test_points_are_tagged_and_reserved_keys_protected(client):
    emb = FakeEmbedder()
    svc = await _service(client, emb)
    point_id = await svc.store_fact_embedding(
        "Alice",
        "WORKS_AT",
        "Acme",
        "",
        "8a7f2a52-6f8c-4c0e-9d3a-0d6b4b2f1a11",
        metadata={"embed_fp": "forged", "embed_template": 99, "source": "test"},
    )
    assert point_id
    point = client.retrieve("mnemodrome_facts_live", ids=[point_id], with_payload=True)[
        0
    ]
    assert point.payload["embed_fp"] == emb.profile.fingerprint
    assert point.payload["embed_template"] == 1
    assert point.payload["source"] == "test"
    health = await svc.health_check()
    assert health["embed_collections"]["facts"]["points_not_tagged_active"] == 0


@pytest.mark.asyncio
async def test_roles_document_for_stores_query_for_search_and_prompts(client):
    emb = FakeEmbedder(prefixes=("doc: ", "q: "))
    svc = await _service(client, emb)
    emb.calls.clear()
    await svc.store_learning_embedding("T", "C", "0b3e6f1c-1d2a-4f5e-8a9b-0c1d2e3f4a5b")
    await svc.store_prompt_embedding("a prompt to remember")
    await svc.search_similar_learnings("find this")
    roles = [role for _, role in emb.calls]
    assert roles == ["document", "query", "query"]


@pytest.mark.asyncio
async def test_dedup_fails_closed_on_mismatch(client):
    await _service(client, FakeEmbedder(model="model-a"))
    svc = await _service(client, FakeEmbedder(model="model-b"))
    dedup = DeduplicationService(
        vector_service=svc, neo4j_service=MagicMock(), file_service=MagicMock()
    )
    svc.search_similar_learnings = MagicMock(
        side_effect=AssertionError("must not search")
    )
    assert await dedup.vector_similarity_search("anything at all") == []


@pytest.mark.asyncio
async def test_ollama_fallback_disabled_when_it_embeds_differently():
    from unittest.mock import AsyncMock

    from src.services.vector_service import OllamaEmbedder

    emb = OllamaEmbedder(host="http://primary", fallback_host="http://fallback")

    async def fake_embed_at(host, text):
        return [1.0, 0.0] if host == "http://primary" else [0.0, 1.0]

    emb._embed_at = AsyncMock(side_effect=fake_embed_at)
    assert await emb.check_fallback_parity() is False
    assert emb.fallback_host is None


@pytest.mark.asyncio
async def test_ollama_fallback_kept_when_identical():
    from unittest.mock import AsyncMock

    from src.services.vector_service import OllamaEmbedder

    emb = OllamaEmbedder(host="http://primary", fallback_host="http://fallback")
    emb._embed_at = AsyncMock(return_value=[0.3, 0.4])
    assert await emb.check_fallback_parity() is True
    assert emb.fallback_host == "http://fallback"


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="EMBED_PROVIDER"):
        VectorService(enabled=False, embed_provider="tei")


@pytest.mark.asyncio
async def test_writes_blocked_into_untrusted_collection(client):
    """#134 review F5: never mix two embedders in one collection."""
    _legacy(client)
    emb = FakeEmbedder(provider="local-onnx", is_remote=False)
    svc = await _service(client, emb, provider="local-onnx")
    before = client.count("mnemodrome_facts").count
    result = await svc.store_fact_embedding(
        "A", "R", "B", "", "1c1d2e3f-4a5b-4c6d-8e7f-8091a2b3c4d5"
    )
    assert result is False
    assert client.count("mnemodrome_facts").count == before
    health = await svc.health_check()
    facts = health["embed_collections"]["facts"]
    assert facts["blocked_writes"] == 1
    assert facts["points_untagged"] == 3
    assert health["embed_blocked_writes"] == 1


@pytest.mark.asyncio
async def test_health_counts_mixed_points_for_every_collection(client):
    emb = FakeEmbedder()
    await _service(client, emb)
    # Another embedder's point lands in the collection (e.g. a stale server
    # after a swap): it must show up as mixed, trusted or not.
    client.upsert(
        "mnemodrome_facts_live",
        points=[PointStruct(id=77, vector=[0.2] * 8, payload={"embed_fp": "other"})],
    )
    svc = await _service(client, FakeEmbedder(model="model-b"))
    facts = (await svc.health_check())["embed_collections"]["facts"]
    assert facts["status"] == registry.STATUS_MISMATCH
    assert facts["points_mixed"] == 1
    assert facts["points_not_tagged_active"] == 1


@pytest.mark.asyncio
async def test_unknown_is_not_trusted_and_is_rechecked(client):
    """#134 review F6: a failed canary must not leave dedup unprotected."""

    class Flaky(FakeEmbedder):
        fail = True

        async def embed_batch(self, texts, role="query"):
            if self.fail:
                return [None] * len(texts)
            return await super().embed_batch(texts, role)

    emb = Flaky()
    svc = await _service(client, emb)
    assert svc.collection_status["facts"]["status"] == registry.STATUS_UNKNOWN
    assert not svc.is_trusted("facts")
    emb.fail = False
    svc._last_recheck.clear()  # skip the rate limit
    assert await svc.ensure_trusted("facts") is True
    assert svc.collection_status["facts"]["status"] == registry.STATUS_OK


@pytest.mark.asyncio
async def test_canary_embedded_once_per_start(client):
    """#134 review F7: one canary run per start, shared by all collections."""
    emb = FakeEmbedder()
    await _service(client, emb)
    emb.calls.clear()
    await _service(client, emb)
    canary_calls = [c for c in emb.calls if c[1] == "document"]
    from src.services.embedding.canary import CANARY_TEXTS

    assert len(canary_calls) == len(CANARY_TEXTS)


@pytest.mark.asyncio
async def test_registry_records_unicode_version(client):
    import unicodedata

    emb = FakeEmbedder()
    await _service(client, emb)
    record = registry.CollectionRegistry(client).get(
        f"mnemodrome_facts__{emb.profile.short_fingerprint}"
    )
    assert record["unicode_version"] == unicodedata.unidata_version

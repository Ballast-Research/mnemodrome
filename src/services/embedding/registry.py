"""
Collection registry and alias indirection (#134).

- Code reads and writes `mnemodrome_<type>_live`, a Qdrant alias. The
  physical collection behind it can be swapped atomically (reindex), and the
  old one stays intact as the rollback target.
- First start after upgrade: if the alias is missing and the legacy
  collection `mnemodrome_<type>` exists, the alias is pointed at it. No data
  changes and no manual step.
- Fresh installs create `mnemodrome_<type>__<fp8>` behind the alias.
- `mnemodrome_embed_registry` holds one record per physical collection:
  fingerprint, profile, canary vectors, provenance. One collection = one
  fingerprint.

All methods are synchronous (QdrantClient is sync); call them through
`asyncio.to_thread` from async code.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from .profile import EmbedProfile

REGISTRY_COLLECTION = "mnemodrome_embed_registry"
FP_KEY = "embed_fp"
TEMPLATE_KEY = "embed_template"
RESERVED_PREFIX = "embed_"

# Collection status values. Only TRUSTED statuses give exact scores.
STATUS_OK = "ok"
STATUS_ADOPTED = "adopted"  # legacy collection, first registered by this version
STATUS_MISMATCH = "mismatch"  # registered with a different fingerprint
STATUS_CANARY_DRIFT = "canary_drift"  # same fingerprint, embedder output moved
STATUS_UNTAGGED = "untagged"  # legacy data, provider cannot vouch for it
STATUS_UNKNOWN = "unknown"  # check could not run
TRUSTED = frozenset({STATUS_OK, STATUS_ADOPTED})


def live_alias(memory_type: str) -> str:
    return f"mnemodrome_{memory_type}_live"


def legacy_name(memory_type: str) -> str:
    return f"mnemodrome_{memory_type}"


def physical_name(memory_type: str, profile: EmbedProfile) -> str:
    return f"mnemodrome_{memory_type}__{profile.short_fingerprint}"


def _record_id(collection: str) -> str:
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"mnemodrome-embed-registry:{collection}")
    )


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class CollectionRegistry:
    def __init__(self, client):
        self.client = client

    # --- registry records -------------------------------------------------

    def ensure(self) -> None:
        if not self.client.collection_exists(REGISTRY_COLLECTION):
            self.client.create_collection(
                collection_name=REGISTRY_COLLECTION,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )

    def get(self, collection: str) -> dict[str, Any] | None:
        points = self.client.retrieve(
            collection_name=REGISTRY_COLLECTION,
            ids=[_record_id(collection)],
            with_payload=True,
        )
        return dict(points[0].payload) if points else None

    def put(self, collection: str, record: dict[str, Any]) -> None:
        self.client.upsert(
            collection_name=REGISTRY_COLLECTION,
            points=[
                PointStruct(
                    id=_record_id(collection),
                    vector=[1.0],
                    payload={**record, "collection": collection},
                )
            ],
        )

    def register(
        self,
        collection: str,
        memory_type: str,
        profile: EmbedProfile,
        canary_vectors: list[list[float]],
        adopted: bool = False,
        built_from: str | None = None,
    ) -> dict[str, Any]:
        record = {
            "memory_type": memory_type,
            "fingerprint": profile.fingerprint,
            "profile": profile.to_dict(),
            "canary_vectors": canary_vectors,
            "adopted": adopted,
            "built_from": built_from,
            "registered_at": now_iso(),
        }
        self.put(collection, record)
        return record

    # --- aliases ----------------------------------------------------------

    def alias_target(self, alias: str) -> str | None:
        for a in self.client.get_aliases().aliases:
            if a.alias_name == alias:
                return a.collection_name
        return None

    def point_alias(self, alias: str, collection: str) -> None:
        """Create or move an alias in one atomic operation."""
        ops = []
        if self.alias_target(alias) is not None:
            ops.append(DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias)))
        ops.append(
            CreateAliasOperation(
                create_alias=CreateAlias(collection_name=collection, alias_name=alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=ops)

    def physical_exists(self, collection: str) -> bool:
        """True only for a real collection (an alias of that name does not count)."""
        names = {c.name for c in self.client.get_collections().collections}
        return collection in names

    def create_physical(self, collection: str, dim: int) -> None:
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        self.ensure_fp_index(collection)

    def ensure_fp_index(self, collection: str) -> None:
        try:
            self.client.create_payload_index(
                collection_name=collection,
                field_name=FP_KEY,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:  # already exists, or local mode without indexes
            pass

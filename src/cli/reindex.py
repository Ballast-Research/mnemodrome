#!/usr/bin/env python3
"""
Rebuild Qdrant collections with the configured embedder (#134, experimental).

The server reads and writes `mnemodrome_<type>_live`, an alias. This tool
builds `mnemodrome_<type>__<fp8>` (fp8 = fingerprint of the embedder named by
EMBED_PROVIDER) from the stored payload text, then moves the alias. The old
collection is never modified, so `rollback` is instant.

Usage:
    python -m src.cli.reindex status                 # per type: live, target, counts
    python -m src.cli.reindex build --dry-run        # what a build would do
    python -m src.cli.reindex build --type procedures --type prompts
    python -m src.cli.reindex catch-up               # copy new writes, fix stale tags
    python -m src.cli.reindex swap                   # catch-up, then move the alias
    python -m src.cli.reindex rollback               # alias back to the old collection

Build is resumable: points already in the target with the target fingerprint
are skipped. Run it with the embedder you want to switch TO, for example
`EMBED_PROVIDER=local-onnx python -m src.cli.reindex build`.

Cutover order: build -> stop writes or accept a short gap -> swap -> restart
the server with the new EMBED_PROVIDER -> catch-up (re-embeds anything the
old server wrote in between; every point carries its fingerprint).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from qdrant_client.models import PointStruct

from src.services.embedding import canary as _canary
from src.services.embedding import registry as _registry
from src.services.embedding import templates as _templates
from src.services.embedding.profile import ROLE_DOCUMENT, ROLE_QUERY


def role_for(memory_type: str) -> str:
    # Prompts are stored with the query role (they are compared with prompts).
    return ROLE_QUERY if memory_type == "prompts" else ROLE_DOCUMENT


@dataclass
class Counters:
    scanned: int = 0
    embedded: int = 0
    would_embed: int = 0
    skipped_current: int = 0
    skipped_no_text: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    pruned: int = 0
    restamped: int = 0


class ReindexError(RuntimeError):
    pass


class Reindexer:
    """Collection rebuild logic. `client` is a sync QdrantClient."""

    def __init__(self, client, embedder, batch_size: int = 32, echo=print):
        self.client = client
        self.embedder = embedder
        self.batch_size = max(1, batch_size)
        self.registry = _registry.CollectionRegistry(client)
        self.echo = echo

    @property
    def profile(self):
        return self.embedder.profile

    def target_name(self, memory_type: str) -> str:
        return _registry.physical_name(memory_type, self.profile)

    def live(self, memory_type: str) -> str:
        target = self.registry.alias_target(_registry.live_alias(memory_type))
        if target is None:
            raise ReindexError(
                f"alias {_registry.live_alias(memory_type)} does not exist; "
                "start the server once so it creates the aliases"
            )
        return target

    def _count(self, collection: str) -> int:
        return self.client.count(collection_name=collection, exact=True).count

    def status(self, memory_type: str) -> dict[str, Any]:
        self.registry.ensure()
        live = self.live(memory_type)
        target = self.target_name(memory_type)
        live_record = self.registry.get(live) or {}
        info: dict[str, Any] = {
            "type": memory_type,
            "alias": _registry.live_alias(memory_type),
            "live": live,
            "live_fingerprint": (live_record.get("fingerprint") or "untagged")[:16],
            "live_points": self._count(live),
            "target": target,
            "target_fingerprint": self.profile.fingerprint[:16],
            "target_exists": self.registry.physical_exists(target),
        }
        if info["target_exists"]:
            info["target_points"] = self._count(target)
            info["target_built_from"] = (self.registry.get(target) or {}).get(
                "built_from"
            )
        return info

    async def _ensure_target(self, memory_type: str, source: str, dry_run: bool) -> str:
        target = self.target_name(memory_type)
        if target == source:
            return target
        record = self.registry.get(target)
        if record is not None and record.get("fingerprint") != self.profile.fingerprint:
            raise ReindexError(f"{target} is registered with another fingerprint")
        if dry_run:
            return target
        if not self.registry.physical_exists(target):
            self.registry.create_physical(target, self.embedder.dimension)
        if record is None:
            vectors = await self.embedder.embed_batch(
                list(_canary.CANARY_TEXTS), ROLE_DOCUMENT
            )
            if not vectors or any(not v for v in vectors):
                raise ReindexError("canary embedding failed; is the embedder healthy?")
            self.registry.register(
                target, memory_type, self.profile, vectors, built_from=source
            )
        return target

    def _payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        clean = {
            k: v
            for k, v in (payload or {}).items()
            if not str(k).startswith(_registry.RESERVED_PREFIX)
        }
        clean[_registry.FP_KEY] = self.profile.fingerprint
        clean[_registry.TEMPLATE_KEY] = _templates.TEMPLATE_VERSION
        return clean

    async def _embed_points(
        self,
        memory_type: str,
        points: list,
        target: str,
        counters: Counters,
        dry_run: bool,
    ) -> None:
        todo: list[tuple[Any, dict[str, Any], str]] = []
        for point in points:
            text = _templates.text_from_payload(memory_type, point.payload or {})
            if not text or not text.strip():
                counters.skipped_no_text.append(str(point.id))
                continue
            todo.append((point.id, point.payload or {}, text))
        if dry_run:
            counters.would_embed += len(todo)
            return
        if not todo:
            return
        vectors = await self.embedder.embed_batch(
            [t for _, _, t in todo], role_for(memory_type)
        )
        structs = []
        for (pid, payload, _), vector in zip(todo, vectors, strict=True):
            if not vector:
                counters.failed.append(str(pid))
                continue
            structs.append(
                PointStruct(id=pid, vector=vector, payload=self._payload(payload))
            )
        if structs:
            self.client.upsert(collection_name=target, points=structs)
            counters.embedded += len(structs)

    async def _copy_missing(
        self,
        memory_type: str,
        source: str,
        target: str,
        counters: Counters,
        dry_run: bool,
    ) -> None:
        """Embed every source point that is absent or stale in the target."""
        fingerprint = self.profile.fingerprint
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=source,
                limit=self.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            counters.scanned += len(points)
            current: set[str] = set()
            if self.registry.physical_exists(target):
                existing = self.client.retrieve(
                    collection_name=target,
                    ids=[p.id for p in points],
                    with_payload=[_registry.FP_KEY],
                )
                current = {
                    str(p.id)
                    for p in existing
                    if (p.payload or {}).get(_registry.FP_KEY) == fingerprint
                }
            pending = [p for p in points if str(p.id) not in current]
            counters.skipped_current += len(points) - len(pending)
            await self._embed_points(memory_type, pending, target, counters, dry_run)
            self.echo(
                f"  {memory_type}: scanned {counters.scanned}, embedded "
                f"{counters.embedded}, current {counters.skipped_current}"
            )
            if offset is None:
                break

    async def _restamp_stale(
        self, memory_type: str, target: str, counters: Counters
    ) -> None:
        """Re-embed points in the target that carry another fingerprint (or none)."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        flt = Filter(
            must_not=[
                FieldCondition(
                    key=_registry.FP_KEY,
                    match=MatchValue(value=self.profile.fingerprint),
                )
            ]
        )
        stale_ids: list[Any] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=target,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=False,
            )
            stale_ids.extend(p.id for p in points)
            if offset is None or not points:
                break
        for i in range(0, len(stale_ids), self.batch_size):
            points = self.client.retrieve(
                collection_name=target,
                ids=stale_ids[i : i + self.batch_size],
                with_payload=True,
            )
            before = counters.embedded
            await self._embed_points(
                memory_type, points, target, counters, dry_run=False
            )
            counters.restamped += counters.embedded - before

    def _prune(self, source: str, target: str, counters: Counters) -> None:
        """Delete target points that no longer exist in the source (pre-swap only)."""
        source_ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=source, limit=256, offset=offset, with_payload=False
            )
            source_ids.update(str(p.id) for p in points)
            if offset is None or not points:
                break
        extra = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=target, limit=256, offset=offset, with_payload=False
            )
            extra.extend(p.id for p in points if str(p.id) not in source_ids)
            if offset is None or not points:
                break
        if extra:
            self.client.delete(collection_name=target, points_selector=extra)
            counters.pruned += len(extra)

    async def build(self, memory_type: str, dry_run: bool = False) -> Counters:
        self.registry.ensure()
        counters = Counters()
        source = self.live(memory_type)
        target = await self._ensure_target(memory_type, source, dry_run)
        if target == source:
            self.echo(f"  {memory_type}: live collection already has this fingerprint")
            return counters
        await self._copy_missing(memory_type, source, target, counters, dry_run)
        return counters

    def _pair(self, memory_type: str) -> tuple[str, str]:
        """(old, new) for the configured fingerprint."""
        target = self.target_name(memory_type)
        record = self.registry.get(target)
        if record is None or not self.registry.physical_exists(target):
            raise ReindexError(f"{target} does not exist; run build first")
        old = record.get("built_from")
        if not old:
            raise ReindexError(f"{target} has no source collection recorded")
        return old, target

    async def catch_up(self, memory_type: str) -> Counters:
        counters = Counters()
        old, new = self._pair(memory_type)
        await self._copy_missing(memory_type, old, new, counters, dry_run=False)
        if self.live(memory_type) == old:
            # Before the swap the old collection is the source of truth.
            self._prune(old, new, counters)
        await self._restamp_stale(memory_type, new, counters)
        return counters

    async def swap(self, memory_type: str, force: bool = False) -> Counters:
        counters = await self.catch_up(memory_type)
        old, new = self._pair(memory_type)
        old_n, new_n = self._count(old), self._count(new)
        gap = old_n - new_n - len(counters.skipped_no_text)
        if gap > 0 and not force:
            raise ReindexError(
                f"{memory_type}: {new} has {new_n} points, {old} has {old_n} "
                f"({len(counters.skipped_no_text)} without text, {len(counters.failed)} "
                "failed). Not swapping; fix or pass --force."
            )
        self.registry.point_alias(_registry.live_alias(memory_type), new)
        self.echo(f"  {memory_type}: alias now -> {new} (was {old})")
        return counters

    def rollback(self, memory_type: str) -> str:
        live = self.live(memory_type)
        record = self.registry.get(live) or {}
        previous = record.get("built_from")
        if not previous or not self.registry.physical_exists(previous):
            raise ReindexError(f"{live} has no previous collection to roll back to")
        self.registry.point_alias(_registry.live_alias(memory_type), previous)
        self.echo(f"  {memory_type}: alias now -> {previous} (was {live})")
        return previous


async def _make_reindexer(batch: int) -> Reindexer:
    from qdrant_client import QdrantClient  # noqa: PLC0415

    from src.services.vector_service import create_vector_service  # noqa: PLC0415

    service = create_vector_service()
    embedder = service._build_embedder()
    if not await embedder.initialize():
        raise ReindexError(f"embedder {service.embed_provider} failed to initialize")
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )
    return Reindexer(client, embedder, batch_size=batch)


async def _run(args: argparse.Namespace) -> int:
    types = args.type or list(_templates.MEMORY_TYPES)
    reindexer = await _make_reindexer(args.batch)
    print(
        f"embedder: {reindexer.profile.provider} {reindexer.profile.model} "
        f"fingerprint {reindexer.profile.short_fingerprint}"
    )
    try:
        for memory_type in types:
            if args.command == "status":
                print(json.dumps(reindexer.status(memory_type)))
            elif args.command == "build":
                result = await reindexer.build(memory_type, dry_run=args.dry_run)
                print(json.dumps({"type": memory_type, **_summary(result)}))
            elif args.command == "catch-up":
                result = await reindexer.catch_up(memory_type)
                print(json.dumps({"type": memory_type, **_summary(result)}))
            elif args.command == "swap":
                result = await reindexer.swap(memory_type, force=args.force)
                print(json.dumps({"type": memory_type, **_summary(result)}))
            elif args.command == "rollback":
                reindexer.rollback(memory_type)
    except ReindexError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        await reindexer.embedder.close()
    return 0


def _summary(c: Counters) -> dict[str, Any]:
    data = asdict(c)
    data["skipped_no_text"] = len(c.skipped_no_text)
    data["failed"] = len(c.failed)
    return data


def main(argv: list[str] | None = None) -> int:
    try:  # honour a local .env like the server does
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()
    except ImportError:
        pass
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.reindex", description=__doc__.split("\n\n")[1]
    )
    parser.add_argument(
        "command", choices=["status", "build", "catch-up", "swap", "rollback"]
    )
    parser.add_argument(
        "--type",
        action="append",
        choices=list(_templates.MEMORY_TYPES),
        help="memory type (repeatable; default: all five)",
    )
    parser.add_argument("--batch", type=int, default=32, help="points per batch")
    parser.add_argument("--dry-run", action="store_true", help="build: count only")
    parser.add_argument(
        "--force", action="store_true", help="swap even if the counts differ"
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

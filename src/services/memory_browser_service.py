"""
Memory Browser service (Feature 039).

Read-only access to every Mnemodrome memory layer for the Admin UI:

- facts and entities: Neo4j (graph layer) via ``neo4j_service.query``
- semantic search: Qdrant (vector layer) via ``get_vector_service().search``
- events: the event log service
- learnings, sessions, procedures, updates, graphs: the markdown file layer
  resolved through ``get_memory_path`` (no copy, no extra index store)

Directory listings of large NFS folders are slow, so the file layer keeps an
in-memory index per memory type with a TTL and a manual refresh.

Nothing in this module writes to any store.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import MemoryType, get_memory_path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FILE_INDEX_TTL_SECONDS = int(os.getenv("MEMORY_BROWSER_FILE_INDEX_TTL", "300"))
MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 50
MAX_FILE_BYTES = 2 * 1024 * 1024  # never stream more than 2 MB into the browser

VALIDITY_CHOICES = ("all", "valid", "invalidated", "superseded")

# Filename safety: plain names only, no separators, no traversal.
_SAFE_NAME = re.compile(r"^[^/\\\0]+$")
_FACT_FILE = re.compile(r"^fact-(?P<slug>.+)-(?P<ts>\d{8}-\d{6})\.md$")
_FRONTMATTER = re.compile(r"\A---\s*\n(?P<yaml>.*?)\n---\s*\n?", re.DOTALL)
_BOLD_KEY_LINE = re.compile(r"^\*\*(?P<key>[^*]+)\*\*:\s*(?P<value>.+?)\s*$")
_TITLE_LINE = re.compile(r"^#\s+(?P<title>.+?)\s*$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(value: Any) -> Any:
    """Return ISO 8601 text for datetimes; pass other values through."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _jsonable(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _iso(v) for k, v in record.items()}


def clamp_page(offset: int | None, limit: int | None) -> tuple[int, int]:
    """Clamp pagination inputs to safe bounds."""
    offset = max(0, int(offset or 0))
    limit = int(limit or DEFAULT_PAGE_SIZE)
    limit = max(1, min(MAX_PAGE_SIZE, limit))
    return offset, limit


def safe_filename(name: str) -> str:
    """Reject anything that is not a plain file name inside one folder."""
    if not name or name in {".", ".."} or not _SAFE_NAME.match(name):
        raise ValueError("Invalid file name")
    return name


def entity_slug(name: str) -> str:
    """Slug used by capture_fact when it names the fact markdown file."""
    return name.lower().replace(" ", "-")


def parse_header(text: str) -> tuple[dict[str, str], str, str]:
    """
    Split a memory markdown file into (header, title, body).

    Supports real YAML frontmatter (``---`` blocks, simple ``key: value``
    lines only) and the ``**Key**: value`` lines Mnemodrome writes today.
    """
    header: dict[str, str] = {}
    title = ""
    body = text

    match = _FRONTMATTER.match(text)
    if match:
        for line in match.group("yaml").splitlines():
            if ":" in line and not line.startswith((" ", "\t", "-")):
                key, _, value = line.partition(":")
                header[key.strip()] = value.strip().strip("\"'")
        body = text[match.end() :]

    for line in body.splitlines()[:40]:
        if not title:
            t = _TITLE_LINE.match(line)
            if t:
                title = t.group("title")
                continue
        b = _BOLD_KEY_LINE.match(line)
        if b:
            header.setdefault(b.group("key").strip(), b.group("value").strip("` "))

    if not title:
        title = header.get("title", "")
    return header, title, body


def render_markdown(text: str) -> str:
    """Render markdown to HTML server-side; fall back to escaped <pre>."""
    try:
        from markdown_it import MarkdownIt

        md = MarkdownIt("commonmark", {"html": False, "linkify": False})
        md.enable("table")
        return md.render(text)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Markdown render failed, using plain text: {e}")
        return f"<pre>{html.escape(text)}</pre>"


# ---------------------------------------------------------------------------
# File-layer index
# ---------------------------------------------------------------------------


@dataclass
class FileEntry:
    name: str
    mtime: float
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modified": datetime.fromtimestamp(self.mtime, tz=UTC).isoformat(),
            "size": self.size,
        }


@dataclass
class _TypeIndex:
    entries: list[FileEntry] = field(default_factory=list)
    built_at: float = 0.0
    scan_seconds: float = 0.0
    error: str | None = None


class FileIndex:
    """In-memory, TTL-cached listing of each memory-type folder."""

    def __init__(self, ttl_seconds: int = FILE_INDEX_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._indexes: dict[MemoryType, _TypeIndex] = {}
        self._locks: dict[MemoryType, asyncio.Lock] = {}

    def path_for(self, memory_type: MemoryType) -> Path:
        return get_memory_path(memory_type)

    def _lock(self, memory_type: MemoryType) -> asyncio.Lock:
        return self._locks.setdefault(memory_type, asyncio.Lock())

    @staticmethod
    def _scan(folder: Path) -> list[FileEntry]:
        entries: list[FileEntry] = []
        with os.scandir(folder) as it:
            for de in it:
                if not de.name.endswith(".md") or de.name.startswith("."):
                    continue
                try:
                    st = de.stat()
                except OSError:
                    continue
                if not de.is_file():
                    continue
                entries.append(FileEntry(de.name, st.st_mtime, st.st_size))
        entries.sort(key=lambda e: e.mtime, reverse=True)
        return entries

    async def get(self, memory_type: MemoryType, force: bool = False) -> _TypeIndex:
        idx = self._indexes.get(memory_type)
        fresh = idx is not None and (time.time() - idx.built_at) < self.ttl_seconds
        if fresh and not force:
            return idx  # type: ignore[return-value]
        async with self._lock(memory_type):
            idx = self._indexes.get(memory_type)
            fresh = idx is not None and (time.time() - idx.built_at) < self.ttl_seconds
            if fresh and not force:
                return idx  # type: ignore[return-value]
            new = _TypeIndex()
            started = time.perf_counter()
            try:
                folder = self.path_for(memory_type)
                if folder.is_dir():
                    new.entries = await asyncio.to_thread(self._scan, folder)
                else:
                    new.error = f"Folder does not exist: {folder}"
            except Exception as e:
                new.error = str(e)
                logger.warning(f"File index scan failed for {memory_type.value}: {e}")
            new.scan_seconds = round(time.perf_counter() - started, 3)
            new.built_at = time.time()
            self._indexes[memory_type] = new
            return new

    async def refresh(self, memory_type: MemoryType | None = None) -> dict[str, Any]:
        types = [memory_type] if memory_type else list(MemoryType)
        out: dict[str, Any] = {}
        for mt in types:
            idx = await self.get(mt, force=True)
            out[mt.value] = {
                "count": len(idx.entries),
                "scan_seconds": idx.scan_seconds,
                "error": idx.error,
            }
        return out

    async def status(self, memory_type: MemoryType) -> dict[str, Any]:
        idx = await self.get(memory_type)
        return {
            "count": len(idx.entries),
            "built_at": datetime.fromtimestamp(idx.built_at, tz=UTC).isoformat(),
            "age_seconds": round(time.time() - idx.built_at, 1),
            "ttl_seconds": self.ttl_seconds,
            "scan_seconds": idx.scan_seconds,
            "error": idx.error,
        }

    async def page(
        self,
        memory_type: MemoryType,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
        query: str | None = None,
    ) -> dict[str, Any]:
        idx = await self.get(memory_type)
        entries = idx.entries
        if query:
            q = query.lower()
            entries = [e for e in entries if q in e.name.lower()]
        total = len(entries)
        page = [e.as_dict() for e in entries[offset : offset + limit]]
        return {
            "type": memory_type.value,
            "path": str(self.path_for(memory_type)),
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": page,
            "index": await self.status(memory_type),
        }

    def resolve(self, memory_type: MemoryType, name: str) -> Path:
        """Return the on-disk path for a file, refusing anything outside the folder."""
        name = safe_filename(name)
        folder = self.path_for(memory_type).resolve()
        target = (folder / name).resolve()
        if target.parent != folder:
            raise ValueError("Invalid file name")
        return target

    async def read(self, memory_type: MemoryType, name: str) -> dict[str, Any]:
        target = self.resolve(memory_type, name)
        if not target.is_file():
            raise FileNotFoundError(name)
        size = target.stat().st_size
        truncated = size > MAX_FILE_BYTES
        raw = await asyncio.to_thread(_read_head, target, MAX_FILE_BYTES)
        header, title, body = parse_header(raw)
        return {
            "type": memory_type.value,
            "name": name,
            "path": str(target),
            "size": size,
            "truncated": truncated,
            "modified": datetime.fromtimestamp(
                target.stat().st_mtime, tz=UTC
            ).isoformat(),
            "title": title or name,
            "header": header,
            "html": render_markdown(body),
            "raw": raw,
        }

    async def find_fact_file(
        self, entity1: str | None, created_at: Any
    ) -> dict[str, Any] | None:
        """
        Find the markdown file capture_fact wrote for a Neo4j relationship.

        The file name is ``fact-<entity1 slug>-<local YYYYmmdd-HHMMSS>.md`` and
        does not embed the relationship uuid, so we match on slug plus a
        timestamp window around ``created_at`` (checked as local time and UTC).
        """
        if not entity1 or created_at is None:
            return None
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                return None
        if not isinstance(created_at, datetime):
            return None
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)

        idx = await self.get(MemoryType.FACT)
        prefix = f"fact-{entity_slug(entity1)}-"
        candidates = []
        for e in idx.entries:
            if not e.name.startswith(prefix):
                continue
            m = _FACT_FILE.match(e.name)
            if not m or m.group("slug") != entity_slug(entity1):
                continue
            candidates.append((e, m.group("ts")))
        if not candidates:
            return None

        local_target = created_at.astimezone().replace(tzinfo=None)
        utc_target = created_at.astimezone(UTC).replace(tzinfo=None)
        window = timedelta(seconds=120)
        best: tuple[timedelta, FileEntry] | None = None
        for entry, ts in candidates:
            try:
                file_ts = datetime.strptime(ts, "%Y%m%d-%H%M%S")
            except ValueError:
                continue
            delta = min(abs(file_ts - local_target), abs(file_ts - utc_target))
            if delta <= window and (best is None or delta < best[0]):
                best = (delta, entry)
        if best is None:
            return None
        return {"type": MemoryType.FACT.value, **best[1].as_dict()}


def _read_head(path: Path, max_bytes: int) -> str:
    with open(path, "rb") as fh:
        data = fh.read(max_bytes)
    return data.decode("utf-8", errors="replace")


file_index = FileIndex()


# ---------------------------------------------------------------------------
# Graph layer (Neo4j)
# ---------------------------------------------------------------------------

# Facts are RELATES_TO edges between any two named nodes. Two generations of
# edges coexist: current ones store the relation in ``r.type`` and context in
# ``r.context``; older imported ones use ``r.name`` and ``r.fact`` and keep
# ``created_at`` as an ISO string. The browser reads both.
_FACT_MATCH = "MATCH (s)-[r:RELATES_TO]->(t)"
_REL_EXPR = "coalesce(r.type, r.name)"
_CREATED_EXPR = "toString(r.created_at)"
_FACT_RETURN = f"""
RETURN r.uuid AS uuid,
       s.name AS entity1, s.uuid AS entity1_uuid,
       coalesce(s.entity_type, head(labels(s))) AS entity1_type,
       {_REL_EXPR} AS relation,
       t.name AS entity2, t.uuid AS entity2_uuid,
       coalesce(t.entity_type, head(labels(t))) AS entity2_type,
       coalesce(r.context, r.fact) AS context,
       r.confidence AS confidence,
       r.created_at AS created_at,
       r.valid_at AS valid_at,
       r.invalid_at AS invalid_at,
       r.expired_at AS expired_at,
       r.invalidation_reason AS invalidation_reason,
       r.superseded_by AS superseded_by,
       r.session_id AS session_id,
       r.source AS source
"""
# An "entity" is any named node that takes part in at least one fact.
_ENTITY_MATCH = (
    "MATCH (e) WHERE e.name IS NOT NULL AND EXISTS {{ (e)-[:RELATES_TO]-() }}"
)


def _validity_clause(validity: str) -> str:
    if validity == "valid":
        return (
            "r.invalid_at IS NULL AND r.superseded_by IS NULL "
            "AND (r.expired_at IS NULL OR datetime(r.expired_at) > datetime())"
        )
    if validity == "invalidated":
        return (
            "r.superseded_by IS NULL AND (r.invalid_at IS NOT NULL "
            "OR (r.expired_at IS NOT NULL AND datetime(r.expired_at) <= datetime()))"
        )
    if validity == "superseded":
        return "r.superseded_by IS NOT NULL"
    return "true"


def fact_status(row: dict[str, Any]) -> str:
    """Derive a display status from the temporal properties of a fact."""
    if row.get("superseded_by"):
        return "superseded"
    if row.get("invalid_at"):
        return "invalidated"
    expired = row.get("expired_at")
    if expired:
        if isinstance(expired, str):
            try:
                expired = datetime.fromisoformat(expired.replace("Z", "+00:00"))
            except ValueError:
                return "invalidated"
        if isinstance(expired, datetime):
            if expired.tzinfo is None:
                expired = expired.replace(tzinfo=UTC)
            if expired <= datetime.now(UTC):
                return "invalidated"
    return "valid"


def _parse_date(value: str | None, end_of_day: bool = False) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"Invalid date: {value}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if end_of_day and len(value) == 10:
        dt = dt + timedelta(days=1)
    return dt.isoformat()


class GraphBrowser:
    """Paginated, filtered read access to facts and entities in Neo4j."""

    def __init__(self, neo4j=None):
        self._neo4j = neo4j

    @property
    def neo4j(self):
        if self._neo4j is None:
            from src.services.neo4j_service import neo4j_service

            self._neo4j = neo4j_service
        return self._neo4j

    async def counts(self) -> dict[str, Any]:
        rows = await self.neo4j.query(
            f"""
            MATCH ()-[r:RELATES_TO]->()
            WITH count(r) AS facts,
                 sum(CASE WHEN {_validity_clause("valid")} THEN 1 ELSE 0 END) AS valid_facts
            {_ENTITY_MATCH.format()}
            RETURN facts, valid_facts, count(e) AS entities
            """
        )
        if not rows:
            return {"facts": 0, "valid_facts": 0, "entities": 0}
        return {k: int(v or 0) for k, v in rows[0].items()}

    async def relation_types(self) -> list[dict[str, Any]]:
        rows = await self.neo4j.query(
            f"""
            MATCH ()-[r:RELATES_TO]->()
            WITH {_REL_EXPR} AS relation, count(r) AS count
            WHERE relation IS NOT NULL
            RETURN relation, count
            ORDER BY count DESC, relation
            LIMIT 500
            """
        )
        return [r for r in rows if r.get("relation")]

    async def list_facts(
        self,
        entity: str | None = None,
        relation: str | None = None,
        validity: str = "all",
        since: str | None = None,
        until: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
    ) -> dict[str, Any]:
        if validity not in VALIDITY_CHOICES:
            raise ValueError(f"validity must be one of {VALIDITY_CHOICES}")
        offset, limit = clamp_page(offset, limit)
        where = [_validity_clause(validity)]
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if entity:
            where.append(
                "(toLower(s.name) CONTAINS toLower($entity) "
                "OR toLower(t.name) CONTAINS toLower($entity))"
            )
            params["entity"] = entity
        if relation:
            where.append(f"{_REL_EXPR} = $relation")
            params["relation"] = relation
        # created_at is a datetime on new edges and an ISO string on imported
        # ones; comparing the ISO text form works for both.
        since_iso = _parse_date(since)
        until_iso = _parse_date(until, end_of_day=True)
        if since_iso:
            where.append(f"{_CREATED_EXPR} >= $since")
            params["since"] = since_iso
        if until_iso:
            where.append(f"{_CREATED_EXPR} < $until")
            params["until"] = until_iso
        where_sql = " AND ".join(where)

        count_rows = await self.neo4j.query(
            f"{_FACT_MATCH} WHERE {where_sql} RETURN count(r) AS total",
            params,
        )
        total = int(count_rows[0]["total"]) if count_rows else 0
        rows = await self.neo4j.query(
            f"{_FACT_MATCH} WHERE {where_sql} {_FACT_RETURN} "
            f"ORDER BY {_CREATED_EXPR} DESC SKIP $offset LIMIT $limit",
            params,
        )
        items = []
        for row in rows:
            item = _jsonable(row)
            item["status"] = fact_status(row)
            items.append(item)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": items,
            "filters": {
                "entity": entity,
                "relation": relation,
                "validity": validity,
                "since": since,
                "until": until,
            },
        }

    async def get_fact(self, uuid: str) -> dict[str, Any] | None:
        rows = await self.neo4j.query(
            f"MATCH (s)-[r:RELATES_TO {{uuid: $uuid}}]->(t) {_FACT_RETURN} LIMIT 1",
            {"uuid": uuid},
        )
        if not rows:
            return None
        row = rows[0]
        item = _jsonable(row)
        item["status"] = fact_status(row)
        # Supersession chain neighbours (cheap, by uuid).
        chain = await self.neo4j.query(
            """
            OPTIONAL MATCH ()-[nxt:RELATES_TO {uuid: $next}]->()
            OPTIONAL MATCH ()-[prev:RELATES_TO {superseded_by: $uuid}]->()
            RETURN nxt.uuid AS next_uuid, collect(DISTINCT prev.uuid) AS previous_uuids
            """,
            {"uuid": uuid, "next": row.get("superseded_by") or ""},
        )
        if chain:
            item["next_uuid"] = chain[0].get("next_uuid")
            item["previous_uuids"] = [
                u for u in chain[0].get("previous_uuids") or [] if u
            ]
        item["file"] = await file_index.find_fact_file(
            row.get("entity1"), row.get("created_at")
        )
        return item

    async def list_entities(
        self, query: str | None = None, offset: int = 0, limit: int = DEFAULT_PAGE_SIZE
    ) -> dict[str, Any]:
        offset, limit = clamp_page(offset, limit)
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        match = _ENTITY_MATCH.format()
        if query:
            match += " AND toLower(e.name) CONTAINS toLower($q)"
            params["q"] = query
        count_rows = await self.neo4j.query(f"{match} RETURN count(e) AS total", params)
        total = int(count_rows[0]["total"]) if count_rows else 0
        rows = await self.neo4j.query(
            f"""
            {match}
            OPTIONAL MATCH (e)-[out:RELATES_TO]->()
            WITH e, count(out) AS outgoing
            OPTIONAL MATCH ()-[inc:RELATES_TO]->(e)
            WITH e, outgoing, count(inc) AS incoming
            RETURN e.uuid AS uuid, e.name AS name,
                   coalesce(e.entity_type, head(labels(e))) AS entity_type,
                   e.summary AS summary, e.created_at AS created_at,
                   outgoing, incoming, outgoing + incoming AS degree
            ORDER BY degree DESC, name
            SKIP $offset LIMIT $limit
            """,
            params,
        )
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [_jsonable(r) for r in rows],
            "query": query,
        }

    async def get_entity(
        self, uuid: str, limit: int = MAX_PAGE_SIZE
    ) -> dict[str, Any] | None:
        rows = await self.neo4j.query(
            """
            MATCH (e {uuid: $uuid})
            RETURN e.uuid AS uuid, e.name AS name,
                   coalesce(e.entity_type, head(labels(e))) AS entity_type,
                   e.summary AS summary, e.created_at AS created_at,
                   e.access_count AS access_count, e.last_accessed AS last_accessed
            LIMIT 1
            """,
            {"uuid": uuid},
        )
        if not rows:
            return None
        entity = _jsonable(rows[0])
        limit = int(limit)
        rel_return = f"""
            RETURN r.uuid AS uuid, {_REL_EXPR} AS relation,
                   o.uuid AS other_uuid, o.name AS other_name,
                   coalesce(o.entity_type, head(labels(o))) AS other_type,
                   r.created_at AS created_at, r.invalid_at AS invalid_at,
                   r.expired_at AS expired_at, r.superseded_by AS superseded_by,
                   r.confidence AS confidence
            ORDER BY {_CREATED_EXPR} DESC
            LIMIT {limit}
        """
        outgoing = await self.neo4j.query(
            f"MATCH (e {{uuid: $uuid}})-[r:RELATES_TO]->(o) {rel_return}",
            {"uuid": uuid},
        )
        incoming = await self.neo4j.query(
            f"MATCH (o)-[r:RELATES_TO]->(e {{uuid: $uuid}}) {rel_return}",
            {"uuid": uuid},
        )
        relationships = []
        for direction, rows_ in (("out", outgoing), ("in", incoming)):
            for r in rows_:
                item = _jsonable(r)
                item["direction"] = direction
                item["status"] = fact_status(r)
                relationships.append(item)
        relationships.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
        truncated = len(relationships) > limit
        entity["relationships"] = relationships[:limit]
        entity["relationship_count"] = len(entity["relationships"])
        entity["truncated"] = (
            truncated or len(outgoing) >= limit or len(incoming) >= limit
        )
        return entity


graph_browser = GraphBrowser()


# ---------------------------------------------------------------------------
# Vector layer (Qdrant)
# ---------------------------------------------------------------------------


_browser_vector_service = None
_browser_vector_lock: asyncio.Lock | None = None


async def get_browser_vector_service():
    """
    Vector service owned by the Admin UI's event loop.

    The MCP server's singleton is created and initialized on the MCP loop and
    its embedder holds a loop-bound httpx.AsyncClient, so the browser keeps its
    own instance (same configuration, same collections). Initialization is
    lazy and idempotent; a failure leaves the instance disabled, not raised.
    """
    global _browser_vector_service, _browser_vector_lock
    if _browser_vector_lock is None:
        _browser_vector_lock = asyncio.Lock()
    async with _browser_vector_lock:
        if _browser_vector_service is None:
            from src.services.vector_service import create_vector_service

            _browser_vector_service = create_vector_service()
        vs = _browser_vector_service
        if vs.enabled and not vs._initialized:
            try:
                await vs.initialize()
            except Exception as e:
                logger.warning(f"Memory browser: vector service init failed: {e}")
        return vs


async def semantic_search(
    query: str,
    search_type: str = "all",
    limit: int = 20,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Run the existing vector search and shape results for the browser."""
    vs = await get_browser_vector_service()
    if not vs.is_enabled():
        return {
            "query": query,
            "search_type": search_type,
            "enabled": False,
            "items": [],
            "error": "Vector service is disabled or not initialized",
        }
    limit = max(1, min(MAX_PAGE_SIZE, int(limit)))
    raw = await vs.search(
        query_text=query, search_type=search_type, limit=limit, min_score=min_score
    )
    items = []
    for r in raw:
        kind = r.get("type") or search_type
        item = {
            "type": kind,
            "score": r.get("score"),
            "neo4j_uuid": r.get("neo4j_uuid"),
            "timestamp": _iso(r.get("timestamp") or r.get("created_at")),
        }
        if kind == "fact":
            item["title"] = (
                f"{r.get('entity1', '?')} {r.get('relation', '?')} {r.get('entity2', '?')}"
            )
            item["snippet"] = r.get("context") or ""
            item["link"] = (
                f"/admin/memory/facts?uuid={r.get('neo4j_uuid')}"
                if r.get("neo4j_uuid")
                else None
            )
        elif kind == "learning":
            item["title"] = r.get("title") or "Learning"
            item["snippet"] = (r.get("content") or "")[:300]
            item["link"] = f"/admin/memory/learnings?q={_q(r.get('title'))}"
        elif kind == "session":
            item["title"] = (r.get("summary") or "Session")[:120]
            item["snippet"] = (r.get("key_insights") or r.get("summary") or "")[:300]
            item["link"] = "/admin/memory/sessions"
        else:
            item["title"] = (r.get("content") or r.get("matched_text") or kind)[:120]
            item["snippet"] = (r.get("content") or "")[:300]
            item["link"] = None
        items.append(item)
    return {
        "query": query,
        "search_type": search_type,
        "enabled": True,
        "items": items,
        "total": len(items),
    }


def _q(value: Any) -> str:
    from urllib.parse import quote_plus

    return quote_plus(str(value or ""))


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


async def overview() -> dict[str, Any]:
    """Counts, paths, and health for every layer. Never raises."""
    out: dict[str, Any] = {"generated_at": datetime.now(UTC).isoformat()}

    # Graph layer
    try:
        out["graph"] = {"status": "healthy", **(await graph_browser.counts())}
    except Exception as e:
        out["graph"] = {"status": "error", "error": str(e)}

    # Vector layer (browser-owned instance, see get_browser_vector_service)
    try:
        health = await (await get_browser_vector_service()).health_check()
        stats = health.get("stats") or {}
        out["vector"] = {
            "status": health.get("status"),
            "collections": {
                name: (info or {}).get("points_count") for name, info in stats.items()
            },
            "total_points": sum(
                int((info or {}).get("points_count") or 0) for info in stats.values()
            ),
            "error": health.get("error"),
        }
    except Exception as e:
        out["vector"] = {"status": "error", "error": str(e)}

    # Event log
    try:
        from src import config
        from src.services.event_log_service import event_log_service

        if event_log_service is None:
            raise RuntimeError("event log service not available")
        health = await event_log_service.get_event_log_health()
        out["events"] = {
            "status": health.get("status"),
            "total": health.get("total_events"),
            "newest": health.get("newest_event"),
            "oldest": health.get("oldest_event"),
            "backend": getattr(config, "EVENT_LOG_BACKEND", "sqlite"),
            "path": getattr(config, "EVENT_LOG_DB_PATH", None),
            "error": health.get("error"),
        }
    except Exception as e:
        out["events"] = {"status": "error", "error": str(e)}

    # File layer
    files: dict[str, Any] = {}
    for mt in MemoryType:
        try:
            status = await file_index.status(mt)
            files[mt.value] = {
                "path": str(file_index.path_for(mt)),
                "status": "error" if status["error"] else "healthy",
                **status,
            }
        except Exception as e:
            files[mt.value] = {"status": "error", "error": str(e)}
    out["files"] = files
    out["file_index_ttl_seconds"] = file_index.ttl_seconds
    return out

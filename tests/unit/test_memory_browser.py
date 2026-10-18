"""Unit tests for the Memory Browser (Feature 039): service helpers and routes."""

import base64
import os
import shutil
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.admin_ui import app
from src.config import MEMORY_PATH_ENV_VARS, MemoryType
from src.services import memory_browser_service as mb

AUTH = {"Authorization": "Basic " + base64.b64encode(b"admin:test-password").decode()}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_clamp_page(self):
        assert mb.clamp_page(None, None) == (0, mb.DEFAULT_PAGE_SIZE)
        assert mb.clamp_page(-5, 9999) == (0, mb.MAX_PAGE_SIZE)
        assert mb.clamp_page(10, 0) == (10, mb.DEFAULT_PAGE_SIZE)

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "x\0y"])
    def test_safe_filename_rejects(self, bad):
        with pytest.raises(ValueError):
            mb.safe_filename(bad)

    def test_safe_filename_accepts_plain(self):
        assert mb.safe_filename("learning-x-20260101-000000.md") == (
            "learning-x-20260101-000000.md"
        )

    def test_parse_header_bold_keys_and_title(self):
        header, title, body = mb.parse_header(
            "# Learning: Foo\n\n**Category**: general\n**UUID**: `abc`\n\nbody\n"
        )
        assert title == "Learning: Foo"
        assert header["Category"] == "general"
        assert header["UUID"] == "abc"
        assert "body" in body

    def test_parse_header_yaml_frontmatter(self):
        header, title, body = mb.parse_header(
            "---\ntitle: Hello\ncategory: ops\n---\n# Heading\ntext\n"
        )
        assert header == {"title": "Hello", "category": "ops"}
        assert title == "Heading"
        assert body.startswith("# Heading")

    def test_render_markdown_disables_raw_html(self):
        out = mb.render_markdown("# T\n\n<script>alert(1)</script>\n\n**b**")
        assert "<h1>" in out
        assert "<strong>b</strong>" in out
        assert "<script>" not in out

    def test_fact_status(self):
        assert mb.fact_status({}) == "valid"
        assert (
            mb.fact_status({"invalid_at": "2026-01-01T00:00:00+00:00"}) == "invalidated"
        )
        assert mb.fact_status({"superseded_by": "x", "invalid_at": "y"}) == "superseded"
        past = datetime(2000, 1, 1, tzinfo=UTC)
        assert mb.fact_status({"expired_at": past}) == "invalidated"
        future = datetime(2999, 1, 1, tzinfo=UTC)
        assert mb.fact_status({"expired_at": future}) == "valid"

    def test_validity_clause_choices(self):
        assert mb._validity_clause("all") == "true"
        assert "superseded_by IS NOT NULL" in mb._validity_clause("superseded")
        assert "invalid_at IS NULL" in mb._validity_clause("valid")


# ---------------------------------------------------------------------------
# File index
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_root(tmp_path, monkeypatch):
    """Umbrella storage path with a few files per type."""
    monkeypatch.setenv("MNEMODROME_STORAGE_PATH", str(tmp_path))
    for var in MEMORY_PATH_ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    for mt in MemoryType:
        (tmp_path / mt.value).mkdir()
    learn = tmp_path / "learnings"
    for i in range(3):
        f = learn / f"learning-item-{i}-2026010{i}-000000.md"
        f.write_text(f"# Learning: Item {i}\n\n**Category**: t\n\ncontent {i}\n")
        ts = time.time() - (10 - i) * 60  # newer file for larger i
        os.utime(f, (ts, ts))
    (learn / "notes.txt").write_text("ignored")
    (learn / ".hidden.md").write_text("ignored")
    facts = tmp_path / "facts"
    (facts / "fact-neo4j-20260515-034653.md").write_text("# Fact: Neo4j USES Cypher\n")
    return tmp_path


@pytest.mark.asyncio
async def test_file_index_pages_newest_first(memory_root):
    idx = mb.FileIndex(ttl_seconds=300)
    page = await idx.page(MemoryType.LEARNING, offset=0, limit=2)
    assert page["total"] == 3
    assert [i["name"] for i in page["items"]] == [
        "learning-item-2-20260102-000000.md",
        "learning-item-1-20260101-000000.md",
    ]
    page2 = await idx.page(MemoryType.LEARNING, offset=2, limit=2)
    assert [i["name"] for i in page2["items"]] == ["learning-item-0-20260100-000000.md"]
    filtered = await idx.page(MemoryType.LEARNING, query="item-1")
    assert filtered["total"] == 1


@pytest.mark.asyncio
async def test_file_index_ttl_and_refresh(memory_root):
    idx = mb.FileIndex(ttl_seconds=3600)
    first = await idx.page(MemoryType.LEARNING)
    assert first["total"] == 3
    (memory_root / "learnings" / "learning-new-20260201-000000.md").write_text(
        "# New\n"
    )
    cached = await idx.page(MemoryType.LEARNING)
    assert cached["total"] == 3, "TTL cache must not rescan on every request"
    result = await idx.refresh(MemoryType.LEARNING)
    assert result["learnings"]["count"] == 4
    assert (await idx.page(MemoryType.LEARNING))["total"] == 4


@pytest.mark.asyncio
async def test_file_index_read_renders_and_blocks_traversal(memory_root):
    idx = mb.FileIndex()
    doc = await idx.read(MemoryType.LEARNING, "learning-item-1-20260101-000000.md")
    assert doc["title"] == "Learning: Item 1"
    assert doc["header"]["Category"] == "t"
    assert "<h1>" in doc["html"]
    with pytest.raises(ValueError):
        await idx.read(MemoryType.LEARNING, "../facts/fact-neo4j-20260515-034653.md")
    with pytest.raises(FileNotFoundError):
        await idx.read(MemoryType.LEARNING, "missing.md")


@pytest.mark.asyncio
async def test_file_index_missing_folder_reports_error(memory_root):
    shutil.rmtree(memory_root / "updates")
    idx = mb.FileIndex()
    page = await idx.page(MemoryType.UPDATE)
    assert page["total"] == 0
    assert "does not exist" in page["index"]["error"]


@pytest.mark.asyncio
async def test_find_fact_file_matches_slug_and_time(memory_root):
    idx = mb.FileIndex()
    # capture_fact names the file with the local wall-clock time.
    local = datetime(2026, 5, 15, 3, 46, 53).astimezone()
    found = await idx.find_fact_file("Neo4j", local)
    assert found and found["name"] == "fact-neo4j-20260515-034653.md"
    assert await idx.find_fact_file("Neo4j", datetime(2020, 1, 1, tzinfo=UTC)) is None
    assert await idx.find_fact_file("Other", local) is None


# ---------------------------------------------------------------------------
# Graph browser (mocked Neo4j)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_facts_builds_filters_and_paginates():
    neo = MagicMock()
    neo.query = AsyncMock(
        side_effect=[
            [{"total": 2}],
            [
                {
                    "uuid": "u1",
                    "entity1": "A",
                    "relation": "USES",
                    "entity2": "B",
                    "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                    "superseded_by": None,
                    "invalid_at": None,
                    "expired_at": None,
                }
            ],
        ]
    )
    gb = mb.GraphBrowser(neo4j=neo)
    out = await gb.list_facts(
        entity="a",
        relation="USES",
        validity="valid",
        since="2026-01-01",
        until="2026-02-01",
        offset=10,
        limit=5,
    )
    assert out["total"] == 2
    assert out["items"][0]["status"] == "valid"
    assert out["items"][0]["created_at"] == "2026-01-01T00:00:00+00:00"
    cypher, params = neo.query.call_args_list[1].args
    assert "SKIP $offset LIMIT $limit" in cypher
    assert "coalesce(r.type, r.name) = $relation" in cypher
    assert "CONTAINS toLower($entity)" in cypher
    assert params["offset"] == 10 and params["limit"] == 5
    assert params["since"].startswith("2026-01-01")
    assert params["until"].startswith("2026-02-02")  # inclusive end of day


@pytest.mark.asyncio
async def test_list_facts_rejects_bad_validity():
    gb = mb.GraphBrowser(neo4j=MagicMock())
    with pytest.raises(ValueError):
        await gb.list_facts(validity="bogus")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with patch("src.admin_ui.initialize_config_manager"):
        with TestClient(app) as c:
            yield c


class TestMemoryRoutes:
    def test_requires_auth(self, client):
        assert client.get("/admin/memory").status_code == 401
        assert client.get("/admin/api/memory/overview").status_code == 401

    def test_pages_render(self, client):
        views = [
            "",
            "/facts",
            "/entities",
            "/search",
            "/learnings",
            "/sessions",
            "/procedures",
            "/updates",
            "/graphs",
        ]
        for view in views:
            r = client.get(f"/admin/memory{view}", headers=AUTH)
            assert r.status_code == 200, view
            assert "memory.js" in r.text
        assert client.get("/admin/memory/nope", headers=AUTH).status_code == 404

    def test_overview_never_fails_when_backends_down(self, client):
        with patch.object(
            mb.graph_browser, "counts", AsyncMock(side_effect=RuntimeError("down"))
        ):
            r = client.get("/admin/api/memory/overview", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["graph"]["status"] == "error"
        assert set(body["files"]) == {mt.value for mt in MemoryType}

    def test_facts_listing_and_validation(self, client):
        fake = {"total": 0, "offset": 0, "limit": 50, "items": [], "filters": {}}
        with patch.object(
            mb.graph_browser, "list_facts", AsyncMock(return_value=fake)
        ) as m:
            r = client.get(
                "/admin/api/memory/facts?entity=x&validity=valid&limit=10&offset=20",
                headers=AUTH,
            )
            assert r.status_code == 200 and r.json()["total"] == 0
            assert m.call_args.kwargs["validity"] == "valid"
            assert m.call_args.kwargs["limit"] == 10
        r = client.get("/admin/api/memory/facts?limit=999", headers=AUTH)
        assert r.status_code == 422
        with patch.object(
            mb.graph_browser, "list_facts", AsyncMock(side_effect=ValueError("bad"))
        ):
            r = client.get("/admin/api/memory/facts?validity=bad", headers=AUTH)
            assert r.status_code == 400
        with patch.object(
            mb.graph_browser, "list_facts", AsyncMock(side_effect=RuntimeError("x"))
        ):
            assert (
                client.get("/admin/api/memory/facts", headers=AUTH).status_code == 503
            )

    def test_fact_detail_404(self, client):
        with patch.object(mb.graph_browser, "get_fact", AsyncMock(return_value=None)):
            r = client.get("/admin/api/memory/facts/abc", headers=AUTH)
            assert r.status_code == 404

    def test_entities(self, client):
        listing = {"total": 1, "offset": 0, "limit": 50, "items": [{"uuid": "e1"}]}
        with patch.object(
            mb.graph_browser, "list_entities", AsyncMock(return_value=listing)
        ):
            r = client.get("/admin/api/memory/entities?q=a", headers=AUTH)
            assert r.json()["total"] == 1
        with patch.object(
            mb.graph_browser,
            "get_entity",
            AsyncMock(return_value={"uuid": "e1", "relationships": []}),
        ):
            r = client.get("/admin/api/memory/entities/e1", headers=AUTH)
            assert r.json()["uuid"] == "e1"

    def test_search_uses_vector_service(self, client):
        vs = MagicMock()
        vs.is_enabled.return_value = True
        vs.search = AsyncMock(
            return_value=[
                {
                    "type": "fact",
                    "score": 0.9,
                    "entity1": "A",
                    "relation": "R",
                    "entity2": "B",
                    "neo4j_uuid": "u1",
                },
                {"type": "learning", "score": 0.8, "title": "L", "content": "c"},
            ]
        )
        with patch.object(mb, "get_browser_vector_service", AsyncMock(return_value=vs)):
            r = client.get(
                "/admin/api/memory/search?q=hello&search_type=all", headers=AUTH
            )
        assert r.status_code == 200
        items = r.json()["items"]
        assert items[0]["link"] == "/admin/memory/facts?uuid=u1"
        assert items[1]["title"] == "L"
        assert vs.search.call_args.kwargs["query_text"] == "hello"
        assert client.get("/admin/api/memory/search", headers=AUTH).status_code == 422
        r = client.get("/admin/api/memory/search?q=x&search_type=nope", headers=AUTH)
        assert r.status_code == 422

    def test_files_listing_detail_and_refresh(self, client, memory_root):
        with patch.object(mb, "file_index", mb.FileIndex()):
            r = client.get("/admin/api/memory/files/learnings?limit=2", headers=AUTH)
            assert r.status_code == 200
            assert r.json()["total"] == 3 and len(r.json()["items"]) == 2
            name = r.json()["items"][0]["name"]
            d = client.get(f"/admin/api/memory/files/learnings/{name}", headers=AUTH)
            assert d.status_code == 200 and "<h1>" in d.json()["html"]
            r = client.get("/admin/api/memory/files/learnings/..%2Fx.md", headers=AUTH)
            assert r.status_code in (400, 404)
            r = client.get("/admin/api/memory/files/learnings/missing.md", headers=AUTH)
            assert r.status_code == 404
            assert (
                client.get("/admin/api/memory/files/bogus", headers=AUTH).status_code
                == 404
            )
            r = client.post(
                "/admin/api/memory/files/refresh?type=learnings", headers=AUTH
            )
            assert r.status_code == 200
            assert r.json()["types"]["learnings"]["count"] == 3
            r = client.post("/admin/api/memory/files/refresh", headers=AUTH)
            assert r.status_code == 200

    def test_no_write_routes_exposed(self):
        """Read-only: the only POST under the memory API is the index refresh."""
        posts = [
            r.path
            for r in app.routes
            if r.path.startswith("/admin/api/memory")
            and "POST" in getattr(r, "methods", set())
        ]
        assert posts == ["/admin/api/memory/files/refresh"]


# ---------------------------------------------------------------------------
# Neo4jService: one driver per event loop (dual-server process)
# ---------------------------------------------------------------------------


def test_neo4j_service_keeps_one_driver_per_event_loop():
    """The Admin UI loop must not reuse connections created on the MCP loop."""
    import asyncio

    from src.services.neo4j_service import Neo4jService

    made = []

    def fake_driver():
        d = MagicMock(name=f"driver{len(made)}")
        d.close = AsyncMock()
        made.append(d)
        return d

    svc = Neo4jService()
    with patch.object(Neo4jService, "_new_driver", side_effect=fake_driver):
        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()
        try:
            loop_a.run_until_complete(svc.connect())
            a1 = loop_a.run_until_complete(_get_driver(svc))
            loop_a.run_until_complete(svc.connect())
            a2 = loop_a.run_until_complete(_get_driver(svc))
            assert a1 is a2 is made[0]
            assert loop_b.run_until_complete(_get_driver(svc)) is None
            loop_b.run_until_complete(svc.connect())
            b1 = loop_b.run_until_complete(_get_driver(svc))
            assert b1 is made[1] and b1 is not a1
            # default driver still answers outside any loop and on loop A
            assert svc.driver is made[0]
            assert loop_a.run_until_complete(_get_driver(svc)) is made[0]
            loop_a.run_until_complete(svc.close())
            assert made[0].close.await_count == 1 and made[1].close.await_count == 1
            assert svc.driver is None
        finally:
            loop_a.close()
            loop_b.close()


async def _get_driver(svc):
    return svc.driver


def test_neo4j_service_driver_setter_keeps_mock_semantics():
    from src.services.neo4j_service import Neo4jService

    svc = Neo4jService()
    mock = MagicMock()
    svc.driver = mock
    assert svc.driver is mock
    svc.driver = None
    assert svc.driver is None


# ---------------------------------------------------------------------------
# PostgreSQLConnection: one pool per event loop (dual-server process)
# ---------------------------------------------------------------------------


def _fake_pool(name):
    conn = MagicMock(name=f"{name}-conn")
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock(name=name)
    pool.acquire = MagicMock(return_value=acquire_cm)
    pool.close = AsyncMock()
    return pool, conn


def test_postgres_connection_uses_one_pool_per_event_loop():
    import asyncio

    from src.database.postgresql import PostgreSQLConnection

    pools = []

    async def fake_create(self, min_size, max_size):
        pool, conn = _fake_pool(f"pool{len(pools)}")
        pools.append((pool, conn, min_size))
        return pool

    async def grab(db):
        async with db._acquire() as conn:
            return conn

    db = PostgreSQLConnection()
    with patch.object(PostgreSQLConnection, "_create_pool", fake_create):
        loop_a = asyncio.new_event_loop()
        loop_b = asyncio.new_event_loop()
        try:
            loop_a.run_until_complete(db.connect())
            assert loop_a.run_until_complete(grab(db)) is pools[0][1]
            assert loop_b.run_until_complete(grab(db)) is pools[1][1]
            assert loop_b.run_until_complete(grab(db)) is pools[1][1]
            assert len(pools) == 2 and pools[1][2] == 1  # small second pool
            loop_a.run_until_complete(db.disconnect())
            assert pools[0][0].close.await_count == 1
            assert pools[1][0].close.await_count == 1
        finally:
            loop_a.close()
            loop_b.close()


def test_postgres_connection_injected_pool_still_used():
    """Tests that set .pool directly (no connect) keep working."""
    import asyncio

    from src.database.postgresql import PostgreSQLConnection

    db = PostgreSQLConnection()
    pool, conn = _fake_pool("injected")
    db.pool = pool
    db._connected = True

    async def grab():
        async with db._acquire() as c:
            return c

    assert asyncio.run(grab()) is conn


class TestAdminHealthNeo4jPing:
    def test_health_reports_real_ping(self, client):
        from src.services.neo4j_service import Neo4jService

        svc = Neo4jService()
        svc.ping = AsyncMock(return_value=True)
        with patch("src.services.neo4j_service.neo4j_service", svc):
            r = client.get("/admin/health", headers=AUTH)
        assert r.json()["neo4j"]["connected"] is True
        svc.ping = AsyncMock(side_effect=OSError("down"))
        with patch("src.services.neo4j_service.neo4j_service", svc):
            r = client.get("/admin/health", headers=AUTH)
        assert r.json()["neo4j"]["connected"] is False

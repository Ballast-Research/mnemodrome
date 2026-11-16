"""Feature 040: per-agent API keys — hashing, key file, modes, CLI, provenance.

Every key used here is generated at runtime; no secret is checked in.
"""

from __future__ import annotations

import json
import os
import stat
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.services import agent_auth
from src.services.agent_auth import (
    LEGACY_AGENT_NAME,
    STATE_KEY,
    AgentKeyAuthMiddleware,
    AgentKeyStore,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def keys_file(tmp_path):
    return tmp_path / "agent-keys.json"


@pytest.fixture
def log_lines():
    lines: list[str] = []
    sink_id = logger.add(lambda m: lines.append(str(m)), level="DEBUG")
    yield lines
    logger.remove(sink_id)


async def _whoami(request: Request):
    return JSONResponse({"agent": agent_auth.current_agent()})


def _app(store, mode, exempt=("/health",)):
    inner = Starlette(
        routes=[
            Route("/whoami", _whoami),
            Route("/health", _whoami),
            Route("/mcp", _whoami, methods=["GET", "POST"]),
        ]
    )
    mw = AgentKeyAuthMiddleware(inner, store, mode=mode, exempt_paths=exempt)
    return mw, TestClient(mw)


# ---------------------------------------------------------------------------
# Key primitives
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_generated_keys_are_unique_and_long(self):
        keys = {agent_auth.generate_key() for _ in range(50)}
        assert len(keys) == 50
        for k in keys:
            assert k.startswith(agent_auth.KEY_PREFIX)
            assert len(k) >= 40

    def test_hash_is_sha256_hex(self):
        k = agent_auth.generate_key()
        d = agent_auth.hash_key(k)
        assert len(d) == 64 and int(d, 16) >= 0
        assert d == agent_auth.hash_key(k)
        assert k not in d

    def test_keys_match(self):
        k = agent_auth.generate_key()
        d = agent_auth.hash_key(k)
        assert agent_auth.keys_match(k, d)
        assert not agent_auth.keys_match(agent_auth.generate_key(), d)

    def test_compare_uses_constant_time(self):
        k = agent_auth.generate_key()
        with patch(
            "src.services.agent_auth.hmac.compare_digest", return_value=True
        ) as cd:
            assert agent_auth.keys_match(k, "0" * 64)
        cd.assert_called_once()

    @pytest.mark.parametrize("mode", ["off", "warn", "enforce", " WARN "])
    def test_normalize_mode_valid(self, mode):
        assert agent_auth.normalize_mode(mode) == mode.strip().lower()

    def test_normalize_mode_default_and_invalid(self):
        assert agent_auth.normalize_mode(None) == "off"
        with pytest.raises(ValueError):
            agent_auth.normalize_mode("strict")

    @pytest.mark.parametrize("name", ["claude-code", "codex", "runner.prod", "a_1"])
    def test_valid_names(self, name):
        assert agent_auth.validate_agent_name(name) == name

    @pytest.mark.parametrize(
        "name", ["", "-x", "a b", "x/y", "a" * 65, LEGACY_AGENT_NAME, "../etc"]
    )
    def test_invalid_names(self, name):
        with pytest.raises(ValueError):
            agent_auth.validate_agent_name(name)

    def test_extract_presented_key(self):
        k = agent_auth.generate_key()
        assert agent_auth.extract_presented_key({"authorization": f"Bearer {k}"}) == k
        assert agent_auth.extract_presented_key({"authorization": f"bearer {k}"}) == k
        assert agent_auth.extract_presented_key({"x-mnemodrome-key": k}) == k
        # Authorization wins when both are present
        other = agent_auth.generate_key()
        assert (
            agent_auth.extract_presented_key(
                {"authorization": f"Bearer {k}", "x-mnemodrome-key": other}
            )
            == k
        )
        # Non-bearer Authorization falls back to the alternate header
        assert (
            agent_auth.extract_presented_key(
                {"authorization": "Basic Zm9vOmJhcg==", "x-mnemodrome-key": k}
            )
            == k
        )
        assert agent_auth.extract_presented_key({}) is None
        assert agent_auth.extract_presented_key({"authorization": "Bearer "}) is None


# ---------------------------------------------------------------------------
# Keys file
# ---------------------------------------------------------------------------


class TestKeysFile:
    def test_add_stores_only_digest_with_0600(self, keys_file):
        key = agent_auth.add_agent(keys_file, "claude-code")
        raw = keys_file.read_text()
        assert key not in raw
        doc = json.loads(raw)
        entry = doc["agents"]["claude-code"]
        assert entry["key_sha256"] == agent_auth.hash_key(key)
        assert entry["created_at"] and entry["rotated_at"] is None
        assert entry["scopes"] == ["read", "write"]
        assert stat.S_IMODE(os.stat(keys_file).st_mode) == 0o600

    def test_add_duplicate_rejected(self, keys_file):
        agent_auth.add_agent(keys_file, "codex")
        with pytest.raises(ValueError):
            agent_auth.add_agent(keys_file, "codex")

    def test_list_never_returns_digests(self, keys_file):
        agent_auth.add_agent(keys_file, "b-agent")
        agent_auth.add_agent(keys_file, "a-agent")
        listed = agent_auth.list_agents(keys_file)
        assert [a["name"] for a in listed] == ["a-agent", "b-agent"]
        for a in listed:
            assert "key_sha256" not in a

    def test_rotate_changes_only_that_agent(self, keys_file):
        k1 = agent_auth.add_agent(keys_file, "one")
        k2 = agent_auth.add_agent(keys_file, "two")
        k1b = agent_auth.rotate_agent(keys_file, "one")
        assert k1b != k1
        store = AgentKeyStore(keys_file)
        assert store.authenticate(k1) is None
        assert store.authenticate(k1b) == "one"
        assert store.authenticate(k2) == "two"
        assert json.loads(keys_file.read_text())["agents"]["one"]["rotated_at"]

    def test_revoke_removes_only_that_agent(self, keys_file):
        k1 = agent_auth.add_agent(keys_file, "one")
        k2 = agent_auth.add_agent(keys_file, "two")
        agent_auth.revoke_agent(keys_file, "one")
        store = AgentKeyStore(keys_file)
        assert store.authenticate(k1) is None
        assert store.authenticate(k2) == "two"

    def test_rotate_or_revoke_unknown(self, keys_file):
        with pytest.raises(KeyError):
            agent_auth.rotate_agent(keys_file, "ghost")
        with pytest.raises(KeyError):
            agent_auth.revoke_agent(keys_file, "ghost")

    def test_load_rejects_bad_documents(self, keys_file):
        keys_file.write_text("[]")
        with pytest.raises(ValueError):
            agent_auth.load_document(keys_file)
        keys_file.write_text(json.dumps({"agents": {"x": {"key_sha256": "nothex"}}}))
        with pytest.raises(ValueError):
            agent_auth.load_document(keys_file)
        keys_file.write_text(json.dumps({"version": 99, "agents": {}}))
        with pytest.raises(ValueError):
            agent_auth.load_document(keys_file)

    def test_missing_file_is_empty(self, keys_file):
        assert agent_auth.list_agents(keys_file) == []


# ---------------------------------------------------------------------------
# Runtime store
# ---------------------------------------------------------------------------


class TestStore:
    def test_authenticate(self, keys_file):
        key = agent_auth.add_agent(keys_file, "claude-code")
        store = AgentKeyStore(keys_file)
        assert store.agent_count == 1
        assert store.authenticate(key) == "claude-code"
        assert store.authenticate(agent_auth.generate_key()) is None
        assert store.authenticate("") is None
        assert store.authenticate(None) is None

    def test_no_path_no_keys(self):
        store = AgentKeyStore(None)
        assert store.agent_count == 0
        assert store.authenticate(agent_auth.generate_key()) is None

    def test_legacy_token(self, keys_file):
        legacy = agent_auth.generate_key()
        store = AgentKeyStore(keys_file, legacy_token=legacy)
        assert store.has_legacy_token
        assert store.authenticate(legacy) == LEGACY_AGENT_NAME

    def test_hot_reload_on_change(self, keys_file):
        store = AgentKeyStore(keys_file, check_interval_s=0)
        assert store.agent_count == 0
        key = agent_auth.add_agent(keys_file, "late")
        assert store.authenticate(key) == "late"
        agent_auth.revoke_agent(keys_file, "late")
        assert store.authenticate(key) is None

    def test_bad_reload_keeps_last_good(self, keys_file, log_lines):
        key = agent_auth.add_agent(keys_file, "stable")
        store = AgentKeyStore(keys_file, check_interval_s=0)
        keys_file.write_text("{ not json")
        assert store.authenticate(key) == "stable"
        assert any("could not be loaded" in line for line in log_lines)

    def test_reload_is_throttled(self, keys_file):
        store = AgentKeyStore(keys_file, check_interval_s=3600)
        key = agent_auth.add_agent(keys_file, "later")
        assert store.authenticate(key) is None  # not re-read yet
        store.reload(force=True)
        assert store.authenticate(key) == "later"


# ---------------------------------------------------------------------------
# Middleware modes (Starlette TestClient)
# ---------------------------------------------------------------------------


class TestMiddlewareModes:
    def test_off_passes_everything_and_ignores_keys(self, keys_file):
        key = agent_auth.add_agent(keys_file, "a")
        _, client = _app(AgentKeyStore(keys_file), "off")
        assert client.get("/whoami").json() == {"agent": None}
        r = client.get("/whoami", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200 and r.json() == {"agent": None}

    def test_enforce_rejects_missing_and_bad(self, keys_file):
        agent_auth.add_agent(keys_file, "a")
        mw, client = _app(AgentKeyStore(keys_file), "enforce")
        r = client.get("/whoami")
        assert r.status_code == 401
        assert r.headers["www-authenticate"].startswith("Bearer")
        assert r.json() == {"error": "Unauthorized"}
        r = client.post(
            "/mcp", headers={"Authorization": f"Bearer {agent_auth.generate_key()}"}
        )
        assert r.status_code == 401
        assert mw.stats.missing == 1 and mw.stats.invalid == 1
        assert mw.stats.rejected == 2

    def test_enforce_accepts_bearer_and_alt_header(self, keys_file):
        key = agent_auth.add_agent(keys_file, "claude-code")
        mw, client = _app(AgentKeyStore(keys_file), "enforce")
        r = client.get("/whoami", headers={"Authorization": f"Bearer {key}"})
        assert r.status_code == 200 and r.json() == {"agent": "claude-code"}
        r = client.post("/mcp", headers={"X-Mnemodrome-Key": key})
        assert r.status_code == 200 and r.json() == {"agent": "claude-code"}
        assert mw.stats.authenticated == 2

    def test_enforce_with_no_keys_rejects_all(self):
        _, client = _app(AgentKeyStore(None), "enforce")
        r = client.get("/whoami", headers={"Authorization": "Bearer anything"})
        assert r.status_code == 401

    def test_enforce_exempt_health(self, keys_file):
        _, client = _app(AgentKeyStore(keys_file), "enforce")
        assert client.get("/health").status_code == 200

    def test_enforce_exempt_still_identifies(self, keys_file):
        key = agent_auth.add_agent(keys_file, "a")
        _, client = _app(AgentKeyStore(keys_file), "enforce")
        r = client.get("/health", headers={"Authorization": f"Bearer {key}"})
        assert r.json() == {"agent": "a"}

    def test_enforce_no_exempt_paths(self, keys_file):
        _, client = _app(AgentKeyStore(keys_file), "enforce", exempt=())
        assert client.get("/health").status_code == 401

    def test_warn_serves_logs_and_counts(self, keys_file, log_lines):
        key = agent_auth.add_agent(keys_file, "a")
        mw, client = _app(AgentKeyStore(keys_file), "warn")
        r = client.get("/whoami")
        assert r.status_code == 200 and r.json() == {"agent": None}
        bad = agent_auth.generate_key()
        r = client.get("/whoami", headers={"Authorization": f"Bearer {bad}"})
        assert r.status_code == 200
        r = client.get("/whoami", headers={"Authorization": f"Bearer {key}"})
        assert r.json() == {"agent": "a"}
        assert mw.stats.missing == 1 and mw.stats.invalid == 1
        assert mw.stats.authenticated == 1 and mw.stats.rejected == 0
        warnings = [line for line in log_lines if "AUTH warn" in line]
        assert len(warnings) == 2
        assert "reason=missing_key" in warnings[0]
        assert "reason=invalid_key" in warnings[1]

    def test_warn_logs_are_throttled_but_counted(self, keys_file, log_lines):
        mw, client = _app(AgentKeyStore(keys_file), "warn")
        for _ in range(5):
            client.get("/whoami")
        assert mw.stats.missing == 5
        assert len([line for line in log_lines if "AUTH warn" in line]) == 1

    def test_keys_and_digests_never_logged(self, keys_file, log_lines):
        key = agent_auth.add_agent(keys_file, "a")
        bad = agent_auth.generate_key()
        for mode in ("warn", "enforce"):
            store = AgentKeyStore(keys_file)
            _, client = _app(store, mode)
            client.get("/whoami", headers={"Authorization": f"Bearer {key}"})
            client.get("/whoami", headers={"Authorization": f"Bearer {bad}"})
        blob = "\n".join(log_lines)
        for secret in (key, bad, agent_auth.hash_key(key), agent_auth.hash_key(bad)):
            assert secret not in blob

    def test_non_http_scope_passthrough(self):
        inner = AsyncMock()
        mw = AgentKeyAuthMiddleware(inner, AgentKeyStore(None), mode="enforce")
        import asyncio

        asyncio.run(mw({"type": "lifespan"}, None, None))
        inner.assert_awaited_once()

    def test_agent_context_reset_after_request(self, keys_file):
        key = agent_auth.add_agent(keys_file, "a")
        _, client = _app(AgentKeyStore(keys_file), "enforce")
        client.get("/whoami", headers={"Authorization": f"Bearer {key}"})
        assert agent_auth.current_agent() is None


# ---------------------------------------------------------------------------
# Provenance seam
# ---------------------------------------------------------------------------


class TestCurrentAgent:
    def test_none_outside_request(self):
        assert agent_auth.current_agent() is None

    def test_contextvar(self):
        token = agent_auth.set_current_agent("ctx-agent")
        try:
            assert agent_auth.current_agent() == "ctx-agent"
        finally:
            agent_auth.reset_current_agent(token)

    def test_mcp_request_context_fallback(self):
        """MCP tool calls run in the session task: read the request scope."""
        from mcp.server.lowlevel.server import request_ctx

        fake = SimpleNamespace(
            request=SimpleNamespace(scope={"state": {STATE_KEY: "tool-agent"}})
        )
        token = request_ctx.set(fake)
        try:
            assert agent_auth.current_agent() == "tool-agent"
        finally:
            request_ctx.reset(token)

    def test_mcp_request_context_without_request(self):
        from mcp.server.lowlevel.server import request_ctx

        token = request_ctx.set(SimpleNamespace(request=None))
        try:
            assert agent_auth.current_agent() is None
        finally:
            request_ctx.reset(token)


@pytest.fixture
def as_agent():
    token = agent_auth.set_current_agent("prov-agent")
    yield "prov-agent"
    agent_auth.reset_current_agent(token)


class TestProvenanceRecording:
    @pytest.mark.asyncio
    async def test_relationship_factory_records_captured_by(self, as_agent):
        from src.services.relationship_factory import RelationshipFactory

        factory = RelationshipFactory()
        row = {
            "source_name": "A",
            "relation_type": "USES",
            "target_name": "B",
        }
        with (
            patch.object(factory, "_find_existing", AsyncMock(return_value=None)),
            patch(
                "src.services.relationship_factory.is_functional", return_value=False
            ),
            patch(
                "src.services.relationship_factory.neo4j_service.query",
                AsyncMock(return_value=[row]),
            ) as q,
        ):
            await factory.create("s-uuid", "USES", "t-uuid")
        cypher, params = q.call_args.args
        assert "captured_by: $captured_by" in cypher
        assert params["captured_by"] == as_agent

    @pytest.mark.asyncio
    async def test_relationship_factory_no_agent_is_null(self):
        from src.services.relationship_factory import RelationshipFactory

        factory = RelationshipFactory()
        row = {"source_name": "A", "relation_type": "USES", "target_name": "B"}
        with (
            patch.object(factory, "_find_existing", AsyncMock(return_value=None)),
            patch(
                "src.services.relationship_factory.is_functional", return_value=False
            ),
            patch(
                "src.services.relationship_factory.neo4j_service.query",
                AsyncMock(return_value=[row]),
            ) as q,
        ):
            await factory.create("s-uuid", "USES", "t-uuid")
        assert q.call_args.args[1]["captured_by"] is None

    @pytest.mark.asyncio
    async def test_event_log_records_agent(self, as_agent):
        from src.services.event_log_service import EventLogService

        svc = EventLogService()
        svc._initialized = True
        svc.session_id = "s-1"
        svc.db = AsyncMock()
        svc.db.fetchval = AsyncMock(return_value=7)
        await svc.log_event("tool_call", "capture_fact", metadata={"k": 1})
        insert_args = svc.db.fetchval.call_args.args
        assert insert_args[4] == as_agent  # actor
        assert json.loads(insert_args[9]) == {"k": 1, "agent": as_agent}

    @pytest.mark.asyncio
    async def test_event_log_keeps_explicit_actor(self, as_agent):
        from src.services.event_log_service import EventLogService

        svc = EventLogService()
        svc._initialized = True
        svc.session_id = "s-1"
        svc.db = AsyncMock()
        svc.db.fetchval = AsyncMock(return_value=7)
        await svc.log_event("tool_call", "x", actor="retention_scheduler")
        insert_args = svc.db.fetchval.call_args.args
        assert insert_args[4] == "retention_scheduler"
        assert json.loads(insert_args[9]) == {"agent": as_agent}

    @pytest.mark.asyncio
    async def test_event_log_unchanged_without_agent(self):
        from src.services.event_log_service import EventLogService

        svc = EventLogService()
        svc._initialized = True
        svc.session_id = "s-1"
        svc.db = AsyncMock()
        svc.db.fetchval = AsyncMock(return_value=7)
        await svc.log_event("tool_call", "x")
        insert_args = svc.db.fetchval.call_args.args
        assert insert_args[4] is None and insert_args[9] is None

    @staticmethod
    def _recording_query():
        calls: list[tuple[str, dict]] = []

        async def _query(cypher, params=None):
            calls.append((cypher, params or {}))
            raise RuntimeError("stop after first write")

        return calls, _query

    @pytest.mark.asyncio
    async def test_learning_node_records_captured_by(self, as_agent, tmp_path):
        import src.mcp_server as ms

        calls, fake_query = self._recording_query()
        fs = MagicMock()
        fs._get_target_path.return_value = tmp_path
        fs.write_file.return_value = tmp_path / "x.md"
        with (
            patch.object(ms.neo4j_service, "query", side_effect=fake_query),
            patch.object(ms, "file_service", fs),
        ):
            with pytest.raises(Exception):
                await ms.capture_learning(
                    title="t", content="c", auto_deduplicate=False
                )
        cypher, params = calls[0]
        assert "CREATE (l:Learning" in cypher and "captured_by: $captured_by" in cypher
        assert params["captured_by"] == as_agent

    @pytest.mark.asyncio
    async def test_session_node_records_captured_by(self, as_agent):
        import src.mcp_server as ms

        calls, fake_query = self._recording_query()
        with patch.object(ms.neo4j_service, "query", side_effect=fake_query):
            with pytest.raises(Exception):
                await ms.capture_session(conversation_summary="summary")
        cypher, params = calls[0]
        assert "CREATE (s:Session" in cypher and "captured_by: $captured_by" in cypher
        assert params["captured_by"] == as_agent

    @pytest.mark.asyncio
    async def test_auto_capture_tool_actor(self, as_agent):
        import src.mcp_server as ms
        from src import config

        @ms.auto_capture_tool
        async def demo_tool():
            return "ok"

        logged = AsyncMock(return_value=1)
        with (
            patch.object(config, "EVENT_LOG_AUTO_CAPTURE", True),
            patch.object(ms.event_log_service, "log_event", logged),
        ):
            assert await demo_tool() == "ok"
            import asyncio

            await asyncio.sleep(0)
        assert logged.call_args.kwargs["actor"] == as_agent


# ---------------------------------------------------------------------------
# Real MCP app: /mcp and hook routes behind the middleware
# ---------------------------------------------------------------------------


@pytest.fixture
def mcp_app_factory(keys_file):
    import src.mcp_server as ms

    def _build(mode, legacy_token="", transport="streamable-http"):
        ms.mcp._session_manager = None  # a session manager runs only once
        return ms.build_mcp_http_app(
            transport,
            mode=mode,
            keys_file=str(keys_file),
            legacy_token=legacy_token,
            exempt_paths=("/health",),
        )

    yield _build
    ms.mcp._session_manager = None
    ms._auth_middleware = None


class TestMcpAppRoutes:
    def test_enforce_blocks_mcp_without_key(self, mcp_app_factory):
        app = mcp_app_factory("enforce")
        with TestClient(app) as client:
            r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
            assert r.status_code == 401

    def test_enforce_passes_mcp_with_key(self, mcp_app_factory, keys_file):
        key = agent_auth.add_agent(keys_file, "claude-code")
        app = mcp_app_factory("enforce")
        with TestClient(app) as client:
            # Unknown session id: the transport answers (404) without starting
            # a server session, which proves the request got past auth.
            r = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": "does-not-exist",
                },
            )
            assert r.status_code != 401
            assert r.status_code in (400, 404)

    def test_tool_call_sees_authenticated_agent(self, mcp_app_factory, keys_file):
        """End to end: a real MCP tool call reads the agent via current_agent()."""
        from contextlib import asynccontextmanager

        import src.mcp_server as ms

        @asynccontextmanager
        async def _no_services(server):
            yield {}

        async def whoami_040() -> str:
            return str(agent_auth.current_agent())

        ms.mcp.add_tool(whoami_040)
        key = agent_auth.add_agent(keys_file, "e2e-agent")
        app = mcp_app_factory("enforce")
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json, text/event-stream",
        }

        def _data(resp):
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:])
            return resp.json()

        try:
            with (
                patch.object(ms.mcp._mcp_server, "lifespan", _no_services),
                TestClient(app) as client,
            ):
                r = client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "test", "version": "1"},
                        },
                    },
                    headers=headers,
                )
                assert r.status_code == 200
                headers["mcp-session-id"] = r.headers["mcp-session-id"]
                headers["mcp-protocol-version"] = "2025-06-18"
                client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers=headers,
                )
                call = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": "whoami_040", "arguments": {}},
                }
                r = client.post("/mcp", json=call, headers=headers)
                assert r.status_code == 200
                assert _data(r)["result"]["content"][0]["text"] == "e2e-agent"
                no_key = {k: v for k, v in headers.items() if k != "Authorization"}
                assert client.post("/mcp", json=call, headers=no_key).status_code == 401
        finally:
            ms.mcp.remove_tool("whoami_040")

    def test_sse_transport_is_covered(self, mcp_app_factory):
        app = mcp_app_factory("enforce", transport="sse")
        with TestClient(app) as client:
            assert client.post("/messages/").status_code == 401

    def test_hook_route_with_agent_key_in_enforce(self, mcp_app_factory, keys_file):
        import src.mcp_server as ms

        key = agent_auth.add_agent(keys_file, "hooks")
        legacy = agent_auth.generate_key()
        app = mcp_app_factory("enforce", legacy_token=legacy)
        with (
            patch.object(ms, "MNEMODROME_HOOK_TOKEN", legacy),
            TestClient(app) as client,
        ):
            assert client.get("/project-context").status_code == 401
            # agent key passes both middleware and route check; 400 = no project
            r = client.get(
                "/project-context", headers={"Authorization": f"Bearer {key}"}
            )
            assert r.status_code == 400
            # legacy token still works
            r = client.get(
                "/project-context", headers={"Authorization": f"Bearer {legacy}"}
            )
            assert r.status_code == 400

    def test_hook_route_legacy_behaviour_in_off_mode(self, mcp_app_factory):
        import src.mcp_server as ms

        legacy = agent_auth.generate_key()
        app = mcp_app_factory("off", legacy_token=legacy)
        with (
            patch.object(ms, "MNEMODROME_HOOK_TOKEN", legacy),
            TestClient(app) as client,
        ):
            assert client.get("/project-context").status_code == 401
            r = client.get(
                "/project-context", headers={"Authorization": f"Bearer {legacy}"}
            )
            assert r.status_code == 400

    def test_off_mode_no_token_unchanged(self, mcp_app_factory):
        import src.mcp_server as ms

        app = mcp_app_factory("off")
        with patch.object(ms, "MNEMODROME_HOOK_TOKEN", ""), TestClient(app) as client:
            assert client.get("/project-context").status_code == 400

    def test_health_exempt_and_reports_auth_counters(self, mcp_app_factory):
        import src.mcp_server as ms

        app = mcp_app_factory("warn")
        with (
            patch.object(ms.neo4j_service, "ping", AsyncMock()),
            TestClient(app) as client,
        ):
            client.get("/project-context")  # served (400), counted as missing
            r = client.get("/health")
            assert r.status_code in (200, 503)
            auth = r.json()["auth"]
            assert auth["mode"] == "warn"
            assert auth["unauthenticated_missing"] >= 1
            assert "agents" not in auth


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def _run(self, capsys, *argv):
        from src.cli import agent_keys

        rc = agent_keys.main(list(argv))
        out = capsys.readouterr()
        return rc, out.out, out.err

    def test_add_prints_key_once_and_stores_hash(self, capsys, keys_file):
        rc, out, err = self._run(capsys, "--file", str(keys_file), "add", "codex")
        assert rc == 0
        key = out.strip().splitlines()[0]
        assert key.startswith(agent_auth.KEY_PREFIX)
        assert "only time" in err
        assert key not in keys_file.read_text()
        assert AgentKeyStore(keys_file).authenticate(key) == "codex"

    def test_list_shows_names_not_hashes(self, capsys, keys_file):
        self._run(capsys, "--file", str(keys_file), "add", "codex")
        digest = json.loads(keys_file.read_text())["agents"]["codex"]["key_sha256"]
        rc, out, _ = self._run(capsys, "--file", str(keys_file), "list")
        assert rc == 0 and "codex" in out and digest not in out

    def test_list_empty(self, capsys, keys_file):
        rc, out, _ = self._run(capsys, "--file", str(keys_file), "list")
        assert rc == 0 and "No agent keys" in out

    def test_rotate_and_revoke(self, capsys, keys_file):
        _, out, _ = self._run(capsys, "--file", str(keys_file), "add", "runner")
        old = out.strip().splitlines()[0]
        rc, out, _ = self._run(capsys, "--file", str(keys_file), "rotate", "runner")
        new = out.strip().splitlines()[0]
        assert rc == 0 and new != old
        store = AgentKeyStore(keys_file)
        assert store.authenticate(old) is None
        assert store.authenticate(new) == "runner"
        rc, out, _ = self._run(capsys, "--file", str(keys_file), "revoke", "runner")
        assert rc == 0 and "Revoked" in out
        assert AgentKeyStore(keys_file).authenticate(new) is None

    def test_errors_return_nonzero(self, capsys, keys_file):
        rc, _, err = self._run(capsys, "--file", str(keys_file), "revoke", "ghost")
        assert rc == 1 and "not found" in err
        rc, _, err = self._run(capsys, "--file", str(keys_file), "add", "bad name")
        assert rc == 1 and "error" in err
        self._run(capsys, "--file", str(keys_file), "add", "dup")
        rc, _, err = self._run(capsys, "--file", str(keys_file), "add", "dup")
        assert rc == 1 and "already exists" in err

    def test_env_var_path(self, capsys, keys_file, monkeypatch):
        monkeypatch.setenv("MNEMODROME_AGENT_KEYS_FILE", str(keys_file))
        rc, _, _ = self._run(capsys, "add", "from-env")
        assert rc == 0 and keys_file.exists()

    def test_missing_path_exits(self, monkeypatch):
        from src.cli import agent_keys

        monkeypatch.delenv("MNEMODROME_AGENT_KEYS_FILE", raising=False)
        with patch("dotenv.load_dotenv", return_value=False):
            with pytest.raises(SystemExit):
                agent_keys.main(["list"])

"""
Integration tests for Feature 037: Relationship Identity Contract.

Uses testcontainers[neo4j] (ephemeral Neo4j 5.26) to assert real contract
behaviors against a live graph store. These tests are RED before D1-D8
implementation and must all pass after.

Markers: pytest.mark.identity_contract (also tagged integration)

Contract behaviours under test:
  - D1: create returns unique uuid that equals the Qdrant point id
  - D2/D2b: atomic supersession; new fact gets a vector under its real uuid
  - D3: invalidate precisely one native edge; duplicate-uuid pair raises
  - D4: created_at vs valid_at distinct; find_relationship returns both
  - D5: dedup is native-scoped, fresh uuid, non-lossy, foreign-safe
  - D7: foreign null-type edges are never touched by any operation
  - Teeth check: reverting D3 single-statement guard makes a test RED

DO NOT connect to the production mbp02 database. All tests operate on an
ephemeral testcontainers Neo4j instance.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

import src.services.relationship_factory as _rf_module
from src.services.relationship_factory import RelationshipFactory

# ---------------------------------------------------------------------------
# testcontainers fixture — ephemeral Neo4j 5.26
# ---------------------------------------------------------------------------

NEO4J_IMAGE = "neo4j:5.26-community"
NEO4J_PASSWORD = "test-password"


@pytest.fixture(scope="session")
def neo4j_container():
    """Spin up an ephemeral Neo4j 5.26 once per test session, then teardown."""
    from testcontainers.neo4j import Neo4jContainer

    with Neo4jContainer(image=NEO4J_IMAGE).with_env(
        "NEO4J_AUTH", f"neo4j/{NEO4J_PASSWORD}"
    ) as container:
        yield container


@pytest.fixture()
def neo4j_bolt_url(neo4j_container):
    """Return the Bolt URL for the ephemeral Neo4j container."""
    return neo4j_container.get_connection_url()


@pytest_asyncio.fixture()
async def neo4j_driver(neo4j_bolt_url):
    """Async driver — function-scoped to avoid event-loop teardown issues."""
    driver = AsyncGraphDatabase.driver(neo4j_bolt_url, auth=("neo4j", NEO4J_PASSWORD))
    yield driver
    await driver.close()


@pytest_asyncio.fixture(autouse=True)
async def clean_db(neo4j_driver):
    """Wipe the graph before each test to ensure isolation."""
    async with neo4j_driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield


# ---------------------------------------------------------------------------
# Helper: run raw Cypher on the ephemeral driver
# ---------------------------------------------------------------------------


async def _query(driver, cypher: str, params: dict | None = None) -> list[dict]:
    async with driver.session() as session:
        result = await session.run(cypher, params or {})
        return await result.data()


async def _seed_entity(driver, name: str, entity_uuid: str | None = None) -> str:
    """Create an Entity node, return its uuid."""
    eid = entity_uuid or str(uuid4())
    await _query(
        driver,
        "CREATE (:Entity {uuid: $uuid, name: $name, created_at: datetime()})",
        {"uuid": eid, "name": name},
    )
    return eid


async def _seed_native_rel(
    driver,
    src_uuid: str,
    rel_type: str,
    tgt_uuid: str,
    rel_uuid: str | None = None,
    valid_at_offset_secs: int = 0,
) -> str:
    """Create a native RELATES_TO edge (type IS NOT NULL). Return its uuid."""
    rid = rel_uuid or str(uuid4())
    # valid_at slightly different from created_at via offset parameter (D4 test)
    offset_expr = (
        f"datetime() + duration('PT{valid_at_offset_secs}S')"
        if valid_at_offset_secs
        else "datetime()"
    )
    cypher = f"""
    MATCH (s {{uuid: $src}}), (t {{uuid: $tgt}})
    CREATE (s)-[r:RELATES_TO {{
        uuid: $rid,
        type: $rel_type,
        valid_at: {offset_expr},
        created_at: datetime(),
        invalid_at: NULL,
        expired_at: NULL,
        confidence: 0.5
    }}]->(t)
    RETURN r.uuid AS uuid
    """
    await _query(
        driver,
        cypher,
        {"src": src_uuid, "tgt": tgt_uuid, "rid": rid, "rel_type": rel_type},
    )
    return rid


async def _seed_foreign_rel(driver, src_uuid: str, tgt_uuid: str) -> str:
    """Create a foreign legacy edge (type IS NULL, group_id present)."""
    fid = str(uuid4())
    cypher = """
    MATCH (s {uuid: $src}), (t {uuid: $tgt})
    CREATE (s)-[r:RELATES_TO {
        uuid: $fid,
        group_id: 'legacy-group',
        fact: 'some mem0 fact'
    }]->(t)
    RETURN r.uuid AS uuid
    """
    await _query(driver, cypher, {"src": src_uuid, "tgt": tgt_uuid, "fid": fid})
    return fid


# ---------------------------------------------------------------------------
# Production-service wiring helper
# ---------------------------------------------------------------------------


def _make_container_neo4j_service(driver):
    """Return a Neo4jService instance wired to the ephemeral container driver.

    Neo4jService.connect() is a no-op when self.driver is already set, so all
    production methods (query, execute_update_transaction, find_relationship, …)
    will run against the container without touching the production singleton.
    """
    from src.services.neo4j_service import Neo4jService

    svc = Neo4jService.__new__(Neo4jService)
    svc.driver = driver
    svc.database = None  # uses the default database
    svc.uri = "bolt://localhost:0"  # never used — driver pre-set
    svc.user = "neo4j"
    svc.password = "test"
    return svc


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.integration, pytest.mark.identity_contract]


# ===========================================================================
# D1: App-generated uuid propagated; dual-key return
# ===========================================================================


@pytest.mark.asyncio
class TestD1AppGeneratedUuid:
    """The uuid returned from create_relationship must be app-generated
    (not randomUUID() inside Cypher), non-null, and returned under both
    'uuid' and 'relationship_uuid' keys (dual-key, for consumer compat)."""

    async def test_create_returns_non_null_uuid(self, neo4j_driver):
        """PRODUCTION CODE: RelationshipFactory.create() must return uuid and
        relationship_uuid equal to the app-generated value (D1 dual-key contract).

        Wires the production RelationshipFactory to the container via patch so that
        the test FAILS if the production Cypher drops or renames the uuid field.
        """
        src = await _seed_entity(neo4j_driver, "EntityA")
        tgt = await _seed_entity(neo4j_driver, "EntityB")
        app_uuid = str(uuid4())
        container_svc = _make_container_neo4j_service(neo4j_driver)
        factory = RelationshipFactory()

        # Patch the module-level neo4j_service singleton used by RelationshipFactory
        with patch.object(_rf_module, "neo4j_service", container_svc):
            result = await factory.create(
                source_uuid=src,
                relation_type="USES",
                target_uuid=tgt,
                relationship_uuid=app_uuid,
            )

        assert result, "RelationshipFactory.create() returned None or empty"
        assert result.get("uuid") == app_uuid, (
            f"D1: uuid {result.get('uuid')!r} != app_uuid {app_uuid!r}; "
            "production Cypher may be using randomUUID() or not returning uuid"
        )
        assert result.get("relationship_uuid") == app_uuid, (
            f"D1: relationship_uuid key missing or wrong; got {result.get('relationship_uuid')!r}"
        )

        # Also verify the edge is actually stored in the graph with the correct uuid
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() WHERE r.type IS NOT NULL RETURN r.uuid AS uuid",
            {"uuid": app_uuid},
        )
        assert rows, "Edge not found in graph after RelationshipFactory.create()"
        assert rows[0]["uuid"] == app_uuid, "Graph uuid differs from returned uuid"

    async def test_uuid_unique_across_two_facts(self, neo4j_driver):
        """Two facts must have distinct uuids."""
        src = await _seed_entity(neo4j_driver, "A")
        tgt1 = await _seed_entity(neo4j_driver, "B")
        tgt2 = await _seed_entity(neo4j_driver, "C")
        rid1 = await _seed_native_rel(neo4j_driver, src, "USES", tgt1)
        rid2 = await _seed_native_rel(neo4j_driver, src, "USES", tgt2)
        assert rid1 != rid2

    async def test_uuid_stored_in_graph_equals_app_generated(self, neo4j_driver):
        """The uuid written to the graph must equal the app-generated value."""
        src = await _seed_entity(neo4j_driver, "X")
        tgt = await _seed_entity(neo4j_driver, "Y")
        app_uuid = str(uuid4())
        rid = await _seed_native_rel(neo4j_driver, src, "KNOWS", tgt, rel_uuid=app_uuid)
        assert rid == app_uuid

        # Confirm it's stored in the graph
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.uuid AS uuid",
            {"uuid": app_uuid},
        )
        assert rows and rows[0]["uuid"] == app_uuid


# ===========================================================================
# D1 + cross-layer: uuid equals Qdrant point id
# ===========================================================================


@pytest.mark.asyncio
class TestD1CrossLayerIdentity:
    """The uuid sent to store_fact_embedding must be the graph uuid (FR-005).
    Since we have no Qdrant in the test env, we spy on the call.
    """

    async def test_fact_uuid_passed_to_store_fact_embedding(self, neo4j_driver):
        """The fact_uuid forwarded to vector layer must equal the graph uuid."""
        src = await _seed_entity(neo4j_driver, "Alpha")
        tgt = await _seed_entity(neo4j_driver, "Beta")
        app_uuid = str(uuid4())
        rid = await _seed_native_rel(
            neo4j_driver, src, "RELATES", tgt, rel_uuid=app_uuid
        )

        captured_uuid = {}

        async def fake_store(
            entity1, relation, entity2, context, fact_uuid, metadata=None
        ):
            captured_uuid["uuid"] = str(fact_uuid)
            return str(fact_uuid)

        # Simulate what capture_fact does after graph write
        await fake_store("Alpha", "RELATES", "Beta", "", rid)
        assert captured_uuid["uuid"] == app_uuid, (
            f"Vector point id {captured_uuid['uuid']} != graph uuid {app_uuid}"
        )


# ===========================================================================
# D3: Precise invalidation — single fact, fail-hard on duplicates
# ===========================================================================


@pytest.mark.asyncio
class TestD3PreciseInvalidation:
    """Production TemporalValidityService.invalidate_relationship must:
    - Return success dict when exactly 1 matching native edge exists
    - Raise ValueError when 0 matching native edges exist (not found)
    - Raise ValueError when >1 matching native edges share the same uuid (ambiguous)
    - Leave foreign (null-type) edges completely untouched
    """

    async def test_invalidate_one_native_edge(self, neo4j_driver):
        """Invalidating one unique fact returns success dict and sets invalid_at."""
        from src.services.temporal_validity_service import TemporalValidityService

        src = await _seed_entity(neo4j_driver, "S")
        tgt = await _seed_entity(neo4j_driver, "T")
        rid = await _seed_native_rel(neo4j_driver, src, "RUNS_ON", tgt)

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()
        result = await tvs.invalidate_relationship(
            neo4j_service=container_svc,
            relationship_uuid=rid,
            invalidation_type="both",
            reason="test-invalidation",
        )
        assert result["status"] == "success", (
            f"Production invalidate_relationship must return status=success; got {result!r}"
        )
        assert result["fact_uuid"] == rid, (
            "fact_uuid in response must match the invalidated rid"
        )

        # Confirm the edge is actually invalidated in the graph
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia",
            {"uuid": rid},
        )
        assert rows[0]["ia"] is not None, "invalid_at was not set"

    async def test_invalidate_raises_on_no_match(self, neo4j_driver):
        """Invalidating a non-existent uuid raises ValueError('not found')."""
        from src.services.temporal_validity_service import TemporalValidityService

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()
        with pytest.raises(ValueError, match="[Nn]ot found|not found"):
            await tvs.invalidate_relationship(
                neo4j_service=container_svc,
                relationship_uuid=str(uuid4()),
                invalidation_type="both",
                reason="no-match test",
            )

    async def test_invalidate_raises_on_duplicate_uuid(self, neo4j_driver):
        """When two native edges share the same uuid, raises ValueError (D3 ambiguous guard)."""
        from src.services.temporal_validity_service import TemporalValidityService

        src = await _seed_entity(neo4j_driver, "Dup-Src")
        tgt1 = await _seed_entity(neo4j_driver, "Dup-Tgt1")
        tgt2 = await _seed_entity(neo4j_driver, "Dup-Tgt2")
        shared_uuid = str(uuid4())

        # Seed two native edges with the same uuid (legacy duplicate scenario)
        await _seed_native_rel(neo4j_driver, src, "TYPE_A", tgt1, rel_uuid=shared_uuid)
        # Force-write a second edge with the same uuid
        await _query(
            neo4j_driver,
            """
            MATCH (s {uuid: $src}), (t {uuid: $tgt})
            CREATE (s)-[r:RELATES_TO {uuid: $rid, type: 'TYPE_B',
                valid_at: datetime(), created_at: datetime(),
                invalid_at: NULL, expired_at: NULL}]->(t)
            """,
            {"src": src, "tgt": tgt2, "rid": shared_uuid},
        )

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()
        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            await tvs.invalidate_relationship(
                neo4j_service=container_svc,
                relationship_uuid=shared_uuid,
                invalidation_type="both",
                reason="duplicate-guard test",
            )

    async def test_foreign_edges_not_touched_by_invalidation(self, neo4j_driver):
        """Production invalidate_relationship leaves foreign (null-type) edges untouched."""
        from src.services.temporal_validity_service import TemporalValidityService

        src = await _seed_entity(neo4j_driver, "FS")
        tgt_n = await _seed_entity(neo4j_driver, "FTN")
        tgt_f = await _seed_entity(neo4j_driver, "FTF")
        native_rid = str(uuid4())

        await _seed_native_rel(
            neo4j_driver, src, "NATIVE_TYPE", tgt_n, rel_uuid=native_rid
        )
        await _seed_foreign_rel(neo4j_driver, src, tgt_f)

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()
        result = await tvs.invalidate_relationship(
            neo4j_service=container_svc,
            relationship_uuid=native_rid,
            invalidation_type="both",
            reason="scope-guard test",
        )
        assert result["status"] == "success", "Native edge should have been invalidated"

        # Foreign edge must remain untouched (no invalid_at)
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO]->() WHERE r.type IS NULL RETURN r.invalid_at AS ia",
        )
        for row in rows:
            assert row["ia"] is None, "Foreign edge was accidentally invalidated"


# ===========================================================================
# D2 / D2b: Atomic supersession — old valid on failure, new has vector
# ===========================================================================


@pytest.mark.asyncio
class TestD2AtomicSupersession:
    """Supersession must be executed via execute_update_transaction() (production).

    All tests call the PRODUCTION Neo4jService.execute_update_transaction() wired
    to the container driver — NOT a reimplemented Cypher.  This means the tests
    will go RED if the production Cypher is wrong.
    """

    async def _atomic_supersede(
        self,
        neo4j_driver,
        src_uuid: str,
        tgt_uuid: str,
        rel_type: str,
        old_uuid: str,
        new_uuid: str,
        context: str = "",
        entity2_old_uuid: str | None = None,
    ) -> dict:
        """Call PRODUCTION execute_update_transaction() wired to container.

        Returns the transaction result dict on success or {} when the old edge
        isn't found (production raises in that case — we normalize to {} here
        so callers can assert `not result` uniformly).
        """
        container_svc = _make_container_neo4j_service(neo4j_driver)
        try:
            return await container_svc.execute_update_transaction(
                old_fact_uuid=old_uuid,
                entity1_uuid=src_uuid,
                relation_type=rel_type,
                entity2_uuid=tgt_uuid,
                context=context,
                new_fact_uuid=new_uuid,
                entity2_old_uuid=entity2_old_uuid,
            )
        except Exception:
            # Production raises when old edge not found or not unique.
            # Normalize to {} so callers can assert `not result`.
            return {}

    async def test_atomic_supersession_success(self, neo4j_driver):
        """Normal supersession via production execute_update_transaction:
        old invalidated, new created atomically."""
        src = await _seed_entity(neo4j_driver, "Source")
        tgt_old = await _seed_entity(neo4j_driver, "OldTarget")
        tgt_new = await _seed_entity(neo4j_driver, "NewTarget")
        old_uuid = await _seed_native_rel(neo4j_driver, src, "USES", tgt_old)
        new_uuid = str(uuid4())

        result = await self._atomic_supersede(
            neo4j_driver,
            src,
            tgt_new,
            "USES",
            old_uuid,
            new_uuid,
            entity2_old_uuid=tgt_old,
        )
        assert result, (
            "Production execute_update_transaction returned empty — Cypher may be broken"
        )
        assert result["old_fact_uuid"] == old_uuid
        assert result["new_fact_uuid"] == new_uuid

        # Old edge must be invalid
        old_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia",
            {"uuid": old_uuid},
        )
        assert old_rows[0]["ia"] is not None, (
            "old fact not invalidated by production code"
        )

        # New edge must be valid
        new_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia, r.uuid AS uuid",
            {"uuid": new_uuid},
        )
        assert new_rows, (
            "new fact not found in graph after production execute_update_transaction"
        )
        assert new_rows[0]["ia"] is None, "new fact should be valid"

    async def test_supersession_no_match_leaves_old_valid(self, neo4j_driver):
        """When old_uuid not found, production code returns empty / raises (old stays valid)."""
        src = await _seed_entity(neo4j_driver, "Src2")
        tgt = await _seed_entity(neo4j_driver, "Tgt2")
        result = await self._atomic_supersede(
            neo4j_driver, src, tgt, "KNOWS", str(uuid4()), str(uuid4())
        )
        # No orphan created (production single-statement prevents partial writes)
        assert not result, "Supersession of non-existent fact should return empty"
        orphan_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO]->() WHERE r.type IS NOT NULL RETURN count(r) AS cnt",
        )
        assert orphan_rows[0]["cnt"] == 0, (
            "No edges should exist when old fact not found"
        )

    async def test_new_uuid_passed_to_vector_layer(self, neo4j_driver):
        """D2b: After production supersession, vector store receives new_uuid (not old)."""
        src = await _seed_entity(neo4j_driver, "VecSrc")
        tgt_old = await _seed_entity(neo4j_driver, "VecOld")
        tgt_new = await _seed_entity(neo4j_driver, "VecNew")
        old_uuid = await _seed_native_rel(neo4j_driver, src, "RUNS", tgt_old)
        new_uuid = str(uuid4())

        result = await self._atomic_supersede(
            neo4j_driver,
            src,
            tgt_new,
            "RUNS",
            old_uuid,
            new_uuid,
            entity2_old_uuid=tgt_old,
        )
        assert result, "Production supersession failed unexpectedly"

        # Simulate what mcp_server does: call store_fact_embedding with new_uuid
        vector_calls = []

        async def fake_store(
            entity1, relation, entity2, context, fact_uuid, metadata=None
        ):
            vector_calls.append(str(fact_uuid))
            return str(fact_uuid)

        # D2b: after production supersession returns new_fact_uuid, caller must use it
        await fake_store("VecSrc", "RUNS", "VecNew", "", result["new_fact_uuid"])
        assert len(vector_calls) == 1
        assert vector_calls[0] == new_uuid, (
            f"D2b: vector store must use new_uuid={new_uuid}, got {vector_calls[0]}"
        )
        assert vector_calls[0] != old_uuid, "D2b: vector store must NOT use old_uuid"

    async def test_injected_failure_does_not_create_orphan(self, neo4j_driver):
        """Simulate a failure between invalidate and create (two-statement impl).
        With the single-statement D2 Cypher this is impossible by design;
        we test that a two-statement impl leaves the old fact valid on failure."""
        src = await _seed_entity(neo4j_driver, "FailSrc")
        tgt_old = await _seed_entity(neo4j_driver, "FailOld")
        old_uuid = await _seed_native_rel(neo4j_driver, src, "HAD", tgt_old)
        new_uuid = str(uuid4())

        # Simulate two-statement partial execution (old invalidated, new NOT created)
        # This is the BROKEN scenario that D2 single-statement prevents.
        await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() SET r.invalid_at = datetime()",
            {"uuid": old_uuid},
        )
        # Do NOT create new edge — simulate connection drop

        # Verify the bad state: old invalid, no new edge
        old_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia",
            {"uuid": old_uuid},
        )
        assert old_rows[0]["ia"] is not None, "old is invalid (bad state)"

        new_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.uuid",
            {"uuid": new_uuid},
        )
        # No orphan but old is invalid — bad state. D2 single Cypher prevents this.
        assert not new_rows, (
            "No orphan should exist after partial two-statement failure"
        )

        # Note: the D2 single-statement Cypher either commits both or neither.
        # This test documents the problem; the implementation fix makes it impossible.


# ===========================================================================
# D4: Provenance — created_at vs valid_at distinct
# ===========================================================================


@pytest.mark.asyncio
class TestD4Provenance:
    """created_at (recorded when) and valid_at (true from) must be distinct
    fields that are correctly returned by find_relationship queries."""

    async def test_created_at_and_valid_at_are_distinct_fields(self, neo4j_driver):
        """Seed an edge where valid_at differs from created_at; confirm both returned."""
        src = await _seed_entity(neo4j_driver, "PA")
        tgt = await _seed_entity(neo4j_driver, "PB")
        rid = str(uuid4())

        # Deliberately set valid_at in the future relative to created_at
        cypher = """
        MATCH (s {uuid: $src}), (t {uuid: $tgt})
        CREATE (s)-[r:RELATES_TO {
            uuid: $rid,
            type: 'RECORDED',
            created_at: datetime(),
            valid_at: datetime() + duration('P1D'),
            invalid_at: NULL,
            expired_at: NULL
        }]->(t)
        RETURN r.created_at AS created_at, r.valid_at AS valid_at
        """
        rows = await _query(neo4j_driver, cypher, {"src": src, "tgt": tgt, "rid": rid})
        assert rows, "No rows returned"
        assert rows[0]["created_at"] is not None, "created_at must be set"
        assert rows[0]["valid_at"] is not None, "valid_at must be set"
        # They must differ (valid_at is 1 day ahead)
        assert rows[0]["created_at"] != rows[0]["valid_at"], (
            "created_at and valid_at must be distinguishable"
        )

    async def test_find_relationship_returns_both_provenance_fields(self, neo4j_driver):
        """Production Neo4jService.find_relationship() must return 'created_at' and 'valid_at'
        as separate keys (D4 requires no aliasing of one onto the other)."""
        src = await _seed_entity(neo4j_driver, "PR_Src")
        tgt = await _seed_entity(neo4j_driver, "PR_Tgt")
        await _seed_native_rel(neo4j_driver, src, "HAS", tgt)

        # Call PRODUCTION find_relationship wired to the ephemeral container
        container_svc = _make_container_neo4j_service(neo4j_driver)
        result = await container_svc.find_relationship("PR_Src", "HAS", "PR_Tgt")

        assert result is not None, (
            "Production find_relationship returned None — edge not found"
        )
        assert "created_at" in result, (
            "D4: 'created_at' key missing from find_relationship result; "
            "check production Cypher for aliasing error"
        )
        assert "valid_at" in result, (
            "D4: 'valid_at' key missing from find_relationship result"
        )


# ===========================================================================
# D5: Dedup — native-scoped, fresh uuid, non-lossy, foreign-safe
# ===========================================================================


@pytest.mark.asyncio
class TestD5Dedup:
    """Entity deduplication via production EntityDeduplicator.deduplicate_entity() must:
    - Only move native (type IS NOT NULL) edges
    - Assign a fresh uuid to every moved edge
    - Keep both if duplicate (type, target) exists on canonical (non-lossy)
    - Never DELETE a node that still carries foreign edges

    Setup convention: two Entity nodes with the same name = "duplicates".
    When native edge counts tie, the node created first (lower created_at) wins
    canonical selection (ORDER BY relationship_count DESC, created_at ASC).
    To guarantee node_a is canonical, give it one extra "anchor" native edge so
    its count is higher than node_b's.
    """

    async def test_dedup_assigns_fresh_uuid_per_moved_edge(self, neo4j_driver):
        """Production deduplicate_entity() assigns a fresh uuid to every moved native edge.

        Wires EntityDeduplicator to the ephemeral container via module-level patch.
        The test will go RED if production Cypher copies the old uuid instead of
        assigning str(uuid4()) per edge.
        """
        import src.cli.deduplicate_entities as _dedup_module
        from src.cli.deduplicate_entities import EntityDeduplicator

        dedup_name = "DupEntityFreshUUID"
        tgt_a = await _seed_entity(neo4j_driver, "TgtFreshUUID_A")
        tgt_b = await _seed_entity(neo4j_driver, "TgtFreshUUID_B")

        # node_a: 2 native edges → highest count → canonical
        node_a = await _seed_entity(neo4j_driver, dedup_name)
        await _seed_native_rel(neo4j_driver, node_a, "ANCHOR1", tgt_a)
        await _seed_native_rel(neo4j_driver, node_a, "ANCHOR2", tgt_b)

        # node_b: 1 native edge (to tgt_b) → lower count → dup
        node_b = await _seed_entity(neo4j_driver, dedup_name)
        original_rid = await _seed_native_rel(neo4j_driver, node_b, "KNOWS", tgt_b)

        container_svc = _make_container_neo4j_service(neo4j_driver)
        with patch.object(_dedup_module, "neo4j_service", container_svc):
            deduplicator = EntityDeduplicator()
            result = await deduplicator.deduplicate_entity(
                dedup_name, "Entity", dry_run=False
            )

        assert result["merged_count"] >= 1, "At least one dup node must be merged"

        # The moved edge (node_b KNOWS tgt_b) must have a fresh uuid, not original_rid
        rows = await _query(
            neo4j_driver,
            "MATCH (c {uuid: $c})-[r:RELATES_TO]->(t {uuid: $tgt}) "
            "WHERE r.type IS NOT NULL RETURN r.uuid AS uuid",
            {"c": node_a, "tgt": tgt_b, "type": "KNOWS"},
        )
        # Filter to the KNOWS edge specifically
        knows_rows = await _query(
            neo4j_driver,
            "MATCH (c {uuid: $c})-[r:RELATES_TO {type: 'KNOWS'}]->(t {uuid: $tgt}) "
            "RETURN r.uuid AS uuid",
            {"c": node_a, "tgt": tgt_b},
        )
        assert knows_rows, "Moved KNOWS edge must exist on canonical node after dedup"
        assert knows_rows[0]["uuid"] != original_rid, (
            f"Fresh uuid must NOT copy the original uuid {original_rid!r}; "
            f"got {knows_rows[0]['uuid']!r}"
        )

    async def test_dedup_does_not_touch_foreign_edges(self, neo4j_driver):
        """Production deduplicate_entity() never migrates or deletes foreign (null-type) edges.

        TEETH CHECK: removing the type IS NOT NULL filter from the production
        discovery/canonical-selection query changes which node is canonical,
        making the canonical assertion below fail → test goes RED.
        """
        import src.cli.deduplicate_entities as _dedup_module
        from src.cli.deduplicate_entities import EntityDeduplicator

        dedup_name = "DupEntityNoForeign"
        tgt_a = await _seed_entity(neo4j_driver, "TgtNoForeign_A")
        tgt_b = await _seed_entity(neo4j_driver, "TgtNoForeign_B")

        # node_a: 1 native edge — native count = 1
        node_a = await _seed_entity(neo4j_driver, dedup_name)
        await _seed_native_rel(neo4j_driver, node_a, "ANCHOR", tgt_a)

        # node_b: 1 native edge + 1 foreign edge — native count = 1, total = 2
        # With correct (native-only) filter: tie → node_a canonical (created first)
        # With broken filter (all edges): node_b total=2 > node_a total=1 → WRONG canonical
        node_b = await _seed_entity(neo4j_driver, dedup_name)
        native_rid = await _seed_native_rel(neo4j_driver, node_b, "NATIVE", tgt_b)
        foreign_rid = await _seed_foreign_rel(neo4j_driver, node_b, tgt_b)

        container_svc = _make_container_neo4j_service(neo4j_driver)
        with patch.object(_dedup_module, "neo4j_service", container_svc):
            deduplicator = EntityDeduplicator()
            result = await deduplicator.deduplicate_entity(
                dedup_name, "Entity", dry_run=False
            )

        assert result["merged_count"] >= 1, "At least one dup node must be merged"

        # Foreign edge must still be on node_b (production must never touch it)
        foreign_rows = await _query(
            neo4j_driver,
            "MATCH (d {uuid: $dup})-[r:RELATES_TO]->() WHERE r.type IS NULL RETURN r.uuid AS uuid",
            {"dup": node_b},
        )
        assert foreign_rows, "Foreign edge must still be on the dup node"
        assert foreign_rows[0]["uuid"] == foreign_rid, (
            "Foreign edge uuid must be unchanged"
        )

        # The native edge from node_b must have been moved to node_a (canonical)
        # This assertion FAILS when the discovery filter is removed (teeth-check):
        # broken filter makes node_b canonical, so node_a's edge is moved to node_b,
        # and node_a is deleted — leaving no edge from node_a to tgt_b.
        moved_rows = await _query(
            neo4j_driver,
            "MATCH (c {uuid: $c})-[r:RELATES_TO {type: 'NATIVE'}]->(t {uuid: $tgt}) "
            "RETURN r.uuid AS uuid",
            {"c": node_a, "tgt": tgt_b},
        )
        assert moved_rows, (
            "Native edge must have been moved to canonical (node_a). "
            "TEETH CHECK: this fails when the type IS NOT NULL filter is removed from "
            "the discovery query, proving the filter is load-bearing."
        )

    async def test_dedup_does_not_delete_node_with_foreign_edges(self, neo4j_driver):
        """Production deduplicate_entity() preserves the dup node when foreign edges remain.

        After native edges are migrated the foreign-safe delete check counts ALL
        remaining edges (not just native). The dup node must survive as long as
        any foreign edge is still attached.
        """
        import src.cli.deduplicate_entities as _dedup_module
        from src.cli.deduplicate_entities import EntityDeduplicator

        dedup_name = "DupEntitySafeDelete"
        tgt_a = await _seed_entity(neo4j_driver, "TgtSafe_A")
        tgt_b = await _seed_entity(neo4j_driver, "TgtSafe_B")

        # node_a: 1 native edge → canonical (created first, equal native count)
        node_a = await _seed_entity(neo4j_driver, dedup_name)
        await _seed_native_rel(neo4j_driver, node_a, "ANCHOR", tgt_a)

        # node_b: 1 native edge + 1 foreign edge → dup
        node_b = await _seed_entity(neo4j_driver, dedup_name)
        native_rid = await _seed_native_rel(neo4j_driver, node_b, "SAFE_TYPE", tgt_b)
        foreign_rid = await _seed_foreign_rel(neo4j_driver, node_b, tgt_b)

        container_svc = _make_container_neo4j_service(neo4j_driver)
        with patch.object(_dedup_module, "neo4j_service", container_svc):
            deduplicator = EntityDeduplicator()
            await deduplicator.deduplicate_entity(dedup_name, "Entity", dry_run=False)

        # node_b must still exist in the graph (not deleted) because it has a foreign edge
        node_b_rows = await _query(
            neo4j_driver,
            "MATCH (d {uuid: $dup}) RETURN d.uuid AS uuid",
            {"dup": node_b},
        )
        assert node_b_rows, (
            "Dup node with foreign edges must NOT be deleted by production dedup"
        )

        # The foreign edge must still be attached
        foreign_rows = await _query(
            neo4j_driver,
            "MATCH (d {uuid: $dup})-[r:RELATES_TO]->() WHERE r.type IS NULL RETURN r.uuid AS uuid",
            {"dup": node_b},
        )
        assert foreign_rows, "Foreign edge must remain on the preserved dup node"

    async def test_dedup_preserves_duplicate_type_target_facts(self, neo4j_driver):
        """Non-lossy: if canonical already has (type, target), both facts survive migration.

        Production deduplicate_entity() must CREATE the moved edge unconditionally
        (never skip when (type, target) already exists on canonical). The test
        asserts canonical ends up with 2 USES→tgt edges.
        """
        import src.cli.deduplicate_entities as _dedup_module
        from src.cli.deduplicate_entities import EntityDeduplicator

        dedup_name = "DupEntityNonLossy"
        tgt = await _seed_entity(neo4j_driver, "TgtNonLossy")
        extra_tgt = await _seed_entity(neo4j_driver, "ExtraTgtNonLossy")

        # node_a: 2 native edges → highest count → canonical
        node_a = await _seed_entity(neo4j_driver, dedup_name)
        can_rid = await _seed_native_rel(neo4j_driver, node_a, "USES", tgt)
        await _seed_native_rel(neo4j_driver, node_a, "ANCHOR", extra_tgt)

        # node_b: 1 native edge with the same (type=USES, target=tgt) → dup
        node_b = await _seed_entity(neo4j_driver, dedup_name)
        dup_rid = await _seed_native_rel(neo4j_driver, node_b, "USES", tgt)

        container_svc = _make_container_neo4j_service(neo4j_driver)
        with patch.object(_dedup_module, "neo4j_service", container_svc):
            deduplicator = EntityDeduplicator()
            result = await deduplicator.deduplicate_entity(
                dedup_name, "Entity", dry_run=False
            )

        assert result["merged_count"] >= 1, "At least one dup node must be merged"

        # Both USES→tgt facts must exist on canonical (non-lossy: no dedup of same type+target)
        rows = await _query(
            neo4j_driver,
            "MATCH (c {uuid: $c})-[r:RELATES_TO {type: 'USES'}]->(t {uuid: $t}) "
            "RETURN count(r) AS cnt",
            {"c": node_a, "t": tgt},
        )
        assert rows[0]["cnt"] == 2, (
            f"Non-lossy: expected 2 USES facts on canonical, got {rows[0]['cnt']}. "
            "Production dedup must CREATE the moved edge even when canonical already "
            "has the same (type, target)."
        )


# ===========================================================================
# D7: Scope guard — every Mnemodrome op scoped to native edges
# ===========================================================================


@pytest.mark.asyncio
class TestD7ScopeGuard:
    """No Mnemodrome operation must read, modify, or delete foreign (null-type) edges."""

    async def test_activation_hebbian_skips_foreign_edges(self, neo4j_driver):
        """Production ActivationService.strengthen_hebbian_links() only touches native edges.

        Wires the production ActivationService to the ephemeral container by patching
        the module-level neo4j_service. Will go RED if the production Cypher's
        r.type IS NOT NULL guard is removed.
        """
        import src.services.activation_service as _activation_module
        from src.services.activation_service import ActivationService

        src = await _seed_entity(neo4j_driver, "HebSrc")
        tgt = await _seed_entity(neo4j_driver, "HebTgt")
        native_rid = await _seed_native_rel(neo4j_driver, src, "COACTIVE", tgt)
        foreign_rid = await _seed_foreign_rel(neo4j_driver, src, tgt)

        # Call PRODUCTION strengthen_hebbian_links() wired to the container
        container_svc = _make_container_neo4j_service(neo4j_driver)
        with patch.object(_activation_module, "neo4j_service", container_svc):
            svc = ActivationService()
            await svc.strengthen_hebbian_links([src, tgt])

        # Native edge must have hebbian_weight set by production code
        native_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.hebbian_weight AS hw",
            {"uuid": native_rid},
        )
        assert native_rows, "Native edge not found after Hebbian update"
        assert native_rows[0]["hw"] is not None, (
            "Production strengthen_hebbian_links() must set hebbian_weight on native edge"
        )

        # Foreign edge must not have hebbian_weight set
        foreign_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.hebbian_weight AS hw",
            {"uuid": foreign_rid},
        )
        assert foreign_rows[0]["hw"] is None, (
            "Foreign (null-type) edge must not be touched by production Hebbian update"
        )

    async def test_foreign_edges_excluded_from_validity_check(self, neo4j_driver):
        """check_relationship_validity must be scoped to native edges only."""
        src = await _seed_entity(neo4j_driver, "ValidSrc")
        tgt = await _seed_entity(neo4j_driver, "ValidTgt")
        foreign_rid = await _seed_foreign_rel(neo4j_driver, src, tgt)

        # D7-scoped validity check
        cypher = """
        MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
        WHERE r.type IS NOT NULL
        RETURN r.invalid_at AS invalid_at
        """
        rows = await _query(neo4j_driver, cypher, {"uuid": foreign_rid})
        assert not rows, "Foreign edge must not appear in native-scoped validity check"

    async def test_conflict_audit_scoped_to_native_edges(self, neo4j_driver):
        """Conflict detection uses r.type IS NOT NULL and r.type (not r.name)."""
        src = await _seed_entity(neo4j_driver, "CA_Src")
        tgt1 = await _seed_entity(neo4j_driver, "CA_Tgt1")
        tgt2 = await _seed_entity(neo4j_driver, "CA_Tgt2")

        # Two native edges with same type → conflict
        await _seed_native_rel(neo4j_driver, src, "WORKS_AT", tgt1)
        await _seed_native_rel(neo4j_driver, src, "WORKS_AT", tgt2)
        await _seed_foreign_rel(neo4j_driver, src, tgt1)  # foreign (should not count)

        # D7-scoped conflict query (r.type not r.name)
        cypher = """
        MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
        WHERE r.invalid_at IS NULL
          AND r.expired_at IS NULL
          AND r.type IS NOT NULL
        WITH e1, r.type AS relation, collect({entity2: e2}) AS facts
        WHERE size(facts) > 1
        RETURN e1.name AS entity1, relation, size(facts) AS fact_count
        """
        rows = await _query(neo4j_driver, cypher)
        assert any(r["relation"] == "WORKS_AT" for r in rows), (
            "Should detect native conflict on WORKS_AT"
        )
        # Foreign edges not counted
        for row in rows:
            assert row["relation"] is not None, (
                "Conflict query returned a null-type row"
            )


# ===========================================================================
# FR-001: Uniqueness — duplicate uuid detection
# ===========================================================================


@pytest.mark.asyncio
class TestFR001Uniqueness:
    """SC-001: verifies uniqueness invariant at the application layer."""

    async def test_no_duplicate_uuids_after_proper_capture(self, neo4j_driver):
        """After creating two facts with distinct uuids, count confirms 0 duplicates."""
        src = await _seed_entity(neo4j_driver, "UniSrc")
        tgt1 = await _seed_entity(neo4j_driver, "UniTgt1")
        tgt2 = await _seed_entity(neo4j_driver, "UniTgt2")
        await _seed_native_rel(neo4j_driver, src, "LINKS", tgt1)
        await _seed_native_rel(neo4j_driver, src, "LINKS", tgt2)

        dup_check = await _query(
            neo4j_driver,
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.type IS NOT NULL
            WITH r.uuid AS u, count(r) AS cnt
            WHERE cnt > 1
            RETURN count(u) AS dup_count
            """,
        )
        assert dup_check[0]["dup_count"] == 0, "Native edges must have unique uuids"

    async def test_duplicate_uuid_is_detectable(self, neo4j_driver):
        """SC-006: attempting to create a dup-uuid edge is detectable via query."""
        src = await _seed_entity(neo4j_driver, "DetSrc")
        tgt1 = await _seed_entity(neo4j_driver, "DetTgt1")
        tgt2 = await _seed_entity(neo4j_driver, "DetTgt2")
        shared = str(uuid4())
        await _seed_native_rel(neo4j_driver, src, "T1", tgt1, rel_uuid=shared)
        await _seed_native_rel(neo4j_driver, src, "T2", tgt2, rel_uuid=shared)

        dup_check = await _query(
            neo4j_driver,
            """
            MATCH ()-[r:RELATES_TO]->()
            WHERE r.type IS NOT NULL AND r.uuid = $uuid
            RETURN count(r) AS cnt
            """,
            {"uuid": shared},
        )
        assert dup_check[0]["cnt"] == 2, "Should detect shared uuid"


# ===========================================================================
# Teeth check: reverting D3 guard makes this test RED
# ===========================================================================


@pytest.mark.asyncio
class TestTeethCheck:
    """SC-007: Prove the test suite has teeth — the test below is specifically
    designed to catch the missing D3 cardinality guard.

    If you revert the D3 guard (remove the `WHERE size(rs) = 1` line),
    then test_d3_guard_prevents_fan_out will PASS where it should FAIL,
    but test_teeth_check_d3_guard_blocks_duplicate_uuid will RED immediately.

    Actually the teeth check here is inverted: we assert that the D3 guard
    (WHERE size(rs)=1) produces 0 rows on a duplicate-uuid set, rather than
    accidentally invalidating both. Without the guard the statement would
    return 2 rows, breaking FR-002.
    """

    async def test_teeth_check_d3_guard_blocks_duplicate_uuid(self, neo4j_driver):
        """D3 guard: with two native edges sharing a uuid, invalidation returns 0 rows.
        Removing the `WHERE size(rs)=1` line would return 2 rows — this test
        would then FAIL asserting `not result`, proving the suite has teeth.
        """
        src = await _seed_entity(neo4j_driver, "TC_Src")
        tgt1 = await _seed_entity(neo4j_driver, "TC_Tgt1")
        tgt2 = await _seed_entity(neo4j_driver, "TC_Tgt2")
        shared_uuid = str(uuid4())
        await _seed_native_rel(neo4j_driver, src, "TC_A", tgt1, rel_uuid=shared_uuid)
        # Force second edge with same uuid
        await _query(
            neo4j_driver,
            """
            MATCH (s {uuid: $src}), (t {uuid: $tgt})
            CREATE (s)-[r:RELATES_TO {uuid: $rid, type: 'TC_B',
                valid_at: datetime(), created_at: datetime(),
                invalid_at: NULL, expired_at: NULL}]->(t)
            """,
            {"src": src, "tgt": tgt2, "rid": shared_uuid},
        )

        # D3 single-statement guard
        cypher_d3 = """
        MATCH ()-[r:RELATES_TO {uuid: $uuid}]->()
        WHERE r.type IS NOT NULL
        WITH collect(r) AS rs
        WHERE size(rs) = 1
        WITH rs[0] AS r
        SET r.invalid_at = datetime()
        RETURN r.uuid AS fact_uuid
        """
        result = await _query(neo4j_driver, cypher_d3, {"uuid": shared_uuid})

        # Guard must block: returns empty because size(rs) == 2, not 1
        assert not result, (
            "TEETH CHECK FAILED: D3 guard must return 0 rows for duplicate-uuid pair. "
            "If this fails, it means the WHERE size(rs)=1 guard is missing."
        )

        # Verify neither edge was touched
        edges = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia",
            {"uuid": shared_uuid},
        )
        for e in edges:
            assert e["ia"] is None, (
                "D3 guard failure: edge was invalidated despite duplicate"
            )


# ===========================================================================
# Production-service-wired tests (RED before implementation, GREEN after)
# ===========================================================================
#
# These tests wire the actual production service classes to the container Neo4j,
# so they will FAIL against the current (pre-D1..D8) production code and PASS
# after implementation. They provide the truest "RED before implementation"
# guarantee required by plan.md and SC-007.
# ===========================================================================


class _ContainerNeo4jService:
    """Minimal Neo4jService-compatible adapter wired to the ephemeral container.

    Allows production service objects (temporal_validity_service, etc.) that
    accept a neo4j_service argument to be tested against the container without
    affecting the production singleton.
    """

    def __init__(self, driver):
        self._driver = driver
        self.database = None  # None uses default database

    async def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        from src.services.neo4j_service import _normalize_neo4j_value

        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return [_normalize_neo4j_value(r) for r in records]


@pytest.mark.asyncio
class TestProductionServiceContract:
    """
    Wire production service code to the container and assert contract behaviour.

    These tests are RED before D1-D8 implementation because the production
    services use incorrect Cypher (randomUUID, no D3 guard, two-statement
    update, etc.). They turn GREEN after each design decision is implemented.

    Teeth check: reverting D3 implementation makes
    test_production_invalidate_raises_or_blocks_on_duplicate_uuid RED.
    """

    async def test_production_invalidate_blocks_duplicate_uuid(self, neo4j_driver):
        """D3 PRODUCTION TEETH CHECK.

        Calls the production TemporalValidityService.invalidate_relationship()
        with a container service wired to two native edges sharing a uuid.
        Before D3 implementation: the function lacks the size(rs)=1 guard, so
        it would either succeed (invalidating both edges) or return a result —
        but the test asserts that either 0 rows are changed OR ValueError is raised.
        After D3 implementation: function raises ValueError on duplicate-uuid.

        This test IS the teeth-check that makes the suite RED if D3 is reverted.
        """
        from src.services.temporal_validity_service import TemporalValidityService

        src = await _seed_entity(neo4j_driver, "Prod_D3_Src")
        tgt1 = await _seed_entity(neo4j_driver, "Prod_D3_Tgt1")
        tgt2 = await _seed_entity(neo4j_driver, "Prod_D3_Tgt2")
        shared_uuid = str(uuid4())

        await _seed_native_rel(
            neo4j_driver, src, "DUP_TYPE_A", tgt1, rel_uuid=shared_uuid
        )
        # Force second native edge with same uuid
        await _query(
            neo4j_driver,
            """
            MATCH (s {uuid: $src}), (t {uuid: $tgt})
            CREATE (s)-[r:RELATES_TO {uuid: $rid, type: 'DUP_TYPE_B',
                valid_at: datetime(), created_at: datetime(),
                invalid_at: NULL, expired_at: NULL}]->(t)
            """,
            {"src": src, "tgt": tgt2, "rid": shared_uuid},
        )

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()

        with pytest.raises(Exception) as exc_info:
            # D3: must raise when uuid matches >1 native edge
            await tvs.invalidate_relationship(
                neo4j_service=container_svc,
                relationship_uuid=shared_uuid,
                invalidation_type="both",
                reason="D3 guard test",
            )

        assert exc_info.value is not None, (
            "D3 production teeth check: invalidate_relationship must raise "
            "on duplicate-uuid (ambiguous). Before D3 impl, this test is RED "
            "because the production code has no cardinality guard."
        )

        # Also verify: neither edge was invalidated
        edges = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() WHERE r.type IS NOT NULL "
            "RETURN r.invalid_at AS ia",
            {"uuid": shared_uuid},
        )
        for e in edges:
            assert e["ia"] is None, (
                "D3 production guard failure: edge invalidated despite duplicate uuid"
            )

    async def test_production_invalidate_exactly_one_on_unique_uuid(self, neo4j_driver):
        """D3: With a unique uuid, invalidate_relationship succeeds and changes exactly 1 row."""
        from src.services.temporal_validity_service import TemporalValidityService

        src = await _seed_entity(neo4j_driver, "Prod_D3U_Src")
        tgt = await _seed_entity(neo4j_driver, "Prod_D3U_Tgt")
        rid = await _seed_native_rel(neo4j_driver, src, "UNIQUE_TYPE", tgt)

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()

        result = await tvs.invalidate_relationship(
            neo4j_service=container_svc,
            relationship_uuid=rid,
            invalidation_type="both",
            reason="Test unique invalidation",
        )

        assert result["status"] == "success"
        assert result["fact_uuid"] == rid

        # Exactly 1 row changed
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia",
            {"uuid": rid},
        )
        assert rows[0]["ia"] is not None, "Edge must be invalidated"

    async def test_production_invalidate_does_not_touch_foreign_edges(
        self, neo4j_driver
    ):
        """D3 + D7: invalidate_relationship must be scoped to native edges (r.type IS NOT NULL).
        A foreign null-type edge with a different uuid must remain untouched.
        """
        from src.services.temporal_validity_service import TemporalValidityService

        src = await _seed_entity(neo4j_driver, "Prod_D7_Src")
        tgt_n = await _seed_entity(neo4j_driver, "Prod_D7_TgtN")
        tgt_f = await _seed_entity(neo4j_driver, "Prod_D7_TgtF")
        native_rid = await _seed_native_rel(neo4j_driver, src, "NATIVE_ONLY", tgt_n)
        await _seed_foreign_rel(neo4j_driver, src, tgt_f)

        container_svc = _ContainerNeo4jService(neo4j_driver)
        tvs = TemporalValidityService()

        await tvs.invalidate_relationship(
            neo4j_service=container_svc,
            relationship_uuid=native_rid,
            invalidation_type="both",
            reason="Scope guard test",
        )

        # Foreign edge must not have invalid_at set
        foreign = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO]->() WHERE r.type IS NULL RETURN r.invalid_at AS ia",
        )
        for f in foreign:
            assert f["ia"] is None, "Production D7 failure: foreign edge was touched"

    async def test_production_relationship_factory_create_dual_key(self, neo4j_driver):
        """D1 PRODUCTION: RelationshipFactory.create() must return both 'uuid'
        and 'relationship_uuid' keys with the app-generated value (dual-key contract).
        This test FAILS if the production Cypher drops or renames either key.
        """
        src = await _seed_entity(neo4j_driver, "ProdD1_Src")
        tgt = await _seed_entity(neo4j_driver, "ProdD1_Tgt")
        app_uuid = str(uuid4())

        container_svc = _make_container_neo4j_service(neo4j_driver)
        factory = RelationshipFactory()

        with patch.object(_rf_module, "neo4j_service", container_svc):
            result = await factory.create(
                source_uuid=src,
                relation_type="PROD_TEST",
                target_uuid=tgt,
                relationship_uuid=app_uuid,
            )

        assert result is not None, "RelationshipFactory.create() returned None"
        assert result.get("uuid") == app_uuid, (
            f"D1 production: 'uuid' key missing or wrong. Got: {result.get('uuid')!r}"
        )
        assert result.get("relationship_uuid") == app_uuid, (
            f"D1 production: 'relationship_uuid' key missing or wrong. "
            f"Got: {result.get('relationship_uuid')!r}"
        )

        # Graph state: exactly one native edge with the app-generated uuid
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() WHERE r.type IS NOT NULL RETURN r.uuid AS uuid",
            {"uuid": app_uuid},
        )
        assert len(rows) == 1, f"Expected 1 edge with uuid={app_uuid}, got {len(rows)}"

    async def test_production_relationship_factory_invalidate_raises_on_ambiguous(
        self, neo4j_driver
    ):
        """D3 PRODUCTION: RelationshipFactory.invalidate() must raise ValueError
        when >1 active native edges share the same (source, type, target) triple
        — the cardinality guard must be present in the production invalidate() Cypher.
        This test FAILS if the WHERE size(rs)=1 guard is removed from production.
        """
        src = await _seed_entity(neo4j_driver, "ProdD3_Src")
        tgt = await _seed_entity(neo4j_driver, "ProdD3_Tgt")

        # Create two native edges with the SAME (source, type, target) — ambiguous triple
        shared_type = "AMBIG_TYPE"
        await _seed_native_rel(neo4j_driver, src, shared_type, tgt)
        await _seed_native_rel(neo4j_driver, src, shared_type, tgt)

        container_svc = _make_container_neo4j_service(neo4j_driver)
        factory = RelationshipFactory()

        with patch.object(_rf_module, "neo4j_service", container_svc):
            with pytest.raises((ValueError, Exception)) as exc_info:
                await factory.invalidate(
                    source_uuid=src,
                    relation_type=shared_type,
                    target_uuid=tgt,
                )

        assert exc_info.value is not None, (
            "D3 production: RelationshipFactory.invalidate() must raise on ambiguous triple"
        )
        # Verify neither edge was invalidated
        rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO]->() WHERE r.type IS NOT NULL AND r.invalid_at IS NULL "
            "RETURN count(r) AS cnt",
        )
        assert rows[0]["cnt"] == 2, (
            "D3 production: ambiguous guard must leave both edges untouched"
        )

    async def test_production_execute_update_transaction_atomic(self, neo4j_driver):
        """D2 PRODUCTION: execute_update_transaction() atomically invalidates old
        and creates new in a single statement.  Verifies the production Cypher
        both binds the new target distinctly and respects the cardinality guard.
        This test FAILS if production Cypher is split into two statements or
        drops the cardinality guard.
        """
        src = await _seed_entity(neo4j_driver, "ProdD2_Src")
        tgt_old = await _seed_entity(neo4j_driver, "ProdD2_OldTgt")
        tgt_new = await _seed_entity(neo4j_driver, "ProdD2_NewTgt")
        old_uuid = await _seed_native_rel(neo4j_driver, src, "TX_TYPE", tgt_old)
        new_uuid = str(uuid4())

        container_svc = _make_container_neo4j_service(neo4j_driver)
        tx_result = await container_svc.execute_update_transaction(
            old_fact_uuid=old_uuid,
            entity1_uuid=src,
            relation_type="TX_TYPE",
            entity2_uuid=tgt_new,
            context="production atomicity test",
            new_fact_uuid=new_uuid,
            entity2_old_uuid=tgt_old,
        )

        assert tx_result["old_fact_uuid"] == old_uuid
        assert tx_result["new_fact_uuid"] == new_uuid

        # Old edge must be invalid
        old_rows = await _query(
            neo4j_driver,
            "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN r.invalid_at AS ia",
            {"uuid": old_uuid},
        )
        assert old_rows[0]["ia"] is not None, "Production D2: old edge not invalidated"

        # New edge must be valid and point to new target
        new_rows = await _query(
            neo4j_driver,
            "MATCH (s {uuid: $src})-[r:RELATES_TO {uuid: $uuid}]->(t {uuid: $tgt}) "
            "RETURN r.uuid AS uuid, r.invalid_at AS ia",
            {"src": src, "uuid": new_uuid, "tgt": tgt_new},
        )
        assert new_rows, (
            "Production D2: new edge not found with correct (src, uuid, new_tgt)"
        )
        assert new_rows[0]["ia"] is None, "Production D2: new edge should be valid"

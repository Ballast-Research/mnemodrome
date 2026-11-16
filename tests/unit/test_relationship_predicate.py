"""C2: fact predicate must come from the relationship `type` property."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.context_formatter import (
    format_context_markdown,
    relationship_predicate,
)
from src.services.context_retrieval_service import ContextRetrievalService
from src.services.neo4j_service import Neo4jService


class FakeRel:
    """Mimics a neo4j Relationship: .type is the edge type, .get reads props."""

    type = "RELATES_TO"

    def __init__(self, props):
        self._props = props

    def get(self, key, default=None):
        return self._props.get(key, default)


class _AsyncRecords:
    def __init__(self, records):
        self._it = iter(records)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _service_with_records(records):
    session = MagicMock()
    session.run = AsyncMock(return_value=_AsyncRecords(records))
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    driver = MagicMock()
    driver.session.return_value = cm
    svc = Neo4jService()
    svc.connect = AsyncMock()
    svc._driver = driver
    return svc


def _record():
    return {
        "source": {"name": "A", "uuid": "1"},
        "r": FakeRel({"type": "USES"}),
        "target": {"name": "B", "uuid": "2"},
    }


def test_prefers_property_over_edge_type():
    assert relationship_predicate(FakeRel({"type": "USES"})) == "USES"


def test_falls_back_to_edge_type_then_default():
    assert relationship_predicate(FakeRel({})) == "RELATES_TO"
    other = FakeRel({})
    other.type = "OTHER"
    assert relationship_predicate(other) == "OTHER"
    assert relationship_predicate(object()) == "RELATES_TO"


def test_dict_property_wins_over_edge_type():
    class DictWithType(dict):
        type = "RELATES_TO"

    assert relationship_predicate(DictWithType({"type": "USES"})) == "USES"


def test_formatter_renders_property_predicate():
    out = format_context_markdown([_record()])
    assert "USES" in out
    assert "RELATES_TO" not in out


@pytest.mark.asyncio
async def test_temporal_query_maps_property_predicate():
    svc = _service_with_records([_record()])
    out = await svc.temporal_query()
    assert out[0]["r"]["type"] == "USES"


@pytest.mark.asyncio
async def test_temporal_query_by_entities_maps_property_predicate():
    svc = _service_with_records([_record()])
    out = await svc.temporal_query_by_entities(["1"])
    assert out[0]["r"]["type"] == "USES"


@pytest.mark.asyncio
async def test_cascade_queries_return_property_not_edge_type():
    with patch("src.services.context_retrieval_service.get_vector_service"):
        with patch("src.services.context_retrieval_service.redis_service"):
            with patch("src.services.context_retrieval_service.neo4j_service"):
                svc = ContextRetrievalService()
    svc.neo4j_service = AsyncMock()
    svc.neo4j_service.query = AsyncMock(return_value=[])

    await svc._query_entity_context({"u1"})
    await svc._get_related_facts({"u1"})

    queries = [c.args[0] for c in svc.neo4j_service.query.await_args_list]
    assert len(queries) == 2
    for q in queries:
        assert "type(r)" not in q
    assert "rel: r.type" in queries[0]
    assert "r.type as relation" in queries[1]

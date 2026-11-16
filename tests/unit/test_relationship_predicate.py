"""C2: fact predicate must come from the relationship `type` property."""

from src.services.context_formatter import _get_relationship_type
from src.services.neo4j_service import _relationship_predicate


class FakeRel:
    """Mimics a neo4j Relationship: .type is the edge type, .get reads props."""

    type = "RELATES_TO"

    def __init__(self, props):
        self._props = props

    def get(self, key, default=None):
        return self._props.get(key, default)


def test_neo4j_service_prefers_property():
    assert _relationship_predicate(FakeRel({"type": "USES"})) == "USES"


def test_neo4j_service_falls_back_to_edge_type():
    assert _relationship_predicate(FakeRel({})) == "RELATES_TO"


def test_context_formatter_prefers_property():
    assert _get_relationship_type(FakeRel({"type": "USES"})) == "USES"


def test_context_formatter_falls_back_to_edge_type():
    assert _get_relationship_type(FakeRel({})) == "RELATES_TO"


def test_context_formatter_dict():
    assert _get_relationship_type({"type": "USES"}) == "USES"

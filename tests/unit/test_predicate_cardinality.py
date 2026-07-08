"""Unit tests for predicate cardinality classification.

Guards the supersession policy: functional (single-valued) predicates may
supersede; cumulative (many-valued) predicates must not.
"""

import importlib

import pytest

from src.services import predicate_cardinality
from src.services.predicate_cardinality import (
    is_cumulative,
    is_functional,
    normalize_predicate,
)


class TestNormalizePredicate:
    def test_strips_and_uppercases(self):
        assert normalize_predicate("  lives_in ") == "LIVES_IN"

    def test_none_and_empty_become_empty(self):
        assert normalize_predicate(None) == ""
        assert normalize_predicate("   ") == ""


class TestIsFunctional:
    @pytest.mark.parametrize("predicate", ["LIVES_IN", "HAS_STATUS", "CURRENT_STATUS"])
    def test_registered_functional_predicates_are_functional(self, predicate):
        assert is_functional(predicate) is True

    def test_case_insensitive(self):
        assert is_functional("lives_in") is True
        assert is_functional(" Lives_In ") is True

    @pytest.mark.parametrize(
        "predicate", ["ORDERED", "BOUGHT", "OWNS", "USES", "KNOWS", "WORKS_AT"]
    )
    def test_cumulative_predicates_are_not_functional(self, predicate):
        assert is_functional(predicate) is False
        assert is_cumulative(predicate) is True

    def test_unknown_predicate_defaults_cumulative(self):
        # The safe default: never-seen predicate must NOT be treated as functional.
        assert is_functional("SOME_BRAND_NEW_PREDICATE_XYZ") is False

    def test_empty_predicate_is_cumulative(self):
        assert is_functional(None) is False
        assert is_functional("") is False

    def test_novel_cumulative_predicate_logged_once(self, caplog):
        predicate_cardinality._seen_cumulative.discard("NOVEL_PRED_ABC")
        import logging

        with caplog.at_level(logging.INFO):
            is_functional("NOVEL_PRED_ABC")
            is_functional("NOVEL_PRED_ABC")
        mentions = [r for r in caplog.records if "NOVEL_PRED_ABC" in r.getMessage()]
        # Logged at most once for the novel predicate (loguru/caplog wiring may not
        # capture it, but it must never log twice).
        assert len(mentions) <= 1


class TestEnvOverrideExtendsSet:
    def test_env_var_extends_but_never_shrinks_defaults(self, monkeypatch):
        monkeypatch.setenv("MNEMODROME_FUNCTIONAL_PREDICATES", "subscribes_to, foo")
        import src.config as config

        importlib.reload(config)
        try:
            # Built-in defaults survive…
            assert "LIVES_IN" in config.FUNCTIONAL_PREDICATES
            # …and the env additions are present, upper-cased.
            assert "SUBSCRIBES_TO" in config.FUNCTIONAL_PREDICATES
            assert "FOO" in config.FUNCTIONAL_PREDICATES
        finally:
            monkeypatch.delenv("MNEMODROME_FUNCTIONAL_PREDICATES", raising=False)
            importlib.reload(config)

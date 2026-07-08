"""Predicate cardinality classification — the supersession decision seam.

A relationship auto-supersedes a prior same-(subject, predicate) edge ONLY when
the predicate is *functional* (single-valued). Cumulative / many-valued
predicates (ORDERED, BOUGHT, OWNS, USES, KNOWS, ...) describe independent facts
that must coexist even when they share a subject and predicate.

Why this matters: keying supersession on (subject, predicate) alone treats every
predicate as single-valued, so capturing ``Carlos -ORDERED-> EliteDesks`` wrongly
invalidated the independent ``Carlos -ORDERED-> RTX PRO 4500``. Cardinality is an
intrinsic *schema* property of the predicate, not a per-instance judgment, so a
registry is the correct representation.

Design: ``is_functional()`` is the single decision point ("the seam"). Today it is
backed by a deterministic registry (``config.FUNCTIONAL_PREDICATES``) with a
safe-by-default policy — anything not registered is treated as cumulative, so an
unknown predicate can never silently cause data loss. Every novel predicate is
logged once so we can later decide, from real data, whether a smarter classifier
(embedding-similarity to the registered seeds, or a one-shot LLM verdict, written
back into the registry as a low-confidence, review-flagged schema fact) is worth
building. Swapping in that backend is a change to this function only.
"""

from loguru import logger

from src import config

# Process-local set of cumulative predicates already logged, to log each novel
# predicate exactly once instead of on every capture. Instrumentation only — it
# has no effect on the classification result.
_seen_cumulative: set[str] = set()


def normalize_predicate(relation_type: str | None) -> str:
    """Canonical form used for cardinality lookup: stripped, upper-cased."""
    return (relation_type or "").strip().upper()


def is_functional(relation_type: str | None) -> bool:
    """Return True iff this predicate is single-valued (supersession allowed).

    Safe default: an empty/unknown predicate returns False (cumulative), so the
    caller will NOT auto-supersede. The first time a given cumulative predicate
    is seen it is logged once for later analysis.
    """
    predicate = normalize_predicate(relation_type)
    if not predicate:
        return False

    if predicate in config.FUNCTIONAL_PREDICATES:
        return True

    if predicate not in _seen_cumulative:
        _seen_cumulative.add(predicate)
        logger.info(
            "predicate_cardinality: treating '{}' as CUMULATIVE (not in "
            "FUNCTIONAL_PREDICATES); supersession disabled for it. Add it to "
            "MNEMODROME_FUNCTIONAL_PREDICATES if it is single-valued.",
            predicate,
        )
    return False


def is_cumulative(relation_type: str | None) -> bool:
    """Inverse of :func:`is_functional` — many-valued predicate (no supersession)."""
    return not is_functional(relation_type)

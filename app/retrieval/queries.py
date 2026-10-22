"""Rewrite a question, split it, and decide which searches to run."""

import re

from app.retrieval.text import content_tokens

_PREFIXES = (
    "please ",
    "can you tell me ",
    "could you tell me ",
    "would you tell me ",
    "can you ",
    "could you ",
    "would you ",
    "tell me ",
    "i need to know ",
    "i want to know ",
    "i would like to know ",
)
_CLAUSE = re.compile(r"\?+|;|\n+|\s+\band\b\s+", re.IGNORECASE)


def rewrite_query(question: str) -> str:
    """Drop politeness and a trailing question mark so search sees the ask."""
    text = " ".join(question.strip().split())
    changed = True
    while changed and text:
        changed = False
        folded = text.casefold()
        for prefix in _PREFIXES:
            if folded.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
                break
    if text.endswith("?"):
        text = text[:-1].strip()
    return text or question.strip()


def decompose_query(question: str) -> tuple[str, ...]:
    """Split a question into clauses that can be retrieved on their own.

    A split is kept only when both sides still contain two content words, so
    ``connect and read timeouts`` stays one query.
    """
    pieces = [piece.strip(" ?.!") for piece in _CLAUSE.split(question) if piece.strip(" ?.!")]
    kept = [piece for piece in pieces if len(content_tokens(piece)) >= 2]
    if len(kept) < 2:
        return ()
    return tuple(dict.fromkeys(kept))


def plan_searches(
    question: str,
    *,
    rewrite: bool = False,
    decompose: bool = False,
    multi_query: bool = False,
    extra: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the queries to run.

    With every flag off and no graph terms, the only search is the original
    question. Multi-query adds each clause and a keyword query, then fuses
    the rankings.
    """
    if not rewrite and not decompose and not multi_query and not extra:
        return (question,)
    base = rewrite_query(question) if rewrite else question
    clauses = decompose_query(base) if decompose else ()
    if not multi_query and not extra:
        return (base,)
    queries = [base, *clauses, *extra]
    if multi_query:
        keywords = " ".join(content_tokens(base))
        if keywords:
            queries.append(keywords)
    ordered: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = query.strip()
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return tuple(ordered) or (question,)

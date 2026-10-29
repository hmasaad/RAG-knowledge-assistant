"""Query rewriting, decomposition, and multi-query retrieval."""

from app.ingestion.vector_store import VectorHit
from app.retrieval.queries import (
    decompose_query,
    extra_topic_queries,
    plan_searches,
    rewrite_query,
)
from app.retrieval.search import SearchService


def test_admin_portal_adds_section_aliases() -> None:
    extras = extra_topic_queries("What are admin portal requirements?")

    assert "web admin" in extras
    assert "platform admin" in extras
    assert extra_topic_queries("which file implements the retry logic") == ()
    assert "acceptance criteria" in extra_topic_queries("what is MVP scope?")
    assert "authentication & profile" in extra_topic_queries("authentication flow")


def test_rewrite_drops_politeness() -> None:
    rewritten = rewrite_query("Can you tell me which file implements the retry logic?")

    assert rewritten == "which file implements the retry logic"


def test_decompose_keeps_a_short_and() -> None:
    assert decompose_query("where are connect and read timeouts configured") == ()


def test_decompose_splits_two_asks() -> None:
    clauses = decompose_query(
        "which file implements the retry logic and where are timeouts configured"
    )

    assert clauses == (
        "which file implements the retry logic",
        "where are timeouts configured",
    )


def test_multi_query_adds_each_clause_once() -> None:
    searches = plan_searches(
        "which file implements the retry logic and where are timeouts configured",
        rewrite=True,
        decompose=True,
        multi_query=True,
    )

    assert searches[0] == "which file implements the retry logic and where are timeouts configured"
    assert "which file implements the retry logic" in searches
    assert "where are timeouts configured" in searches
    assert len(searches) == len({query.casefold() for query in searches})


def test_default_plan_is_the_original_question() -> None:
    question = "which file implements the retry logic"
    assert plan_searches(question) == (question,)


def test_multi_query_finds_a_clause_the_full_question_misses() -> None:
    pair = _Pair()
    service = SearchService(
        vectors=_Store(pair),
        embedder=pair,
        decompose=True,
        multi_query=True,
    )

    result = service.retrieve(
        "which file implements the retry logic and where are timeouts configured",
        top_k=3,
    )

    assert "where are timeouts configured" in result.queries
    assert any("connect timeout" in passage.text for passage in result.passages)


class _Pair:
    model = "local-hash-bow"
    query = ""

    def embed_query(self, text: str) -> list[float]:
        self.query = text
        return [1.0]


class _Store:
    def __init__(self, pair: _Pair) -> None:
        self._pair = pair

    def query(self, embedding: list[float], top_k: int, document_ids: set[str] | None = None):
        text = self._pair.query.casefold()
        if text.startswith("where are timeouts"):
            return [_hit("timeout", "The HTTP client applies the connect timeout.")]
        if text.startswith("which file implements"):
            return [_hit("retry", "The file that implements retry logic is app/http/retry.py.")]
        return [_hit("other", "Schema changes ship as SQL files.")]

    def list_chunks(self, document_ids: set[str] | None = None):
        return [
            _hit("retry", "The file that implements retry logic is app/http/retry.py."),
            _hit("timeout", "The HTTP client applies the connect timeout."),
            _hit("other", "Schema changes ship as SQL files."),
        ]


def _hit(chunk_id: str, text: str) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        text=text,
        score=0.8,
        metadata={
            "document_id": "doc",
            "filename": f"{chunk_id}.md",
            "content_type": "text/markdown",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len(text),
            "page": -1,
            "heading": "",
            "version": 1,
        },
    )

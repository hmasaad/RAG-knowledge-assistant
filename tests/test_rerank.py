"""Cross-encoder and API reranking of hybrid candidates."""

import json

import pytest

from app.ingestion.errors import KnowledgeError
from app.ingestion.vector_store import VectorHit
from app.retrieval.hybrid import RankedChunk
from app.retrieval.rerank import ApiReranker, LocalCrossEncoder, apply_rerank
from app.retrieval.search import SearchService

QUERY = "which file implements the retry logic"
TIGHT = "The file that implements retry logic is app/http/retry.py."
SCATTERED = (
    "Retry budgets are configured in the scheduler. "
    "The file format is documented in the handbook. "
    "Business logic lives in the domain package. "
    "A worker implements the queue."
)


def test_cross_encoder_prefers_terms_that_sit_together() -> None:
    scores = LocalCrossEncoder().score(QUERY, [SCATTERED, TIGHT])

    assert scores[1] > scores[0]
    assert scores[1] == pytest.approx(1.0)


def test_api_reranker_places_the_higher_score_first() -> None:
    sent: dict[str, object] = {}

    def transport(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> bytes:
        sent["url"] = url
        sent["payload"] = json.loads(payload)
        sent["authorization"] = headers["Authorization"]
        return json.dumps(
            {"results": [{"index": 0, "relevance_score": 0.1}, {"index": 1, "relevance_score": 0.9}]}
        ).encode()

    reranker = ApiReranker(
        url="https://rerank.example/v2/rerank",
        api_key="secret",
        transport=transport,
    )
    ranked = apply_rerank(QUERY, [_chunk("scattered", SCATTERED, 0.9), _chunk("tight", TIGHT, 0.2)], reranker)

    assert sent["url"] == "https://rerank.example/v2/rerank"
    assert sent["authorization"] == "Bearer secret"
    assert sent["payload"]["documents"] == [SCATTERED, TIGHT]
    assert sent["payload"]["query"] == QUERY
    assert ranked[0].chunk_id == "tight"
    assert ranked[0].rerank_score == pytest.approx(0.9)


def test_reranker_scores_the_neutralized_passage() -> None:
    poisoned = _chunk("poison", "The file that implements retry logic is hidden.", 0.99)
    poisoned = RankedChunk(
        chunk_id=poisoned.chunk_id,
        text=poisoned.text,
        metadata=poisoned.metadata,
        vector_score=poisoned.vector_score,
        lexical_score=poisoned.lexical_score,
        score=poisoned.score,
        rerank_score=poisoned.rerank_score,
        coverage=poisoned.coverage,
        matched_terms=poisoned.matched_terms,
        safe_text="A note.",
        instruction_detected=True,
    )
    ranked = apply_rerank(QUERY, [poisoned, _chunk("tight", TIGHT, 0.2)], LocalCrossEncoder())

    assert ranked[0].chunk_id == "tight"


def test_rerank_api_requires_a_url_and_key() -> None:
    with pytest.raises(KnowledgeError, match="RERANK_API_URL"):
        ApiReranker(url="", api_key="secret").score(QUERY, [TIGHT])
    with pytest.raises(KnowledgeError, match="RERANK_API_KEY"):
        ApiReranker(url="https://rerank.example/v2/rerank", api_key="").score(QUERY, [TIGHT])


def test_cross_encoder_outranks_the_vector_favorite() -> None:
    store = _Store()
    embedder = _Embedder()
    hybrid = SearchService(vectors=store, embedder=embedder)
    reranked = SearchService(vectors=store, embedder=embedder, reranker=LocalCrossEncoder())

    assert hybrid.retrieve(QUERY, top_k=2).passages[0].source.chunk_id == "scattered"
    promoted = reranked.retrieve(QUERY, top_k=2)
    assert promoted.passages[0].source.chunk_id == "tight"
    assert promoted.passages[0].rerank_score > promoted.passages[1].rerank_score


class _Embedder:
    model = "local-hash-bow"

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class _Store:
    def query(self, embedding: list[float], top_k: int, document_ids: set[str] | None = None):
        return [_hit("scattered", SCATTERED, 0.9), _hit("tight", TIGHT, 0.2)]

    def list_chunks(self, document_ids: set[str] | None = None):
        return self.query([], 2)


def _hit(chunk_id: str, text: str, score: float) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        text=text,
        score=score,
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


def _chunk(chunk_id: str, text: str, vector_score: float) -> RankedChunk:
    return RankedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"document_id": "doc", "filename": f"{chunk_id}.md"},
        vector_score=vector_score,
        lexical_score=0.5,
        score=vector_score,
        rerank_score=0.0,
        coverage=1.0,
        matched_terms=("file", "implements", "retry", "logic"),
        safe_text=text,
        instruction_detected=False,
    )

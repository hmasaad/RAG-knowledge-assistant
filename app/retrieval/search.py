"""Vector search over the indexed knowledge collection."""

import time
from dataclasses import dataclass

from app.ingestion.embeddings import Embedder
from app.ingestion.errors import IngestionError
from app.ingestion.vector_store import VectorHit, VectorStore
from app.observability.logging import log_event


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    filename: str
    content_type: str
    text: str
    score: float
    chunk_index: int
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


class SearchService:
    def __init__(self, *, vectors: VectorStore, embedder: Embedder) -> None:
        self._vectors = vectors
        self._embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        cleaned = query.strip()
        if not cleaned:
            raise IngestionError("Query must not be empty.")
        if top_k < 1 or top_k > 20:
            raise IngestionError("top_k must be between 1 and 20.")

        started = time.perf_counter()
        embedding = self._embedder.embed_query(cleaned)
        matches = self._vectors.query(embedding, top_k)
        hits = [_hit(match) for match in matches]
        top_score = hits[0].score if hits else 0.0
        log_event(
            "rag.retrieval",
            "search_completed",
            top_k=top_k,
            hit_count=len(hits),
            top_score=round(top_score, 6),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            embedding_model=getattr(self._embedder, "model", "unknown"),
        )
        return hits


def _hit(match: VectorHit) -> SearchHit:
    metadata = match.metadata
    page = metadata.get("page", -1)
    heading = metadata.get("heading") or ""
    return SearchHit(
        chunk_id=match.chunk_id,
        document_id=str(metadata.get("document_id", "")),
        filename=str(metadata.get("filename", "")),
        content_type=str(metadata.get("content_type", "")),
        text=match.text,
        score=match.score,
        chunk_index=int(metadata.get("chunk_index", 0)),
        start_char=int(metadata.get("start_char", 0)),
        end_char=int(metadata.get("end_char", 0)),
        page=None if int(page) < 0 else int(page),
        heading=str(heading) or None,
    )

"""Hybrid retrieval over the indexed knowledge collection."""

import time
from dataclasses import dataclass

from app.ingestion.embeddings import Embedder
from app.ingestion.errors import KnowledgeError
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.vector_store import VectorStore, chunk_version
from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.retrieval.conflicts import documents_conflict
from app.retrieval.hybrid import RankedChunk, rank_chunks
from app.retrieval.inspect import covers_question
from app.retrieval.text import content_tokens


@dataclass(frozen=True)
class SourceMetadata:
    chunk_id: str
    document_id: str
    filename: str
    content_type: str
    chunk_index: int
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


@dataclass(frozen=True)
class RankedPassage:
    rank: int
    text: str
    score: float
    vector_score: float
    lexical_score: float
    answers_question: bool
    matched_terms: tuple[str, ...]
    source: SourceMetadata
    safe_text: str = ""
    instruction_detected: bool = False

    def model_text(self) -> str:
        """Text the model may see. Instruction lines are already removed."""
        if self.instruction_detected or self.safe_text:
            return self.safe_text
        return self.text


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    sufficient_evidence: bool
    inspection: str
    passages: list[RankedPassage]
    conflict: bool = False


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
    def __init__(
        self,
        *,
        vectors: VectorStore,
        embedder: Embedder,
        metadata: MetadataStore | None = None,
        default_user_id: str = "local",
        candidate_k: int = 20,
        rrf_k: int = 60,
        coverage_weight: float = 0.35,
        min_coverage: float = 0.5,
        metrics: Metrics | None = None,
    ) -> None:
        self._vectors = vectors
        self._embedder = embedder
        self._metadata = metadata
        self._default_user_id = default_user_id
        self._candidate_k = candidate_k
        self._rrf_k = rrf_k
        self._coverage_weight = coverage_weight
        self._min_coverage = min_coverage
        self._metrics = metrics

    def search(self, query: str, top_k: int = 5, *, user_id: str | None = None) -> list[SearchHit]:
        """Return ranked passages in the original search shape."""
        retrieved = self.retrieve(query, top_k, user_id=user_id)
        return [_search_hit(passage) for passage in retrieved.passages]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
    ) -> RetrievalResult:
        """Embed a question, fuse vector and lexical rankings, and inspect coverage.

        Only documents the caller can read are searched.
        """
        cleaned = _validate(query, top_k)
        caller = user_id or self._default_user_id
        versions = self._visible_documents(caller)
        if versions is not None and not versions:
            return RetrievalResult(
                query=cleaned,
                sufficient_evidence=False,
                inspection="No documents are visible to this caller.",
                passages=[],
            )
        document_ids = None if versions is None else set(versions)

        started = time.perf_counter()
        try:
            embedding = self._embedder.embed_query(cleaned)
            candidate_k = max(self._candidate_k, top_k)
            vector_hits = _keep_versions(
                self._vectors.query(
                    embedding,
                    candidate_k,
                    document_ids=document_ids,
                ),
                versions,
            )
            ranked = rank_chunks(
                query_tokens=content_tokens(cleaned),
                vector_hits=vector_hits,
                chunks=_keep_versions(
                    self._vectors.list_chunks(document_ids=document_ids),
                    versions,
                ),
                candidate_k=candidate_k,
                rrf_k=self._rrf_k,
                coverage_weight=self._coverage_weight,
            )
        except KnowledgeError:
            raise
        except Exception as exc:
            log_event(
                "rag.retrieval",
                "retrieval_failed",
                user_id=caller,
                error=type(exc).__name__,
            )
            raise KnowledgeError("Retrieval failed.", status_code=503) from exc

        passages = [
            _passage(rank, chunk, min_coverage=self._min_coverage)
            for rank, chunk in enumerate(ranked[:top_k], start=1)
        ]
        sufficient, inspection = _inspection(passages)
        conflict = documents_conflict(
            [
                (passage.source.filename, passage.model_text(), passage.answers_question)
                for passage in passages
            ]
        )
        if conflict:
            inspection = f"{inspection} The visible documents disagree."
        result = RetrievalResult(
            query=cleaned,
            sufficient_evidence=sufficient,
            inspection=inspection,
            passages=passages,
            conflict=conflict,
        )
        top_score = passages[0].score if passages else 0.0
        log_event(
            "rag.retrieval",
            "retrieval_completed",
            user_id=caller,
            top_k=top_k,
            hit_count=len(passages),
            vector_hits=len(vector_hits),
            sufficient_evidence=sufficient,
            conflict=conflict,
            top_score=round(top_score, 6),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            embedding_model=getattr(self._embedder, "model", "unknown"),
        )
        if self._metrics is not None:
            self._metrics.observe(
                "rag_retrieval_seconds",
                time.perf_counter() - started,
            )
        return result

    def _visible_documents(self, user_id: str) -> dict[str, int] | None:
        if self._metadata is None:
            return None
        return self._metadata.readable_versions(user_id)


def _validate(query: str, top_k: int) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise KnowledgeError("Query must not be empty.")
    if top_k < 1 or top_k > 20:
        raise KnowledgeError("top_k must be between 1 and 20.")
    return cleaned


def _passage(rank: int, chunk: RankedChunk, *, min_coverage: float) -> RankedPassage:
    return RankedPassage(
        rank=rank,
        text=chunk.text,
        score=_round(chunk.score),
        vector_score=_round(chunk.vector_score),
        lexical_score=_round(chunk.lexical_score),
        answers_question=covers_question(
            chunk.coverage,
            chunk.matched_terms,
            min_coverage=min_coverage,
        ),
        matched_terms=chunk.matched_terms,
        source=_source(chunk.chunk_id, chunk.metadata),
        safe_text=chunk.safe_text,
        instruction_detected=chunk.instruction_detected,
    )


def _inspection(passages: list[RankedPassage]) -> tuple[bool, str]:
    if not passages:
        return False, "The collection has no indexed passages."
    supporting = sum(1 for passage in passages if passage.answers_question)
    if supporting == 0:
        return False, "None of the retrieved passages cover enough of the question to answer it."
    return True, f"{supporting} of {len(passages)} retrieved passages cover the question."


def _search_hit(passage: RankedPassage) -> SearchHit:
    source = passage.source
    return SearchHit(
        chunk_id=source.chunk_id,
        document_id=source.document_id,
        filename=source.filename,
        content_type=source.content_type,
        text=passage.text,
        score=passage.score,
        chunk_index=source.chunk_index,
        start_char=source.start_char,
        end_char=source.end_char,
        page=source.page,
        heading=source.heading,
    )


def _source(chunk_id: str, metadata: dict[str, object]) -> SourceMetadata:
    page = metadata.get("page", -1)
    heading = metadata.get("heading") or ""
    return SourceMetadata(
        chunk_id=chunk_id,
        document_id=str(metadata.get("document_id", "")),
        filename=str(metadata.get("filename", "")),
        content_type=str(metadata.get("content_type", "")),
        chunk_index=int(metadata.get("chunk_index", 0)),
        start_char=int(metadata.get("start_char", 0)),
        end_char=int(metadata.get("end_char", 0)),
        page=None if int(page) < 0 else int(page),
        heading=str(heading) or None,
    )


def _round(value: float) -> float:
    return round(float(value), 6)


def _keep_versions(hits: list, versions: dict[str, int] | None) -> list:
    if versions is None:
        return hits
    kept = []
    for hit in hits:
        document_id = str(hit.metadata.get("document_id", ""))
        expected = versions.get(document_id)
        if expected is not None and chunk_version(hit.metadata) == expected:
            kept.append(hit)
    return kept

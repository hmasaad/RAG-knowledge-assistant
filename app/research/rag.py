"""The search operation the research agent calls on the knowledge collection."""

import math
import time
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.errors import KnowledgeError
from app.retrieval.search import RankedPassage, RetrievalResult, SearchService, SourceMetadata


class RagContractError(KnowledgeError):
    """The knowledge collection broke the search contract."""


class RagTimeout(RagContractError):
    def __init__(self) -> None:
        super().__init__("Retrieval timed out.", status_code=504)


class RagMalformed(RagContractError):
    def __init__(self) -> None:
        super().__init__("Malformed retrieval response.", status_code=502)


@dataclass(frozen=True)
class SearchFilters:
    """Which documents a research search may use.

    ``document_ids`` is ``None`` when every document the caller can read is
    eligible. An empty tuple matches nothing.
    """

    user_id: str | None = None
    document_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SourceLocation:
    document_id: str
    chunk_id: str
    filename: str
    content_type: str
    chunk_index: int
    start_char: int
    end_char: int
    page: int | None
    heading: str | None


@dataclass(frozen=True)
class RagPassage:
    """One ranked passage from the knowledge collection."""

    text: str
    document_id: str
    score: float
    source: SourceLocation
    answers_question: bool
    matched_terms: tuple[str, ...] = ()
    safe_text: str = ""
    instruction_detected: bool = False
    vector_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 1


@dataclass(frozen=True)
class RagSearchResult:
    query: str
    passages: tuple[RagPassage, ...]
    sufficient_evidence: bool
    conflict: bool = False
    inspection: str = ""


class RagClient(Protocol):
    def search(self, query: str, filters: SearchFilters, top_k: int) -> RagSearchResult:
        """Return ranked passages, document ids, source locations, and scores."""


class KnowledgeRagClient:
    """Adapt the knowledge collection to the research search contract."""

    def __init__(self, retrieval: SearchService, *, timeout: float | None = None) -> None:
        self._retrieval = retrieval
        self._timeout = timeout

    def search(self, query: str, filters: SearchFilters, top_k: int) -> RagSearchResult:
        if filters.document_ids is not None and not filters.document_ids:
            return RagSearchResult(
                query=query.strip(),
                passages=(),
                sufficient_evidence=False,
                inspection="No documents match the filter.",
            )
        started = time.perf_counter()
        retrieved = self._retrieval.retrieve(query, top_k, user_id=filters.user_id)
        if self._timeout is not None and time.perf_counter() - started >= self._timeout:
            raise RagTimeout()
        allowed = None if filters.document_ids is None else set(filters.document_ids)
        passages = tuple(
            _from_ranked(passage)
            for passage in retrieved.passages
            if allowed is None or passage.source.document_id in allowed
        )
        if allowed is None:
            return RagSearchResult(
                query=retrieved.query,
                passages=passages,
                sufficient_evidence=retrieved.sufficient_evidence,
                conflict=retrieved.conflict,
                inspection=retrieved.inspection,
            )
        sufficient = any(passage.answers_question for passage in passages)
        return RagSearchResult(
            query=retrieved.query,
            passages=passages,
            sufficient_evidence=sufficient,
            conflict=False,
            inspection=retrieved.inspection if passages else "No documents match the filter.",
        )


def as_retrieval(result: RagSearchResult) -> RetrievalResult:
    """Rebuild the answerer's retrieval result from the contract."""
    passages = [_to_ranked(passage) for passage in result.passages]
    return RetrievalResult(
        query=result.query,
        sufficient_evidence=result.sufficient_evidence,
        inspection=result.inspection,
        passages=passages,
        conflict=result.conflict,
        queries=(result.query,),
    )


def _from_ranked(passage: RankedPassage) -> RagPassage:
    source = passage.source
    ranked = RagPassage(
        text=passage.text,
        document_id=source.document_id,
        score=passage.score,
        source=SourceLocation(
            document_id=source.document_id,
            chunk_id=source.chunk_id,
            filename=source.filename,
            content_type=source.content_type,
            chunk_index=source.chunk_index,
            start_char=source.start_char,
            end_char=source.end_char,
            page=source.page,
            heading=source.heading,
        ),
        answers_question=passage.answers_question,
        matched_terms=passage.matched_terms,
        safe_text=passage.safe_text,
        instruction_detected=passage.instruction_detected,
        vector_score=passage.vector_score,
        lexical_score=passage.lexical_score,
        rerank_score=passage.rerank_score,
        rank=passage.rank,
    )
    _validate(ranked)
    return ranked


def _validate(passage: RagPassage) -> None:
    source = passage.source
    if not passage.document_id or not source.document_id or not source.chunk_id:
        raise RagMalformed()
    if source.start_char < 0 or source.end_char < source.start_char:
        raise RagMalformed()
    if not math.isfinite(passage.score):
        raise RagMalformed()
    if passage.document_id != source.document_id:
        raise RagMalformed()


def _to_ranked(passage: RagPassage) -> RankedPassage:
    source = passage.source
    return RankedPassage(
        rank=passage.rank,
        text=passage.text,
        score=passage.score,
        vector_score=passage.vector_score,
        lexical_score=passage.lexical_score,
        answers_question=passage.answers_question,
        matched_terms=passage.matched_terms,
        source=SourceMetadata(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            content_type=source.content_type,
            chunk_index=source.chunk_index,
            start_char=source.start_char,
            end_char=source.end_char,
            page=source.page,
            heading=source.heading,
        ),
        safe_text=passage.safe_text,
        instruction_detected=passage.instruction_detected,
        rerank_score=passage.rerank_score,
    )

"""End-to-end question answering over the knowledge collection."""

import time

from app.generation.answerers import Answerer
from app.generation.grounding import AnswerResult
from app.ingestion.metadata_store import MetadataStore
from app.observability.cache import LruCache
from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.retrieval.search import SearchService


class GenerationService:
    def __init__(
        self,
        *,
        retrieval: SearchService,
        answerer: Answerer,
        metadata: MetadataStore,
        cache: LruCache[tuple[object, ...], AnswerResult],
        metrics: Metrics,
        default_user_id: str = "local",
    ) -> None:
        self._retrieval = retrieval
        self._answerer = answerer
        self._metadata = metadata
        self._cache = cache
        self._metrics = metrics
        self._default_user_id = default_user_id

    def answer(self, question: str, top_k: int = 4, *, user_id: str | None = None) -> AnswerResult:
        caller = user_id or self._default_user_id
        key = (self._metadata.generation(), caller, question.strip(), top_k)
        cached = self._cache.get(key)
        if cached is not None:
            self._metrics.inc("rag_answer_cache_hits_total")
            log_event(
                "rag.generation",
                "answer_cache_hit",
                user_id=caller,
                catalog_generation=key[0],
            )
            return cached

        self._metrics.inc("rag_answer_cache_misses_total")
        started = time.perf_counter()
        retrieved = self._retrieval.retrieve(question, top_k, user_id=caller)
        result = self._answerer.generate(question, retrieved)
        self._cache.put(key, result)
        log_event(
            "rag.generation",
            "answer_completed",
            user_id=caller,
            sufficient_evidence=result.sufficient_evidence,
            conflict=result.conflict,
            citation_count=len(result.citations),
            passage_count=len(result.passages),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            model=getattr(self._answerer, "model", "unknown"),
        )
        return result

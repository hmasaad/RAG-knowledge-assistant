"""End-to-end question answering over the knowledge collection."""

import time

from app.generation.agent import clarification, needs_clarification, reformulate
from app.generation.answerers import Answerer
from app.generation.grounding import AnswerResult, refusal
from app.ingestion.metadata_store import MetadataStore
from app.observability.cache import LruCache
from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.observability.tracing import TraceLog
from app.retrieval.search import SearchService
from app.security.leakage import answer_leaks


class GenerationService:
    def __init__(
        self,
        *,
        retrieval: SearchService,
        answerer: Answerer,
        metadata: MetadataStore,
        cache: LruCache[tuple[object, ...], AnswerResult],
        metrics: Metrics,
        traces: TraceLog | None = None,
        default_user_id: str = "local",
        agent_enabled: bool = True,
    ) -> None:
        self._retrieval = retrieval
        self._answerer = answerer
        self._metadata = metadata
        self._cache = cache
        self._metrics = metrics
        self._traces = traces if traces is not None else TraceLog()
        self._default_user_id = default_user_id
        self._agent_enabled = agent_enabled

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
        trace_id = self._traces.start(user_id=caller, question=question.strip())
        if self._agent_enabled and needs_clarification(question):
            result = clarification(question)
            result.trace_id = trace_id
            self._traces.step(trace_id, action="clarify")
            self._finish(key, caller, result, started, trace_id)
            return result

        retrieved = self._retrieval.retrieve(question, top_k, user_id=caller)
        self._traces.step(
            trace_id,
            action="search",
            query=question.strip(),
            sufficient_evidence=retrieved.sufficient_evidence,
            hit_count=len(retrieved.passages),
        )
        action = "search"
        if (
            self._agent_enabled
            and not retrieved.sufficient_evidence
            and retrieved.passages
        ):
            alternative = reformulate(question)
            if alternative is not None:
                retried = self._retrieval.retrieve(alternative, top_k, user_id=caller)
                self._traces.step(
                    trace_id,
                    action="retry",
                    query=alternative,
                    sufficient_evidence=retried.sufficient_evidence,
                    hit_count=len(retried.passages),
                )
                if retried.sufficient_evidence:
                    retrieved = retried
                    action = "retry"
        result = self._answerer.generate(question, retrieved)
        forbidden = self._retrieval.hidden_sensitive_tokens(caller)
        if answer_leaks(result.answer, forbidden):
            result = refusal(question, result.passages)
            log_event(
                "rag.generation",
                "answer_leak_blocked",
                user_id=caller,
                trace_id=trace_id,
            )
        if action != "retry":
            action = "answer" if result.sufficient_evidence else "refuse"
        result.action = action
        result.trace_id = trace_id
        self._traces.step(trace_id, action=action, sufficient_evidence=result.sufficient_evidence)
        self._finish(key, caller, result, started, trace_id)
        return result

    def _finish(
        self,
        key: tuple[object, ...],
        caller: str,
        result: AnswerResult,
        started: float,
        trace_id: str,
    ) -> None:
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
            action=result.action,
            trace_id=trace_id,
        )

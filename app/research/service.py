"""Investigate a question in steps and consolidate the evidence."""

import re
import time
from dataclasses import dataclass, replace

from app.generation.agent import CLARIFICATION_TEXT, needs_clarification, reformulate
from app.generation.answerers import Answerer
from app.generation.grounding import AnswerResult, Citation
from app.research.verification import (
    ClaimAssessment,
    Contradiction,
    find_contradictions,
    passage_contradiction,
    verify_answer,
)
from app.ingestion.graph import path_entities
from app.ingestion.metadata_store import MetadataStore
from app.observability.cache import LruCache
from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.observability.tracing import TraceLog
from app.research.evidence import ClaimLink, EvidenceRecord, EvidenceStore
from app.research.graph import EvidenceGraph, ProvenanceEngine
from app.research.orchestrator import SearchBudget, run_search
from app.research.planner import GRAPH_COMPLETION, KNOWLEDGE_COMPLETION, build_plan
from app.research.rag import KnowledgeRagClient, RagClient, RagSearchResult, SearchFilters, as_retrieval
from app.retrieval.search import RankedPassage, SearchService
from app.security.leakage import answer_leaks
from app.security.prompts import REFUSAL_TEXT

_MARKER = re.compile(r"\[(\d+)\]")
_QUOTE_LIMIT = 500
_GRAPH_LIMIT = 4


@dataclass(frozen=True)
class ResearchStep:
    question: str
    tool: str
    status: str
    evidence: str
    completion: str = ""
    complete: bool = False


@dataclass
class ResearchReport:
    question: str
    report: str
    sufficient_evidence: bool
    conflict: bool
    tools: tuple[str, ...]
    subquestions: tuple[ResearchStep, ...]
    contradictions: tuple[str, ...]
    gaps: tuple[str, ...]
    catalog: tuple[str, ...]
    citations: list[Citation]
    passages: list[RankedPassage]
    action: str
    trace_id: str = ""
    evidence_records: tuple[EvidenceRecord, ...] = ()
    links: tuple[ClaimLink, ...] = ()
    conflicts: tuple[Contradiction, ...] = ()
    graph: EvidenceGraph | None = None


class _EvidenceBoard:
    """Renumber citation markers as findings are accepted."""

    def __init__(self) -> None:
        self.passages: list[RankedPassage] = []
        self._by_chunk: dict[str, int] = {}

    def rewrite(self, answer: AnswerResult) -> str:
        accepted = {int(citation.marker.strip("[]")) for citation in answer.citations}

        def replace(match: re.Match[str]) -> str:
            number = int(match.group(1))
            if number not in accepted or number > len(answer.passages):
                return match.group(0)
            return f"[{self._place(answer.passages[number - 1])}]"

        return _MARKER.sub(replace, answer.answer).strip()

    def citations(self) -> list[Citation]:
        cited: list[Citation] = []
        for number, passage in enumerate(self.passages, start=1):
            source = passage.source
            cited.append(
                Citation(
                    marker=f"[{number}]",
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    filename=source.filename,
                    content_type=source.content_type,
                    heading=source.heading,
                    page=source.page,
                    quote=_quote(passage.model_text() or passage.text),
                )
            )
        return cited

    def _place(self, passage: RankedPassage) -> int:
        chunk_id = passage.source.chunk_id
        existing = self._by_chunk.get(chunk_id)
        if existing is not None:
            return existing
        rank = len(self.passages) + 1
        self.passages.append(_with_rank(passage, rank))
        self._by_chunk[chunk_id] = rank
        return rank


class ResearchService:
    """Decompose a question, gather evidence, and write one research report."""

    def __init__(
        self,
        *,
        retrieval: SearchService,
        answerer: Answerer,
        metadata: MetadataStore,
        cache: LruCache[tuple[object, ...], ResearchReport],
        metrics: Metrics,
        traces: TraceLog | None = None,
        default_user_id: str = "local",
        rag: RagClient | None = None,
        search_budget: int = 16,
        search_timeout: float | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._answerer = answerer
        self._metadata = metadata
        self._cache = cache
        self._metrics = metrics
        self._traces = traces if traces is not None else TraceLog()
        self._default_user_id = default_user_id
        self._rag = rag if rag is not None else KnowledgeRagClient(retrieval, timeout=search_timeout)
        self._budget_limit = search_budget

    def research(
        self,
        question: str,
        top_k: int = 4,
        *,
        user_id: str | None = None,
    ) -> ResearchReport:
        caller = user_id or self._default_user_id
        asked = question.strip()
        key = ("research", self._metadata.generation(), caller, asked, top_k)
        cached = self._cache.get(key)
        if cached is not None:
            self._metrics.inc("rag_research_cache_hits_total")
            log_event("rag.research", "research_cache_hit", user_id=caller)
            return cached

        self._metrics.inc("rag_research_requests_total")
        started = time.perf_counter()
        trace_id = self._traces.start(user_id=caller, question=asked)
        if needs_clarification(asked):
            report = _empty_report(asked, CLARIFICATION_TEXT, action="clarify", trace_id=trace_id)
            self._traces.step(trace_id, action="clarify")
            self._finish(key, caller, report, started)
            return report

        plan = build_plan(asked, budget=self._budget_limit)
        budget = SearchBudget(plan.budget)
        store = EvidenceStore()
        self._traces.step(
            trace_id,
            action="plan",
            query=" | ".join(task.question for task in plan.tasks),
            tool=",".join(plan.tools),
            hit_count=len(plan.tasks),
        )
        board = _EvidenceBoard()
        steps: list[ResearchStep] = []
        contradictions: list[str] = []
        conflicts: list[Contradiction] = []
        gaps: list[str] = []
        evidence: list[str] = []
        searched: set[str] = set()

        catalog = self._catalog(caller, plan.tools, trace_id)
        for task in plan.tasks:
            step = self._investigate(
                task.question,
                tool=task.tool,
                completion=task.completion,
                top_k=top_k,
                caller=caller,
                trace_id=trace_id,
                board=board,
                contradictions=contradictions,
                conflicts=conflicts,
                gaps=gaps,
                evidence=evidence,
                budget=budget,
                store=store,
            )
            steps.append(step)
            searched.add(task.question.casefold())
        for subquestion in plan.deferred:
            gaps.append(subquestion)
            steps.append(
                ResearchStep(subquestion, "knowledge", "gap", "", KNOWLEDGE_COMPLETION, False)
            )
        if "graph" in plan.tools:
            self._investigate_graph(
                asked,
                caller=caller,
                top_k=top_k,
                trace_id=trace_id,
                board=board,
                steps=steps,
                contradictions=contradictions,
                conflicts=conflicts,
                gaps=gaps,
                evidence=evidence,
                searched=searched,
                budget=budget,
                store=store,
            )

        verified = [step for step in steps if step.status in {"supported", "contradicted"}]
        action = "research" if verified or contradictions or catalog else "refuse"
        report_text = _render(evidence, contradictions, gaps, catalog, refused=not verified)
        report = ResearchReport(
            question=asked,
            report=report_text,
            sufficient_evidence=bool(verified) and not gaps,
            conflict=bool(contradictions),
            tools=plan.tools,
            subquestions=tuple(steps),
            contradictions=tuple(contradictions),
            gaps=tuple(gaps),
            catalog=catalog,
            citations=board.citations(),
            passages=list(board.passages),
            action=action,
            trace_id=trace_id,
            evidence_records=tuple(store.records),
            links=tuple(store.links),
            conflicts=tuple(conflicts),
            graph=ProvenanceEngine(self._metadata).from_research(store.records, conflicts),
        )
        report = self._block_leaks(report, caller, trace_id)
        self._traces.step(
            trace_id,
            action=report.action,
            sufficient_evidence=report.sufficient_evidence,
            hit_count=len(report.citations),
        )
        self._finish(key, caller, report, started)
        return report

    def _investigate(
        self,
        question: str,
        *,
        tool: str,
        completion: str,
        top_k: int,
        caller: str,
        trace_id: str,
        board: _EvidenceBoard,
        contradictions: list[str],
        conflicts: list[Contradiction],
        gaps: list[str],
        evidence: list[str],
        budget: SearchBudget,
        store: EvidenceStore,
    ) -> ResearchStep:
        found, error = self._retrieve(question, top_k, caller, trace_id, tool, budget)
        if error or found is None:
            gaps.append(question)
            self._traces.step(
                trace_id,
                action="gap",
                query=question,
                tool=tool,
                sufficient_evidence=False,
                hit_count=0,
            )
            return ResearchStep(question, tool, "gap", error, completion, False)
        retrieved = as_retrieval(found)
        answer = self._answerer.generate(question, retrieved)
        claims = verify_answer(answer)
        verified = bool(claims)
        self._traces.step(
            trace_id,
            action="verify",
            query=question,
            tool=tool,
            sufficient_evidence=verified,
            hit_count=len(claims),
        )
        linked = _record_supported(claims, found.passages, store)
        found_conflicts = find_contradictions(question, linked)
        if not found_conflicts and retrieved.conflict:
            found_conflicts = [passage_contradiction(question, retrieved.passages)]
        if found_conflicts:
            conflicts.extend(found_conflicts)
            contradictions.extend(item.summary for item in found_conflicts)
            self._traces.step(
                trace_id,
                action="contradiction",
                query=question,
                tool=tool,
                sufficient_evidence=verified,
            )
        if verified:
            text = board.rewrite(answer)
            evidence.append(text)
            status = "contradicted" if found_conflicts else "supported"
            return ResearchStep(question, tool, status, text, completion, True)
        gaps.append(question)
        self._traces.step(
            trace_id,
            action="gap",
            query=question,
            tool=tool,
            sufficient_evidence=False,
            hit_count=len(retrieved.passages),
        )
        return ResearchStep(question, tool, "gap", "", completion, False)

    def _retrieve(
        self,
        question: str,
        top_k: int,
        caller: str,
        trace_id: str,
        tool: str,
        budget: SearchBudget,
    ) -> tuple[RagSearchResult | None, str]:
        filters = SearchFilters(user_id=caller)
        found, error = run_search(self._rag, budget, question, filters, top_k)
        if error == "Search budget exhausted.":
            return None, error
        self._traces.step(
            trace_id,
            action="search",
            query=question,
            tool=tool,
            sufficient_evidence=bool(found and found.sufficient_evidence),
            hit_count=0 if found is None else len(found.passages),
        )
        if error or found is None:
            return None, error
        if found.sufficient_evidence or not found.passages:
            return found, ""
        alternative = reformulate(question)
        if alternative is None:
            return found, ""
        retried, retry_error = run_search(self._rag, budget, alternative, filters, top_k)
        if retry_error == "Search budget exhausted.":
            return found, ""
        self._traces.step(
            trace_id,
            action="retry",
            query=alternative,
            tool=tool,
            sufficient_evidence=bool(retried and retried.sufficient_evidence),
            hit_count=0 if retried is None else len(retried.passages),
        )
        if retried is not None and retried.sufficient_evidence:
            return retried, ""
        return found, ""

    def _catalog(self, caller: str, tools: tuple[str, ...], trace_id: str) -> tuple[str, ...]:
        if "catalog" not in tools:
            return ()
        names = tuple(
            record.filename
            for record in self._metadata.list_visible(caller)
            if record.indexed_version >= 1
        )
        self._traces.step(
            trace_id,
            action="catalog",
            tool="catalog",
            hit_count=len(names),
            query=", ".join(names),
        )
        return names

    def _investigate_graph(
        self,
        question: str,
        *,
        caller: str,
        top_k: int,
        trace_id: str,
        board: _EvidenceBoard,
        steps: list[ResearchStep],
        contradictions: list[str],
        conflicts: list[Contradiction],
        gaps: list[str],
        evidence: list[str],
        searched: set[str],
        budget: SearchBudget,
        store: EvidenceStore,
    ) -> None:
        names = path_entities(question)
        visible = set(self._metadata.readable_versions(caller))
        neighbors = [
            name
            for name in self._metadata.neighbors(names, visible)[:_GRAPH_LIMIT]
            if name.casefold() not in searched
        ]
        self._traces.step(
            trace_id,
            action="graph",
            tool="graph",
            query=", ".join(neighbors),
            hit_count=len(neighbors),
        )
        for neighbor in neighbors:
            searched.add(neighbor.casefold())
            steps.append(
                self._investigate(
                    neighbor,
                    tool="graph",
                    completion=GRAPH_COMPLETION,
                    top_k=top_k,
                    caller=caller,
                    trace_id=trace_id,
                    board=board,
                    contradictions=contradictions,
                    conflicts=conflicts,
                    gaps=gaps,
                    evidence=evidence,
                    budget=budget,
                    store=store,
                )
            )

    def _block_leaks(self, report: ResearchReport, caller: str, trace_id: str) -> ResearchReport:
        hidden = self._retrieval.hidden_sensitive_tokens(caller)
        if not answer_leaks(report.report, hidden):
            return report
        log_event("rag.research", "research_leak_blocked", user_id=caller, trace_id=trace_id)
        report.report = REFUSAL_TEXT
        report.sufficient_evidence = False
        report.action = "refuse"
        report.citations = []
        report.passages = []
        report.evidence_records = ()
        report.links = ()
        report.conflicts = ()
        report.graph = EvidenceGraph()
        return report

    def _finish(self, key, caller: str, report: ResearchReport, started: float) -> None:
        self._cache.put(key, report)
        log_event(
            "rag.research",
            "research_completed",
            user_id=caller,
            action=report.action,
            sufficient_evidence=report.sufficient_evidence,
            conflict=report.conflict,
            citation_count=len(report.citations),
            gap_count=len(report.gaps),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            trace_id=report.trace_id,
        )


def _record_supported(
    claims: list[ClaimAssessment],
    passages: tuple,
    store: EvidenceStore,
) -> list[ClaimAssessment]:
    by_chunk = {passage.source.chunk_id: passage for passage in passages}
    linked: list[ClaimAssessment] = []
    for claim in claims:
        supporting = by_chunk.get(claim.passage.source.chunk_id)
        if supporting is None:
            continue
        record = store.record_claim(
            claim.claim,
            supporting,
            verdict=claim.verdict,
            support=claim.support,
        )
        if record is None:
            continue
        linked.append(replace(claim, claim_id=record.claim_id))
    return linked


def _render(
    evidence: list[str],
    contradictions: list[str],
    gaps: list[str],
    catalog: tuple[str, ...],
    *,
    refused: bool,
) -> str:
    lines: list[str] = []
    if catalog:
        lines.append("Visible documents: " + ", ".join(catalog) + ".")
    if evidence:
        lines.extend(evidence)
    elif refused:
        lines.append(REFUSAL_TEXT)
    if contradictions:
        lines.append("Contradictions: " + " ".join(contradictions))
    if gaps:
        lines.append("Evidence gaps: " + "; ".join(gaps) + ".")
    return "\n".join(lines)


def _empty_report(question: str, text: str, *, action: str, trace_id: str) -> ResearchReport:
    return ResearchReport(
        question=question,
        report=text,
        sufficient_evidence=False,
        conflict=False,
        tools=(),
        subquestions=(),
        contradictions=(),
        gaps=(),
        catalog=(),
        citations=[],
        passages=[],
        action=action,
        trace_id=trace_id,
    )


def _quote(text: str) -> str:
    stripped = text.strip()
    if len(stripped) <= _QUOTE_LIMIT:
        return stripped
    return stripped[: _QUOTE_LIMIT - 1].rstrip() + "…"


def _with_rank(passage: RankedPassage, rank: int) -> RankedPassage:
    return RankedPassage(
        rank=rank,
        text=passage.text,
        score=passage.score,
        vector_score=passage.vector_score,
        lexical_score=passage.lexical_score,
        answers_question=passage.answers_question,
        matched_terms=passage.matched_terms,
        source=passage.source,
        safe_text=passage.safe_text,
        instruction_detected=passage.instruction_detected,
        rerank_score=passage.rerank_score,
    )

"""Investigate a question in steps and consolidate the evidence."""

import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

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
from app.ingestion.metadata_store import MetadataStore
from app.observability.cache import LruCache
from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.observability.tracing import TraceLog
from app.research.evidence import ClaimLink, EvidenceRecord, EvidenceStore
from app.research.graph import EvidenceGraph, ProvenanceEngine
from app.research.loop import PROBE_CONFLICT, REPORT, RETRY, STOP, Decision, LoopState, decide
from app.research.observability import ResearchObserver
from app.research.quality import apply_quality_gate, evaluate_research
from app.research.orchestrator import SearchBudget
from app.research.coordination import CoordinationReport, MultiAgentCoordinator, findings_from_report
from app.research.strategy import StrategyMemory, StrategyStat
from app.research.planner import GRAPH_COMPLETION, KNOWLEDGE_COMPLETION, build_plan
from app.research.rag import KnowledgeRagClient, RagClient, RagSearchResult, SearchFilters, as_retrieval
from app.research.tools import ToolContext, approved_tools
from app.ingestion.errors import KnowledgeError
from app.research.contracts import ClaimVerdict, EvidenceRef, ResearchState
from app.research.cycle import CycleCancelled, CycleDeadline, CycleHalted, CycleResult
from app.research.jobs import ProductionResearchOrchestrator
from app.research.sessions import ResearchSession, ResearchSessionStore
from app.research.state_manager import DurableResearchStateManager
from app import __version__
from app.engineering.service import EngineeringService
from app.research.worker import ResearchWorker
from app.retrieval.search import RankedPassage, SearchService
from app.security.leakage import answer_leaks
from app.security.prompts import REFUSAL_TEXT

_MARKER = re.compile(r"\[(\d+)\]")
_QUOTE_LIMIT = 500
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
    decision: str = "report"
    decisions: tuple[str, ...] = ()
    remaining_budget: int = 0
    session_id: str = ""
    status: str = "completed"
    side_effects: tuple[dict, ...] = ()
    quality_action: str = "stop"
    quality_reason: str = ""
    quality_failures: tuple[str, ...] = ()
    quality_claims: tuple = ()
    quality_confident: bool = True
    reliability: float = 0.0
    coordination: CoordinationReport | None = None


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
        adapt_rounds: int = 2,
        worker: ResearchWorker | None = None,
        indexing_mode: str = "inline",
        job_timeout_seconds: float = 900,
        sessions: ResearchSessionStore | None = None,
        worker_concurrency: int = 2,
        user_concurrency: int = 2,
        worker_id: str = "worker",
        engineering: EngineeringService | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._answerer = answerer
        self._metadata = metadata
        self._cache = cache
        self._metrics = metrics
        self._traces = traces if traces is not None else TraceLog()
        self._observer = ResearchObserver(metrics, self._traces)
        self._default_user_id = default_user_id
        self._rag = rag if rag is not None else KnowledgeRagClient(retrieval, timeout=search_timeout)
        self._budget_limit = search_budget
        self._adapt_rounds = adapt_rounds
        self._worker = worker if worker is not None else ResearchWorker(
            approved_tools(rag=self._rag, metadata=metadata),
            budget=search_budget,
            max_iterations=max(8, search_budget),
            observer=self._observer,
        )
        self._worker.observer = self._observer
        self._worker._dispatcher.observer = self._observer
        self._indexing_mode = indexing_mode
        self._job_timeout_seconds = job_timeout_seconds
        self._sessions = sessions if sessions is not None else ResearchSessionStore(metadata.path)
        self._state = DurableResearchStateManager(
            self._sessions,
            lease_seconds=job_timeout_seconds,
        )
        self._jobs = ProductionResearchOrchestrator(
            self._state,
            self._run_session,
            concurrency=worker_concurrency,
            user_concurrency=user_concurrency,
            worker_id=worker_id,
            metrics=metrics,
            traces=self._traces,
        )
        self._engineering = engineering if engineering is not None else EngineeringService(metadata.path)
        self._strategy = StrategyMemory(metadata.path)
        self._worker.strategy = self._strategy

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
        with self._observer.on_trace(trace_id):
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

            decisions = self._adapt(
                top_k=top_k,
                caller=caller,
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
            last = decisions[-1] if decisions else Decision(REPORT, "Evidence is sufficient.")
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
                decision=last.action,
                decisions=tuple(item.action for item in decisions),
                remaining_budget=budget.remaining(),
            )
            report = self._block_leaks(report, caller, trace_id)
            _attach_quality(report)
            _attach_coordination(report, budget=self._budget_limit)
            self._strategy.learn(report)
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
        context = ToolContext(
            user_id=caller,
            top_k=top_k,
            filters=SearchFilters(user_id=caller),
            budget=budget,
            document_ids=set(self._metadata.readable_versions(caller)),
        )
        first = self._worker.execute("knowledge", question, context)
        if first.error == "Search budget exhausted.":
            return None, first.error
        found = first.search
        self._traces.step(
            trace_id,
            action="search",
            query=question,
            tool=tool,
            sufficient_evidence=bool(found and found.sufficient_evidence),
            hit_count=0 if found is None else len(found.passages),
        )
        if not first.ok or found is None:
            return None, first.error
        if found.sufficient_evidence or not found.passages:
            return found, ""
        alternative = reformulate(question)
        if alternative is None:
            return found, ""
        retried_call = self._worker.execute("knowledge", alternative, context)
        if retried_call.error == "Search budget exhausted.":
            return found, ""
        retried = retried_call.search
        self._traces.step(
            trace_id,
            action="retry",
            query=alternative,
            tool=tool,
            sufficient_evidence=bool(retried and retried.sufficient_evidence),
            hit_count=0 if retried is None else len(retried.passages),
        )
        self._observer.record_retry("search")
        if retried is not None and retried.sufficient_evidence:
            return retried, ""
        return found, ""

    def _catalog(self, caller: str, tools: tuple[str, ...], trace_id: str) -> tuple[str, ...]:
        if "catalog" not in tools:
            return ()
        result = self._worker.execute(
            "catalog",
            "",
            ToolContext(user_id=caller),
        )
        names = result.names
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
        result = self._worker.execute(
            "graph",
            question,
            ToolContext(
                user_id=caller,
                document_ids=set(self._metadata.readable_versions(caller)),
            ),
        )
        neighbors = [name for name in result.names if name.casefold() not in searched]
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

    def _adapt(
        self,
        *,
        top_k: int,
        caller: str,
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
    ) -> list[Decision]:
        decisions: list[Decision] = []
        gap_retries = 0
        conflict_probes = 0
        while True:
            quality_action, quality_reason, quality_failures = _quality_feedback(
                board,
                evidence,
                contradictions,
                gaps,
                steps,
            )
            state = _loop_state(
                steps,
                gaps,
                conflicts,
                store,
                budget,
                gap_retries=gap_retries,
                conflict_probes=conflict_probes,
                max_adapts=self._adapt_rounds,
                quality_action=quality_action,
                quality_reason=quality_reason,
                quality_failures=quality_failures,
            )
            choice = decide(state, strategy=self._strategy)
            decisions.append(choice)
            self._traces.step(
                trace_id,
                action="decide",
                query=choice.query or choice.action,
                tool=choice.action,
                sufficient_evidence=state.complete and not state.gaps,
                hit_count=budget.remaining(),
            )
            if choice.action in {REPORT, STOP} or not choice.query:
                break
            if choice.query.casefold() in searched:
                break
            searched.add(choice.query.casefold())
            step = self._investigate(
                choice.query,
                tool="knowledge",
                completion=KNOWLEDGE_COMPLETION,
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
            if choice.action == RETRY:
                gap_retries += 1
                self._observer.record_retry("gap")
                _replace_gap(steps, gaps, step)
            else:
                conflict_probes += 1
                self._observer.record_retry("conflict")
                if step.complete:
                    steps.append(step)
                else:
                    _drop_gap(gaps, step.question)
        return decisions

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
        report.decision = "stop"
        report.decisions = ()
        return report

    def propose_plan(
        self,
        question: str,
        *,
        user_id: str | None = None,
        top_k: int = 4,
    ):
        from app.research.autonomous import AutonomousPlanner

        caller = user_id or self._default_user_id
        planner = AutonomousPlanner(self._worker.guardrails)
        return planner.generate(
            question.strip(),
            budget=self._budget_limit,
            user_id=caller,
            top_k=top_k,
        )

    def learn_strategy(self, situation: str, action: str, reward: float) -> StrategyStat:
        return self._strategy.observe(situation, action, reward)

    def recommend_strategy(self, situation: str, fallback: str = "stop") -> str:
        denied = self._worker.guardrails.policy.denied_actions
        return self._strategy.recommend(situation, fallback, denied=denied)

    def strategy_snapshot(self) -> list[StrategyStat]:
        return self._strategy.snapshot()

    def coordinate(
        self,
        question: str,
        *,
        findings: list | None = None,
        user_id: str | None = None,
    ) -> CoordinationReport:
        denied = self._worker.guardrails.policy.denied_actions
        return MultiAgentCoordinator(denied_actions=denied).coordinate(
            question.strip(),
            findings=tuple(findings or ()),
            budget=self._budget_limit,
        )

    def cycle(
        self,
        question: str,
        top_k: int = 4,
        *,
        user_id: str | None = None,
        halt_after: int | None = None,
        priority: int = 0,
    ) -> ResearchReport:
        return self.submit_cycle(
            question,
            top_k,
            user_id=user_id,
            halt_after=halt_after,
            priority=priority,
        )

    def submit_cycle(
        self,
        question: str,
        top_k: int = 4,
        *,
        user_id: str | None = None,
        halt_after: int | None = None,
        priority: int = 0,
    ) -> ResearchReport:
        caller = user_id or self._default_user_id
        asked = question.strip()
        self._metrics.inc("rag_research_cycle_requests_total")
        started = time.perf_counter()
        trace_id = self._traces.start(user_id=caller, question=asked)
        session = self._sessions.create(
            user_id=caller,
            question=asked,
            top_k=top_k,
            trace_id=trace_id,
            priority=priority,
            deadline=_deadline_at(self._job_timeout_seconds),
        )
        self._traces.update(trace_id, session_id=session.id)
        self._begin_run(session)
        if needs_clarification(asked):
            report = _empty_report(asked, CLARIFICATION_TEXT, action="clarify", trace_id=trace_id)
            report.session_id = session.id
            report.status = "completed"
            self._traces.update(trace_id, session_id=session.id)
            self._state.complete(
                session.id,
                {
                    "action": "clarify",
                    "report": CLARIFICATION_TEXT,
                    "sufficient_evidence": False,
                    "steps": ["clarify"],
                    "gaps": [],
                    "contradictions": [],
                    "cost": 0,
                    "verdicts": [],
                    "evidence": [],
                },
                ResearchState(),
            )
            self._traces.step(trace_id, action="clarify")
            self._finish(("research_cycle", session.id), caller, report, started)
            return report
        if self._indexing_mode == "inline":
            return self._run_session(session.id, started=started, halt_after=halt_after)
        return self._with_effects(_pending_report(session), session.id)

    def get_session(self, session_id: str, *, user_id: str | None = None) -> ResearchReport:
        session = self._owned_session(session_id, user_id)
        return self._report_from_session(session)

    def list_sessions(self, *, user_id: str | None = None) -> list[ResearchSession]:
        caller = user_id or self._default_user_id
        return self._sessions.list_for_user(caller)

    def resume(self, session_id: str, *, user_id: str | None = None) -> ResearchReport:
        session = self._owned_session(session_id, user_id)
        if session.status == "completed":
            return self._report_from_session(session)
        self._state.requeue(session.id)
        if self._indexing_mode == "inline":
            return self._run_session(session.id)
        stored = self._sessions.get(session.id)
        pending = _pending_report(stored if stored is not None else session)
        return self._with_effects(pending, session.id)

    def job_counts(self) -> dict[str, int]:
        return self._state.job_counts()

    def cancel(self, session_id: str, *, user_id: str | None = None) -> ResearchReport:
        session = self._owned_session(session_id, user_id)
        if session.status == "completed":
            return self._report_from_session(session)
        stored = self._state.cancel(session.id)
        return self._report_from_session(stored if stored is not None else session)

    def process_next_job(self) -> bool:
        """Run one queued research cycle. Returns False when the queue is empty."""
        return self._jobs.process_next()

    def process_available(self) -> int:
        """Claim and run up to the worker concurrency limit."""
        return self._jobs.process_available()

    def _owned_session(self, session_id: str, user_id: str | None) -> ResearchSession:
        caller = user_id or self._default_user_id
        session = self._sessions.get(session_id)
        if session is None or session.user_id != caller:
            raise KnowledgeError("Research session not found.", status_code=404)
        return session

    def _run_session(
        self,
        session_id: str,
        *,
        started: float | None = None,
        claimed: bool = False,
        halt_after: int | None = None,
    ) -> ResearchReport:
        session = self._sessions.get(session_id)
        if session is None:
            raise KnowledgeError("Research session not found.", status_code=404)
        clock = started if started is not None else time.perf_counter()
        self._traces.start(
            trace_id=session.trace_id,
            user_id=session.user_id,
            question=session.question,
            session_id=session.id,
        )
        with self._observer.on_trace(session.trace_id):
            return self._run_session_traced(
                session,
                clock,
                claimed=claimed,
                halt_after=halt_after,
            )

    def _run_session_traced(
        self,
        session,
        clock: float,
        *,
        claimed: bool,
        halt_after: int | None,
    ) -> ResearchReport:
        if not claimed:
            self._state.acquire(session.id)
        self._begin_run(session)
        state = self._state.restore(session.id)
        try:
            result = self._worker.cycle(
                session.question,
                user_id=session.user_id,
                top_k=session.top_k,
                state=state,
                on_progress=lambda progress, request=None, outcome=None: (
                    self._state.checkpoint(session.id, progress, request, outcome),
                    self._state.heartbeat(session.id),
                    self._record_progress(session.id, request, outcome),
                ),
                halt_after=halt_after,
                should_cancel=lambda: self._sessions.cancel_requested(session.id),
                deadline=session.deadline,
                strategy=self._strategy,
            )
        except CycleCancelled as cancelled:
            self._state.finish_cancelled(session.id, _result_payload(cancelled.result), state)
            report = _cycle_report(
                session.question,
                cancelled.result,
                trace_id=session.trace_id,
                budget=self._budget_limit,
            )
            report.session_id = session.id
            report.status = "cancelled"
            report.decision = "stop"
            self._close_run(session, report, status="cancelled")
            self._observer.record_request(
                "cycle",
                seconds=time.perf_counter() - clock,
                cost=max(0, self._budget_limit - report.remaining_budget),
                status="cancelled",
                reliability=report.reliability,
                quality_confident=report.quality_confident,
                quality_failures=report.quality_failures,
            )
            return self._with_effects(report, session.id)
        except CycleDeadline as expired:
            self._state.fail(session.id, "Deadline exceeded.", state)
            report = _cycle_report(
                session.question,
                expired.result,
                trace_id=session.trace_id,
                budget=self._budget_limit,
            )
            report.session_id = session.id
            report.status = "failed"
            report.decision = "stop"
            self._close_run(session, report, status="failed", error="Deadline exceeded.")
            self._observer.record_request(
                "cycle",
                seconds=time.perf_counter() - clock,
                cost=max(0, self._budget_limit - report.remaining_budget),
                status="failed",
                reliability=report.reliability,
                quality_confident=report.quality_confident,
                quality_failures=report.quality_failures,
            )
            return self._with_effects(report, session.id)
        except CycleHalted as halted:
            self._state.halt(session.id, _result_payload(halted.result), state)
            report = _cycle_report(
                session.question,
                halted.result,
                trace_id=session.trace_id,
                budget=self._budget_limit,
            )
            report.session_id = session.id
            report.status = "halted"
            report.decision = "stop"
            self._close_run(session, report, status="halted")
            self._observer.record_request(
                "cycle",
                seconds=time.perf_counter() - clock,
                cost=max(0, self._budget_limit - report.remaining_budget),
                status="halted",
                reliability=report.reliability,
                quality_confident=report.quality_confident,
                quality_failures=report.quality_failures,
            )
            return self._with_effects(report, session.id)
        except Exception as exc:
            self._observer.record_action(
                "cycle",
                ok=False,
                seconds=time.perf_counter() - clock,
                error=str(exc) or type(exc).__name__,
            )
            self._state.fail(session.id, str(exc) or type(exc).__name__, state)
            self._close_run(session, None, status="failed", error=str(exc) or type(exc).__name__)
            raise
        report = _cycle_report(
            session.question,
            result,
            trace_id=session.trace_id,
            budget=self._budget_limit,
        )
        report.session_id = session.id
        report.status = "completed"
        self._state.complete(session.id, _result_payload(result), state)
        self._traces.step(session.trace_id, action=report.action)
        self._close_run(session, report, status="completed")
        self._finish(("research_cycle", session.id), session.user_id, report, clock)
        return self._with_effects(report, session.id)

    def _begin_run(self, session) -> None:
        if self._engineering is None:
            return
        try:
            self._engineering.start_run(
                run_id=session.id,
                agent_version=__version__,
                session_id=session.id,
                trace_id=session.trace_id,
                question=session.question,
            )
        except Exception:
            return

    def _record_progress(self, session_id: str, request, outcome) -> None:
        if self._engineering is None or request is None or outcome is None:
            return
        try:
            error = getattr(outcome, "error", "") or ""
            if "permitted" in error.casefold() or "denied" in error.casefold():
                self._engineering.record_policy(session_id, error, action=request.action)
            self._engineering.record_tool(
                session_id,
                request.action,
                ok=bool(outcome.ok),
                error=error,
                query=getattr(request, "query", "") or getattr(request, "claim", ""),
                cost=int(getattr(outcome, "cost", 0) or 0),
            )
            for verdict in getattr(outcome, "verdicts", ()) or ():
                self._engineering.record_verification(
                    session_id,
                    verdict.claim,
                    ok=verdict.verdict == "supported",
                    evidence_ids=tuple(verdict.evidence_ids),
                )
        except Exception:
            return

    def _close_run(self, session, report, *, status: str, error: str = "") -> None:
        if self._engineering is None:
            return
        try:
            cost = 0
            duration = 0.0
            tokens = 0
            if report is not None:
                cost = float(max(0, self._budget_limit - report.remaining_budget))
                tokens = int(cost * 250)
            finished = self._engineering.finish_run(
                session.id,
                status=status,
                duration_ms=duration,
                token_usage=tokens,
                estimated_cost=cost,
                error=error,
            )
            self._engineering.evaluate(
                session.id,
                workflow_state=status,
                reliability=getattr(report, "reliability", None) if report is not None else None,
                quality_confident=getattr(report, "quality_confident", None) if report is not None else None,
                sufficient=getattr(report, "sufficient_evidence", None) if report is not None else None,
            )
            if report is not None:
                self._strategy.learn(report)
            _ = finished
        except Exception:
            return

    def _with_effects(self, report: ResearchReport, session_id: str) -> ResearchReport:
        report.side_effects = tuple(self._state.side_effects(session_id))
        return report

    def _report_from_session(self, session: ResearchSession) -> ResearchReport:
        if session.status in {"completed", "halted", "cancelled"} and session.result:
            result = _cycle_result(session)
            report = _cycle_report(
                session.question,
                result,
                trace_id=session.trace_id,
                budget=self._budget_limit,
            )
            report.session_id = session.id
            report.status = session.status
            if session.action == "clarify":
                report.action = "clarify"
                report.report = session.report
            if session.status in {"halted", "cancelled"}:
                report.decision = "stop"
            return self._with_effects(report, session.id)
        return self._with_effects(_pending_report(session), session.id)

    def _finish(self, key, caller: str, report: ResearchReport, started: float) -> None:
        self._cache.put(key, report)
        path = "cycle" if isinstance(key, tuple) and key[0] == "research_cycle" else "research"
        elapsed = time.perf_counter() - started
        self._observer.record_request(
            path,
            seconds=elapsed,
            cost=max(0, self._budget_limit - report.remaining_budget),
            status=report.status or report.action,
            reliability=report.reliability,
            quality_confident=report.quality_confident,
            quality_failures=report.quality_failures,
        )
        log_event(
            "rag.research",
            "research_completed",
            user_id=caller,
            action=report.action,
            sufficient_evidence=report.sufficient_evidence,
            conflict=report.conflict,
            citation_count=len(report.citations),
            gap_count=len(report.gaps),
            duration_ms=round(elapsed * 1000, 2),
            trace_id=report.trace_id,
        )


def _quality_feedback(
    board: _EvidenceBoard,
    evidence: list[str],
    contradictions: list[str],
    gaps: list[str],
    steps: list[ResearchStep],
) -> tuple[str, str, tuple[str, ...]]:
    verified = [step for step in steps if step.status in {"supported", "contradicted"}]
    snapshot = ResearchReport(
        question="",
        report=_render(evidence, contradictions, gaps, (), refused=not verified),
        sufficient_evidence=bool(verified) and not gaps,
        conflict=bool(contradictions),
        tools=(),
        subquestions=(),
        contradictions=tuple(contradictions),
        gaps=tuple(gaps),
        catalog=(),
        citations=board.citations(),
        passages=[],
        action="research",
        status="running",
    )
    score = evaluate_research(snapshot)
    return score.recommendation, score.recommendation_reason, score.failures


def _attach_quality(report: ResearchReport) -> ResearchReport:
    score = apply_quality_gate(report)
    report.reliability = score.reliability
    return report


def _attach_coordination(report: ResearchReport, *, budget: int) -> ResearchReport:
    report.coordination = MultiAgentCoordinator().coordinate(
        report.question,
        findings=findings_from_report(report),
        conflicts=report.conflicts,
        budget=budget,
    )
    return report


def _loop_state(
    steps: list[ResearchStep],
    gaps: list[str],
    conflicts: list[Contradiction],
    store: EvidenceStore,
    budget: SearchBudget,
    *,
    gap_retries: int,
    conflict_probes: int,
    max_adapts: int,
    quality_action: str = "",
    quality_reason: str = "",
    quality_failures: tuple[str, ...] = (),
) -> LoopState:
    verified = [step for step in steps if step.status in {"supported", "contradicted"}]
    files = {record.source.filename for record in store.records}
    unresolved = tuple(item.summary for item in conflicts if len(files) < 2)
    return LoopState(
        complete=bool(verified) and not gaps,
        supported=len(verified),
        gaps=tuple(gaps),
        unresolved_conflicts=unresolved,
        remaining_budget=budget.remaining(),
        spent_cost=budget.used,
        gap_retries=gap_retries,
        conflict_probes=conflict_probes,
        max_adapts=max_adapts,
        quality_action=quality_action,
        quality_reason=quality_reason,
        quality_failures=quality_failures,
    )


def _replace_gap(steps: list[ResearchStep], gaps: list[str], step: ResearchStep) -> None:
    if not step.complete:
        _drop_gap(gaps, step.question)
        return
    for index, existing in enumerate(steps):
        alternative = reformulate(existing.question)
        if existing.status == "gap" and (
            existing.question.casefold() == step.question.casefold()
            or alternative == step.question
        ):
            steps[index] = step
            _drop_gap(gaps, existing.question)
            _drop_gap(gaps, step.question)
            return
    steps.append(step)
    _drop_gap(gaps, step.question)


def _drop_gap(gaps: list[str], question: str) -> None:
    key = question.casefold()
    keep = [gap for gap in gaps if gap.casefold() != key]
    gaps[:] = keep


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


def _pending_report(session: ResearchSession) -> ResearchReport:
    report = _empty_report(session.question, session.report, action=session.action, trace_id=session.trace_id)
    report.session_id = session.id
    report.status = session.status
    report.gaps = session.gaps
    report.contradictions = session.contradictions
    report.decisions = session.steps
    report.subquestions = tuple(
        ResearchStep(question=session.question, tool=name, status="ok", evidence="", complete=False)
        for name in session.steps
    )
    return report


def _result_payload(result: CycleResult) -> dict:
    return {
        "action": result.action,
        "report": result.report,
        "sufficient_evidence": result.sufficient_evidence,
        "steps": list(result.steps),
        "gaps": list(result.gaps),
        "contradictions": list(result.contradictions),
        "cost": result.cost,
        "verdicts": [
            {"claim": item.claim, "verdict": item.verdict, "evidence_ids": list(item.evidence_ids)}
            for item in result.verdicts
        ],
        "evidence": [
            {
                "evidence_id": item.evidence_id,
                "document_id": item.document_id,
                "source": item.source,
                "text": item.text,
            }
            for item in result.evidence
        ],
    }


def _cycle_result(session: ResearchSession) -> CycleResult:
    payload = session.result or {}
    state = session.state or ResearchState()
    verdicts = tuple(
        ClaimVerdict(
            claim=str(item.get("claim", "")),
            verdict=str(item.get("verdict", "")),
            evidence_ids=tuple(str(value) for value in item.get("evidence_ids", ())),
        )
        for item in payload.get("verdicts", [])
    ) or tuple(state.verdicts)
    evidence = tuple(
        EvidenceRef(
            evidence_id=str(item.get("evidence_id", "")),
            document_id=str(item.get("document_id", "")),
            source=str(item.get("source", "")),
            text=str(item.get("text", "")),
        )
        for item in payload.get("evidence", [])
    ) or tuple(state.evidence)
    return CycleResult(
        question=session.question,
        report=str(payload.get("report", session.report)),
        sufficient_evidence=bool(payload.get("sufficient_evidence", session.sufficient_evidence)),
        steps=tuple(payload.get("steps", session.steps)),
        gaps=tuple(payload.get("gaps", session.gaps)),
        verdicts=verdicts,
        evidence=evidence,
        contradictions=tuple(payload.get("contradictions", session.contradictions)),
        cost=int(payload.get("cost", session.cost) or 0),
        action=str(payload.get("action", session.action)),
    )


def _cycle_report(
    question: str,
    result: CycleResult,
    *,
    trace_id: str,
    budget: int,
) -> ResearchReport:
    citations: list[Citation] = []
    seen: set[str] = set()
    by_id = {item.evidence_id: item for item in result.evidence}
    for verdict in result.verdicts:
        if verdict.verdict != "supported":
            continue
        for evidence_id in verdict.evidence_ids:
            if evidence_id in seen:
                continue
            passage = by_id.get(evidence_id)
            if passage is None:
                continue
            seen.add(evidence_id)
            citations.append(
                Citation(
                    marker=f"[{len(citations) + 1}]",
                    chunk_id=passage.evidence_id,
                    document_id=passage.document_id,
                    filename=passage.source,
                    content_type="text/markdown",
                    heading=None,
                    page=None,
                    quote=_quote(passage.text),
                )
            )
    steps = tuple(
        ResearchStep(question=question, tool=name, status="ok", evidence="", complete=True)
        for name in result.steps
    )
    report = ResearchReport(
        question=question,
        report=result.report,
        sufficient_evidence=result.sufficient_evidence,
        conflict=bool(result.contradictions),
        tools=("knowledge",),
        subquestions=steps,
        contradictions=result.contradictions,
        gaps=result.gaps,
        catalog=(),
        citations=citations,
        passages=[],
        action=result.action,
        trace_id=trace_id,
        decision="report" if result.sufficient_evidence else "stop",
        decisions=result.steps,
        remaining_budget=max(0, budget - result.cost),
        graph=result.graph if getattr(result, "graph", None) is not None else None,
    )
    _attach_quality(report)
    return _attach_coordination(report, budget=budget)


def _deadline_at(seconds: float) -> str:
    if seconds <= 0:
        return ""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


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

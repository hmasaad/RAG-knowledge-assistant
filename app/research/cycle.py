"""Run question → retrieve → verify → fill gaps → retrieve → synthesize."""

from dataclasses import dataclass
from datetime import datetime, timezone

from app.research.adapters import cited_answer
from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.autonomous import AutonomousPlan, AutonomousPlanner
from app.research.strategy import StrategyMemory, classify_situation
from app.research.dag import ResearchDag
from app.research.dispatcher import ActionDispatcher
from app.research.pool import BoundedWorkerPool


class CycleHalted(Exception):
    """Stop after a committed checkpoint so a later process can resume."""

    def __init__(self, result: "CycleResult") -> None:
        super().__init__("Halted after a committed checkpoint.")
        self.result = result


class CycleCancelled(Exception):
    """Stop because the caller cancelled the session."""

    def __init__(self, result: "CycleResult") -> None:
        super().__init__("Cancelled.")
        self.result = result


class CycleDeadline(Exception):
    """Stop because the session or task deadline has passed."""

    def __init__(self, result: "CycleResult") -> None:
        super().__init__("Deadline exceeded.")
        self.result = result


@dataclass(frozen=True)
class CycleResult:
    question: str
    report: str
    sufficient_evidence: bool
    steps: tuple[str, ...]
    gaps: tuple[str, ...]
    verdicts: tuple[ClaimVerdict, ...]
    evidence: tuple[EvidenceRef, ...]
    contradictions: tuple[str, ...]
    cost: int
    action: str
    graph: object | None = None


def run_cycle(
    dispatcher: ActionDispatcher,
    question: str,
    *,
    state: ResearchState | None = None,
    user_id: str = "local",
    top_k: int = 4,
    on_progress=None,
    halt_after: int | None = None,
    should_cancel=None,
    deadline: str = "",
    pool: BoundedWorkerPool | None = None,
    strategy: StrategyMemory | None = None,
) -> CycleResult:
    """Retrieve, verify, fill gaps, retrieve again, then cite an answer.

    Actions come from a DAG. Already-done nodes and executed fingerprints are
    skipped so a persisted session can resume after a crash or a restart.
    """
    session = state if state is not None else ResearchState()
    asked = question.strip()
    if deadline and _now() >= deadline:
        raise CycleDeadline(_cycle_snapshot(asked, session))
    workers = pool if pool is not None else BoundedWorkerPool(1)
    planner = AutonomousPlanner(dispatcher.guardrails, strategy=strategy)
    plan = AutonomousPlan.from_payload(session.dag)
    if plan is None:
        plan = planner.generate(
            asked,
            budget=dispatcher.budget,
            user_id=user_id,
            top_k=top_k,
            deadline=deadline,
        )
    plan.dag.recover()
    plan.dag.expire_due(_now())
    dag = plan.dag
    session.dag = plan.to_payload()
    ran = 0

    def step(request: ActionRequest, task_id: str = "") -> None:
        nonlocal ran
        if should_cancel is not None and should_cancel():
            raise CycleCancelled(_cycle_snapshot(asked, session))
        if deadline and _now() >= deadline:
            if task_id:
                dag.mark_expired(task_id)
                session.dag = dag.to_payload()
            raise CycleDeadline(_cycle_snapshot(asked, session))
        if task_id:
            dag.mark_running(task_id)
            session.dag = dag.to_payload()
            if on_progress is not None:
                try:
                    on_progress(session)
                except TypeError:
                    pass
        if _step(dispatcher, session, request, on_progress=on_progress) is None:
            if task_id:
                dag.mark_done(task_id)
                session.dag = dag.to_payload()
            return
        if task_id:
            dag.mark_done(task_id)
            session.dag = dag.to_payload()
        ran += 1
        if should_cancel is not None and should_cancel():
            raise CycleCancelled(_cycle_snapshot(asked, session))
        if deadline and _now() >= deadline:
            raise CycleDeadline(_cycle_snapshot(asked, session))
        if halt_after is not None and ran >= halt_after:
            raise CycleHalted(_cycle_snapshot(asked, session))

    _drain(dag, session, asked, step, workers)
    if "synthesize" not in dag.tasks:
        plan.dag = dag
        plan = planner.revise(
            plan,
            quality_action=_revise_action(session, strategy),
            quality_reason="",
            gaps=tuple(session.gaps),
            conflicts=tuple(session.contradictions),
            executed=set(session.executed),
            remaining_budget=max(0, dispatcher.budget - session.cost),
            spent_cost=session.cost,
            user_id=user_id,
            top_k=top_k,
            deadline=deadline,
        )
        dag = plan.dag
        session.dag = plan.to_payload()
        _drain(dag, session, asked, step, workers)
    return _cycle_snapshot(asked, session)


def _drain(
    dag: ResearchDag,
    session: ResearchState,
    asked: str,
    step,
    pool: BoundedWorkerPool,
) -> None:
    while True:
        task = dag.next_ready()
        if task is None:
            return
        request = task.request()
        if task.action == SYNTHESIZE and not request.query.strip():
            request = ActionRequest(
                SYNTHESIZE,
                query=cited_answer(asked, session),
                user_id=task.user_id,
                top_k=task.top_k,
            )
        pool.map((task,), lambda item, req=request: step(req, item.id))


def _cycle_snapshot(asked: str, session: ResearchState) -> CycleResult:
    supported = any(item.verdict == "supported" for item in session.verdicts)
    action = "research" if supported or session.report else "refuse"
    return CycleResult(
        question=asked,
        report=session.report or cited_answer(asked, session),
        sufficient_evidence=supported and not session.gaps,
        steps=tuple(session.steps),
        gaps=tuple(session.gaps),
        verdicts=tuple(session.verdicts),
        evidence=tuple(session.evidence),
        contradictions=tuple(session.contradictions),
        cost=session.cost,
        action=action,
    )


def _step(
    dispatcher: ActionDispatcher,
    state: ResearchState,
    request: ActionRequest,
    *,
    on_progress=None,
) -> ActionOutcome | None:
    if state.fingerprint(request) in state.executed:
        return None
    outcome = dispatcher.dispatch(request, state)
    state.steps.append(request.action)
    if on_progress is not None:
        try:
            on_progress(state, request, outcome)
        except TypeError:
            on_progress(state)
    return outcome


def _revise_action(session: ResearchState, strategy: StrategyMemory | None = None) -> str:
    if session.contradictions:
        fallback = RESOLVE_CONTRADICTIONS
    elif session.gaps:
        fallback = FILL_GAPS
    else:
        fallback = "stop"
    if strategy is None:
        return fallback
    return strategy.recommend(
        classify_situation(gaps=session.gaps, conflicts=session.contradictions),
        fallback,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

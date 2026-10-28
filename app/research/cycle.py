"""Run question → retrieve → verify → fill gaps → retrieve → synthesize."""

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from app.research.adapters import cited_answer
from app.research.contracts import (
    FILL_GAPS,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.autonomous import AutonomousPlan, AutonomousPlanner
from app.research.coordination import MultiAgentCoordinator
from app.research.strategy import StrategyMemory
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
    coordination: dict | None = None


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
    team = MultiAgentCoordinator(denied_actions=dispatcher.guardrails.policy.denied_actions)
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
    plan.dag.propagate_all()
    dag = plan.dag
    session.dag = plan.to_payload()
    if on_progress is not None:
        try:
            on_progress(session)
        except TypeError:
            on_progress(session)
    ran = 0
    lock = threading.Lock()

    def step(request: ActionRequest, task_id: str = "") -> None:
        nonlocal ran
        dag = plan.dag
        with lock:
            if should_cancel is not None and should_cancel():
                raise CycleCancelled(_cycle_snapshot(asked, session))
            if deadline and _now() >= deadline:
                if task_id:
                    dag.mark_expired(task_id)
                    session.dag = plan.to_payload()
                raise CycleDeadline(_cycle_snapshot(asked, session))
            if task_id:
                dag.mark_running(task_id)
                session.dag = plan.to_payload()
                if on_progress is not None:
                    try:
                        on_progress(session)
                    except TypeError:
                        pass
            if session.fingerprint(request) in session.executed:
                if task_id:
                    dag.mark_done(task_id)
                    session.dag = plan.to_payload()
                return
            blocked = dispatcher.validate(request, session)
        if blocked:
            outcome = ActionOutcome(action=request.action, ok=False, error=blocked, cost=0)
        else:
            outcome = dispatcher.invoke(request, session)
        with lock:
            if session.fingerprint(request) in session.executed:
                if task_id:
                    dag.mark_done(task_id)
                    session.dag = plan.to_payload()
                return
            session.apply(request, outcome)
            session.steps.append(request.action)
            dispatcher.observer.record_action(
                request.action,
                ok=outcome.ok,
                cost=outcome.cost,
                error=outcome.error,
            )
            if not blocked and request.action == FILL_GAPS:
                dispatcher.observer.record_retry("gap")
            if on_progress is not None:
                try:
                    on_progress(session, request, outcome)
                except TypeError:
                    on_progress(session)
            task = dag.tasks.get(task_id)
            team.publish(
                session,
                outcome,
                agent_id=(task.agent_id if task is not None else task_id) or task_id,
                role=(task.role if task is not None else "") or "",
            )
            if task_id:
                if outcome.ok:
                    dag.mark_done(task_id)
                elif dag.can_retry(task_id, outcome.error):
                    session.executed.discard(session.fingerprint(request))
                    dag.reschedule(task_id)
                else:
                    dag.mark_failed(task_id, outcome.error)
                session.dag = plan.to_payload()
            ran += 1

    def drain() -> None:
        nonlocal plan
        while True:
            wave = plan.dag.ready_wave()
            if halt_after is not None:
                remaining = halt_after - ran
                if remaining <= 0:
                    raise CycleHalted(_cycle_snapshot(asked, session))
                wave = wave[:remaining]
            if not wave:
                return
            prepared = [_request_for(task, session, asked, team, dispatcher.budget) for task in wave]
            workers.map(
                list(zip(wave, prepared)),
                lambda item: step(item[1], item[0].id),
            )
            plan = planner.replan(
                plan,
                state=session,
                remaining_budget=max(0, dispatcher.budget - session.cost),
                spent_cost=session.cost,
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
            )
            session.dag = plan.to_payload()
            if on_progress is not None:
                try:
                    on_progress(session)
                except TypeError:
                    on_progress(session)
            if should_cancel is not None and should_cancel():
                raise CycleCancelled(_cycle_snapshot(asked, session))
            if deadline and _now() >= deadline:
                raise CycleDeadline(_cycle_snapshot(asked, session))
            if halt_after is not None and ran >= halt_after:
                raise CycleHalted(_cycle_snapshot(asked, session))

    drain()
    if plan.dag.next_ready() is None and "synthesize" not in plan.dag.tasks:
        plan = planner.replan(
            plan,
            state=session,
            remaining_budget=max(0, dispatcher.budget - session.cost),
            spent_cost=session.cost,
            user_id=user_id,
            top_k=top_k,
            deadline=deadline,
        )
        session.dag = plan.to_payload()
        drain()
    team.settle(session, asked, budget=dispatcher.budget)
    return _cycle_snapshot(asked, session)


def _request_for(task, session: ResearchState, asked: str, team: MultiAgentCoordinator, budget: int) -> ActionRequest:
    request = task.request()
    if task.action != SYNTHESIZE:
        return request
    team.settle(session, asked, budget=budget)
    if request.query.strip():
        return request
    return ActionRequest(
        SYNTHESIZE,
        query=cited_answer(asked, session),
        user_id=task.user_id,
        top_k=task.top_k,
    )


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
        coordination=dict(session.coordination) if session.coordination else None,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

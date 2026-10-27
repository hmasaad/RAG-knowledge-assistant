"""Run question → retrieve → verify → fill gaps → retrieve → synthesize."""

from dataclasses import dataclass
from datetime import datetime, timezone

from app.research.adapters import cited_answer
from app.research.contracts import (
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
    dag = plan.dag
    session.dag = plan.to_payload()
    ran = 0

    def step(request: ActionRequest, task_id: str = "") -> None:
        nonlocal ran, plan
        dag = plan.dag
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
        outcome = _step(dispatcher, session, request, on_progress=on_progress)
        if outcome is None:
            if task_id:
                dag.mark_done(task_id)
                session.dag = plan.to_payload()
            return
        task = dag.tasks.get(task_id)
        team.publish(
            session,
            outcome,
            agent_id=(task.agent_id if task is not None else task_id) or task_id,
            role=(task.role if task is not None else "") or "",
        )
        if task_id:
            dag.mark_done(task_id)
            session.dag = plan.to_payload()
        ran += 1
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
        if should_cancel is not None and should_cancel():
            raise CycleCancelled(_cycle_snapshot(asked, session))
        if deadline and _now() >= deadline:
            raise CycleDeadline(_cycle_snapshot(asked, session))
        if halt_after is not None and ran >= halt_after:
            raise CycleHalted(_cycle_snapshot(asked, session))

    _drain(plan, session, asked, step, workers, team=team, budget=dispatcher.budget)
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
        _drain(plan, session, asked, step, workers, team=team, budget=dispatcher.budget)
    team.settle(session, asked, budget=dispatcher.budget)
    return _cycle_snapshot(asked, session)


def _drain(
    plan: AutonomousPlan,
    session: ResearchState,
    asked: str,
    step,
    pool: BoundedWorkerPool,
    *,
    team: MultiAgentCoordinator,
    budget: int,
) -> None:
    while True:
        task = plan.dag.next_ready()
        if task is None:
            return
        request = task.request()
        if task.action == SYNTHESIZE:
            team.settle(session, asked, budget=budget)
            if not request.query.strip():
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
        coordination=dict(session.coordination) if session.coordination else None,
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

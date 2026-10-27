"""Generate, revise, and optimize a research DAG under quality and safety limits."""

from dataclasses import dataclass, field, replace

from app.generation.agent import reformulate
from app.research.contracts import (
    FILL_GAPS,
    RESOLVE_CONTRADICTIONS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionRequest,
    ResearchState,
)
from app.research.dag import DagTask, ResearchDag, build_search_dag, extend_gap_dag
from app.research.guardrails import ResearchGuardrails
from app.research.planner import build_plan, select_tools
from app.research.strategy import StrategyMemory, classify_situation

_COSTLY = {SEARCH, FILL_GAPS}


@dataclass
class AutonomousPlan:
    """A versioned research DAG the agent can regenerate and tighten."""

    question: str
    dag: ResearchDag
    version: int = 1
    revisions: list[str] = field(default_factory=list)
    reason: str = "Initial plan."
    tools: tuple[str, ...] = ()
    budget: int = 16

    def actions(self) -> tuple[str, ...]:
        return tuple(self.dag.tasks[task_id].action for task_id in self.dag.order)

    def pending(self) -> list[DagTask]:
        return [self.dag.tasks[task_id] for task_id in self.dag.order if self.dag.tasks[task_id].status == "pending"]

    def to_payload(self) -> dict:
        payload = self.dag.to_payload()
        payload.update(
            {
                "question": self.question,
                "version": self.version,
                "revisions": list(self.revisions),
                "reason": self.reason,
                "tools": list(self.tools),
                "budget": self.budget,
            }
        )
        return payload

    @classmethod
    def from_payload(cls, payload: dict | None) -> "AutonomousPlan | None":
        data = payload or {}
        if not data.get("tasks"):
            return None
        return cls(
            question=str(data.get("question", "")),
            dag=ResearchDag.from_payload(data),
            version=int(data.get("version", 1) or 1),
            revisions=list(data.get("revisions", []) or []),
            reason=str(data.get("reason", "")),
            tools=tuple(str(item) for item in data.get("tools", ()) or ()),
            budget=int(data.get("budget", 16) or 16),
        )


class AutonomousPlanner:
    """Write a research plan, then change it only when quality and safety allow."""

    def __init__(
        self,
        guardrails: ResearchGuardrails | None = None,
        strategy: StrategyMemory | None = None,
    ) -> None:
        self.guardrails = guardrails or ResearchGuardrails()
        self.strategy = strategy

    def generate(
        self,
        question: str,
        *,
        budget: int = 16,
        user_id: str = "local",
        top_k: int = 4,
        deadline: str = "",
    ) -> AutonomousPlan:
        asked = question.strip()
        outline = build_plan(asked, budget=budget)
        dag = build_search_dag(asked, user_id=user_id, top_k=top_k, deadline=deadline)
        plan = AutonomousPlan(
            question=asked,
            dag=dag,
            version=1,
            revisions=[],
            reason="Generated from the question clauses and approved tools.",
            tools=outline.tools or select_tools(asked),
            budget=budget,
        )
        return self.optimize(
            plan,
            executed=set(),
            remaining_budget=budget,
            spent_cost=0,
        )

    def revise(
        self,
        plan: AutonomousPlan,
        *,
        quality_action: str,
        quality_reason: str = "",
        gaps: tuple[str, ...] = (),
        conflicts: tuple[str, ...] = (),
        executed: set[tuple[str, str, str]] | None = None,
        remaining_budget: int = 16,
        spent_cost: int = 0,
        user_id: str = "local",
        top_k: int = 4,
        deadline: str = "",
    ) -> AutonomousPlan:
        """Add only the next quality-backed actions that guardrails still allow."""
        done = executed if executed is not None else set()
        if self.guardrails.should_stop(remaining_budget, spent_cost):
            self._ensure_synthesize(plan, user_id=user_id, top_k=top_k, deadline=deadline)
            plan.version += 1
            plan.revisions.append("Stopped: budget or cost ceiling.")
            plan.reason = "Guardrails stopped further planning."
            return self.optimize(plan, executed=done, remaining_budget=remaining_budget, spent_cost=spent_cost)

        action = (quality_action or "stop").strip()
        denied = self.guardrails.policy.denied_actions
        if self.strategy is not None:
            action = self.strategy.recommend(
                classify_situation(gaps=gaps, conflicts=conflicts),
                action,
                denied=denied,
            )
        if action in denied:
            action = "stop"
        if action == FILL_GAPS and gaps:
            alternatives = tuple(reformulate(gap) or gap for gap in gaps)
            extend_gap_dag(
                plan.dag,
                gaps,
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                alternatives=alternatives,
            )
            plan.revisions.append(quality_reason or "Fill remaining evidence gaps.")
        elif action == RESOLVE_CONTRADICTIONS and conflicts:
            self._add_unique(
                plan,
                DagTask(
                    id=f"resolve:{plan.version}",
                    action=RESOLVE_CONTRADICTIONS,
                    depends_on=self._tail(plan.dag),
                    query=conflicts[0],
                    user_id=user_id,
                    top_k=top_k,
                    deadline=deadline,
                    agent_id="referee",
                    role="referee",
                ),
            )
            plan.revisions.append(quality_reason or "Resolve a documented contradiction.")
        elif action == REVERIFY:
            claim = gaps[0] if gaps else plan.question
            self._add_unique(
                plan,
                DagTask(
                    id=f"reverify:plan:{plan.version}",
                    action=REVERIFY,
                    depends_on=self._tail(plan.dag),
                    claim=claim,
                    query=reformulate(claim) or "",
                    user_id=user_id,
                    top_k=top_k,
                    deadline=deadline,
                    agent_id=f"verifier:plan:{plan.version}",
                    role="verifier",
                ),
            )
            plan.revisions.append(quality_reason or "Reverify an unsupported claim.")
        else:
            plan.revisions.append(quality_reason or "No further research is required.")

        self._ensure_synthesize(plan, user_id=user_id, top_k=top_k, deadline=deadline)
        plan.version += 1
        plan.reason = plan.revisions[-1] if plan.revisions else "Revised plan."
        return self.optimize(
            plan,
            executed=done,
            remaining_budget=remaining_budget,
            spent_cost=spent_cost,
        )

    def replan(
        self,
        plan: AutonomousPlan,
        *,
        state: ResearchState,
        remaining_budget: int,
        spent_cost: int = 0,
        user_id: str = "local",
        top_k: int = 4,
        deadline: str = "",
    ) -> AutonomousPlan:
        """Grow or shrink the DAG from evidence gathered so far."""
        pruned = self._prune_stale(plan, state)
        action = _revise_action(state, self.strategy)
        if self.guardrails.should_stop(remaining_budget, spent_cost):
            action = "stop"
        if not pruned and not self._needs_extension(plan, action, state) and "synthesize" in plan.dag.tasks:
            return self.optimize(
                plan,
                executed=set(state.executed),
                remaining_budget=remaining_budget,
                spent_cost=spent_cost,
            )
        return self.revise(
            plan,
            quality_action=action,
            quality_reason=_replan_reason(action, state),
            gaps=tuple(gap for gap in state.gaps if not _gap_covered(gap, _covered_fills(plan))),
            conflicts=tuple(state.contradictions),
            executed=set(state.executed),
            remaining_budget=remaining_budget,
            spent_cost=spent_cost,
            user_id=user_id,
            top_k=top_k,
            deadline=deadline,
        )

    def optimize(
        self,
        plan: AutonomousPlan,
        *,
        executed: set[tuple[str, str, str]],
        remaining_budget: int,
        spent_cost: int = 0,
    ) -> AutonomousPlan:
        """Drop denied, duplicate, or over-budget work. Keep finished nodes."""
        kept: list[DagTask] = []
        seen: set[tuple[str, str, str]] = set(executed)
        costly = 0
        for task_id in plan.dag.order:
            task = plan.dag.tasks[task_id]
            if task.status != "pending":
                kept.append(task)
                continue
            if task.action == SYNTHESIZE:
                kept.append(task)
                continue
            dummy = ResearchState()
            dummy.executed = set(seen)
            dummy.cost = spent_cost
            dummy.iterations = len(seen)
            denied = self.guardrails.check_action(
                task.request(),
                dummy,
                adapters=set(self.guardrails.policy.allowed_actions) - self.guardrails.policy.denied_actions,
            )
            fingerprint = (task.action, task.query.strip().casefold(), task.claim.strip().casefold())
            if denied or fingerprint in seen:
                continue
            if task.action in _COSTLY:
                if costly >= max(0, remaining_budget):
                    continue
                costly += 1
            seen.add(fingerprint)
            kept.append(task)
        kept_ids = {task.id for task in kept}
        last = ""
        rebuilt: list[DagTask] = []
        pending_synth: DagTask | None = None
        for task in kept:
            if task.action == SYNTHESIZE and task.status == "pending":
                pending_synth = task
                continue
            if task.status == "pending":
                if any(parent not in kept_ids for parent in task.depends_on):
                    continue
                last = task.id
                rebuilt.append(task)
                continue
            last = task.id
            rebuilt.append(task)
        if pending_synth is not None:
            rebuilt.append(replace(pending_synth, depends_on=(last,) if last else ()))
        plan.dag = ResearchDag()
        for task in rebuilt:
            plan.dag.add(task)
        return plan

    def _ensure_synthesize(
        self,
        plan: AutonomousPlan,
        *,
        user_id: str,
        top_k: int,
        deadline: str,
    ) -> None:
        if "synthesize" in plan.dag.tasks:
            return
        plan.dag.add(
            DagTask(
                "synthesize",
                SYNTHESIZE,
                depends_on=self._tail(plan.dag),
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                agent_id="synthesizer",
                role="synthesizer",
            )
        )

    def _add_unique(self, plan: AutonomousPlan, task: DagTask) -> None:
        if task.id in plan.dag.tasks:
            return
        plan.dag.add(task)

    def _tail(self, dag: ResearchDag) -> tuple[str, ...]:
        if not dag.order:
            return ()
        return (dag.order[-1],)

    def _prune_stale(self, plan: AutonomousPlan, state: ResearchState) -> bool:
        gaps = {gap.casefold() for gap in state.gaps}
        supported = {
            verdict.claim.casefold()
            for verdict in state.verdicts
            if verdict.verdict == "supported"
        }
        kept: list[DagTask] = []
        removed = False
        for task_id in plan.dag.order:
            task = plan.dag.tasks[task_id]
            if task.status != "pending" or _initial_wave(task):
                kept.append(task)
                continue
            if task.action == FILL_GAPS and (task.claim or task.query).casefold() not in gaps:
                removed = True
                continue
            if task.action == RESOLVE_CONTRADICTIONS and not state.contradictions:
                removed = True
                continue
            if task.id.startswith("reverify:plan:") and task.claim.casefold() in supported:
                removed = True
                continue
            kept.append(task)
        if not removed:
            return False
        plan.dag = ResearchDag()
        for task in kept:
            plan.dag.add(task)
        return True

    def _needs_extension(self, plan: AutonomousPlan, action: str, state: ResearchState) -> bool:
        if action == FILL_GAPS:
            covered = _covered_fills(plan)
            return any(not _gap_covered(gap, covered) for gap in state.gaps)
        if action == RESOLVE_CONTRADICTIONS:
            return bool(state.contradictions) and not any(
                task.action == RESOLVE_CONTRADICTIONS for task in plan.dag.tasks.values()
            )
        if action == REVERIFY:
            claim = (state.gaps[0] if state.gaps else plan.question).casefold()
            return not any(
                task.action == REVERIFY and task.claim.casefold() == claim
                for task in plan.dag.tasks.values()
            )
        return False


def _revise_action(state: ResearchState, strategy: StrategyMemory | None) -> str:
    if state.contradictions:
        fallback = RESOLVE_CONTRADICTIONS
    elif state.gaps:
        fallback = FILL_GAPS
    else:
        fallback = "stop"
    if strategy is None:
        return fallback
    return strategy.recommend(classify_situation(gaps=state.gaps, conflicts=state.contradictions), fallback)


def _replan_reason(action: str, state: ResearchState) -> str:
    if action == FILL_GAPS:
        return "Evidence is still missing; the DAG adds a gap fill."
    if action == RESOLVE_CONTRADICTIONS:
        return "New contradictions require a resolve step."
    if action == REVERIFY:
        return "A claim still needs verification."
    return "Evidence is sufficient to synthesize."


def _covered_fills(plan: AutonomousPlan) -> set[str]:
    covered: set[str] = set()
    for task in plan.dag.tasks.values():
        if task.action != FILL_GAPS:
            continue
        for value in (task.claim, task.query):
            key = value.strip().casefold()
            if key:
                covered.add(key)
            alternative = reformulate(value)
            if alternative:
                covered.add(alternative.casefold())
    return covered


def _gap_covered(gap: str, covered: set[str]) -> bool:
    key = gap.strip().casefold()
    if key in covered:
        return True
    alternative = reformulate(gap)
    return bool(alternative and alternative.casefold() in covered)


def _initial_wave(task: DagTask) -> bool:
    if task.action == SEARCH and task.id.startswith("search:"):
        return True
    return task.action == REVERIFY and task.id.startswith("reverify:") and ":gap:" not in task.id and ":plan:" not in task.id

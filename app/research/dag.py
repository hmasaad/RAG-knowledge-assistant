"""A research plan as a DAG of actions with durable task status."""

from dataclasses import dataclass, field, replace

from app.research.contracts import (
    FILL_GAPS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionRequest,
)
from app.research.planner import subquestions

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
EXPIRED = "expired"
TERMINAL = frozenset({DONE, FAILED, EXPIRED})
_RETRYABLE = ("timed out", "timeout")


@dataclass
class DagTask:
    """One scheduled action and the tasks that must finish first."""

    id: str
    action: str
    depends_on: tuple[str, ...] = ()
    query: str = ""
    claim: str = ""
    user_id: str = "local"
    top_k: int = 4
    status: str = PENDING
    deadline: str = ""
    error: str = ""
    agent_id: str = ""
    role: str = ""
    retries: int = 0

    def request(self) -> ActionRequest:
        return ActionRequest(
            self.action,
            query=self.query,
            claim=self.claim,
            user_id=self.user_id,
            top_k=self.top_k,
        )

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "depends_on": list(self.depends_on),
            "query": self.query,
            "claim": self.claim,
            "user_id": self.user_id,
            "top_k": self.top_k,
            "status": self.status,
            "deadline": self.deadline,
            "error": self.error,
            "agent_id": self.agent_id,
            "role": self.role,
            "retries": self.retries,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DagTask":
        return cls(
            id=str(payload.get("id", "")),
            action=str(payload.get("action", "")),
            depends_on=tuple(str(item) for item in payload.get("depends_on", ()) or ()),
            query=str(payload.get("query", "")),
            claim=str(payload.get("claim", "")),
            user_id=str(payload.get("user_id", "") or "local"),
            top_k=int(payload.get("top_k", 4) or 4),
            status=str(payload.get("status", PENDING) or PENDING),
            deadline=str(payload.get("deadline", "") or ""),
            error=str(payload.get("error", "") or ""),
            agent_id=str(payload.get("agent_id", "") or ""),
            role=str(payload.get("role", "") or ""),
            retries=int(payload.get("retries", 0) or 0),
        )


@dataclass
class ResearchDag:
    """Ready tasks have no unfinished parents. Insertion order is the schedule."""

    tasks: dict[str, DagTask] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, task: DagTask) -> DagTask:
        if task.id in self.tasks:
            return self.tasks[task.id]
        self.tasks[task.id] = task
        self.order.append(task.id)
        return task

    def parents_done(self, task: DagTask) -> bool:
        if task.action == SYNTHESIZE:
            if not task.depends_on:
                return True
            return all(
                parent in self.tasks and self.tasks[parent].status in TERMINAL
                for parent in task.depends_on
            )
        return all(
            parent in self.tasks and self.tasks[parent].status == DONE
            for parent in task.depends_on
        )

    def ready(self) -> list[DagTask]:
        ready: list[DagTask] = []
        for task_id in self.order:
            task = self.tasks[task_id]
            if task.status != PENDING:
                continue
            if self.parents_done(task):
                ready.append(task)
        return ready

    def ready_wave(self) -> list[DagTask]:
        """Independent pending work. Synthesize waits until it is the only ready task."""
        waiting = self.ready()
        work = [task for task in waiting if task.action != SYNTHESIZE]
        return work or waiting

    def next_ready(self) -> DagTask | None:
        waiting = self.ready_wave()
        return waiting[0] if waiting else None

    def frontier(self) -> tuple[str, ...]:
        """Open leaves that later work should wait for."""
        children: set[str] = set()
        for task in self.tasks.values():
            if task.action == SYNTHESIZE:
                continue
            for parent in task.depends_on:
                children.add(parent)
        return tuple(
            task_id
            for task_id in self.order
            if self.tasks[task_id].action != SYNTHESIZE and task_id not in children
        )

    def mark_running(self, task_id: str) -> None:
        self._set(task_id, RUNNING)

    def mark_done(self, task_id: str) -> None:
        self._set(task_id, DONE)

    def mark_failed(self, task_id: str, error: str = "") -> None:
        self._set(task_id, FAILED, error=error)
        self.propagate_failure(task_id)

    def mark_expired(self, task_id: str) -> None:
        self._set(task_id, EXPIRED, error="Deadline exceeded.")
        self.propagate_failure(task_id)

    def can_retry(self, task_id: str, error: str = "") -> bool:
        task = self.tasks.get(task_id)
        if task is None or task.action == SYNTHESIZE:
            return False
        reason = (error or task.error).casefold()
        return task.retries < 1 and any(token in reason for token in _RETRYABLE)

    def reschedule(self, task_id: str) -> bool:
        """Return a failed or just-failed task to pending for one more attempt."""
        task = self.tasks.get(task_id)
        if task is None:
            return False
        self.tasks[task_id] = replace(task, status=PENDING, error="", retries=task.retries + 1)
        return True

    def propagate_failure(self, task_id: str) -> int:
        """Fail pending dependents when a prerequisite failed or expired."""
        parent = self.tasks.get(task_id)
        if parent is None or parent.status not in {FAILED, EXPIRED}:
            return 0
        blocked = 0
        changed = True
        while changed:
            changed = False
            for task in list(self.tasks.values()):
                if task.status != PENDING or task.action == SYNTHESIZE:
                    continue
                failed = [
                    dep
                    for dep in task.depends_on
                    if dep in self.tasks and self.tasks[dep].status in {FAILED, EXPIRED}
                ]
                if not failed:
                    continue
                self._set(task.id, FAILED, error=f"Blocked by failed prerequisite {failed[0]}.")
                blocked += 1
                changed = True
        return blocked

    def propagate_all(self) -> int:
        blocked = 0
        for task_id, task in list(self.tasks.items()):
            if task.status in {FAILED, EXPIRED}:
                blocked += self.propagate_failure(task_id)
        return blocked

    def recover(self) -> int:
        """Return interrupted tasks to pending so a new worker can run them."""
        reset = 0
        for task in self.tasks.values():
            if task.status == RUNNING:
                task.status = PENDING
                task.error = ""
                reset += 1
        return reset

    def expire_due(self, now: str) -> int:
        expired = 0
        for task in list(self.tasks.values()):
            if task.status == PENDING and task.deadline and task.deadline < now:
                task.status = EXPIRED
                task.error = "Deadline exceeded."
                expired += 1
                self.propagate_failure(task.id)
        return expired

    def complete(self) -> bool:
        return bool(self.tasks) and all(task.status in {DONE, FAILED, EXPIRED} for task in self.tasks.values())

    def to_payload(self) -> dict:
        return {"tasks": [self.tasks[task_id].to_payload() for task_id in self.order]}

    @classmethod
    def from_payload(cls, payload: dict | None) -> "ResearchDag":
        dag = cls()
        data = payload or {}
        for item in data.get("tasks", []) or []:
            task = DagTask.from_payload(item)
            if task.id:
                dag.add(task)
        return dag

    def _set(self, task_id: str, status: str, error: str = "") -> None:
        task = self.tasks.get(task_id)
        if task is None:
            return
        self.tasks[task_id] = replace(task, status=status, error=error or task.error)


def build_search_dag(
    question: str,
    *,
    user_id: str = "local",
    top_k: int = 4,
    deadline: str = "",
) -> ResearchDag:
    """Search each clause independently; each reverify waits only on its search."""
    dag = ResearchDag()
    for index, clause in enumerate(subquestions(question.strip())):
        search_id = f"search:{index}"
        verify_id = f"reverify:{index}"
        dag.add(
            DagTask(
                search_id,
                SEARCH,
                query=clause,
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                agent_id=f"searcher:{index}",
                role="searcher",
            )
        )
        dag.add(
            DagTask(
                verify_id,
                REVERIFY,
                depends_on=(search_id,),
                claim=clause,
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                agent_id=f"verifier:{index}",
                role="verifier",
            )
        )
    return dag


def extend_gap_dag(
    dag: ResearchDag,
    gaps: tuple[str, ...],
    *,
    user_id: str = "local",
    top_k: int = 4,
    deadline: str = "",
    alternatives: tuple[str, ...] = (),
) -> ResearchDag:
    """Fill remaining gaps in parallel. Each fill waits on the current frontier."""
    parents = tuple(
        task_id
        for task_id in dag.order
        if dag.tasks[task_id].action == REVERIFY and ":gap:" not in task_id
    ) or dag.frontier()
    verify_ids: list[str] = []
    for index, gap in enumerate(gaps):
        fill_id = f"fill:{index}"
        verify_id = f"reverify:gap:{index}"
        query = alternatives[index] if index < len(alternatives) else gap
        dag.add(
            DagTask(
                fill_id,
                FILL_GAPS,
                depends_on=parents,
                query=query,
                claim=gap,
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                agent_id=f"searcher:gap:{index}",
                role="searcher",
            )
        )
        dag.add(
            DagTask(
                verify_id,
                REVERIFY,
                depends_on=(fill_id,),
                query=query,
                claim=gap,
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                agent_id=f"verifier:gap:{index}",
                role="verifier",
            )
        )
        verify_ids.append(verify_id)
    _bind_synthesize(dag, tuple(verify_ids) or parents, user_id=user_id, top_k=top_k, deadline=deadline)
    return dag


def _bind_synthesize(
    dag: ResearchDag,
    parents: tuple[str, ...],
    *,
    user_id: str,
    top_k: int,
    deadline: str,
) -> None:
    existing = dag.tasks.get("synthesize")
    if existing is not None:
        if existing.status == PENDING:
            dag.tasks["synthesize"] = replace(existing, depends_on=parents or existing.depends_on)
        return
    dag.add(
        DagTask(
            "synthesize",
            SYNTHESIZE,
            depends_on=parents,
            user_id=user_id,
            top_k=top_k,
            deadline=deadline,
            agent_id="synthesizer",
            role="synthesizer",
        )
    )

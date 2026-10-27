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

    def ready(self) -> list[DagTask]:
        ready: list[DagTask] = []
        for task_id in self.order:
            task = self.tasks[task_id]
            if task.status != PENDING:
                continue
            if all(self.tasks[parent].status == DONE for parent in task.depends_on):
                ready.append(task)
        return ready

    def next_ready(self) -> DagTask | None:
        waiting = self.ready()
        return waiting[0] if waiting else None

    def mark_running(self, task_id: str) -> None:
        self._set(task_id, RUNNING)

    def mark_done(self, task_id: str) -> None:
        self._set(task_id, DONE)

    def mark_failed(self, task_id: str, error: str = "") -> None:
        self._set(task_id, FAILED, error=error)

    def mark_expired(self, task_id: str) -> None:
        self._set(task_id, EXPIRED, error="Deadline exceeded.")

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
        for task in self.tasks.values():
            if task.status == PENDING and task.deadline and task.deadline < now:
                task.status = EXPIRED
                task.error = "Deadline exceeded."
                expired += 1
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
    """Search each clause, then reverify that clause. Later clauses wait."""
    dag = ResearchDag()
    previous = ""
    for index, clause in enumerate(subquestions(question.strip())):
        search_id = f"search:{index}"
        verify_id = f"reverify:{index}"
        dag.add(
            DagTask(
                search_id,
                SEARCH,
                depends_on=(previous,) if previous else (),
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
        previous = verify_id
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
    """Fill remaining gaps, then synthesize. Parents are the finished verify wave."""
    parents = tuple(
        task_id
        for task_id in dag.order
        if dag.tasks[task_id].action == REVERIFY and ":gap:" not in task_id
    )
    last = parents[-1] if parents else (dag.order[-1] if dag.order else "")
    for index, gap in enumerate(gaps):
        fill_id = f"fill:{index}"
        verify_id = f"reverify:gap:{index}"
        query = alternatives[index] if index < len(alternatives) else gap
        dag.add(
            DagTask(
                fill_id,
                FILL_GAPS,
                depends_on=(last,) if last else (),
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
        last = verify_id
    if "synthesize" not in dag.tasks:
        dag.add(
            DagTask(
                "synthesize",
                SYNTHESIZE,
                depends_on=(last,) if last else (),
                user_id=user_id,
                top_k=top_k,
                deadline=deadline,
                agent_id="synthesizer",
                role="synthesizer",
            )
        )
    return dag

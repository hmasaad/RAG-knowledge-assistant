"""Run one approved research tool or a validated research action."""

import threading
from dataclasses import replace

from app.research.contracts import ActionOutcome, ActionRequest, ResearchState
from app.research.cycle import CycleResult, run_cycle
from app.research.pool import BoundedWorkerPool
from app.research.dispatcher import ActionDispatcher
from app.research.graph import EvidenceGraph
from app.research.guardrails import GuardrailPolicy, ResearchGuardrails
from app.research.mocks import mock_action_adapters
from app.research.observability import ResearchObserver, timed
from app.research.tools import ToolContext, ToolRegistry, ToolResult


class ResearchWorker:
    """Route tools and actions. Default action adapters are mocks.

    RAG search, claim verification, and the evidence graph are registered
    separately. The worker does not treat those modules as wired just because
    they exist in the process.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        *,
        dispatcher: ActionDispatcher | None = None,
        adapters: list | None = None,
        budget: int = 16,
        max_iterations: int = 8,
        user_id: str = "local",
        graph: EvidenceGraph | None = None,
        guardrails: ResearchGuardrails | None = None,
        denied_tools: frozenset[str] | None = None,
        max_cost: int = 0,
        observer: ResearchObserver | None = None,
    ) -> None:
        self._registry = registry if registry is not None else ToolRegistry()
        self.state = ResearchState()
        self.graph = graph
        self.observer = observer or getattr(dispatcher, "observer", None) or ResearchObserver()
        self.guardrails = guardrails or getattr(dispatcher, "guardrails", None) or ResearchGuardrails(
            GuardrailPolicy(
                budget=budget,
                max_iterations=max_iterations,
                max_cost=max_cost,
                denied_tools=denied_tools or frozenset(),
            )
        )
        self._dispatcher = dispatcher or ActionDispatcher(
            adapters if adapters is not None else mock_action_adapters(),
            budget=budget,
            max_iterations=max_iterations,
            guardrails=self.guardrails,
            observer=self.observer,
        )
        self._dispatcher.observer = self.observer
        self._cycle_lock = threading.Lock()
        self.pool = BoundedWorkerPool(1)

    def execute(self, tool: str, query: str, context: ToolContext) -> ToolResult:
        with timed() as clock:
            error = self.guardrails.check_tool(tool, context)
            if error:
                result = ToolResult(tool, query, False, error=error)
            else:
                result = self._registry.run(tool, query, context)
        self.observer.record_action(
            tool,
            ok=result.ok,
            cost=1 if result.ok and tool == "knowledge" else 0,
            seconds=clock.seconds,
            error=result.error,
            kind="tool",
        )
        return result

    def dispatch(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        return self._dispatcher.dispatch(request, state if state is not None else self.state)

    def cycle(
        self,
        question: str,
        *,
        user_id: str = "local",
        top_k: int = 4,
        state: ResearchState | None = None,
        on_progress=None,
        halt_after: int | None = None,
        should_cancel=None,
        deadline: str = "",
        pool: BoundedWorkerPool | None = None,
    ) -> CycleResult:
        with self._cycle_lock:
            if self.graph is not None:
                self.graph.clear()
            result = run_cycle(
                self._dispatcher,
                question,
                state=state if state is not None else ResearchState(),
                user_id=user_id,
                top_k=top_k,
                on_progress=on_progress,
                halt_after=halt_after,
                should_cancel=should_cancel,
                deadline=deadline,
                pool=pool if pool is not None else self.pool,
            )
            return replace(result, graph=self.graph)

    def tools(self) -> tuple[str, ...]:
        return self._registry.names()

    def actions(self) -> tuple[str, ...]:
        return self._dispatcher.actions()

"""Validate a research action, route it, and record the outcome."""

from app.research.contracts import (
    FILL_GAPS,
    ActionAdapter,
    ActionOutcome,
    ActionRequest,
    ResearchState,
)
from app.research.guardrails import GuardrailPolicy, ResearchGuardrails
from app.research.observability import ResearchObserver, timed


class ActionDispatcher:
    """Send each allowed action to its adapter and update research state."""

    def __init__(
        self,
        adapters: list[ActionAdapter] | None = None,
        *,
        budget: int = 16,
        max_iterations: int = 8,
        guardrails: ResearchGuardrails | None = None,
        max_cost: int = 0,
        denied_actions: frozenset[str] | None = None,
        observer: ResearchObserver | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("Search budget must be at least 1.")
        if max_iterations < 1:
            raise ValueError("Iteration limit must be at least 1.")
        self.budget = budget
        self.max_iterations = max_iterations
        self.guardrails = guardrails or ResearchGuardrails(
            GuardrailPolicy(
                budget=budget,
                max_iterations=max_iterations,
                max_cost=max_cost,
                denied_actions=denied_actions or frozenset(),
            )
        )
        self.observer = observer or ResearchObserver()
        self._adapters = {adapter.action: adapter for adapter in adapters or []}

    def register(self, adapter: ActionAdapter) -> None:
        self._adapters[adapter.action] = adapter

    def actions(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def dispatch(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        session = state if state is not None else ResearchState()
        with timed() as clock:
            error = self.validate(request, session)
            if error:
                outcome = ActionOutcome(action=request.action, ok=False, error=error, cost=0)
                session.errors.append(error)
            else:
                outcome = self.invoke(request, session)
                session.apply(request, outcome)
        self.observer.record_action(
            request.action,
            ok=outcome.ok,
            cost=outcome.cost,
            seconds=clock.seconds,
            error=outcome.error,
        )
        if not error and request.action == FILL_GAPS:
            self.observer.record_retry("gap")
        return outcome

    def invoke(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        """Run the adapter without writing state so a cycle can apply under a lock."""
        adapter = self._adapters[request.action]
        try:
            try:
                return adapter.run(request, state)
            except TypeError:
                return adapter.run(request)
        except TimeoutError:
            return ActionOutcome(action=request.action, ok=False, error="Action timed out.", cost=1)
        except Exception as exc:
            return ActionOutcome(
                action=request.action,
                ok=False,
                error=str(exc) or type(exc).__name__,
                cost=1,
            )

    def validate(self, request: ActionRequest, state: ResearchState) -> str:
        return self.guardrails.check_action(request, state, adapters=set(self._adapters))

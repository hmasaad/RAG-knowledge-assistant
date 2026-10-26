"""Dispatcher routing and validation use fake adapters, not a live model."""

from app.research.contracts import (
    FILL_GAPS,
    REVERIFY,
    RESOLVE_CONTRADICTIONS,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.dispatcher import ActionDispatcher


class FakeAdapter:
    def __init__(self, action: str, *, handler=None) -> None:
        self.action = action
        self.calls: list[ActionRequest] = []
        self._handler = handler

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        self.calls.append(request)
        if self._handler is not None:
            return self._handler(request)
        return ActionOutcome(
            action=self.action,
            ok=True,
            evidence=(EvidenceRef("e1", "d1", "notes.md", "retry limit is three"),),
            verdicts=(ClaimVerdict("retry limit is three", "supported", ("e1",)),),
            provenance=("e1",),
            cost=1,
        )


def test_dispatcher_routes_each_action() -> None:
    adapters = {
        SEARCH: FakeAdapter(SEARCH),
        FILL_GAPS: FakeAdapter(FILL_GAPS),
        RESOLVE_CONTRADICTIONS: FakeAdapter(RESOLVE_CONTRADICTIONS),
        REVERIFY: FakeAdapter(REVERIFY),
        SYNTHESIZE: FakeAdapter(
            SYNTHESIZE,
            handler=lambda request: ActionOutcome(
                action=SYNTHESIZE,
                ok=True,
                report=request.query,
                cost=0,
            ),
        ),
    }
    dispatcher = ActionDispatcher(list(adapters.values()), budget=16, max_iterations=8)
    state = ResearchState()
    state.evidence.append(EvidenceRef("seed", "d1", "notes.md"))

    search = dispatcher.dispatch(ActionRequest(SEARCH, query="retry limit"), state)
    gaps = dispatcher.dispatch(ActionRequest(FILL_GAPS, query="wifi password"), state)
    conflict = dispatcher.dispatch(ActionRequest(RESOLVE_CONTRADICTIONS, query="retry limit"), state)
    verify = dispatcher.dispatch(ActionRequest(REVERIFY, claim="retry limit is three"), state)
    report = dispatcher.dispatch(ActionRequest(SYNTHESIZE, query="The retry limit is three. [1]"), state)

    assert [search.action, gaps.action, conflict.action, verify.action, report.action] == [
        SEARCH, FILL_GAPS, RESOLVE_CONTRADICTIONS, REVERIFY, SYNTHESIZE
    ]
    assert all(outcome.ok for outcome in (search, gaps, conflict, verify, report))
    assert [len(adapter.calls) for adapter in adapters.values()] == [1, 1, 1, 1, 1]
    assert state.report == "The retry limit is three. [1]"
    assert state.cost == 4
    assert state.iterations == 5


def test_invalid_actions_and_missing_inputs_are_rejected() -> None:
    dispatcher = ActionDispatcher([FakeAdapter(SEARCH)], budget=4, max_iterations=4)
    state = ResearchState()

    unknown = dispatcher.dispatch(ActionRequest("browse", query="retry"), state)
    blank = dispatcher.dispatch(ActionRequest(SEARCH, query="   "), state)
    missing_claim = dispatcher.dispatch(ActionRequest(REVERIFY, query="retry"), state)

    assert unknown.error == "Invalid action: browse."
    assert blank.error == "search requires a query."
    assert missing_claim.error == "Unknown action: reverify."
    assert unknown.ok is blank.ok is missing_claim.ok is False
    assert state.cost == 0
    assert state.errors == [unknown.error, blank.error, missing_claim.error]


def test_duplicate_execution_and_limits_are_blocked() -> None:
    dispatcher = ActionDispatcher([FakeAdapter(SEARCH)], budget=2, max_iterations=8)
    state = ResearchState()
    first = dispatcher.dispatch(ActionRequest(SEARCH, query="retry limit"), state)
    duplicate = dispatcher.dispatch(ActionRequest(SEARCH, query="Retry Limit"), state)
    second = dispatcher.dispatch(ActionRequest(SEARCH, query="timeouts"), state)
    over_budget = dispatcher.dispatch(ActionRequest(SEARCH, query="wifi"), state)

    tight = ActionDispatcher([FakeAdapter(SEARCH)], budget=8, max_iterations=1)
    limited = ResearchState()
    tight.dispatch(ActionRequest(SEARCH, query="retry limit"), limited)
    over_iterations = tight.dispatch(ActionRequest(SEARCH, query="timeouts"), limited)

    assert first.ok is True
    assert duplicate.error == "Action already executed."
    assert second.ok is True
    assert over_budget.error == "Search budget exhausted."
    assert over_iterations.error == "Iteration limit reached."
    assert state.cost == 2


def test_timeouts_and_partial_failures_update_state() -> None:
    def timeout(_request: ActionRequest) -> ActionOutcome:
        raise TimeoutError

    def partial(_request: ActionRequest) -> ActionOutcome:
        return ActionOutcome(
            action=FILL_GAPS,
            ok=False,
            error="One source failed.",
            evidence=(EvidenceRef("e2", "d2", "retry.md", "retry.py"),),
            gaps=("office wifi password",),
            provenance=("e2",),
            cost=1,
        )

    dispatcher = ActionDispatcher(
        [FakeAdapter(SEARCH, handler=timeout), FakeAdapter(FILL_GAPS, handler=partial)],
        budget=8,
        max_iterations=8,
    )
    state = ResearchState()
    timed_out = dispatcher.dispatch(ActionRequest(SEARCH, query="retry limit"), state)
    failed = dispatcher.dispatch(ActionRequest(FILL_GAPS, query="wifi password"), state)

    assert timed_out.ok is False
    assert timed_out.error == "Action timed out."
    assert failed.ok is False
    assert failed.error == "One source failed."
    assert [item.evidence_id for item in state.evidence] == ["e2"]
    assert state.gaps == ["office wifi password"]
    assert state.errors == ["Action timed out.", "One source failed."]
    assert state.cost == 2
    assert state.provenance == ["e2"]

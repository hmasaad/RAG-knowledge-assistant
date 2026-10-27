"""Cost, tool, and permission limits stop research work before it runs."""

from app.research.contracts import (
    FILL_GAPS,
    SEARCH,
    SYNTHESIZE,
    ActionRequest,
    ResearchState,
)
from app.research.dispatcher import ActionDispatcher
from app.research.guardrails import (
    ACTION_DENIED,
    BUDGET_EXHAUSTED,
    CALLER_DENIED,
    TOOL_DENIED,
    GuardrailPolicy,
    ResearchGuardrails,
    parse_names,
)
from app.research.loop import STOP, LoopState, decide
from app.research.rag import SearchFilters
from app.research.tools import ToolContext, ToolRegistry
from app.research.worker import ResearchWorker

from tests.test_research_dispatcher import FakeAdapter


def test_guardrails_stop_the_loop_even_when_quality_wants_more_evidence() -> None:
    choice = decide(
        LoopState(
            complete=False,
            supported=1,
            gaps=("what is the office wifi password",),
            unresolved_conflicts=(),
            remaining_budget=0,
            quality_action=FILL_GAPS,
            quality_reason="The answer is missing evidence.",
        )
    )

    assert choice.action == STOP
    assert choice.reason == BUDGET_EXHAUSTED


def test_denied_tools_and_foreign_filters_are_blocked() -> None:
    rails = ResearchGuardrails(
        GuardrailPolicy(denied_tools=parse_names("web, knowledge"))
    )
    worker = ResearchWorker(ToolRegistry(), guardrails=rails)
    denied = worker.execute("knowledge", "retry limit", ToolContext(user_id="ada"))
    unknown = worker.execute("browse", "retry limit", ToolContext(user_id="ada"))
    foreign = worker.execute(
        "catalog",
        "which documents",
        ToolContext(
            user_id="ada",
            filters=SearchFilters(user_id="grace"),
        ),
    )
    missing = worker.execute("catalog", "which documents", ToolContext(user_id=""))

    assert denied.ok is False
    assert denied.error == TOOL_DENIED.format(name="knowledge")
    assert unknown.error == "Unknown tool: browse."
    assert foreign.error == CALLER_DENIED
    assert missing.error == CALLER_DENIED
    assert rails.remaining_cost(4) == 12


def test_document_filters_outside_the_caller_scope_are_blocked() -> None:
    rails = ResearchGuardrails()
    worker = ResearchWorker(ToolRegistry(), guardrails=rails)
    denied = worker.execute(
        "knowledge",
        "retry limit",
        ToolContext(
            user_id="ada",
            filters=SearchFilters(user_id="ada", document_ids=("secret-doc",)),
            document_ids={"ada-doc"},
        ),
    )
    allowed_scope = worker.execute(
        "knowledge",
        "retry limit",
        ToolContext(
            user_id="ada",
            filters=SearchFilters(user_id="ada", document_ids=("ada-doc",)),
            document_ids={"ada-doc"},
        ),
    )
    spent = decide(
        LoopState(
            complete=False,
            supported=0,
            gaps=("what is the office wifi password",),
            unresolved_conflicts=(),
            remaining_budget=2,
            spent_cost=16,
        )
    )

    assert denied.error == CALLER_DENIED
    assert allowed_scope.error == "Unknown tool: knowledge."
    assert spent.action == STOP
    assert spent.reason == BUDGET_EXHAUSTED


def test_denied_actions_and_cost_ceiling_are_blocked() -> None:
    dispatcher = ActionDispatcher(
        [FakeAdapter(SEARCH), FakeAdapter(FILL_GAPS), FakeAdapter(SYNTHESIZE)],
        budget=8,
        max_iterations=8,
        max_cost=1,
        denied_actions=frozenset({FILL_GAPS}),
    )
    state = ResearchState()
    first = dispatcher.dispatch(ActionRequest(SEARCH, query="retry limit"), state)
    blocked = dispatcher.dispatch(ActionRequest(FILL_GAPS, query="wifi"), state)
    over_cost = dispatcher.dispatch(ActionRequest(SEARCH, query="timeouts"), state)

    assert first.ok is True
    assert blocked.ok is False
    assert blocked.error == ACTION_DENIED.format(name=FILL_GAPS)
    assert over_cost.ok is False
    assert over_cost.error == BUDGET_EXHAUSTED
    assert state.cost == 1
    assert dispatcher.guardrails.remaining_cost(state.cost) == 0

"""Enforce cost, tool, and permission limits on research work."""

from dataclasses import dataclass

from app.research.contracts import (
    ACTIONS,
    FILL_GAPS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionRequest,
    ResearchState,
)
from app.research.planner import APPROVED_TOOLS
from app.research.tools import ToolContext

BUDGET_EXHAUSTED = "Search budget exhausted."
ITERATION_LIMIT = "Iteration limit reached."
CALLER_DENIED = "Caller is not permitted."
TOOL_UNKNOWN = "Unknown tool: {name}."
TOOL_DENIED = "Tool is not permitted: {name}."
ACTION_DENIED = "Action is not permitted: {name}."

_REQUIRES_QUERY = {SEARCH, FILL_GAPS, "resolve_contradictions"}


@dataclass(frozen=True)
class GuardrailPolicy:
    """Limits that apply to every tool call and dispatcher action."""

    budget: int = 16
    max_iterations: int = 8
    max_cost: int = 0
    allowed_tools: frozenset[str] = frozenset(APPROVED_TOOLS)
    allowed_actions: frozenset[str] = frozenset(ACTIONS)
    denied_tools: frozenset[str] = frozenset()
    denied_actions: frozenset[str] = frozenset()

    def cost_limit(self) -> int:
        if self.max_cost < 1:
            return self.budget
        return min(self.budget, self.max_cost)


def parse_names(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in value.split(",") if part.strip())


class ResearchGuardrails:
    """Reject work that exceeds cost, uses a blocked tool, or crosses callers."""

    def __init__(self, policy: GuardrailPolicy | None = None) -> None:
        self.policy = policy or GuardrailPolicy()

    def remaining_cost(self, spent: int) -> int:
        return max(0, self.policy.cost_limit() - max(0, spent))

    def should_stop(self, remaining_budget: int, spent: int = 0) -> bool:
        return remaining_budget < 1 or self.remaining_cost(spent) < 1

    def check_tool(self, tool: str, context: ToolContext) -> str:
        name = (tool or "").strip()
        if not (context.user_id or "").strip():
            return CALLER_DENIED
        filters = context.filters
        if filters is not None and filters.user_id and filters.user_id != context.user_id:
            return CALLER_DENIED
        if (
            filters is not None
            and filters.document_ids is not None
            and context.document_ids is not None
            and set(filters.document_ids) - context.document_ids
        ):
            return CALLER_DENIED
        if name in self.policy.denied_tools:
            return TOOL_DENIED.format(name=name)
        if name not in self.policy.allowed_tools:
            if name not in APPROVED_TOOLS:
                return TOOL_UNKNOWN.format(name=name or "(empty)")
            return TOOL_DENIED.format(name=name)
        return ""

    def check_action(
        self,
        request: ActionRequest,
        state: ResearchState,
        *,
        adapters: set[str],
    ) -> str:
        action = (request.action or "").strip()
        if action not in ACTIONS:
            return f"Invalid action: {action or '(empty)'}."
        if action not in adapters:
            return f"Unknown action: {action}."
        if action not in self.policy.allowed_actions or action in self.policy.denied_actions:
            return ACTION_DENIED.format(name=action)
        if action in _REQUIRES_QUERY and not request.query.strip():
            return f"{action} requires a query."
        if action == REVERIFY and not request.claim.strip():
            return "reverify requires a claim."
        if action == SYNTHESIZE and not state.evidence and not request.query.strip():
            return "synthesize requires evidence or a query."
        if state.iterations >= self.policy.max_iterations:
            return ITERATION_LIMIT
        if state.cost >= self.policy.cost_limit():
            return BUDGET_EXHAUSTED
        if state.fingerprint(request) in state.executed:
            return "Action already executed."
        return ""

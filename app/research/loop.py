"""Choose the next research step from evidence quality and remaining cost."""

from dataclasses import dataclass

from app.generation.agent import reformulate
from app.research.contracts import FILL_GAPS, RESOLVE_CONTRADICTIONS
from app.research.guardrails import BUDGET_EXHAUSTED, ResearchGuardrails

REPORT = "report"
RETRY = "retry"
PROBE_CONFLICT = "probe_conflict"
STOP = "stop"


@dataclass(frozen=True)
class LoopState:
    """What the agent knows after the latest investigation."""

    complete: bool
    supported: int
    gaps: tuple[str, ...]
    unresolved_conflicts: tuple[str, ...]
    remaining_budget: int
    gap_retries: int = 0
    conflict_probes: int = 0
    max_adapts: int = 2
    quality_action: str = ""
    quality_reason: str = ""


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    query: str = ""


def decide(state: LoopState, guardrails: ResearchGuardrails | None = None) -> Decision:
    """Pick the next action from quality, contradictions, gaps, and cost.

    Further research is skipped when the budget is gone or another search
    is unlikely to change the report. Guardrails win over a quality retry.
    """
    rails = guardrails or ResearchGuardrails()
    if rails.should_stop(state.remaining_budget):
        return Decision(STOP, BUDGET_EXHAUSTED)
    if state.complete and not state.gaps and not state.unresolved_conflicts:
        return Decision(REPORT, "Evidence is sufficient.")
    adapts = state.gap_retries + state.conflict_probes
    if adapts >= state.max_adapts:
        return _finish(state, "The adaptive research budget is spent.")
    prefer_conflict = (
        state.quality_action == RESOLVE_CONTRADICTIONS or bool(state.unresolved_conflicts)
    )
    if (
        prefer_conflict
        and state.unresolved_conflicts
        and state.conflict_probes < 1
        and state.remaining_budget >= 1
    ):
        return Decision(
            PROBE_CONFLICT,
            state.quality_reason or "Unresolved contradiction.",
            query=state.unresolved_conflicts[0],
        )
    next_gap = retryable_gap(state.gaps)
    if next_gap and state.gap_retries < 1 and state.quality_action != RESOLVE_CONTRADICTIONS:
        reason = state.quality_reason if state.quality_action == FILL_GAPS else "Missing information."
        return Decision(RETRY, reason, query=next_gap)
    return _finish(state, "Further research is not worth the remaining budget.")


def retryable_gap(gaps: tuple[str, ...]) -> str:
    """A gap that still has a shorter query to try."""
    for gap in gaps:
        alternative = reformulate(gap)
        if alternative:
            return alternative
    return ""


def _finish(state: LoopState, costly: str) -> Decision:
    if state.supported:
        return Decision(REPORT, costly)
    if state.complete and not state.gaps:
        return Decision(REPORT, "Evidence is sufficient.")
    return Decision(STOP, costly)

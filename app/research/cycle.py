"""Run question → retrieve → verify → fill gaps → retrieve → synthesize."""

from dataclasses import dataclass

from app.generation.agent import reformulate
from app.research.adapters import cited_answer
from app.research.contracts import (
    FILL_GAPS,
    REVERIFY,
    SEARCH,
    SYNTHESIZE,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.planner import subquestions
from app.research.dispatcher import ActionDispatcher


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
) -> CycleResult:
    """Retrieve, verify, fill gaps, retrieve again, then cite an answer.

    Already-executed actions on ``state`` are skipped so a persisted session
    can resume after a crash or a background worker restart.
    """
    session = state if state is not None else ResearchState()
    asked = question.strip()
    ran = 0

    def step(request: ActionRequest) -> None:
        nonlocal ran
        if should_cancel is not None and should_cancel():
            raise CycleCancelled(_cycle_snapshot(asked, session))
        if _step(dispatcher, session, request, on_progress=on_progress) is None:
            return
        ran += 1
        if should_cancel is not None and should_cancel():
            raise CycleCancelled(_cycle_snapshot(asked, session))
        if halt_after is not None and ran >= halt_after:
            raise CycleHalted(_cycle_snapshot(asked, session))

    for clause in subquestions(asked):
        step(ActionRequest(SEARCH, query=clause, user_id=user_id, top_k=top_k))
        step(ActionRequest(REVERIFY, claim=clause, user_id=user_id, top_k=top_k))
    for gap in tuple(session.gaps):
        alternative = reformulate(gap) or gap
        step(ActionRequest(FILL_GAPS, query=alternative, user_id=user_id, top_k=top_k))
        step(
            ActionRequest(
                REVERIFY,
                query=alternative,
                claim=gap,
                user_id=user_id,
                top_k=top_k,
            )
        )
    report = cited_answer(asked, session)
    step(ActionRequest(SYNTHESIZE, query=report, user_id=user_id, top_k=top_k))
    return _cycle_snapshot(asked, session)


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

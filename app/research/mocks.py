"""Action adapters that do not call RAG, verification, or the evidence graph."""

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


class MockAdapter:
    """Return a canned outcome. Tests replace handlers; the worker uses these by default."""

    def __init__(
        self,
        action: str,
        *,
        handler=None,
        evidence: tuple[EvidenceRef, ...] = (),
        verdicts: tuple[ClaimVerdict, ...] = (),
        gaps: tuple[str, ...] = (),
        contradictions: tuple[str, ...] = (),
        report: str = "",
        cost: int = 1,
        error: str = "",
    ) -> None:
        self.action = action
        self.calls: list[ActionRequest] = []
        self._handler = handler
        self._evidence = evidence
        self._verdicts = verdicts
        self._gaps = gaps
        self._contradictions = contradictions
        self._report = report
        self._cost = cost
        self._error = error

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        self.calls.append(request)
        if self._handler is not None:
            return self._handler(request, state)
        if self._error:
            return ActionOutcome(action=self.action, ok=False, error=self._error, cost=self._cost)
        report = request.query.strip() if self.action == SYNTHESIZE and request.query.strip() else self._report
        evidence = self._evidence
        gaps = self._gaps
        verdicts = self._verdicts
        if self.action == REVERIFY:
            pool = list(state.evidence) if state is not None else []
            supporting = tuple(item.evidence_id for item in pool if request.claim)
            verdict = "supported" if supporting else "insufficient"
            verdicts = (ClaimVerdict(request.claim, verdict, supporting[:1]),)
            gaps = () if supporting else (request.claim,)
            evidence = ()
            cost = 0
        else:
            cost = self._cost
            if self.action in {SEARCH, FILL_GAPS} and not evidence:
                gaps = gaps or ((request.query or request.claim),)
        return ActionOutcome(
            action=self.action,
            ok=True,
            evidence=evidence,
            verdicts=verdicts,
            gaps=gaps,
            contradictions=self._contradictions,
            provenance=tuple(item.evidence_id for item in evidence),
            report=report,
            cost=cost,
        )


def mock_action_adapters(*, handler=None) -> list[MockAdapter]:
    """One mock for each dispatcher action. None of them import a live source."""
    return [
        MockAdapter(
            SEARCH,
            handler=handler,
            evidence=(EvidenceRef("mock-1", "doc-1", "notes.md", "The retry limit is three attempts."),),
        ),
        MockAdapter(
            FILL_GAPS,
            handler=handler,
            evidence=(EvidenceRef("mock-1", "doc-1", "notes.md", "The retry limit is three attempts."),),
        ),
        MockAdapter(
            RESOLVE_CONTRADICTIONS,
            handler=handler,
            cost=0,
        ),
        MockAdapter(REVERIFY, handler=handler, cost=0),
        MockAdapter(SYNTHESIZE, handler=handler, cost=0),
    ]

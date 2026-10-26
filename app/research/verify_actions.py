"""Verification adapters. They score state evidence; they do not search."""

from app.research.contracts import (
    REVERIFY,
    RESOLVE_CONTRADICTIONS,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.verification import assess_claim, passage_contradiction
from app.retrieval.search import RankedPassage, SourceMetadata


class VerifyAction:
    """Judge a claim against evidence already on the research session."""

    action = REVERIFY

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        pool = list(state.evidence) if state is not None else []
        supporting: list[str] = []
        for item in pool:
            assessment = assess_claim(request.claim, as_passage(item))
            if assessment.verdict == "supported":
                supporting.append(item.evidence_id)
        verdict = "supported" if supporting else "insufficient"
        return ActionOutcome(
            action=self.action,
            ok=True,
            verdicts=(
                ClaimVerdict(
                    claim=request.claim,
                    verdict=verdict,
                    evidence_ids=tuple(supporting),
                ),
            ),
            gaps=() if supporting else (request.claim,),
            provenance=tuple(supporting),
            cost=0,
        )


class ResolveConflictsAction:
    """Compare retrieved passages. This adapter does not call RAG."""

    action = RESOLVE_CONTRADICTIONS

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        pool = list(state.evidence) if state is not None else []
        if len({item.source for item in pool}) < 2:
            return ActionOutcome(action=self.action, ok=True, cost=0)
        conflict = passage_contradiction(request.query, [as_passage(item) for item in pool])
        if not conflict.left_filename or not conflict.right_filename:
            return ActionOutcome(action=self.action, ok=True, cost=0)
        return ActionOutcome(
            action=self.action,
            ok=True,
            contradictions=(conflict.summary,),
            cost=0,
        )


def as_passage(item: EvidenceRef) -> RankedPassage:
    """Lift a stored evidence ref into the verification passage shape."""
    return RankedPassage(
        rank=1,
        text=item.text,
        score=1.0,
        vector_score=0.0,
        lexical_score=0.0,
        answers_question=True,
        matched_terms=(),
        source=SourceMetadata(
            chunk_id=item.evidence_id,
            document_id=item.document_id,
            filename=item.source,
            content_type="text/markdown",
            chunk_index=0,
            start_char=0,
            end_char=len(item.text),
            page=None,
            heading=None,
        ),
    )

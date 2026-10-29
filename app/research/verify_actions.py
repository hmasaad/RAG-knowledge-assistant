"""Verification adapters. They score state evidence; they do not search."""

from app.observability.logging import log_event
from app.research.claim_verifier import verify_resolved
from app.research.context_resolver import resolve_context
from app.research.contracts import (
    REVERIFY,
    RESOLVE_CONTRADICTIONS,
    ActionOutcome,
    ActionRequest,
    ClaimVerdict,
    EvidenceRef,
    ResearchState,
)
from app.research.ledger import EvidenceLedger
from app.research.verification import passage_contradiction
from app.retrieval.search import RankedPassage, SourceMetadata


class VerifyAction:
    """Resolve section context, score the claim, and write the evidence ledger."""

    action = REVERIFY

    def run(self, request: ActionRequest, state: ResearchState | None = None) -> ActionOutcome:
        pool = list(state.evidence) if state is not None else []
        ledger = EvidenceLedger.from_payload(list(state.ledger) if state is not None else [])
        supporting: list[str] = []
        decisions: list[dict] = []
        for item in pool:
            resolved = resolve_context(item, pool)
            entry = verify_resolved(request.claim, resolved, pool)
            ledger.record(entry)
            decisions.append(entry.to_payload())
            if entry.verdict == "supported":
                supporting.append(item.evidence_id)
        rows = tuple(ledger.to_payload())
        log_event(
            "rag.research.stages",
            "reverify",
            claim=request.claim,
            supporting=list(supporting),
            rejected=[row["evidence_id"] for row in decisions if row["evidence_id"] not in supporting],
            decisions=decisions,
        )
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
            ledger=rows,
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
            chunk_index=item.chunk_index,
            start_char=item.start_char,
            end_char=item.end_char or len(item.text),
            page=item.page,
            heading=item.heading or None,
            section_id=item.section_id,
            section_title=item.section_title,
            section_number=item.section_number,
        ),
    )

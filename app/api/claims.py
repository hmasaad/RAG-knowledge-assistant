"""Judge one claim with the retrieval contract."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_services, get_user_id
from app.api.research import _graph_response
from app.api.schemas import ClaimJudgeRequest, ClaimJudgeResponse, JudgeEvidenceResponse
from app.research.rag import SearchFilters
from app.wiring import Services

router = APIRouter(tags=["claims"])


@router.post("/claims/judge", response_model=ClaimJudgeResponse)
def judge_claim(
    body: ClaimJudgeRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ClaimJudgeResponse:
    verdict = services.claim_judge.judge_claim(
        body.claim,
        SearchFilters(user_id=user_id),
        body.top_k,
    )
    return ClaimJudgeResponse(
        claim=verdict.claim,
        verdict=verdict.verdict,
        evidence_ids=list(verdict.evidence_ids),
        support_ids=list(verdict.support_ids),
        oppose_ids=list(verdict.oppose_ids),
        contradiction=verdict.contradiction,
        reason=verdict.reason,
        evidence=[
            JudgeEvidenceResponse(
                evidence_id=item.evidence_id,
                document_id=item.document_id,
                filename=item.filename,
                text=item.text,
                role=item.role,
                score=item.score,
                start_char=item.start_char,
                end_char=item.end_char,
                heading=item.heading,
                page=item.page,
                instruction_detected=item.instruction_detected,
            )
            for item in verdict.evidence
        ],
        accepted=verdict.accepted,
        graph=_graph_response(verdict.graph),
    )

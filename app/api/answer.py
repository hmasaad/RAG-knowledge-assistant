"""Answer a question from the indexed collection."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_services, get_user_id
from app.api.schemas import AnswerRequest, AnswerResponse, CitationResponse
from app.api.search import passage_response
from app.wiring import Services

router = APIRouter(tags=["answer"])


@router.post("/answer", response_model=AnswerResponse)
def answer(
    body: AnswerRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> AnswerResponse:
    result = services.generation.answer(body.question, body.top_k, user_id=user_id)
    return AnswerResponse(
        question=result.query,
        answer=result.answer,
        sufficient_evidence=result.sufficient_evidence,
        conflict=result.conflict,
        citations=[
            CitationResponse(
                marker=citation.marker,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                filename=citation.filename,
                content_type=citation.content_type,
                heading=citation.heading,
                page=citation.page,
                quote=citation.quote,
            )
            for citation in result.citations
        ],
        passages=[passage_response(passage) for passage in result.passages],
        action=result.action,
        trace_id=result.trace_id,
    )

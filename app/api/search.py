"""Search the indexed knowledge collection."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_services, get_user_id
from app.api.schemas import (
    PassageResponse,
    RetrieveResponse,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
    SourceResponse,
)
from app.retrieval.search import RankedPassage
from app.wiring import Services

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> SearchResponse:
    hits = services.search.search(body.query, body.top_k, user_id=user_id)
    return SearchResponse(
        query=body.query,
        hits=[
            SearchHitResponse(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                content_type=hit.content_type,
                text=hit.text,
                score=hit.score,
                chunk_index=hit.chunk_index,
                start_char=hit.start_char,
                end_char=hit.end_char,
                page=hit.page,
                heading=hit.heading,
            )
            for hit in hits
        ],
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    body: SearchRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> RetrieveResponse:
    result = services.search.retrieve(body.query, body.top_k, user_id=user_id)
    return RetrieveResponse(
        query=result.query,
        sufficient_evidence=result.sufficient_evidence,
        conflict=result.conflict,
        inspection=result.inspection,
        passages=[passage_response(passage) for passage in result.passages],
        queries=list(result.queries),
    )


def passage_response(passage: RankedPassage) -> PassageResponse:
    source = passage.source
    return PassageResponse(
        rank=passage.rank,
        text=passage.text,
        score=passage.score,
        vector_score=passage.vector_score,
        lexical_score=passage.lexical_score,
        answers_question=passage.answers_question,
        matched_terms=list(passage.matched_terms),
        instruction_detected=passage.instruction_detected,
        rerank_score=passage.rerank_score,
        source=SourceResponse(
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            filename=source.filename,
            content_type=source.content_type,
            chunk_index=source.chunk_index,
            start_char=source.start_char,
            end_char=source.end_char,
            page=source.page,
            heading=source.heading,
        ),
    )

"""Search the indexed knowledge collection."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_services
from app.api.schemas import SearchHitResponse, SearchRequest, SearchResponse
from app.wiring import Services

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    services: Services = Depends(get_services),
) -> SearchResponse:
    hits = services.search.search(body.query, body.top_k)
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

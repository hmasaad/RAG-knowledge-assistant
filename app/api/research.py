"""Investigate a question and return a consolidated research report."""

from fastapi import APIRouter, Depends

from app.api.answer import _citation_response
from app.api.dependencies import get_services, get_user_id
from app.api.schemas import (
    AnswerRequest,
    ClaimLinkResponse,
    ConflictResponse,
    EvidenceGraphResponse,
    EvidenceResponse,
    GraphEdgeResponse,
    GraphNodeResponse,
    ResearchResponse,
    ResearchSubquestionResponse,
)
from app.api.search import passage_response
from app.wiring import Services

router = APIRouter(tags=["research"])


@router.post("/research", response_model=ResearchResponse)
def research(
    body: AnswerRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchResponse:
    result = services.research.research(body.question, body.top_k, user_id=user_id)
    return ResearchResponse(
        question=result.question,
        report=result.report,
        sufficient_evidence=result.sufficient_evidence,
        conflict=result.conflict,
        tools=list(result.tools),
        subquestions=[
            ResearchSubquestionResponse(
                question=step.question,
                tool=step.tool,
                status=step.status,
                evidence=step.evidence,
                completion=step.completion,
                complete=step.complete,
            )
            for step in result.subquestions
        ],
        contradictions=list(result.contradictions),
        gaps=list(result.gaps),
        catalog=list(result.catalog),
        citations=[_citation_response(citation) for citation in result.citations],
        passages=[passage_response(passage) for passage in result.passages],
        evidence=[
            EvidenceResponse(
                evidence_id=record.evidence_id,
                claim_id=record.claim_id,
                claim=record.claim,
                passage_id=record.passage_id,
                document_id=record.source.document_id,
                filename=record.source.filename,
                content_type=record.source.content_type,
                heading=record.source.heading,
                page=record.source.page,
                start_char=record.source.start_char,
                end_char=record.source.end_char,
                score=record.score,
                created_at=record.created_at,
                verdict=record.verdict,
                support=record.support,
            )
            for record in result.evidence_records
        ],
        links=[
            ClaimLinkResponse(
                claim_id=link.claim_id,
                passage_id=link.passage_id,
                evidence_id=link.evidence_id,
            )
            for link in result.links
        ],
        conflicts=[
            ConflictResponse(
                summary=conflict.summary,
                left_filename=conflict.left_filename,
                right_filename=conflict.right_filename,
                left_values=list(conflict.left_values),
                right_values=list(conflict.right_values),
                left_claim_id=conflict.left_claim_id,
                right_claim_id=conflict.right_claim_id,
            )
            for conflict in result.conflicts
        ],
        graph=_graph_response(result.graph),
        action=result.action,
        trace_id=result.trace_id,
    )


def _graph_response(graph) -> EvidenceGraphResponse:
    payload = {"nodes": [], "edges": []} if graph is None else graph.to_payload()
    return EvidenceGraphResponse(
        nodes=[GraphNodeResponse(**node) for node in payload["nodes"]],
        edges=[GraphEdgeResponse(**edge) for edge in payload["edges"]],
    )

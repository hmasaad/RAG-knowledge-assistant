"""Investigate a question and return a consolidated research report."""

from fastapi import APIRouter, Depends, Response

from app.api.answer import _citation_response
from app.api.dependencies import get_services, get_user_id
from app.api.schemas import (
    AnswerRequest,
    AgentAssignmentResponse,
    AgentFindingResponse,
    ClaimLinkResponse,
    ConflictResolutionResponse,
    ConflictResponse,
    CoordinateRequest,
    CoordinationResponse,
    EvidenceGraphResponse,
    EvidenceResponse,
    GraphEdgeResponse,
    GraphNodeResponse,
    QualityClaimResponse,
    PlanTaskResponse,
    ResearchPlanResponse,
    ResearchResponse,
    ResearchSessionListItem,
    ResearchSessionListResponse,
    ResearchSubquestionResponse,
    SideEffectResponse,
    StrategyObserveRequest,
    StrategyRecommendRequest,
    StrategyRecommendResponse,
    StrategyStatResponse,
)
from app.research.coordination import finding_from_payload
from app.api.search import passage_response
from app.wiring import Services

router = APIRouter(tags=["research"])


@router.post("/research/plan", response_model=ResearchPlanResponse)
def research_plan(
    body: AnswerRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchPlanResponse:
    plan = services.research.propose_plan(body.question, user_id=user_id, top_k=body.top_k)
    return ResearchPlanResponse(
        question=plan.question,
        version=plan.version,
        reason=plan.reason,
        revisions=list(plan.revisions),
        tools=list(plan.tools),
        budget=plan.budget,
        actions=list(plan.actions()),
        tasks=[
            PlanTaskResponse(
                id=plan.dag.tasks[task_id].id,
                action=plan.dag.tasks[task_id].action,
                depends_on=list(plan.dag.tasks[task_id].depends_on),
                query=plan.dag.tasks[task_id].query,
                claim=plan.dag.tasks[task_id].claim,
                status=plan.dag.tasks[task_id].status,
                agent_id=plan.dag.tasks[task_id].agent_id,
                role=plan.dag.tasks[task_id].role,
            )
            for task_id in plan.dag.order
        ],
    )


@router.post("/research/coordinate", response_model=CoordinationResponse)
def research_coordinate(
    body: CoordinateRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> CoordinationResponse:
    report = services.research.coordinate(
        body.question,
        findings=[finding_from_payload(item.model_dump()) for item in body.findings],
        user_id=user_id,
    )
    return _coordination_response(report)


@router.get("/research/strategy", response_model=list[StrategyStatResponse])
def read_strategy(services: Services = Depends(get_services)) -> list[StrategyStatResponse]:
    return [StrategyStatResponse(**item.to_payload()) for item in services.research.strategy_snapshot()]


@router.post("/research/strategy", response_model=StrategyStatResponse)
def observe_strategy(
    body: StrategyObserveRequest,
    services: Services = Depends(get_services),
) -> StrategyStatResponse:
    return StrategyStatResponse(**services.research.learn_strategy(body.situation, body.action, body.reward).to_payload())


@router.post("/research/strategy/recommend", response_model=StrategyRecommendResponse)
def recommend_strategy(
    body: StrategyRecommendRequest,
    services: Services = Depends(get_services),
) -> StrategyRecommendResponse:
    action = services.research.recommend_strategy(body.situation, body.fallback)
    return StrategyRecommendResponse(situation=body.situation, action=action, fallback=body.fallback)


@router.post("/research", response_model=ResearchResponse)
def research(
    body: AnswerRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchResponse:
    return _research_response(services.research.research(body.question, body.top_k, user_id=user_id))


@router.post("/research/cycle", response_model=ResearchResponse)
def research_cycle(
    body: AnswerRequest,
    response: Response,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchResponse:
    result = services.research.cycle(
        body.question,
        body.top_k,
        user_id=user_id,
        halt_after=body.halt_after,
        priority=body.priority,
    )
    if result.status in {"queued", "running"}:
        response.status_code = 202
    return _research_response(result)


@router.get("/research/sessions", response_model=ResearchSessionListResponse)
def list_research_sessions(
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchSessionListResponse:
    return ResearchSessionListResponse(
        sessions=[
            ResearchSessionListItem(
                session_id=session.id,
                question=session.question,
                status=session.status,
                action=session.action,
                created_at=session.created_at,
                updated_at=session.updated_at,
                priority=session.priority,
            )
            for session in services.research.list_sessions(user_id=user_id)
        ]
    )


@router.get("/research/sessions/{session_id}", response_model=ResearchResponse)
def get_research_session(
    session_id: str,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchResponse:
    return _research_response(services.research.get_session(session_id, user_id=user_id))


@router.post("/research/sessions/{session_id}/resume", response_model=ResearchResponse)
def resume_research_session(
    session_id: str,
    response: Response,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchResponse:
    result = services.research.resume(session_id, user_id=user_id)
    if result.status in {"queued", "running"}:
        response.status_code = 202
    return _research_response(result)


@router.post("/research/sessions/{session_id}/cancel", response_model=ResearchResponse)
def cancel_research_session(
    session_id: str,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> ResearchResponse:
    return _research_response(services.research.cancel(session_id, user_id=user_id))


def _research_response(result) -> ResearchResponse:
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
        decision=result.decision,
        decisions=list(result.decisions),
        remaining_budget=result.remaining_budget,
        action=result.action,
        trace_id=result.trace_id,
        session_id=result.session_id,
        status=result.status,
        side_effects=[SideEffectResponse(**item) for item in result.side_effects],
        quality_action=result.quality_action,
        quality_reason=result.quality_reason,
        quality_failures=list(result.quality_failures),
        quality_claims=[
            QualityClaimResponse(
                claim=item.claim,
                marker=item.marker,
                citation_exists=item.citation_exists,
                supports=item.supports,
                support_ratio=item.support_ratio,
                filename=item.filename,
                failures=list(item.failures),
            )
            for item in result.quality_claims
        ],
        quality_confident=result.quality_confident,
        coordination=_coordination_response(result.coordination) if result.coordination else None,
    )


def _coordination_response(report) -> CoordinationResponse:
    return CoordinationResponse(
        question=report.question,
        assignments=[AgentAssignmentResponse(**item.to_payload()) for item in report.assignments],
        findings=[AgentFindingResponse(**item.to_payload()) for item in report.findings],
        resolutions=[ConflictResolutionResponse(**item.to_payload()) for item in report.resolutions],
        consensus=list(report.consensus),
        resolved=report.resolved,
    )


def _graph_response(graph) -> EvidenceGraphResponse:
    payload = {"nodes": [], "edges": []} if graph is None else graph.to_payload()
    return EvidenceGraphResponse(
        nodes=[GraphNodeResponse(**node) for node in payload["nodes"]],
        edges=[GraphEdgeResponse(**edge) for edge in payload["edges"]],
    )

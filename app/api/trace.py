"""Source-trace API over the evidence graph."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_services, get_user_id
from app.research.trace import answer_from_trace, library_scene, named_demo_asset, trace_scene
from app.wiring import Services

router = APIRouter(tags=["trace"])


class TraceAssetModel(BaseModel):
    id: str
    name: str
    color: str


class TraceMarkerModel(BaseModel):
    n: int
    claim_id: str


class TraceVersionModel(BaseModel):
    version: int
    checksum: str
    label: str
    recorded_at: str = ""


class TraceLineageModel(BaseModel):
    kind: str
    id: str
    label: str
    version: int = 0
    recorded_at: str = ""
    checksum: str = ""


class TraceCitationModel(BaseModel):
    marker: int | None = None
    quote: str = ""
    filename: str = ""
    version: int = 0
    chunk_id: str = ""
    start_char: int = 0
    end_char: int = 0
    recorded_at: str = ""


class TraceAuditModel(BaseModel):
    decision: str
    detail: str = ""
    recorded_at: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class TraceProvenanceModel(BaseModel):
    document_id: str = ""
    filename: str = ""
    version: int = 0
    checksum: str = ""
    chunk_id: str = ""
    heading: str | None = None
    start_char: int = 0
    end_char: int = 0
    recorded_at: str = ""
    lineage: list[TraceLineageModel] = Field(default_factory=list)
    history: list[TraceVersionModel] = Field(default_factory=list)


class TraceClaimModel(BaseModel):
    id: str
    text: str
    short: str
    agent_id: str
    band: str
    upstream: list[str]
    upstream_count: int
    marker: int | None = None
    entities: list[str]
    contradicts: list[str]
    provenance: TraceProvenanceModel
    citation: TraceCitationModel | None = None
    audit: list[TraceAuditModel] = Field(default_factory=list)


class TraceAgentModel(BaseModel):
    id: str
    label: str
    role: str


class TraceEntityModel(BaseModel):
    id: str
    label: str


class TraceChunkModel(BaseModel):
    id: str
    agent_id: str
    filename: str
    version: int
    label: str
    checksum: str = ""
    chunk_id: str = ""
    document_id: str = ""


class TraceEdgeModel(BaseModel):
    source: str
    relation: str
    target: str


class TraceSceneModel(BaseModel):
    asset: str
    assets: list[TraceAssetModel]
    question: str
    answer: str
    markers: list[TraceMarkerModel]
    agents: list[TraceAgentModel]
    entities: list[TraceEntityModel]
    claims: list[TraceClaimModel]
    chunks: list[TraceChunkModel]
    edges: list[TraceEdgeModel]
    selected_agent: str
    selected_claim: str
    decisions: list[TraceAuditModel] = Field(default_factory=list)


class TraceAskRequest(BaseModel):
    question: str = Field(min_length=1)
    asset: str = "NVDA"


class TraceAskModel(BaseModel):
    asset: str
    answer: str
    markers: list[TraceMarkerModel]
    scene: TraceSceneModel


def _scene_or_404(asset: str) -> dict:
    scene = trace_scene(asset)
    if scene is None:
        raise HTTPException(status_code=404, detail=f"Unknown asset {asset.strip().upper()}.")
    return scene


@router.get("/trace", response_model=TraceSceneModel)
def read_trace(asset: str = "NVDA") -> dict:
    return _scene_or_404(asset)


@router.post("/trace/ask", response_model=TraceAskModel)
def ask_trace(
    body: TraceAskRequest,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> dict:
    question = body.question.strip()
    if named_demo_asset(question) is None:
        scene = library_scene(services.generation.answer(question, 4, user_id=user_id), services.metadata)
        if scene is not None:
            return {
                "asset": scene["asset"],
                "answer": scene["answer"],
                "markers": scene["markers"],
                "scene": scene,
            }
    result = answer_from_trace(question, body.asset)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Unknown asset {body.asset.strip().upper()}.")
    return result

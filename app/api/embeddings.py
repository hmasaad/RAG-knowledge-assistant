"""Inspect, evaluate, and migrate the embedding model."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_services
from app.ingestion.embedding_eval import evaluate_embedder
from app.ingestion.migration import migrate_embeddings
from app.ingestion.similarity import embedding_matches
from app.wiring import Services

router = APIRouter(tags=["embeddings"])


class EmbeddingStatusResponse(BaseModel):
    active_model: str
    active_dimension: int | None
    stored_model: str | None
    stored_dimension: int | None
    migration_required: bool
    chunk_strategy: str


class MigrationResponse(BaseModel):
    model: str
    dimension: int
    documents: int
    chunks: int


class PairScoreResponse(BaseModel):
    id: str
    positive_similarity: float
    negative_similarity: float
    margin: float
    correct: bool


class EmbeddingEvaluationResponse(BaseModel):
    model: str
    dimension: int
    pair_accuracy: float
    mean_positive_similarity: float
    mean_negative_similarity: float
    mean_margin: float
    pairs: list[PairScoreResponse]


@router.get("/embeddings", response_model=EmbeddingStatusResponse)
def embedding_status(services: Services = Depends(get_services)) -> EmbeddingStatusResponse:
    embedder = services.ingestion.embedder
    stored = services.metadata.embedding_identity()
    dimension = getattr(embedder, "dimension", None)
    return EmbeddingStatusResponse(
        active_model=str(getattr(embedder, "model", "unknown")),
        active_dimension=None if dimension is None else int(dimension),
        stored_model=None if stored is None else stored[0],
        stored_dimension=None if stored is None else stored[1],
        migration_required=not embedding_matches(embedder, stored),
        chunk_strategy=services.settings.chunk_strategy,
    )


@router.post("/embeddings/migrate", response_model=MigrationResponse)
def migrate(services: Services = Depends(get_services)) -> MigrationResponse:
    result = migrate_embeddings(
        metadata=services.metadata,
        vectors=services.vectors,
        embedder=services.ingestion.embedder,
    )
    return MigrationResponse(
        model=result.model,
        dimension=result.dimension,
        documents=result.documents,
        chunks=result.chunks,
    )


@router.post("/embeddings/evaluate", response_model=EmbeddingEvaluationResponse)
def evaluate(services: Services = Depends(get_services)) -> EmbeddingEvaluationResponse:
    report = evaluate_embedder(services.ingestion.embedder)
    return EmbeddingEvaluationResponse(
        model=report.model,
        dimension=report.dimension,
        pair_accuracy=report.pair_accuracy,
        mean_positive_similarity=report.mean_positive_similarity,
        mean_negative_similarity=report.mean_negative_similarity,
        mean_margin=report.mean_margin,
        pairs=[
            PairScoreResponse(
                id=score.id,
                positive_similarity=score.positive_similarity,
                negative_similarity=score.negative_similarity,
                margin=score.margin,
                correct=score.correct,
            )
            for score in report.pairs
        ],
    )

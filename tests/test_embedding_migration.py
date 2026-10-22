"""Embedding similarity, evaluation, and model migration."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.errors import IngestionError, KnowledgeError
from app.ingestion.migration import migrate_embeddings
from app.ingestion.similarity import cosine_distance, cosine_similarity
from app.ingestion.vector_store import _distance_to_score
from app.main import create_app
from app.retrieval.search import SearchService
from app.wiring import build_services
from evaluation.embeddings import evaluate_embedder

ROOT = Path(__file__).resolve().parents[1]


def test_cosine_distance_matches_a_positive_retrieval_score() -> None:
    embedder = LocalEmbedder(dimension=32)
    left, right = embedder.embed_documents(
        ["retry logic in the client", "the client retry logic"]
    )

    assert cosine_distance(left, left) == pytest.approx(0.0, abs=1e-9)
    similarity = cosine_similarity(left, right)
    assert similarity > 0
    assert _distance_to_score(cosine_distance(left, right)) == pytest.approx(similarity)


def test_local_embedder_meets_the_pair_baseline() -> None:
    report = evaluate_embedder(LocalEmbedder(dimension=128))
    floor = json.loads((ROOT / "evaluation" / "baselines" / "embedding_pairs.json").read_text())

    assert report.pair_accuracy >= floor["pair_accuracy"]
    assert report.mean_margin > 0
    assert all(score.correct for score in report.pairs)


def test_changed_embedding_width_requires_migration_before_search(tmp_path: Path) -> None:
    settings = Settings(
        embedding_provider="local",
        embedding_dimension=32,
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "metadata.db",
        blob_path=tmp_path / "blobs",
        chunk_size=500,
        chunk_overlap=40,
        max_upload_bytes=100_000,
        log_level="WARNING",
        llm_provider="local",
    )
    services = build_services(settings, embedder=LocalEmbedder(dimension=32))
    services.ingestion.ingest(
        "notes.md",
        b"# Retry logic\n\nThe file that implements retry logic is app/http/retry.py.\n",
    )
    assert services.metadata.embedding_identity() == ("local-hash-bow", 32)

    wider = LocalEmbedder(dimension=64)
    blocked = SearchService(
        vectors=services.vectors,
        embedder=wider,
        metadata=services.metadata,
        default_user_id="local",
    )
    with pytest.raises(KnowledgeError, match="Migrate embeddings") as error:
        blocked.retrieve("which file implements the retry logic", top_k=2)
    assert error.value.status_code == 409

    from app.ingestion.service import IngestionService

    other = IngestionService(
        metadata=services.metadata,
        vectors=services.vectors,
        embedder=wider,
        blobs=services.blobs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )
    with pytest.raises(IngestionError, match="Migrate embeddings"):
        other.ingest(
            "other.md",
            b"# Migrations\n\nSchema changes ship as SQL files under db/migrations.\n",
        )

    migrated = migrate_embeddings(
        metadata=services.metadata,
        vectors=services.vectors,
        embedder=wider,
    )
    hits = blocked.retrieve("which file implements the retry logic", top_k=2)

    assert migrated.dimension == 64
    assert migrated.documents == 1
    assert migrated.chunks >= 1
    assert services.metadata.embedding_identity() == ("local-hash-bow", 64)
    assert "retry.py" in hits.passages[0].text
    with pytest.raises(KnowledgeError, match="Migrate embeddings"):
        services.search.retrieve("which file implements the retry logic", top_k=2)


def test_embedding_endpoints_report_status_and_evaluation(client: TestClient) -> None:
    status = client.get("/api/embeddings")
    evaluation = client.post("/api/embeddings/evaluate")

    assert status.status_code == 200
    assert status.json()["chunk_strategy"] == "structure"
    assert status.json()["migration_required"] is False
    assert evaluation.status_code == 200
    assert evaluation.json()["pair_accuracy"] == 1.0

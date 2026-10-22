"""Shared fixtures for the ingestion tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.main import create_app
from app.wiring import Services, build_services

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        embedding_provider="local",
        embedding_dimension=128,
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "metadata.db",
        chunk_size=500,
        chunk_overlap=40,
        max_upload_bytes=100_000,
        log_level="WARNING",
    )


@pytest.fixture
def services(settings: Settings) -> Services:
    return build_services(settings, embedder=LocalEmbedder(dimension=128))


@pytest.fixture
def client(settings: Settings) -> TestClient:
    application = create_app(settings, embedder=LocalEmbedder(dimension=128))
    with TestClient(application) as test_client:
        yield test_client

"""RAG is plugged into the worker only when search adapters are registered."""

from pathlib import Path

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.research.adapters import compose_action_adapters
from app.research.contracts import REVERIFY, SEARCH, ActionRequest, ResearchState
from app.research.rag import KnowledgeRagClient
from app.research.tools import approved_tools
from app.research.worker import ResearchWorker
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_rag_search_is_not_used_by_the_default_worker(tmp_path: Path) -> None:
    services = _services(tmp_path)
    services.ingestion.ingest("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes())
    worker = ResearchWorker()
    found = worker.dispatch(ActionRequest(SEARCH, query="which file implements the retry logic"))

    assert found.ok is True
    assert found.evidence[0].source == "notes.md"
    assert "retry.py" not in found.evidence[0].text


def test_rag_adapter_retrieves_indexed_passages(tmp_path: Path) -> None:
    worker, services = _rag_worker(tmp_path)
    services.ingestion.ingest("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes())
    found = worker.dispatch(ActionRequest(SEARCH, query="which file implements the retry logic", top_k=3))

    assert found.ok is True
    assert found.evidence
    assert found.evidence[0].source == "api-errors.md"
    assert "retry.py" in found.evidence[0].text


def test_mock_reverify_still_runs_after_rag_search(tmp_path: Path) -> None:
    worker, services = _rag_worker(tmp_path)
    services.ingestion.ingest("api-errors.md", (ROOT / "samples" / "api-errors.md").read_bytes())
    state = ResearchState()
    worker.dispatch(ActionRequest(SEARCH, query="which file implements the retry logic", top_k=3), state)
    verify = worker.dispatch(ActionRequest(REVERIFY, claim="what is the office wifi password"), state)

    assert state.evidence
    assert "retry.py" in state.evidence[0].text
    assert verify.verdicts[0].verdict == "supported"


def _rag_worker(tmp_path: Path):
    services = _services(tmp_path)
    registry = approved_tools(rag=KnowledgeRagClient(services.search), metadata=services.metadata)
    worker = ResearchWorker(
        registry,
        adapters=compose_action_adapters(registry=registry, use_rag=True),
    )
    return worker, services


def _services(tmp_path: Path):
    return build_services(
        Settings(
            embedding_provider="local",
            embedding_dimension=128,
            chroma_path=tmp_path / "chroma",
            sqlite_path=tmp_path / "metadata.db",
            blob_path=tmp_path / "blobs",
            chunk_size=500,
            chunk_overlap=40,
            max_upload_bytes=100_000,
            log_level="WARNING",
            llm_provider="local",
        ),
        embedder=LocalEmbedder(dimension=128),
    )

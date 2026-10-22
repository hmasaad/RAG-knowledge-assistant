"""Parent-child retrieval, contextual embeddings, and the knowledge graph."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.graph import extract_edges
from app.ingestion.vector_store import VectorHit
from app.retrieval.search import SearchService
from app.wiring import build_services


def test_edges_link_a_caller_to_the_file_it_calls() -> None:
    edges = extract_edges("alpha/retry.py calls beta/client.py.", "chunk-1")

    assert len(edges) == 1
    assert edges[0].source == "alpha/retry.py"
    assert edges[0].relation == "calls"
    assert edges[0].target == "beta/client.py"


def test_parent_retrieval_returns_the_whole_section(tmp_path: Path) -> None:
    services = _services(tmp_path, parent_retrieval=True, child_chunk_size=80, chunk_size=400)
    services.ingestion.ingest(
        "retry.md",
        (
            b"# Retry logic\n\n"
            b"The file that implements retry logic is app/http/retry.py. "
            b"Callers are allowed three retry attempts before the client gives up "
            b"on the request and returns an error to the caller.\n"
        ),
    )

    result = services.search.retrieve("three retry attempts", top_k=2)

    assert "app/http/retry.py" in result.passages[0].text
    assert "three retry attempts" in result.passages[0].text
    assert len(result.passages[0].text) > 80


def test_contextual_retrieval_matches_the_document_name(tmp_path: Path) -> None:
    services = _services(tmp_path, contextual_retrieval=True)
    services.ingestion.ingest("retry-logic.md", b"Callers are allowed three attempts.\n")
    services.ingestion.ingest("billing.md", b"The ledger stores invoices.\n")

    result = services.search.retrieve("retry logic", top_k=2)

    assert result.passages[0].source.filename == "retry-logic.md"


def test_graph_neighbor_is_searched_with_the_question() -> None:
    pair = _Pair()
    metadata = _Graph()
    service = SearchService(
        vectors=_Store(pair),
        embedder=pair,
        metadata=metadata,
        knowledge_graph=True,
    )

    result = service.retrieve("alpha/retry.py", top_k=2)

    assert "beta/client.py" in result.queries
    assert any("connect timeout" in passage.text for passage in result.passages)


def test_indexed_graph_records_the_call_and_forgets_it_on_delete(tmp_path: Path) -> None:
    services = _services(tmp_path, knowledge_graph=True)
    record, _ = services.ingestion.ingest("notes.md", b"alpha/retry.py calls beta/client.py.\n")

    assert services.metadata.neighbors(["alpha/retry.py"], None) == ["beta/client.py"]

    services.ingestion.delete_document(record.id)
    assert services.metadata.neighbors(["alpha/retry.py"], None) == []


def test_retrieve_reports_the_original_query(client: TestClient) -> None:
    response = client.post(
        "/api/retrieve",
        json={"query": "which file implements the retry logic", "top_k": 2},
    )

    assert response.status_code == 200
    assert response.json()["queries"] == ["which file implements the retry logic"]


def _services(tmp_path: Path, **overrides: object):
    values: dict[str, object] = {
        "embedding_provider": "local",
        "embedding_dimension": 128,
        "chroma_path": tmp_path / "chroma",
        "sqlite_path": tmp_path / "metadata.db",
        "blob_path": tmp_path / "blobs",
        "chunk_size": 500,
        "chunk_overlap": 40,
        "max_upload_bytes": 100_000,
        "log_level": "WARNING",
        "llm_provider": "local",
    }
    values.update(overrides)
    settings = Settings(**values)
    return build_services(settings, embedder=LocalEmbedder(dimension=128))


class _Pair:
    model = "local-hash-bow"
    query = ""

    def embed_query(self, text: str) -> list[float]:
        self.query = text
        return [1.0]


class _Graph:
    def readable_versions(self, user_id: str) -> dict[str, int]:
        return {"doc": 1}

    def embedding_identity(self) -> None:
        return None

    def neighbors(self, names: list[str], document_ids: set[str] | None) -> list[str]:
        if any(name.casefold() == "alpha/retry.py" for name in names):
            return ["beta/client.py"]
        return []


class _Store:
    def __init__(self, pair: _Pair) -> None:
        self._pair = pair

    def query(self, embedding: list[float], top_k: int, document_ids: set[str] | None = None):
        if "beta/client.py" in self._pair.query:
            return [_hit("client", "beta/client.py sets the connect timeout.")]
        return [_hit("retry", "alpha/retry.py calls beta/client.py.")]

    def list_chunks(self, document_ids: set[str] | None = None):
        return [
            _hit("retry", "alpha/retry.py calls beta/client.py."),
            _hit("client", "beta/client.py sets the connect timeout."),
        ]


def _hit(chunk_id: str, text: str) -> VectorHit:
    return VectorHit(
        chunk_id=chunk_id,
        text=text,
        score=0.8,
        metadata={
            "document_id": "doc",
            "filename": f"{chunk_id}.md",
            "content_type": "text/markdown",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len(text),
            "page": -1,
            "heading": "",
            "version": 1,
        },
    )

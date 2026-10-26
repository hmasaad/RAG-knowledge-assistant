"""Claims, passages, documents, and versions form one provenance graph."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.research.graph import CONTRADICTS, EXTRACTED_FROM, HAS_VERSION, SUPPORTS, ProvenanceEngine
from app.research.judge import ClaimJudgeAdapter, LocalClaimJudge
from app.research.rag import KnowledgeRagClient, SearchFilters
from app.wiring import build_services
from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder

ROOT = Path(__file__).resolve().parents[1]


def test_research_graph_links_claims_to_document_versions(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    uploaded = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    body = client.post(
        "/api/research",
        json={"question": "which file implements the retry logic", "top_k": 3},
    ).json()
    graph = body["graph"]
    kinds = {node["kind"] for node in graph["nodes"]}
    relations = {edge["relation"] for edge in graph["edges"]}
    passage = next(node for node in graph["nodes"] if node["kind"] == "passage")
    version = next(node for node in graph["nodes"] if node["kind"] == "version")

    assert uploaded.status_code == 201
    assert kinds == {"claim", "passage", "document", "version", "source"}
    assert SUPPORTS in relations
    assert EXTRACTED_FROM in relations
    assert HAS_VERSION in relations
    assert passage["filename"] == "api-errors.md"
    assert passage["start_char"] <= passage["end_char"]
    assert version["version"] == 1
    assert version["checksum"]


def test_contradiction_connects_two_document_versions(client: TestClient) -> None:
    for name in ("retry-limit-three.md", "retry-limit-five.md"):
        payload = (ROOT / "evaluation" / "fixtures" / name).read_bytes()
        uploaded = client.post(
            "/api/documents",
            files={"file": (name, payload, "text/markdown")},
        )
        assert uploaded.status_code == 201
    body = client.post(
        "/api/research",
        json={"question": "what is the retry limit", "top_k": 4},
    ).json()
    graph = body["graph"]
    documents = {node["filename"] for node in graph["nodes"] if node["kind"] == "document"}
    contradict = [edge for edge in graph["edges"] if edge["relation"] == CONTRADICTS]
    supports = [edge for edge in graph["edges"] if edge["relation"] == SUPPORTS]

    assert documents == {"retry-limit-three.md", "retry-limit-five.md"}
    assert contradict
    assert len(supports) >= 2
    assert {node["kind"] for node in graph["nodes"]} >= {"claim", "passage", "document", "version"}


def test_judge_graph_records_supporting_and_opposing_passages(tmp_path: Path) -> None:
    services = build_services(
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
    for name in ("retry-limit-three.md", "retry-limit-five.md"):
        path = ROOT / "evaluation" / "fixtures" / name
        services.ingestion.ingest(name, path.read_bytes())
    adapter = ClaimJudgeAdapter(
        rag=KnowledgeRagClient(services.search),
        judge=LocalClaimJudge(),
        metadata=services.metadata,
    )
    verdict = adapter.judge_claim(
        "The retry limit is three attempts.",
        SearchFilters(user_id="local"),
        4,
    )
    graph = verdict.graph
    relations = {edge.relation for edge in graph.edges}
    claim = next(node for node in graph.nodes.values() if node.kind == "claim")

    assert verdict.contradiction is True
    assert SUPPORTS in relations
    assert CONTRADICTS in relations
    assert graph.neighbors(claim.id, "cited_in")
    engine = ProvenanceEngine(services.metadata)
    rebuilt = engine.from_verdict(verdict)
    assert {node.kind for node in rebuilt.nodes.values()} >= {"claim", "passage", "document", "version", "source"}

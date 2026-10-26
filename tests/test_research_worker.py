"""The research worker only runs approved tool adapters."""

from pathlib import Path

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.research.contracts import REVERIFY, SEARCH, SYNTHESIZE, ActionRequest
from app.research.orchestrator import SearchBudget
from app.research.rag import KnowledgeRagClient, SearchFilters
from app.research.tools import ToolContext, approved_tools
from app.research.worker import ResearchWorker
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_worker_uses_mock_adapters_until_real_ones_are_registered() -> None:
    worker = ResearchWorker()
    search = worker.dispatch(ActionRequest(SEARCH, query="retry limit"))
    verify = worker.dispatch(ActionRequest(REVERIFY, claim="the retry limit is three"))
    report = worker.dispatch(ActionRequest(SYNTHESIZE, query="The retry limit is three. [1]"))
    cycle = worker.cycle("what is the retry limit")

    assert worker.tools() == ()
    assert worker.actions() == ("fill_gaps", "resolve_contradictions", "reverify", "search", "synthesize")
    assert search.ok is True
    assert search.evidence[0].source == "notes.md"
    assert verify.verdicts[0].verdict == "supported"
    assert report.report == "The retry limit is three. [1]"
    assert worker.state.report == report.report
    assert SEARCH in cycle.steps
    assert REVERIFY in cycle.steps
    assert SYNTHESIZE in cycle.steps
    assert cycle.graph is None


def test_worker_rejects_an_unknown_tool(tmp_path: Path) -> None:
    worker, _ = _worker(tmp_path)
    result = worker.execute("web", "retry limit", ToolContext(user_id="local"))

    assert result.ok is False
    assert result.error == "Unknown tool: web."
    assert worker.tools() == ("catalog", "graph", "knowledge")


def test_catalog_and_graph_adapters_stay_inside_visible_documents(tmp_path: Path) -> None:
    worker, services = _worker(tmp_path, knowledge_graph=True)
    services.ingestion.ingest(
        "calls.md",
        b"alpha/retry.py calls beta/client.py before it returns.\n",
        owner="ada",
    )
    services.ingestion.ingest(
        "secret.md",
        b"The canary: violet-heron-1942 must stay private.\n",
        owner="grace",
    )
    catalog = worker.execute("catalog", "which documents", ToolContext(user_id="ada"))
    graph = worker.execute("graph", "what does alpha/retry.py call", ToolContext(user_id="ada"))
    hidden = worker.execute("catalog", "which documents", ToolContext(user_id="grace"))

    assert catalog.ok is True
    assert catalog.names == ("calls.md",)
    assert "violet-heron-1942" not in catalog.names
    assert graph.names == ("beta/client.py",)
    assert hidden.names == ("secret.md",)


def test_knowledge_adapter_charges_the_search_budget(tmp_path: Path) -> None:
    worker, services = _worker(tmp_path)
    services.ingestion.ingest(
        "api-errors.md",
        (ROOT / "samples" / "api-errors.md").read_bytes(),
    )
    budget = SearchBudget(1)
    context = ToolContext(
        user_id="local",
        top_k=3,
        filters=SearchFilters(user_id="local"),
        budget=budget,
    )
    found = worker.execute("knowledge", "which file implements the retry logic", context)
    denied = worker.execute("knowledge", "where are timeouts configured", context)

    assert found.ok is True
    assert found.search is not None
    assert found.search.passages
    assert found.search.passages[0].source.filename == "api-errors.md"
    assert denied.ok is False
    assert denied.error == "Search budget exhausted."
    assert budget.remaining() == 0


def _worker(tmp_path: Path, *, knowledge_graph: bool = False):
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
            knowledge_graph=knowledge_graph,
        ),
        embedder=LocalEmbedder(dimension=128),
    )
    rag = KnowledgeRagClient(services.search)
    return ResearchWorker(approved_tools(rag=rag, metadata=services.metadata)), services

"""The research agent searches the knowledge collection through one contract."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.answerers import LocalAnswerer
from app.ingestion.embeddings import LocalEmbedder
from app.ingestion.errors import KnowledgeError
from app.observability.cache import LruCache
from app.observability.metrics import Metrics
from app.observability.tracing import TraceLog
from app.research.planner import CATALOG_COMPLETION, KNOWLEDGE_COMPLETION, build_plan
from app.research.rag import KnowledgeRagClient, RagMalformed, RagTimeout, SearchFilters
from app.research.service import ResearchReport, ResearchService
from app.retrieval.search import RankedPassage, RetrievalResult, SourceMetadata
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_plan_assigns_independent_tasks_and_completion_criteria() -> None:
    plan = build_plan(
        "which file implements the retry logic and where are timeouts configured",
        budget=16,
    )
    longer = build_plan(
        "alpha beta and gamma delta and epsilon zeta and eta theta and iota kappa",
        budget=4,
    )
    catalog = build_plan("which documents mention retry", budget=4)

    assert [task.task_id for task in plan.tasks] == ["q1", "q2"]
    assert plan.tasks[0].question != plan.tasks[1].question
    assert {task.completion for task in plan.tasks} == {KNOWLEDGE_COMPLETION}
    assert plan.tasks[0].tool == "knowledge"
    assert plan.budget == 16
    assert plan.deferred == ()
    assert len(longer.tasks) == 4
    assert len(longer.deferred) == 1
    assert catalog.catalog_completion == CATALOG_COMPLETION


def test_search_returns_ranked_passages_and_honors_filters(tmp_path: Path) -> None:
    services = _services(tmp_path)
    retry, _ = services.ingestion.ingest(
        "retry.md",
        b"# Retry\n\nThe file that implements retry logic is app/http/retry.py.\n",
    )
    services.ingestion.ingest(
        "database.md",
        b"# Database\n\nThe database stores migration history in sqlite.\n",
    )
    rag = KnowledgeRagClient(services.search)

    found = rag.search(
        "which file implements the retry logic",
        SearchFilters(document_ids=(retry.id,)),
        3,
    )
    empty = rag.search("retry logic", SearchFilters(document_ids=()), 3)

    assert found.passages
    assert {passage.document_id for passage in found.passages} == {retry.id}
    passage = found.passages[0]
    assert passage.score >= 0
    assert passage.source.filename == "retry.md"
    assert passage.source.start_char <= passage.source.end_char
    assert passage.source.chunk_id
    assert empty.passages == ()
    assert empty.sufficient_evidence is False


def test_research_stores_evidence_for_a_successful_search(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    client.post("/api/documents", files={"file": ("api-errors.md", payload, "text/markdown")})
    body = client.post(
        "/api/research",
        json={"question": "which file implements the retry logic", "top_k": 3},
    ).json()
    record = body["evidence"][0]
    link = body["links"][0]

    assert body["action"] == "research"
    assert body["subquestions"][0]["complete"] is True
    assert body["subquestions"][0]["completion"] == KNOWLEDGE_COMPLETION
    assert record["evidence_id"]
    assert record["claim"]
    assert record["document_id"]
    assert record["created_at"]
    assert record["start_char"] <= record["end_char"]
    assert "retry.py" in record["claim"]
    assert link == {
        "claim_id": record["claim_id"],
        "passage_id": record["passage_id"],
        "evidence_id": record["evidence_id"],
    }


def test_research_reports_no_results_conflict_and_insufficient_evidence(client: TestClient) -> None:
    missing = client.post(
        "/api/research",
        json={"question": "which file implements the retry logic", "top_k": 3},
    ).json()
    for name in ("retry-limit-three.md", "retry-limit-five.md"):
        payload = (ROOT / "evaluation" / "fixtures" / name).read_bytes()
        uploaded = client.post(
            "/api/documents",
            files={"file": (name, payload, "text/markdown")},
        )
        assert uploaded.status_code == 201
    conflict = client.post(
        "/api/research",
        json={"question": "what is the retry limit", "top_k": 4},
    ).json()
    client.post(
        "/api/documents",
        files={
            "file": (
                "api-errors.md",
                (ROOT / "samples" / "api-errors.md").read_bytes(),
                "text/markdown",
            )
        },
    )
    insufficient = client.post(
        "/api/research",
        json={"question": "what is the office wifi password", "top_k": 3},
    ).json()

    assert missing["action"] == "refuse"
    assert missing["evidence"] == []
    assert missing["links"] == []
    assert missing["subquestions"][0]["complete"] is False
    assert missing["gaps"]
    assert conflict["conflict"] is True
    assert {record["filename"] for record in conflict["evidence"]} == {
        "retry-limit-three.md",
        "retry-limit-five.md",
    }
    assert len(conflict["links"]) == len(conflict["evidence"])
    assert insufficient["sufficient_evidence"] is False
    assert insufficient["evidence"] == []
    assert any("wifi" in gap for gap in insufficient["gaps"])


def test_malformed_and_timeout_responses_become_gaps(tmp_path: Path) -> None:
    services = _services(tmp_path)
    malformed = _research(services, KnowledgeRagClient(_BrokenRetrieval()))
    timed_out = _research(services, KnowledgeRagClient(_EmptyRetrieval(), timeout=0))

    malformed_report = malformed.research("what is the retry limit", top_k=3)
    timeout_report = timed_out.research("what is the retry limit", top_k=3)

    assert malformed_report.action == "refuse"
    assert malformed_report.subquestions[0].evidence == "Malformed retrieval response."
    assert malformed_report.evidence_records == ()
    assert timed_out_gap(timeout_report)
    with pytest.raises(RagMalformed):
        KnowledgeRagClient(_BrokenRetrieval()).search("retry limit", SearchFilters(), 3)
    with pytest.raises(RagTimeout):
        KnowledgeRagClient(_EmptyRetrieval(), timeout=0).search("retry limit", SearchFilters(), 3)


def test_retry_stays_inside_the_search_budget(tmp_path: Path) -> None:
    services = _services(tmp_path)
    services.ingestion.ingest(
        "api-errors.md",
        (ROOT / "samples" / "api-errors.md").read_bytes(),
    )
    research = _research(services, budget=1)
    report = research.research("please tell me something about app/http/retry.py now", top_k=3)
    actions = [step["action"] for step in research._traces.get(report.trace_id)["steps"]]

    assert "search" in actions
    assert "retry" not in actions
    assert report.subquestions[0].complete is False


def test_retrieval_failure_still_raises(tmp_path: Path) -> None:
    services = _services(tmp_path)
    research = _research(services, KnowledgeRagClient(_DownRetrieval()))

    with pytest.raises(KnowledgeError, match="Retrieval failed.") as raised:
        research.research("what is the retry limit", top_k=3)

    assert raised.value.status_code == 503


def timed_out_gap(report: ResearchReport) -> bool:
    return (
        report.action == "refuse"
        and report.subquestions[0].evidence == "Retrieval timed out."
        and report.evidence_records == ()
    )


def _research(services, rag=None, *, budget: int = 4) -> ResearchService:
    return ResearchService(
        retrieval=services.search,
        answerer=LocalAnswerer(),
        metadata=services.metadata,
        cache=LruCache(8),
        metrics=Metrics(),
        traces=TraceLog(),
        rag=rag,
        search_budget=budget,
    )


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


class _BrokenRetrieval:
    def retrieve(self, query: str, top_k: int = 4, user_id: str | None = None) -> RetrievalResult:
        source = SourceMetadata(
            chunk_id="chunk",
            document_id="",
            filename="notes.md",
            content_type="text/markdown",
            chunk_index=0,
            start_char=0,
            end_char=4,
            page=None,
            heading=None,
        )
        return RetrievalResult(
            query=query,
            sufficient_evidence=True,
            inspection="",
            passages=[
                RankedPassage(
                    rank=1,
                    text="note",
                    score=0.4,
                    vector_score=0.4,
                    lexical_score=0.4,
                    answers_question=True,
                    matched_terms=("note",),
                    source=source,
                )
            ],
        )


class _EmptyRetrieval:
    def retrieve(self, query: str, top_k: int = 4, user_id: str | None = None) -> RetrievalResult:
        return RetrievalResult(query=query, sufficient_evidence=False, inspection="", passages=[])


class _DownRetrieval:
    def retrieve(self, query: str, top_k: int = 4, user_id: str | None = None) -> RetrievalResult:
        raise KnowledgeError("Retrieval failed.", status_code=503)

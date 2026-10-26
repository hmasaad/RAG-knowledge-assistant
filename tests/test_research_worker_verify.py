"""Verification is plugged into the worker only when those adapters are registered."""

from pathlib import Path

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.research.adapters import compose_action_adapters
from app.research.contracts import (
    REVERIFY,
    RESOLVE_CONTRADICTIONS,
    SEARCH,
    ActionRequest,
    EvidenceRef,
    ResearchState,
)
from app.research.mocks import MockAdapter
from app.research.rag import KnowledgeRagClient
from app.research.tools import approved_tools
from app.research.verify_actions import ResolveConflictsAction, VerifyAction
from app.research.worker import ResearchWorker
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_verification_does_not_run_until_it_is_registered() -> None:
    worker = ResearchWorker()
    state = ResearchState()
    worker.dispatch(ActionRequest(SEARCH, query="retry limit"), state)
    verify = worker.dispatch(ActionRequest(REVERIFY, claim="what is the office wifi password"), state)

    assert verify.verdicts[0].verdict == "supported"


def test_verify_action_scores_existing_evidence_without_searching() -> None:
    searches = MockAdapter(
        SEARCH,
        evidence=(
            EvidenceRef(
                "e1",
                "d1",
                "api-errors.md",
                "The file that implements retry logic is app/http/retry.py.",
            ),
        ),
    )
    worker = ResearchWorker(
        adapters=[searches, VerifyAction(), ResolveConflictsAction()],
    )
    state = ResearchState()
    worker.dispatch(ActionRequest(SEARCH, query="retry logic"), state)
    supported = worker.dispatch(
        ActionRequest(REVERIFY, claim="which file implements the retry logic"),
        state,
    )
    missing = worker.dispatch(
        ActionRequest(REVERIFY, claim="what is the office wifi password"),
        state,
    )

    assert searches.calls
    assert supported.verdicts[0].verdict == "supported"
    assert supported.verdicts[0].evidence_ids == ("e1",)
    assert missing.verdicts[0].verdict == "insufficient"
    assert missing.gaps == ("what is the office wifi password",)
    assert len(searches.calls) == 1


def test_resolve_conflicts_reads_state_not_the_index(tmp_path: Path) -> None:
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
    registry = approved_tools(rag=KnowledgeRagClient(services.search), metadata=services.metadata)
    worker = ResearchWorker(
        registry,
        adapters=compose_action_adapters(registry=registry, use_verification=True),
    )
    state = ResearchState()
    state.evidence.extend(
        [
            EvidenceRef("a", "d1", "retry-limit-three.md", "The retry limit is 3 attempts."),
            EvidenceRef("b", "d2", "retry-limit-five.md", "The retry limit is 5 attempts."),
        ]
    )
    found = worker.dispatch(ActionRequest(RESOLVE_CONTRADICTIONS, query="what is the retry limit"), state)

    assert found.ok is True
    assert found.contradictions
    assert "3" in found.contradictions[0]
    assert "5" in found.contradictions[0]
    assert services.metadata.list_documents() == []

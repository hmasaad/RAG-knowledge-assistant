"""The research DAG orders work, recovers interrupted nodes, and honors deadlines."""

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings
from app.ingestion.embeddings import LocalEmbedder
from app.research.contracts import SEARCH
from app.research.dag import (
    DONE,
    PENDING,
    RUNNING,
    DagTask,
    ResearchDag,
    build_search_dag,
)
from app.research.pool import BoundedWorkerPool
from app.wiring import build_services

ROOT = Path(__file__).resolve().parents[1]


def test_search_dag_keeps_reverify_behind_its_search() -> None:
    dag = build_search_dag(
        "which file implements the retry logic and where are timeouts configured"
    )
    first = dag.next_ready()

    assert first is not None
    assert first.action == SEARCH
    assert first.id == "search:0"
    assert [task.id for task in dag.ready()] == ["search:0"]

    dag.mark_done("search:0")
    assert dag.next_ready() is not None
    assert dag.next_ready().id == "reverify:0"


def test_dag_recovers_running_nodes_and_skips_finished_work() -> None:
    dag = ResearchDag()
    dag.add(DagTask("search:0", SEARCH, query="retry"))
    dag.add(DagTask("reverify:0", "reverify", depends_on=("search:0",), claim="retry"))
    dag.mark_running("search:0")

    reset = dag.recover()
    payload = dag.to_payload()
    restored = ResearchDag.from_payload(payload)

    assert reset == 1
    assert dag.tasks["search:0"].status == PENDING
    assert restored.next_ready() is not None
    assert restored.next_ready().id == "search:0"
    restored.mark_done("search:0")
    assert restored.next_ready().id == "reverify:0"
    assert restored.tasks["search:0"].status == DONE


def test_bounded_pool_never_exceeds_its_worker_cap() -> None:
    pool = BoundedWorkerPool(2)
    filled = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    started = 0

    def hold(_item: int) -> int:
        nonlocal started
        with lock:
            started += 1
            if started >= 2:
                filled.set()
        release.wait(timeout=2)
        return _item

    worker = threading.Thread(target=lambda: pool.map((1, 2, 3), hold))
    worker.start()
    assert filled.wait(timeout=2)
    in_flight = pool.in_flight
    release.set()
    worker.join(timeout=2)

    assert in_flight == 2
    assert pool.peak == 2
    assert pool.in_flight == 0
    assert worker.is_alive() is False


def test_deadline_expires_a_queued_session_before_claim(tmp_path: Path) -> None:
    settings = Settings(
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
        indexing_mode="background",
        worker_in_process=False,
        job_timeout_seconds=900,
    )
    services = build_services(settings, embedder=LocalEmbedder(dimension=128))
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    session = services.research._sessions.create(
        user_id="local",
        question="which file implements the retry logic",
        top_k=3,
        deadline=past,
    )
    claimed = services.research._jobs.claim_batch()
    stored = services.research._sessions.get(session.id)

    assert claimed == []
    assert stored is not None
    assert stored.status == "failed"
    assert stored.error == "Deadline exceeded."


def test_cycle_stops_when_the_deadline_has_passed() -> None:
    from app.research.cycle import CycleDeadline, run_cycle
    from app.research.dispatcher import ActionDispatcher
    from app.research.mocks import mock_action_adapters

    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    try:
        run_cycle(
            ActionDispatcher(mock_action_adapters()),
            "which file implements the retry logic",
            deadline=past,
        )
    except CycleDeadline as expired:
        assert expired.result.steps == ()
    else:
        raise AssertionError("Expected the cycle to stop at the deadline.")

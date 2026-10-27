"""Coordinate concurrent long-running research sessions."""

from collections.abc import Callable
from datetime import datetime, timezone

from app.observability.logging import log_event
from app.observability.metrics import Metrics
from app.observability.tracing import TraceLog
from app.research.pool import BoundedWorkerPool
from app.research.sessions import ResearchSession
from app.research.state_manager import DurableResearchStateManager


class ProductionResearchOrchestrator:
    """Claim, run, and cancel research jobs without duplicating work.

    One worker process can hold several leases at once. Users are capped so
    one caller cannot fill the queue. Priority decides which queued session
    starts next. A failed session is marked failed and the rest of the batch
    still runs.
    """

    def __init__(
        self,
        manager: DurableResearchStateManager,
        run_session: Callable[..., object],
        *,
        concurrency: int = 2,
        user_concurrency: int = 2,
        worker_id: str = "worker",
        metrics: Metrics | None = None,
        traces: TraceLog | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("Worker concurrency must be at least 1.")
        if user_concurrency < 1:
            raise ValueError("User concurrency must be at least 1.")
        self.manager = manager
        self._run_session = run_session
        self.concurrency = concurrency
        self.user_concurrency = user_concurrency
        self.worker_id = worker_id
        self.metrics = metrics
        self.traces = traces
        self.pool = BoundedWorkerPool(concurrency)

    def claim_batch(self) -> list[ResearchSession]:
        """Mark up to ``concurrency`` queued sessions as running."""
        self.manager.recover()
        claimed: list[ResearchSession] = []
        for _ in range(self.concurrency):
            session = self.manager.claim(
                owner=self.worker_id,
                user_concurrency=self.user_concurrency,
            )
            if session is None:
                break
            claimed.append(session)
            self._record_claim(session)
        return claimed

    def process_available(self) -> int:
        """Run every session this worker just claimed. Returns how many ran."""
        claimed = self.claim_batch()
        self.pool.map(claimed, self._run_claimed)
        return len(claimed)

    def process_next(self) -> bool:
        """Run one queued session. False when nothing is eligible."""
        self.manager.recover()
        session = self.manager.claim(
            owner=self.worker_id,
            user_concurrency=self.user_concurrency,
        )
        if session is None:
            return False
        self._record_claim(session)
        self._run_claimed(session)
        return True

    def _run_claimed(self, session: ResearchSession) -> None:
        try:
            self._run_session(session.id, claimed=True)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            stored = self.manager.get(session.id)
            if stored is not None and stored.status == "running":
                self.manager.fail(session.id, error)
            self._record_finish(session, "failed", error=error)
            return
        stored = self.manager.get(session.id)
        self._record_finish(session, stored.status if stored is not None else "completed")

    def _record_claim(self, session: ResearchSession) -> None:
        wait = _queue_wait_seconds(session)
        if self.metrics is not None:
            self.metrics.inc(
                "rag_research_jobs_claimed_total",
                priority=str(session.priority),
            )
            self.metrics.observe(
                "rag_research_jobs_queue_wait_seconds",
                wait,
                priority=str(session.priority),
            )
        if self.traces is not None and session.trace_id:
            self.traces.start(
                trace_id=session.trace_id,
                session_id=session.id,
                user_id=session.user_id,
                question=session.question,
            )
            self.traces.step(
                session.trace_id,
                action="claim",
                tool="orchestrator",
                status="running",
                duration_ms=round(wait * 1000, 2),
                kind="orchestrator",
            )
        log_event(
            "rag.research",
            "research_claim",
            session_id=session.id,
            trace_id=session.trace_id,
            user_id=session.user_id,
            priority=session.priority,
            queue_wait_ms=round(wait * 1000, 2),
            worker=self.worker_id,
        )

    def _record_finish(self, session: ResearchSession, status: str, *, error: str = "") -> None:
        if self.metrics is not None:
            self.metrics.inc("rag_research_jobs_finished_total", status=status)
        if self.traces is not None and session.trace_id and error:
            self.traces.step(
                session.trace_id,
                action="fail",
                tool="orchestrator",
                status="failed",
                error=error,
                kind="orchestrator",
            )
        log_event(
            "rag.research",
            "research_job",
            session_id=session.id,
            trace_id=session.trace_id,
            status=status,
            error=error,
            worker=self.worker_id,
        )


def _queue_wait_seconds(session: ResearchSession) -> float:
    raw = getattr(session, "created_at", "") or ""
    try:
        created = datetime.fromisoformat(raw)
    except ValueError:
        return 0.0
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - created).total_seconds())

"""Coordinate concurrent long-running research sessions."""

from collections.abc import Callable

from app.research.sessions import ResearchSession
from app.research.state_manager import DurableResearchStateManager


class ProductionResearchOrchestrator:
    """Claim, run, and cancel research jobs without duplicating work.

    One worker process can hold several leases at once. Users are capped so
    one caller cannot fill the queue. Priority decides which queued session
    starts next.
    """

    def __init__(
        self,
        manager: DurableResearchStateManager,
        run_session: Callable[..., object],
        *,
        concurrency: int = 2,
        user_concurrency: int = 2,
        worker_id: str = "worker",
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
        return claimed

    def process_available(self) -> int:
        """Run every session this worker just claimed. Returns how many ran."""
        claimed = self.claim_batch()
        for session in claimed:
            self._run_session(session.id, claimed=True)
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
        self._run_session(session.id, claimed=True)
        return True

"""Durable checkpoints for a research cycle.

The worker and adapters stay in memory. This manager is what survives a
crash, a timeout, a retry, or an application restart: it writes each
finished action and the session snapshot in one transaction, holds a lease
while a worker is alive, and restores executed fingerprints so the same
search is not run again.
"""

from app.research.contracts import ActionOutcome, ActionRequest, ResearchState
from app.research.sessions import ResearchSession, ResearchSessionStore


class DurableResearchStateManager:
    """Persist, lease, and restore research progress."""

    def __init__(self, store: ResearchSessionStore, *, lease_seconds: float = 900) -> None:
        if lease_seconds <= 0:
            raise ValueError("Lease duration must be positive.")
        self.store = store
        self.lease_seconds = lease_seconds

    def create(
        self,
        *,
        user_id: str,
        question: str,
        top_k: int,
        trace_id: str = "",
        status: str = "queued",
        priority: int = 0,
    ) -> ResearchSession:
        return self.store.create(
            user_id=user_id,
            question=question,
            top_k=top_k,
            trace_id=trace_id,
            status=status,
            priority=priority,
        )

    def get(self, session_id: str) -> ResearchSession | None:
        return self.store.get(session_id)

    def restore(self, session_id: str) -> ResearchState:
        """Load the last checkpoint. A missing session starts empty."""
        session = self.store.get(session_id)
        if session is None or session.state is None:
            return ResearchState()
        return session.state

    def already_done(self, session_id: str, request: ActionRequest) -> bool:
        """True when this action was checkpointed. Safe to skip the adapter."""
        if self.store.has_action(session_id, request):
            return True
        state = self.restore(session_id)
        return state.fingerprint(request) in state.executed

    def checkpoint(
        self,
        session_id: str,
        state: ResearchState,
        request: ActionRequest | None = None,
        outcome: ActionOutcome | None = None,
    ) -> None:
        """Save progress. When request and outcome are present they are logged once."""
        self.store.persist_step(session_id, state, request, outcome)

    def acquire(self, session_id: str, *, owner: str = "worker") -> ResearchSession | None:
        self.store.mark_running(session_id, owner=owner, lease_seconds=self.lease_seconds)
        return self.store.get(session_id)

    def heartbeat(self, session_id: str, *, owner: str = "worker") -> None:
        self.store.heartbeat(session_id, owner=owner, seconds=self.lease_seconds)

    def recover(self) -> int:
        """Requeue sessions whose lease expired. Does not claim the next job."""
        return self.store.requeue_expired(self.lease_seconds)

    def halt(self, session_id: str, result: dict, state: ResearchState) -> None:
        self.store.halt(session_id, result, state)

    def side_effects(self, session_id: str) -> list[dict]:
        return self.store.list_effects(session_id)

    def complete(self, session_id: str, result: dict, state: ResearchState) -> None:
        self.store.complete(session_id, result, state)

    def fail(self, session_id: str, error: str, state: ResearchState | None = None) -> None:
        self.store.fail(session_id, error, state)

    def requeue(self, session_id: str) -> ResearchSession | None:
        return self.store.requeue(session_id)

    def cancel(self, session_id: str) -> ResearchSession | None:
        return self.store.cancel(session_id)

    def finish_cancelled(self, session_id: str, result: dict, state: ResearchState) -> None:
        self.store.finish_cancelled(session_id, result, state)

    def claim(
        self,
        *,
        owner: str = "worker",
        user_concurrency: int = 8,
    ) -> ResearchSession | None:
        return self.store.claim(
            self.lease_seconds,
            owner=owner,
            user_concurrency=user_concurrency,
        )

    def job_counts(self) -> dict[str, int]:
        return self.store.job_counts()

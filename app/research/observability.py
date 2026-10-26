"""Record research latency, cost, retries, and failure patterns."""

import time

from app.observability.logging import log_event
from app.observability.metrics import Metrics


def failure_kind(error: str) -> str:
    text = (error or "").casefold()
    if "budget" in text:
        return "budget"
    if "iteration" in text:
        return "iteration"
    if "permitted" in text or "caller" in text:
        return "permission"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "unknown" in text or "invalid" in text:
        return "invalid"
    if "already executed" in text:
        return "duplicate"
    if error:
        return "error"
    return "ok"


class ResearchObserver:
    """Write research timings and counters onto the shared metrics collector."""

    def __init__(self, metrics: Metrics | None = None) -> None:
        self.metrics = metrics

    def record_action(
        self,
        action: str,
        *,
        ok: bool,
        cost: int = 0,
        seconds: float = 0.0,
        error: str = "",
        kind: str = "action",
    ) -> None:
        status = "ok" if ok else "error"
        if self.metrics is not None:
            self.metrics.observe(
                "rag_research_action_seconds",
                seconds,
                action=action,
                kind=kind,
                status=status,
            )
            if cost:
                self.metrics.inc("rag_research_cost_total", float(cost), action=action, kind=kind)
            if not ok:
                self.metrics.inc(
                    "rag_research_failures_total",
                    action=action,
                    kind=kind,
                    error=failure_kind(error),
                )
        log_event(
            "rag.research",
            "research_action",
            action=action,
            kind=kind,
            ok=ok,
            cost=cost,
            duration_ms=round(seconds * 1000, 2),
            error=error,
        )

    def record_retry(self, kind: str) -> None:
        if self.metrics is not None:
            self.metrics.inc("rag_research_retries_total", kind=kind)
        log_event("rag.research", "research_retry", kind=kind)

    def record_request(
        self,
        path: str,
        *,
        seconds: float,
        cost: int = 0,
        status: str = "completed",
    ) -> None:
        if self.metrics is not None:
            self.metrics.observe("rag_research_seconds", seconds, path=path, status=status)
            if cost:
                self.metrics.inc("rag_research_cost_total", float(cost), action=path, kind="request")
        log_event(
            "rag.research",
            "research_request",
            path=path,
            status=status,
            cost=cost,
            duration_ms=round(seconds * 1000, 2),
        )


class timed:
    """Measure elapsed seconds for one action."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def __enter__(self) -> "timed":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.seconds = time.perf_counter() - self._started

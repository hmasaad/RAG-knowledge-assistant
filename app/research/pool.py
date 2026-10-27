"""A worker pool that never starts more tasks than its bound."""

import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


class BoundedWorkerPool:
    """Run callables with a hard cap on in-flight work."""

    def __init__(self, max_workers: int = 2) -> None:
        if max_workers < 1:
            raise ValueError("Worker pool size must be at least 1.")
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._in_flight = 0
        self._peak = 0

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def peak(self) -> int:
        with self._lock:
            return self._peak

    def map(self, items: Iterable[T], fn: Callable[[T], R]) -> list[R]:
        rows = list(items)
        if not rows:
            return []
        if self.max_workers == 1 or len(rows) == 1:
            return [self._run(fn, item) for item in rows]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._run, fn, item) for item in rows]
            return [future.result() for future in futures]

    def _run(self, fn: Callable[[T], R], item: T) -> R:
        with self._lock:
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        try:
            return fn(item)
        finally:
            with self._lock:
                self._in_flight -= 1

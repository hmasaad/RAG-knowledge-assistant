"""In-memory traces for one answer or recovery."""

import threading
import uuid
from collections import OrderedDict
from typing import Any


class TraceLog:
    """Keep the latest traces so a caller can read the steps for an id."""

    def __init__(self, limit: int = 200) -> None:
        self._limit = limit
        self._lock = threading.Lock()
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def start(self, **fields: Any) -> str:
        trace_id = uuid.uuid4().hex[:16]
        with self._lock:
            self._traces[trace_id] = {"id": trace_id, "steps": [], **fields}
            self._traces.move_to_end(trace_id)
            while len(self._traces) > self._limit:
                self._traces.popitem(last=False)
        return trace_id

    def step(self, trace_id: str, **fields: Any) -> None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return
            trace["steps"].append(dict(fields))

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return None
            return {
                "id": trace["id"],
                "steps": [dict(step) for step in trace["steps"]],
            }

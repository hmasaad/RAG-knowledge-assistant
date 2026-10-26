"""In-process counters and timings exposed on ``/metrics``."""

import threading
from typing import Any


class Metrics:
    """Small Prometheus text collector.

    Gauges are supplied at scrape time from the stores. Counters and timings
    live for the life of the process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._observations: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, int]] = {}
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            total, count = self._observations.get(key, (0.0, 0))
            self._observations[key] = (total + value, count + 1)

    def value(self, name: str, **labels: str) -> float:
        key = _key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def summary(self, name: str, **labels: str) -> tuple[float, int]:
        key = _key(name, labels)
        with self._lock:
            return self._observations.get(key, (0.0, 0))

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        key = _key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def clear_gauges(self) -> None:
        with self._lock:
            self._gauges.clear()

    def render(self, gauges: dict[str, float] | None = None) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            observations = dict(self._observations)
            stored = dict(self._gauges)
        if gauges:
            for name, value in gauges.items():
                stored[_key(name, {})] = value
        gauge_names = sorted({key[0] for key in stored})
        for name in gauge_names:
            lines.append(f"# TYPE {name} gauge")
            for key, value in sorted(stored.items()):
                if key[0] == name:
                    lines.append(f"{_format(key)} {value}")
        names = sorted({key[0] for key in counters})
        for name in names:
            lines.append(f"# TYPE {name} counter")
            for key, value in sorted(counters.items()):
                if key[0] == name:
                    lines.append(f"{_format(key)} {value}")
        names = sorted({key[0] for key in observations})
        for name in names:
            lines.append(f"# TYPE {name} summary")
            for key, (total, count) in sorted(observations.items()):
                if key[0] != name:
                    continue
                lines.append(f"{_format(key, suffix='_sum')} {total}")
                lines.append(f"{_format(key, suffix='_count')} {count}")
        return "\n".join(lines) + "\n"


def _key(name: str, labels: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    pairs = tuple(sorted((str(label), str(value)) for label, value in labels.items()))
    return name, pairs


def _format(key: tuple[str, tuple[tuple[str, str], ...]], *, suffix: str = "") -> str:
    name, labels = key
    if not labels:
        return f"{name}{suffix}"
    rendered = ",".join(f'{label}="{_escape(value)}"' for label, value in labels)
    return f"{name}{suffix}{{{rendered}}}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

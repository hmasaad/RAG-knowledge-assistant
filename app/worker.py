"""Background indexer.

Run beside the API when ``INDEXING_MODE=background`` and
``WORKER_IN_PROCESS=false``:

    python -m app.worker

The API and this process must share ``SQLITE_PATH``, ``BLOB_PATH``, and the
vector store. Use ``CHROMA_HOST`` when more than one process opens the vectors.
"""

import signal
import threading

from app.config import get_settings
from app.observability.logging import configure_logging, log_event
from app.wiring import Services, build_services


def process_forever(services: Services, stop: threading.Event) -> None:
    """Index queued documents and run queued research cycles until ``stop`` is set."""
    poll = services.settings.worker_poll_seconds
    while not stop.is_set():
        try:
            indexed = services.ingestion.process_next_job()
            researched = services.research.process_available()
            worked = indexed or researched > 0
        except Exception as exc:
            log_event("rag.worker", "worker_error", error=type(exc).__name__)
            worked = False
        if stop.is_set():
            return
        if not worked:
            stop.wait(poll)


def start_worker(services: Services) -> tuple[threading.Event, threading.Thread]:
    """Start a daemon thread that drains the index queue."""
    stop = threading.Event()
    thread = threading.Thread(
        target=process_forever,
        args=(services, stop),
        name="index-worker",
        daemon=True,
    )
    thread.start()
    return stop, thread


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    services = build_services(settings)
    stop = threading.Event()

    def _stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    log_event("rag.worker", "worker_started", indexing_mode=settings.indexing_mode)
    process_forever(services, stop)
    log_event("rag.worker", "worker_stopped")


if __name__ == "__main__":
    main()

import json
import logging

from app.observability.logging import JsonFormatter, log_event


def test_json_formatter_includes_metric_fields() -> None:
    record = logging.LogRecord(
        name="rag.ingestion",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="document_indexed",
        args=(),
        exc_info=None,
    )
    record.duration_ms = 12.5
    record.chunk_count = 3

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "document_indexed"
    assert payload["logger"] == "rag.ingestion"
    assert payload["duration_ms"] == 12.5
    assert payload["chunk_count"] == 3
    assert "timestamp" in payload


def test_log_event_does_not_overwrite_reserved_filename(caplog) -> None:
    caplog.set_level(logging.INFO, logger="rag.ingestion")
    log_event("rag.ingestion", "document_indexed", filename="notes.pdf", chunk_count=2)

    assert "Attempt to overwrite" not in caplog.text
    assert caplog.records[0].event_filename == "notes.pdf"
    assert caplog.records[0].chunk_count == 2

import logging

from app.observability.logging import JsonFormatter


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

    payload = __import__("json").loads(JsonFormatter().format(record))

    assert payload["message"] == "document_indexed"
    assert payload["logger"] == "rag.ingestion"
    assert payload["duration_ms"] == 12.5
    assert payload["chunk_count"] == 3
    assert "timestamp" in payload

"""JSON logs for ingestion latency and retrieval metrics."""

import json
import logging
from datetime import datetime, timezone
from typing import Any

_STANDARD_ATTRS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Attach a JSON handler to the application logger once."""
    global _CONFIGURED
    logger = logging.getLogger("rag")
    logger.setLevel(level.upper())
    if not _CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True


def log_event(logger_name: str, message: str, **fields: Any) -> None:
    """Write one structured event. ``fields`` become JSON properties."""
    logging.getLogger(logger_name).info(message, extra=fields)

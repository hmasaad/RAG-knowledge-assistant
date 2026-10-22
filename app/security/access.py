"""Caller identity and document visibility."""

import re

from app.ingestion.errors import KnowledgeError

_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_VISIBILITY = {"private", "shared"}


def normalize_user_id(value: str | None, *, default: str) -> str:
    """Return a safe caller id. A missing header uses the configured default."""
    raw = value.strip() if value and value.strip() else default
    if not _USER_ID.fullmatch(raw):
        raise KnowledgeError("Invalid user id.", status_code=400)
    return raw


def normalize_visibility(value: str) -> str:
    cleaned = value.strip().lower()
    if cleaned not in _VISIBILITY:
        raise KnowledgeError("Visibility must be private or shared.")
    return cleaned

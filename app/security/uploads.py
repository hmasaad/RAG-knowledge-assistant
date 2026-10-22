"""Reject uploads that ingestion should not parse."""

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.errors import IngestionError

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".text"}


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    extension: str
    data: bytes


def validate_upload(
    filename: str | None,
    data: bytes,
    *,
    max_bytes: int,
) -> ValidatedUpload:
    """Return a safe filename and the raw bytes, or raise ``IngestionError``."""
    if filename is None or not str(filename).strip():
        raise IngestionError("A filename is required.")

    safe_name = Path(str(filename).replace("\\", "/")).name
    if (
        not safe_name
        or safe_name in {".", ".."}
        or safe_name.startswith(".")
        or "\x00" in safe_name
    ):
        raise IngestionError("Invalid filename.")

    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise IngestionError(
            "Unsupported file type. Upload a PDF, Markdown, or text file.",
            status_code=415,
        )
    if not data:
        raise IngestionError("The file is empty.")
    if len(data) > max_bytes:
        raise IngestionError(
            f"File exceeds the {max_bytes} byte upload limit.",
            status_code=413,
        )
    return ValidatedUpload(filename=safe_name, extension=extension, data=data)

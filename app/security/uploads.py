"""Reject uploads that ingestion should not parse."""

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.errors import IngestionError

TEXT_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
}
SOURCE_TYPES = {
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".jsx": "text/javascript",
    ".ts": "text/typescript",
    ".tsx": "text/typescript",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".java": "text/x-java",
    ".rb": "text/x-ruby",
    ".sql": "text/x-sql",
    ".sh": "text/x-shell",
}
ARCHITECTURE_TYPES = {
    ".adoc": "text/asciidoc",
    ".asciidoc": "text/asciidoc",
}
ALLOWED_EXTENSIONS = set(TEXT_TYPES) | set(SOURCE_TYPES) | set(ARCHITECTURE_TYPES) | {".pdf"}
UNSUPPORTED_TYPE = (
    "Unsupported file type. Upload a PDF, Markdown, text, AsciiDoc, or source file."
)


def content_type_for(extension: str) -> str | None:
    """Media type for a lowercase extension, including the leading dot."""
    if extension == ".pdf":
        return "application/pdf"
    for table in (TEXT_TYPES, SOURCE_TYPES, ARCHITECTURE_TYPES):
        if extension in table:
            return table[extension]
    return None


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
            UNSUPPORTED_TYPE,
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

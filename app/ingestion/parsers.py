"""Extract text from PDF, Markdown, text, AsciiDoc, and source uploads."""

import pymupdf

from app.ingestion.errors import IngestionError
from app.ingestion.models import ParsedDocument, ParsedPage
from app.security.uploads import UNSUPPORTED_TYPE, content_type_for


def parse_document(filename: str, data: bytes) -> ParsedDocument:
    """Extract text pages from an already validated upload."""
    extension = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    content_type = content_type_for(extension)
    if content_type is None:
        raise IngestionError(UNSUPPORTED_TYPE, status_code=415)
    if content_type == "application/pdf":
        return _parse_pdf(data)
    return ParsedDocument(
        content_type=content_type,
        pages=[ParsedPage(number=None, text=_decode_utf8(data))],
    )


def _decode_utf8(data: bytes) -> str:
    payload = data[3:] if data.startswith(b"\xef\xbb\xbf") else data
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError("Text and Markdown files must be UTF-8.") from exc


def _parse_pdf(data: bytes) -> ParsedDocument:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise IngestionError("Could not read the PDF.") from exc
    try:
        pages = [
            ParsedPage(number=index, text=page.get_text("text") or "")
            for index, page in enumerate(document, start=1)
        ]
    finally:
        document.close()
    return ParsedDocument(content_type="application/pdf", pages=pages)

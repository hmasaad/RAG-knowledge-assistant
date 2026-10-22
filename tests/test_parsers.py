import pymupdf
import pytest

from app.ingestion.errors import IngestionError
from app.ingestion.parsers import parse_document
from app.security.uploads import validate_upload


def test_markdown_and_text_types() -> None:
    markdown = parse_document("Guide.MD", b"\xef\xbb\xbf# Title\n")
    plain = parse_document("notes.txt", "plain words".encode())
    source = parse_document("client.py", b"def retry_request():\n    return 1\n")
    architecture = parse_document("platform.adoc", b"= Platform\n")

    assert markdown.content_type == "text/markdown"
    assert markdown.pages[0].text == "# Title\n"
    assert plain.content_type == "text/plain"
    assert plain.pages[0].number is None
    assert source.content_type == "text/x-python"
    assert architecture.content_type == "text/asciidoc"


def test_pdf_preserves_page_numbers() -> None:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Alpha page mentions zephyr-token-one.")
    second = document.new_page()
    second.insert_text((72, 72), "Beta page mentions quillon-token-two.")
    payload = document.tobytes()
    document.close()

    parsed = parse_document("spec.pdf", payload)

    assert parsed.content_type == "application/pdf"
    assert [page.number for page in parsed.pages] == [1, 2]
    assert "zephyr-token-one" in parsed.pages[0].text
    assert "quillon-token-two" in parsed.pages[1].text


def test_invalid_utf8_is_rejected() -> None:
    with pytest.raises(IngestionError, match="UTF-8"):
        parse_document("notes.txt", b"\xff")


def test_upload_validation() -> None:
    upload = validate_upload("../secret.md", b"hello", max_bytes=100)
    assert upload.filename == "secret.md"

    with pytest.raises(IngestionError, match="Unsupported") as unsupported:
        validate_upload("notes.docx", b"hello", max_bytes=100)
    assert unsupported.value.status_code == 415

    with pytest.raises(IngestionError, match="upload limit") as too_large:
        validate_upload("notes.md", b"hello", max_bytes=3)
    assert too_large.value.status_code == 413

    with pytest.raises(IngestionError, match="empty"):
        validate_upload("notes.md", b"", max_bytes=100)

    with pytest.raises(IngestionError, match="Invalid filename"):
        validate_upload(".env", b"secret", max_bytes=100)

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.ingestion.cleaning import clean_text


def test_clean_text_normalizes_whitespace() -> None:
    raw = "Line one  \r\n\r\n\r\nLine two\u00a0tail\x00\n"

    assert clean_text(raw) == "Line one\n\nLine two tail"


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError):
        Settings(chunk_size=100, chunk_overlap=100)

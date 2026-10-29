import pytest

from app.ingestion.chunking import build_chunks


def test_short_text_is_one_chunk() -> None:
    drafts = build_chunks(
        "Retry logic lives here.",
        content_type="text/plain",
        chunk_size=100,
        overlap=10,
    )

    assert len(drafts) == 1
    assert drafts[0].text == "Retry logic lives here."
    assert drafts[0].start == 0


def test_chunks_are_slices_with_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    drafts = build_chunks(text, content_type="text/plain", chunk_size=20, overlap=5)

    assert len(drafts) > 1
    assert all(len(draft.text) <= 20 for draft in drafts)
    assert all(text[draft.start : draft.end] == draft.text for draft in drafts)
    assert drafts[1].text.startswith(drafts[0].text[-5:])
    assert drafts[-1].end == len(text)


def test_prefers_paragraph_breaks() -> None:
    paragraph = "word " * 30
    text = paragraph.rstrip() + "\n\n" + paragraph.rstrip()
    drafts = build_chunks(text, content_type="text/plain", chunk_size=80, overlap=10)

    assert all(len(draft.text) <= 80 for draft in drafts)
    assert all(text[draft.start : draft.end] == draft.text for draft in drafts)
    assert drafts[-1].end == len(text)


def test_markdown_sections_keep_headings_and_ignore_fences() -> None:
    text = (
        "# Real\n\nintro\n\n```\n# Not a heading\n```\n\n"
        "## Retry logic\n\napp/http/retry.py\n"
    )
    drafts = build_chunks(text, content_type="text/markdown", chunk_size=500, overlap=20)

    headings = {draft.heading for draft in drafts}
    assert headings == {"Real", "Retry logic"}
    fenced = next(draft for draft in drafts if "Not a heading" in draft.text)
    assert fenced.heading == "Real"
    retry = next(draft for draft in drafts if "retry.py" in draft.text)
    assert retry.heading == "Retry logic"
    assert text[retry.start : retry.end] == retry.text


def test_pdf_keeps_mvp_scope_with_its_bullets() -> None:
    text = (
        "10. Notifications Push booking notifications. Payment details and invoices.\n"
        "11. MVP Scope\n"
        "1. Customer registration/login and profile.\n"
        "2. Salon discovery and search.\n"
        "10. Basic admin management.\n"
    )
    drafts = build_chunks(text, content_type="application/pdf", chunk_size=80, overlap=10)

    mvp = [draft for draft in drafts if draft.section_title == "MVP Scope"]
    assert mvp
    assert all(draft.section_id == "section_11" for draft in mvp)
    assert all(draft.heading == "11. MVP Scope" for draft in mvp)
    assert all("Payment details" not in draft.text for draft in mvp)
    assert any("Customer registration" in draft.text for draft in mvp)
    assert len({draft.section_id for draft in mvp}) == 1


def test_empty_text_yields_no_chunks() -> None:
    assert build_chunks("  \n\n  ", content_type="text/plain", chunk_size=100, overlap=10) == []


def test_overlap_is_validated() -> None:
    with pytest.raises(ValueError):
        build_chunks("hello", content_type="text/plain", chunk_size=10, overlap=10)

"""Fixed, recursive, semantic, and structure-aware chunking."""

import pytest

from app.ingestion.chunking import build_chunks
from app.ingestion.embeddings import LocalEmbedder


def test_fixed_windows_ignore_paragraph_breaks() -> None:
    text = ("a" * 50) + "\n\n" + ("b" * 50)
    drafts = build_chunks(text, content_type="text/plain", chunk_size=70, overlap=0, strategy="fixed")

    assert drafts[0].text == text[:70]
    assert "a" in drafts[0].text and "b" in drafts[0].text
    assert all(text[draft.start : draft.end] == draft.text for draft in drafts)


def test_recursive_breaks_on_the_paragraph() -> None:
    text = ("a" * 50) + "\n\n" + ("b" * 50)
    drafts = build_chunks(
        text,
        content_type="text/plain",
        chunk_size=70,
        overlap=0,
        strategy="recursive",
    )

    assert drafts[0].text == "a" * 50
    assert "b" not in drafts[0].text
    assert drafts[-1].end == len(text)
    assert all(text[draft.start : draft.end] == draft.text for draft in drafts)


def test_structure_keeps_markdown_headings() -> None:
    text = "# Retry logic\n\nThe retry file is app/http/retry.py.\n\n# Migrations\n\nSQL files.\n"
    drafts = build_chunks(
        text,
        content_type="text/markdown",
        chunk_size=500,
        overlap=20,
        strategy="structure",
    )

    assert {draft.heading for draft in drafts} == {"Retry logic", "Migrations"}
    recursive = build_chunks(
        text,
        content_type="text/markdown",
        chunk_size=500,
        overlap=20,
        strategy="recursive",
    )
    assert all(draft.heading is None for draft in recursive)


def test_semantic_groups_separate_topics() -> None:
    text = (
        "The file that implements retry logic is app/http/retry.py.\n"
        "Callers are allowed three retry attempts.\n"
        "Schema changes ship as SQL files under db/migrations.\n"
        "Each migration version runs inside one transaction.\n"
    )
    drafts = build_chunks(
        text,
        content_type="text/plain",
        chunk_size=500,
        overlap=20,
        strategy="semantic",
        embedder=LocalEmbedder(dimension=128),
        semantic_similarity=0.35,
    )

    retry = next(draft for draft in drafts if "retry.py" in draft.text)
    schema = next(draft for draft in drafts if "SQL files" in draft.text)
    assert retry is not schema
    assert "SQL files" not in retry.text
    assert "retry.py" not in schema.text
    assert all(text[draft.start : draft.end] == draft.text for draft in drafts)


def test_semantic_chunking_requires_an_embedder() -> None:
    with pytest.raises(ValueError, match="embedder"):
        build_chunks(
            "A sentence about retry logic.",
            content_type="text/plain",
            chunk_size=100,
            overlap=10,
            strategy="semantic",
        )


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="strategy"):
        build_chunks("hello", content_type="text/plain", chunk_size=20, overlap=0, strategy="tokens")

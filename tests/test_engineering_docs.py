"""Source code and architecture documents join the ingestion pipeline."""

from pathlib import Path

from app.config import Settings
from app.ingestion.chunking import build_chunks
from app.ingestion.cleaning import clean_text
from app.ingestion.embeddings import LocalEmbedder
from app.wiring import build_services

SOURCE = """\
\"\"\"Helpers for the HTTP client.\"\"\"

def retry_request(attempts):
    \"\"\"Allow a fixed number of attempts.\"\"\"
    return attempts


class Client:
    def call(self):
        return "ok"
"""

ARCHITECTURE = """\
= Platform

The platform stores engineering documents.

== Ingestion

Source files are parsed before they are embedded.
"""


def test_source_chunks_follow_top_level_symbols() -> None:
    drafts = build_chunks(
        SOURCE,
        content_type="text/x-python",
        chunk_size=500,
        overlap=20,
        strategy="structure",
    )

    assert {draft.heading for draft in drafts} == {None, "retry_request", "Client"}
    client = next(draft for draft in drafts if draft.heading == "Client")
    assert "def call" in client.text
    assert "def retry_request" not in client.text
    assert all(SOURCE[draft.start : draft.end] == draft.text for draft in drafts)


def test_asciidoc_chunks_follow_headings() -> None:
    drafts = build_chunks(
        ARCHITECTURE,
        content_type="text/asciidoc",
        chunk_size=500,
        overlap=20,
        strategy="structure",
    )

    assert {draft.heading for draft in drafts} == {"Platform", "Ingestion"}
    ingestion = next(draft for draft in drafts if draft.heading == "Ingestion")
    assert "parsed before they are embedded" in ingestion.text
    assert all(ARCHITECTURE[draft.start : draft.end] == draft.text for draft in drafts)


def test_source_and_architecture_are_embedded_and_searchable(tmp_path: Path) -> None:
    services = build_services(
        Settings(
            embedding_provider="local",
            embedding_dimension=128,
            chroma_path=tmp_path / "chroma",
            sqlite_path=tmp_path / "metadata.db",
            blob_path=tmp_path / "blobs",
            chunk_size=500,
            chunk_overlap=40,
            max_upload_bytes=100_000,
            log_level="WARNING",
            llm_provider="local",
        ),
        embedder=LocalEmbedder(dimension=128),
    )
    source, _ = services.ingestion.ingest("client.py", SOURCE.encode())
    architecture, _ = services.ingestion.ingest("platform.adoc", ARCHITECTURE.encode())
    source_chunks = services.ingestion.chunks_for(source.id)
    found = services.search.search("what does retry_request do", top_k=2)
    section = services.search.search("where are source files parsed", top_k=2)

    assert source.content_type == "text/x-python"
    assert architecture.content_type == "text/asciidoc"
    cleaned = clean_text(SOURCE)
    assert {chunk["heading"] for chunk in source_chunks} == {None, "retry_request", "Client"}
    assert all(
        cleaned[chunk["start_char"] : chunk["end_char"]] == chunk["text"]
        for chunk in source_chunks
    )
    assert found[0].heading == "retry_request"
    assert found[0].filename == "client.py"
    assert "return attempts" in found[0].text
    assert section[0].heading == "Ingestion"
    assert section[0].filename == "platform.adoc"

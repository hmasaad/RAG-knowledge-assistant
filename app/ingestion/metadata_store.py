"""SQLite records for uploaded documents."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion.models import DocumentRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    byte_size INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    error TEXT
);
"""


class MetadataStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def find_by_checksum(self, checksum: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE checksum = ?",
                (checksum,),
            ).fetchone()
        return _record(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return _record(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_record(row) for row in rows]

    def insert_indexing(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        checksum: str,
        byte_size: int,
    ) -> DocumentRecord:
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, filename, content_type, checksum, status,
                    chunk_count, byte_size, created_at, error
                ) VALUES (?, ?, ?, ?, 'indexing', 0, ?, ?, NULL)
                """,
                (document_id, filename, content_type, checksum, byte_size, created_at),
            )
        record = self.get(document_id)
        if record is None:
            raise RuntimeError("Inserted document could not be read back.")
        return record

    def mark_indexing(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'indexing', chunk_count = 0, error = NULL
                WHERE id = ?
                """,
                (document_id,),
            )

    def mark_indexed(self, document_id: str, chunk_count: int) -> DocumentRecord:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'indexed', chunk_count = ?, error = NULL
                WHERE id = ?
                """,
                (chunk_count, document_id),
            )
        record = self.get(document_id)
        if record is None:
            raise RuntimeError("Indexed document is missing from metadata.")
        return record

    def mark_failed(self, document_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'failed', chunk_count = 0, error = ?
                WHERE id = ?
                """,
                (error[:1000], document_id),
            )

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM documents WHERE id = ?",
                (document_id,),
            )
        return cursor.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection


def _record(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        filename=row["filename"],
        content_type=row["content_type"],
        checksum=row["checksum"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        byte_size=row["byte_size"],
        created_at=row["created_at"],
        error=row["error"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

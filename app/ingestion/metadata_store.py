"""SQLite records for documents, versions, and index jobs."""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.ingestion.models import DocumentRecord, JobRecord, VersionRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    checksum TEXT NOT NULL,
    status TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    byte_size INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    error TEXT,
    owner TEXT NOT NULL DEFAULT 'local',
    visibility TEXT NOT NULL DEFAULT 'private',
    version INTEGER NOT NULL DEFAULT 1,
    indexed_version INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT ''
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_owner_checksum
    ON documents(owner, checksum);
CREATE TABLE IF NOT EXISTS document_versions (
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    checksum TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (document_id, version)
);
CREATE TABLE IF NOT EXISTS index_jobs (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    UNIQUE (document_id, version)
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key TEXT PRIMARY KEY,
    vector TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS catalog_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    generation INTEGER NOT NULL
);
"""


class MetadataStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            _ensure_schema(connection)

    def find_by_checksum(self, checksum: str, owner: str = "local") -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE checksum = ? AND owner = ?",
                (checksum, owner),
            ).fetchone()
        return _record(row) if row else None

    def readable_ids(self, user_id: str) -> set[str]:
        """Indexed documents this caller is allowed to search."""
        return set(self.readable_versions(user_id))

    def readable_versions(self, user_id: str) -> dict[str, int]:
        """Current searchable version for each document this caller can read."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, indexed_version FROM documents
                WHERE indexed_version > 0
                  AND (owner = ? OR visibility = 'shared')
                """,
                (user_id,),
            ).fetchall()
        return {row["id"]: int(row["indexed_version"]) for row in rows}

    def list_visible(self, user_id: str) -> list[DocumentRecord]:
        return [
            record
            for record in self.list_documents()
            if record.owner == user_id or record.visibility == "shared"
        ]

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
        return _record(row) if row else None

    def get_version(self, document_id: str, version: int) -> VersionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            ).fetchone()
        return _version(row) if row else None

    def list_versions(self, document_id: str) -> list[VersionRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM document_versions
                WHERE document_id = ?
                ORDER BY version
                """,
                (document_id,),
            ).fetchall()
        return [_version(row) for row in rows]

    def list_documents(self) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_record(row) for row in rows]

    def document_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM documents GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def job_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM index_jobs GROUP BY status"
            ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def generation(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT generation FROM catalog_state WHERE id = 1"
            ).fetchone()
        return int(row["generation"]) if row else 1

    def create_document(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        checksum: str,
        byte_size: int,
        owner: str,
        visibility: str,
    ) -> DocumentRecord:
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, filename, content_type, checksum, status,
                    chunk_count, byte_size, created_at, error, owner, visibility,
                    version, indexed_version, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, NULL, ?, ?, 1, 0, ?)
                """,
                (
                    document_id,
                    filename,
                    content_type,
                    checksum,
                    byte_size,
                    created_at,
                    owner,
                    visibility,
                    created_at,
                ),
            )
            _insert_version(
                connection,
                document_id=document_id,
                version=1,
                filename=filename,
                content_type=content_type,
                checksum=checksum,
                byte_size=byte_size,
                status="queued",
                created_at=created_at,
            )
            _enqueue(connection, document_id, 1, created_at)
        return self._required(document_id)

    def add_version(
        self,
        document_id: str,
        *,
        filename: str,
        content_type: str,
        checksum: str,
        byte_size: int,
        version: int,
    ) -> DocumentRecord:
        """Point the document at a new version without dropping the searchable one."""
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET filename = ?, content_type = ?, checksum = ?, byte_size = ?,
                    status = 'queued', version = ?, error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (filename, content_type, checksum, byte_size, version, created_at, document_id),
            )
            _insert_version(
                connection,
                document_id=document_id,
                version=version,
                filename=filename,
                content_type=content_type,
                checksum=checksum,
                byte_size=byte_size,
                status="queued",
                created_at=created_at,
            )
            _enqueue(connection, document_id, version, created_at)
        return self._required(document_id)

    def requeue(self, document_id: str, version: int) -> DocumentRecord:
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = 'queued', error = NULL, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (created_at, document_id, version),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'queued', error = NULL
                WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            )
            _enqueue(connection, document_id, version, created_at)
        return self._required(document_id)

    def set_visibility(self, document_id: str, visibility: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET visibility = ?, updated_at = ?
                WHERE id = ?
                """,
                (visibility, _now(), document_id),
            )
            _bump(connection)

    def insert_indexing(
        self,
        *,
        document_id: str,
        filename: str,
        content_type: str,
        checksum: str,
        byte_size: int,
        owner: str,
        visibility: str,
    ) -> DocumentRecord:
        """Insert a row that is already being indexed. Used by schema migration tests."""
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id, filename, content_type, checksum, status,
                    chunk_count, byte_size, created_at, error, owner, visibility,
                    version, indexed_version, updated_at
                ) VALUES (?, ?, ?, ?, 'indexing', 0, ?, ?, NULL, ?, ?, 1, 0, ?)
                """,
                (
                    document_id,
                    filename,
                    content_type,
                    checksum,
                    byte_size,
                    created_at,
                    owner,
                    visibility,
                    created_at,
                ),
            )
            _insert_version(
                connection,
                document_id=document_id,
                version=1,
                filename=filename,
                content_type=content_type,
                checksum=checksum,
                byte_size=byte_size,
                status="indexing",
                created_at=created_at,
            )
        return self._required(document_id)

    def mark_indexing(self, document_id: str, version: int | None = None) -> None:
        now = _now()
        with self._connect() as connection:
            if version is None:
                connection.execute(
                    """
                    UPDATE documents
                    SET status = 'indexing', error = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, document_id),
                )
                return
            connection.execute(
                """
                UPDATE documents
                SET status = 'indexing', error = NULL, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (now, document_id, version),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'indexing', error = NULL
                WHERE document_id = ? AND version = ?
                """,
                (document_id, version),
            )

    def mark_indexed(self, document_id: str, chunk_count: int) -> DocumentRecord:
        record = self.get(document_id)
        if record is None:
            raise RuntimeError("Indexed document is missing from metadata.")
        return self.complete_version(document_id, record.version, chunk_count)

    def complete_version(self, document_id: str, version: int, chunk_count: int) -> DocumentRecord:
        now = _now()
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE documents
                SET status = 'indexed', chunk_count = ?, error = NULL,
                    indexed_version = ?, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (chunk_count, version, now, document_id, version),
            )
            if updated.rowcount != 1:
                raise RuntimeError("Indexed document is missing from metadata.")
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'indexed', chunk_count = ?, error = NULL
                WHERE document_id = ? AND version = ?
                """,
                (chunk_count, document_id, version),
            )
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'superseded'
                WHERE document_id = ? AND version < ? AND status = 'indexed'
                """,
                (document_id, version),
            )
            _bump(connection)
        return self._required(document_id)

    def mark_failed(self, document_id: str, error: str) -> None:
        record = self.get(document_id)
        version = record.version if record is not None else 1
        self.mark_version_failed(document_id, version, error)

    def mark_version_failed(self, document_id: str, version: int, error: str) -> None:
        message = error[:1000]
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE document_versions
                SET status = 'failed', error = ?
                WHERE document_id = ? AND version = ?
                """,
                (message, document_id, version),
            )
            connection.execute(
                """
                UPDATE documents
                SET status = 'failed', error = ?, updated_at = ?,
                    chunk_count = CASE WHEN indexed_version = 0 THEN 0 ELSE chunk_count END
                WHERE id = ? AND version = ?
                """,
                (message, _now(), document_id, version),
            )

    def mark_job_running(self, document_id: str, version: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs
                SET status = 'running', started_at = ?, attempts = attempts + 1
                WHERE document_id = ? AND version = ? AND status = 'queued'
                """,
                (_now(), document_id, version),
            )

    def finish_job(self, document_id: str, version: int, status: str, error: str | None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE index_jobs
                SET status = ?, error = ?, finished_at = ?
                WHERE document_id = ? AND version = ?
                """,
                (status, error[:1000] if error else None, _now(), document_id, version),
            )

    def claim_job(self, stale_after_seconds: float) -> JobRecord | None:
        """Take the oldest queued job. A job left running too long is queued again."""
        stale_before = _now_minus(stale_after_seconds)
        connection = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE index_jobs
                SET status = 'queued', error = 'Requeued after the worker stopped.'
                WHERE status = 'running'
                  AND started_at IS NOT NULL
                  AND started_at < ?
                """,
                (stale_before,),
            )
            row = connection.execute(
                """
                SELECT * FROM index_jobs
                WHERE status = 'queued'
                ORDER BY created_at, id
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            started = _now()
            cursor = connection.execute(
                """
                UPDATE index_jobs
                SET status = 'running', started_at = ?, attempts = attempts + 1
                WHERE id = ? AND status = 'queued'
                """,
                (started, row["id"]),
            )
            claimed = cursor.rowcount == 1
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        if not claimed:
            return None
        return JobRecord(
            id=row["id"],
            document_id=row["document_id"],
            version=int(row["version"]),
            status="running",
            attempts=int(row["attempts"]) + 1,
        )

    def checksum_in_use(self, checksum: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1 AS found FROM document_versions WHERE checksum = ?
                UNION ALL
                SELECT 1 FROM documents WHERE checksum = ?
                LIMIT 1
                """,
                (checksum, checksum),
            ).fetchone()
        return row is not None

    def delete(self, document_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("DELETE FROM index_jobs WHERE document_id = ?", (document_id,))
            connection.execute(
                "DELETE FROM document_versions WHERE document_id = ?",
                (document_id,),
            )
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                _bump(connection)
        return deleted

    def get_embedding(self, cache_key: str) -> list[float] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT vector FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["vector"])
        return [float(value) for value in payload]

    def put_embedding(self, cache_key: str, vector: list[float], *, max_entries: int) -> None:
        if max_entries < 1:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO embedding_cache (cache_key, vector, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    vector = excluded.vector,
                    created_at = excluded.created_at
                """,
                (cache_key, json.dumps(vector), _now()),
            )
            count = connection.execute("SELECT COUNT(*) AS count FROM embedding_cache").fetchone()
            extra = int(count["count"]) - max_entries
            if extra > 0:
                connection.execute(
                    """
                    DELETE FROM embedding_cache
                    WHERE cache_key IN (
                        SELECT cache_key FROM embedding_cache
                        ORDER BY created_at, cache_key
                        LIMIT ?
                    )
                    """,
                    (extra,),
                )

    def _required(self, document_id: str) -> DocumentRecord:
        record = self.get(document_id)
        if record is None:
            raise RuntimeError("Inserted document could not be read back.")
        return record

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection


def _record(row: sqlite3.Row) -> DocumentRecord:
    keys = set(row.keys())
    created_at = row["created_at"]
    return DocumentRecord(
        id=row["id"],
        filename=row["filename"],
        content_type=row["content_type"],
        checksum=row["checksum"],
        status=row["status"],
        chunk_count=row["chunk_count"],
        byte_size=row["byte_size"],
        created_at=created_at,
        error=row["error"],
        owner=row["owner"] if "owner" in keys else "local",
        visibility=row["visibility"] if "visibility" in keys else "private",
        version=int(row["version"]) if "version" in keys else 1,
        indexed_version=int(row["indexed_version"]) if "indexed_version" in keys else 0,
        updated_at=row["updated_at"] if "updated_at" in keys and row["updated_at"] else created_at,
    )


def _version(row: sqlite3.Row) -> VersionRecord:
    return VersionRecord(
        document_id=row["document_id"],
        version=int(row["version"]),
        filename=row["filename"],
        content_type=row["content_type"],
        checksum=row["checksum"],
        byte_size=row["byte_size"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        created_at=row["created_at"],
        error=row["error"],
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_minus(seconds: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _bump(connection: sqlite3.Connection) -> None:
    connection.execute("UPDATE catalog_state SET generation = generation + 1 WHERE id = 1")


def _enqueue(connection: sqlite3.Connection, document_id: str, version: int, created_at: str) -> None:
    connection.execute(
        """
        INSERT INTO index_jobs (
            id, document_id, version, status, attempts, error, created_at, started_at, finished_at
        ) VALUES (?, ?, ?, 'queued', 0, NULL, ?, NULL, NULL)
        ON CONFLICT(document_id, version) DO UPDATE SET
            status = 'queued',
            error = NULL,
            started_at = NULL,
            finished_at = NULL,
            created_at = excluded.created_at
        """,
        (uuid.uuid4().hex, document_id, version, created_at),
    )


def _insert_version(
    connection: sqlite3.Connection,
    *,
    document_id: str,
    version: int,
    filename: str,
    content_type: str,
    checksum: str,
    byte_size: int,
    status: str,
    created_at: str,
) -> None:
    connection.execute(
        """
        INSERT INTO document_versions (
            document_id, version, filename, content_type, checksum,
            byte_size, chunk_count, status, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, ?)
        ON CONFLICT(document_id, version) DO UPDATE SET
            filename = excluded.filename,
            content_type = excluded.content_type,
            checksum = excluded.checksum,
            byte_size = excluded.byte_size,
            status = excluded.status,
            error = NULL
        """,
        (document_id, version, filename, content_type, checksum, byte_size, status, created_at),
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    existing = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
    ).fetchone()
    if existing is None:
        connection.executescript(_SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO catalog_state (id, generation) VALUES (1, 1)"
        )
        return
    columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
    sql = existing["sql"] or ""
    if "owner" not in columns or "checksum TEXT NOT NULL UNIQUE" in sql:
        _rebuild_documents(connection, columns)
    else:
        _add_document_columns(connection, columns)
    connection.executescript(_SCHEMA)
    connection.execute("INSERT OR IGNORE INTO catalog_state (id, generation) VALUES (1, 1)")
    connection.execute(
        """
        UPDATE documents
        SET updated_at = created_at
        WHERE updated_at IS NULL OR updated_at = ''
        """
    )
    connection.execute(
        """
        UPDATE documents
        SET indexed_version = version
        WHERE status = 'indexed' AND indexed_version = 0
        """
    )
    connection.execute(
        """
        INSERT OR IGNORE INTO document_versions (
            document_id, version, filename, content_type, checksum,
            byte_size, chunk_count, status, error, created_at
        )
        SELECT
            id, version, filename, content_type, checksum,
            byte_size, chunk_count, status, error, created_at
        FROM documents
        """
    )


def _add_document_columns(connection: sqlite3.Connection, columns: set[str]) -> None:
    if "version" not in columns:
        connection.execute(
            "ALTER TABLE documents ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
        )
    if "indexed_version" not in columns:
        connection.execute(
            "ALTER TABLE documents ADD COLUMN indexed_version INTEGER NOT NULL DEFAULT 0"
        )
    if "updated_at" not in columns:
        connection.execute(
            "ALTER TABLE documents ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"
        )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_owner_checksum
            ON documents(owner, checksum)
        """
    )


def _rebuild_documents(connection: sqlite3.Connection, columns: set[str]) -> None:
    connection.execute("ALTER TABLE documents RENAME TO documents_old")
    connection.executescript(_SCHEMA)
    owner = "owner" if "owner" in columns else "'local'"
    visibility = "visibility" if "visibility" in columns else "'private'"
    connection.execute(
        f"""
        INSERT INTO documents (
            id, filename, content_type, checksum, status, chunk_count,
            byte_size, created_at, error, owner, visibility,
            version, indexed_version, updated_at
        )
        SELECT
            id, filename, content_type, checksum, status, chunk_count,
            byte_size, created_at, error, {owner}, {visibility},
            1,
            CASE WHEN status = 'indexed' THEN 1 ELSE 0 END,
            created_at
        FROM documents_old
        """
    )
    connection.execute("DROP TABLE documents_old")

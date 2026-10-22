"""Content-addressed storage for original uploads.

Bytes are stored by SHA-256 so identical versions share one file. The directory
can sit on a local disk or a shared volume mounted by the API and the worker.
"""

from pathlib import Path
from typing import Protocol

from app.ingestion.errors import IngestionError


class BlobStore(Protocol):
    def put(self, checksum: str, data: bytes) -> None: ...

    def get(self, checksum: str) -> bytes: ...

    def delete(self, checksum: str) -> None: ...


class LocalBlobStore:
    """Store bytes under ``root/ab/cd/<checksum>``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, checksum: str, data: bytes) -> None:
        path = self._path(checksum)
        if path.is_file():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)

    def get(self, checksum: str) -> bytes:
        path = self._path(checksum)
        if not path.is_file():
            raise IngestionError("Stored document bytes are missing.", status_code=500)
        return path.read_bytes()

    def delete(self, checksum: str) -> None:
        self._path(checksum).unlink(missing_ok=True)

    def _path(self, checksum: str) -> Path:
        if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
            raise IngestionError("Invalid document checksum.", status_code=500)
        return self.root / checksum[:2] / checksum[2:4] / checksum


class MemoryBlobStore:
    """In-memory bytes for callers that construct ingestion without a data directory."""

    def __init__(self) -> None:
        self._items: dict[str, bytes] = {}

    def put(self, checksum: str, data: bytes) -> None:
        self._items[checksum] = data

    def get(self, checksum: str) -> bytes:
        try:
            return self._items[checksum]
        except KeyError as exc:
            raise IngestionError("Stored document bytes are missing.", status_code=500) from exc

    def delete(self, checksum: str) -> None:
        self._items.pop(checksum, None)

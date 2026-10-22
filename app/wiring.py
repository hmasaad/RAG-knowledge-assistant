"""Compose stores, embeddings, and services."""

from dataclasses import dataclass

from app.config import Settings
from app.ingestion.embeddings import Embedder, build_embedder
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.service import IngestionService
from app.ingestion.vector_store import VectorStore
from app.retrieval.search import SearchService


@dataclass
class Services:
    settings: Settings
    metadata: MetadataStore
    vectors: VectorStore
    ingestion: IngestionService
    search: SearchService


class LazyEmbedder:
    """Build the configured embedder on the first embed call."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder: Embedder | None = None

    @property
    def model(self) -> str:
        if self._embedder is not None:
            return self._embedder.model
        if self._settings.embedding_provider == "local":
            return "local-hash-bow"
        return self._settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._resolve().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._resolve().embed_query(text)

    def _resolve(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(
                provider=self._settings.embedding_provider,
                api_key=self._settings.openai_api_key,
                model=self._settings.embedding_model,
                dimension=self._settings.embedding_dimension,
            )
        return self._embedder


def build_services(settings: Settings, embedder: Embedder | None = None) -> Services:
    resolved = embedder if embedder is not None else LazyEmbedder(settings)
    metadata = MetadataStore(settings.sqlite_path)
    vectors = VectorStore(settings.chroma_path, settings.collection_name)
    ingestion = IngestionService(
        metadata=metadata,
        vectors=vectors,
        embedder=resolved,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )
    search = SearchService(vectors=vectors, embedder=resolved)
    return Services(
        settings=settings,
        metadata=metadata,
        vectors=vectors,
        ingestion=ingestion,
        search=search,
    )

"""Runtime configuration loaded from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for ingestion, storage, and embeddings.

    Environment variables use the field names in upper case, for example
    ``CHUNK_SIZE`` and ``OPENAI_API_KEY``.
    """

    openai_api_key: str = ""
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=384, ge=8, le=4096)
    chroma_path: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/metadata.db")
    collection_name: str = "knowledge"
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def overlap_fits_in_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

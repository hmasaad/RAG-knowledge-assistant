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
    chunk_strategy: Literal["fixed", "recursive", "semantic", "structure"] = "structure"
    semantic_similarity: float = Field(default=0.35, ge=-1, le=1)
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    llm_provider: Literal["openai", "local"] = "openai"
    chat_model: str = "gpt-4o-mini"
    retrieval_candidates: int = Field(default=20, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1)
    rerank_coverage_weight: float = Field(default=0.35, ge=0, le=1)
    rerank_provider: Literal["coverage", "cross_encoder", "api"] = "coverage"
    rerank_model: str = ""
    rerank_api_url: str = ""
    rerank_api_key: str = ""
    rerank_timeout_seconds: float = Field(default=15, gt=0, le=120)
    evidence_min_coverage: float = Field(default=0.5, ge=0, le=1)
    query_rewrite: bool = False
    query_decompose: bool = False
    multi_query: bool = False
    parent_retrieval: bool = False
    child_chunk_size: int = Field(default=240, ge=1)
    contextual_retrieval: bool = False
    knowledge_graph: bool = False
    blob_path: Path = Path("data/blobs")
    chroma_host: str = ""
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    indexing_mode: Literal["inline", "background"] = "inline"
    worker_in_process: bool = True
    worker_poll_seconds: float = Field(default=1.0, gt=0, le=60)
    job_timeout_seconds: float = Field(default=900, gt=0)
    embedding_cache_size: int = Field(default=4096, ge=0)
    answer_cache_size: int = Field(default=256, ge=0)
    agent_enabled: bool = True
    research_search_budget: int = Field(default=16, ge=1, le=100)
    research_max_cost: int = Field(default=0, ge=0, le=200)
    research_denied_tools: str = ""
    research_denied_actions: str = ""
    research_timeout_seconds: float = Field(default=30, gt=0, le=120)
    research_adapt_rounds: int = Field(default=2, ge=0, le=8)
    research_worker_concurrency: int = Field(default=2, ge=1, le=16)
    research_user_concurrency: int = Field(default=2, ge=1, le=16)
    governance_hold: bool = False
    governance_require_approval: bool = True
    governance_operators: str = ""
    default_user_id: str = "local"
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
        if self.parent_retrieval and self.child_chunk_size >= self.chunk_size:
            raise ValueError("CHILD_CHUNK_SIZE must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

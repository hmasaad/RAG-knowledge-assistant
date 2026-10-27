"""Compose stores, embeddings, and services."""

from dataclasses import dataclass

from app.config import Settings
from app.generation.answerers import Answerer, build_answerer
from app.generation.grounding import AnswerResult
from app.generation.service import GenerationService
from app.ingestion.cache import CachingEmbedder
from app.ingestion.embeddings import Embedder, build_embedder
from app.ingestion.metadata_store import MetadataStore
from app.ingestion.service import IngestionService
from app.ingestion.vector_store import VectorStore
from app.observability.cache import LruCache
from app.observability.metrics import Metrics
from app.observability.tracing import TraceLog
from app.research.judge import ClaimJudgeAdapter, LazyClaimJudge
from app.research.rag import KnowledgeRagClient
from app.engineering.service import EngineeringService
from app.research.service import ResearchReport, ResearchService
from app.research.adapters import compose_action_adapters
from app.research.graph import EvidenceGraph
from app.research.guardrails import GuardrailPolicy, ResearchGuardrails, parse_names
from app.research.observability import ResearchObserver
from app.research.tools import approved_tools
from app.research.worker import ResearchWorker
from app.retrieval.rerank import build_reranker
from app.retrieval.search import RetrievalResult, SearchService
from app.storage.blobs import BlobStore, LocalBlobStore


@dataclass
class Services:
    settings: Settings
    metadata: MetadataStore
    vectors: VectorStore
    blobs: BlobStore
    metrics: Metrics
    ingestion: IngestionService
    search: SearchService
    generation: GenerationService
    research: ResearchService
    claim_judge: ClaimJudgeAdapter
    traces: TraceLog
    engineering: EngineeringService


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


def build_services(
    settings: Settings,
    embedder: Embedder | None = None,
    answerer: Answerer | None = None,
) -> Services:
    resolved = embedder if embedder is not None else LazyEmbedder(settings)
    resolved_answerer = answerer if answerer is not None else LazyAnswerer(settings)
    metadata = MetadataStore(settings.sqlite_path)
    blobs = LocalBlobStore(settings.blob_path)
    metrics = Metrics()
    traces = TraceLog(settings.sqlite_path)
    engineering = EngineeringService(settings.sqlite_path)
    vectors = VectorStore(
        settings.chroma_path,
        settings.collection_name,
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    cached_embedder = CachingEmbedder(
        resolved,
        metadata,
        metrics,
        memory_size=settings.embedding_cache_size,
    )
    ingestion = IngestionService(
        metadata=metadata,
        vectors=vectors,
        embedder=cached_embedder,
        blobs=blobs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
        indexing_mode=settings.indexing_mode,
        metrics=metrics,
        job_timeout_seconds=settings.job_timeout_seconds,
        chunk_strategy=settings.chunk_strategy,
        semantic_similarity=settings.semantic_similarity,
        parent_retrieval=settings.parent_retrieval,
        child_chunk_size=settings.child_chunk_size,
        contextual_retrieval=settings.contextual_retrieval,
        knowledge_graph=settings.knowledge_graph,
    )
    search = SearchService(
        vectors=vectors,
        embedder=cached_embedder,
        metadata=metadata,
        default_user_id=settings.default_user_id,
        candidate_k=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
        coverage_weight=settings.rerank_coverage_weight,
        min_coverage=settings.evidence_min_coverage,
        metrics=metrics,
        reranker=build_reranker(
            provider=settings.rerank_provider,
            model=settings.rerank_model,
            api_url=settings.rerank_api_url,
            api_key=settings.rerank_api_key,
            timeout=settings.rerank_timeout_seconds,
        ),
        rewrite=settings.query_rewrite,
        decompose=settings.query_decompose,
        multi_query=settings.multi_query,
        knowledge_graph=settings.knowledge_graph,
    )
    generation = GenerationService(
        retrieval=search,
        answerer=resolved_answerer,
        metadata=metadata,
        cache=LruCache(settings.answer_cache_size),
        metrics=metrics,
        traces=traces,
        default_user_id=settings.default_user_id,
        agent_enabled=settings.agent_enabled,
    )
    rag_client = KnowledgeRagClient(search, timeout=settings.research_timeout_seconds)
    tool_registry = approved_tools(rag=rag_client, metadata=metadata)
    evidence_graph = EvidenceGraph()
    research_observer = ResearchObserver(metrics, traces)
    research_guardrails = ResearchGuardrails(
        GuardrailPolicy(
            budget=settings.research_search_budget,
            max_iterations=max(8, settings.research_search_budget),
            max_cost=settings.research_max_cost,
            denied_tools=parse_names(settings.research_denied_tools),
            denied_actions=parse_names(settings.research_denied_actions),
        )
    )
    research = ResearchService(
        retrieval=search,
        answerer=resolved_answerer,
        metadata=metadata,
        cache=LruCache[tuple[object, ...], ResearchReport](settings.answer_cache_size),
        metrics=metrics,
        traces=traces,
        default_user_id=settings.default_user_id,
        rag=rag_client,
        search_budget=settings.research_search_budget,
        search_timeout=settings.research_timeout_seconds,
        adapt_rounds=settings.research_adapt_rounds,
        worker=ResearchWorker(
            tool_registry,
            adapters=compose_action_adapters(
                registry=tool_registry,
                use_rag=True,
                use_verification=True,
                graph=evidence_graph,
                metadata=metadata,
            ),
            budget=settings.research_search_budget,
            max_iterations=max(8, settings.research_search_budget),
            graph=evidence_graph,
            guardrails=research_guardrails,
            observer=research_observer,
        ),
        indexing_mode=settings.indexing_mode,
        job_timeout_seconds=settings.job_timeout_seconds,
        worker_concurrency=settings.research_worker_concurrency,
        user_concurrency=settings.research_user_concurrency,
        worker_id=f"worker-{id(metrics)}",
        engineering=engineering,
    )
    claim_judge = ClaimJudgeAdapter(
        rag=rag_client,
        judge=LazyClaimJudge(
            provider=settings.llm_provider,
            api_key=settings.openai_api_key,
            model=settings.chat_model,
        ),
        metadata=metadata,
    )
    return Services(
        settings=settings,
        metadata=metadata,
        vectors=vectors,
        blobs=blobs,
        metrics=metrics,
        ingestion=ingestion,
        search=search,
        generation=generation,
        research=research,
        claim_judge=claim_judge,
        traces=traces,
        engineering=engineering,
    )


class LazyAnswerer:
    """Build the configured answerer on the first question."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._answerer: Answerer | None = None

    @property
    def model(self) -> str:
        if self._answerer is not None:
            return self._answerer.model
        if self._settings.llm_provider == "local":
            return "local-extractive"
        return self._settings.chat_model

    def generate(self, question: str, retrieved: RetrievalResult) -> AnswerResult:
        return self._resolve().generate(question, retrieved)

    def _resolve(self) -> Answerer:
        if self._answerer is None:
            self._answerer = build_answerer(
                provider=self._settings.llm_provider,
                api_key=self._settings.openai_api_key,
                model=self._settings.chat_model,
            )
        return self._answerer


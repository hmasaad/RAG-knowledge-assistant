"""Application entry point."""

import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app import __version__
from app.api.answer import router as answer_router
from app.api.claims import router as claims_router
from app.api.documents import router as documents_router
from app.api.embeddings import router as embeddings_router
from app.api.operations import router as operations_router
from app.api.research import router as research_router
from app.api.search import router as search_router
from app.config import Settings, get_settings
from app.generation.answerers import Answerer
from app.ingestion.embeddings import Embedder
from app.ingestion.errors import KnowledgeError
from app.observability.logging import configure_logging
from app.wiring import Services, build_services

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health(request: Request) -> dict[str, object]:
    services: Services = request.app.state.services
    jobs = services.metadata.job_counts()
    return {
        "status": "ok",
        "documents": len(services.ingestion.list_documents()),
        "chunks": services.vectors.count(),
        "embedding_provider": services.settings.embedding_provider,
        "llm_provider": services.settings.llm_provider,
        "indexing_mode": services.settings.indexing_mode,
        "jobs_queued": jobs.get("queued", 0),
        "jobs_running": jobs.get("running", 0),
        "catalog_generation": services.metadata.generation(),
    }


@health_router.get("/ready")
def ready(request: Request) -> JSONResponse:
    services: Services = request.app.state.services
    try:
        services.metadata.generation()
        services.vectors.count()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": type(exc).__name__},
        )
    return JSONResponse(status_code=200, content={"status": "ok"})


@health_router.get("/metrics")
def metrics(request: Request) -> PlainTextResponse:
    services: Services = request.app.state.services
    services.metrics.clear_gauges()
    services.metrics.set_gauge("rag_chunks", float(services.vectors.count()))
    services.metrics.set_gauge("rag_catalog_generation", float(services.metadata.generation()))
    for status, count in services.metadata.document_counts().items():
        services.metrics.set_gauge("rag_documents", float(count), status=status)
    for status, count in services.metadata.job_counts().items():
        services.metrics.set_gauge("rag_jobs", float(count), status=status)
    return PlainTextResponse(
        services.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    answerer: Answerer | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    services = build_services(resolved_settings, embedder, answerer)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        running = application.state.services
        stop = None
        thread = None
        if running.settings.indexing_mode == "background" and running.settings.worker_in_process:
            from app.worker import start_worker

            stop, thread = start_worker(running)
        yield
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=5)

    application = FastAPI(
        title="RAG Knowledge Assistant",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.services = services
    application.add_exception_handler(KnowledgeError, _knowledge_error)
    application.middleware("http")(_observe_request)
    application.include_router(health_router)
    application.include_router(documents_router, prefix="/api")
    application.include_router(embeddings_router, prefix="/api")
    application.include_router(search_router, prefix="/api")
    application.include_router(answer_router, prefix="/api")
    application.include_router(claims_router, prefix="/api")
    application.include_router(research_router, prefix="/api")
    application.include_router(operations_router, prefix="/api")
    return application


async def _observe_request(request: Request, call_next):
    services: Services = request.app.state.services
    started = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    path = getattr(route, "path", "unmatched")
    services.metrics.inc(
        "rag_http_requests_total",
        method=request.method,
        path=path,
        status=str(response.status_code),
    )
    services.metrics.observe(
        "rag_http_request_seconds",
        time.perf_counter() - started,
        path=path,
    )
    return response


async def _knowledge_error(_request: Request, exc: KnowledgeError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def __getattr__(name: str) -> FastAPI:
    """Build the default app on first use so imports do not open the vector store."""
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

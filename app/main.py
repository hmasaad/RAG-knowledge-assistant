"""Application entry point."""

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from app import __version__
from app.api.documents import router as documents_router
from app.api.search import router as search_router
from app.config import Settings, get_settings
from app.ingestion.embeddings import Embedder
from app.ingestion.errors import IngestionError
from app.observability.logging import configure_logging
from app.wiring import Services, build_services

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
def health(request: Request) -> dict[str, object]:
    services: Services = request.app.state.services
    return {
        "status": "ok",
        "documents": len(services.ingestion.list_documents()),
        "chunks": services.vectors.count(),
        "embedding_provider": services.settings.embedding_provider,
    }


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    services = build_services(resolved_settings, embedder)
    application = FastAPI(title="RAG Knowledge Assistant", version=__version__)
    application.state.services = services
    application.add_exception_handler(IngestionError, _ingestion_error)
    application.include_router(health_router)
    application.include_router(documents_router, prefix="/api")
    application.include_router(search_router, prefix="/api")
    return application


async def _ingestion_error(_request: Request, exc: IngestionError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


def __getattr__(name: str) -> FastAPI:
    """Build the default app on first use so imports do not open the vector store."""
    if name == "app":
        application = create_app()
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

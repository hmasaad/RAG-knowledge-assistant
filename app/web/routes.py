"""Serve the browser UI."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).resolve().parent / "static"
router = APIRouter(tags=["ui"])


@router.get("/", include_in_schema=False)
def ui_home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


def mount_ui(application) -> None:
    application.include_router(router)
    application.mount("/ui", StaticFiles(directory=STATIC), name="ui")

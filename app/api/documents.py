"""Upload, list, inspect, and delete indexed documents."""

from fastapi import APIRouter, Depends, File, Response, UploadFile

from app.api.dependencies import get_services
from app.api.schemas import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
)
from app.wiring import Services

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentResponse)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    services: Services = Depends(get_services),
) -> DocumentResponse:
    data = file.file.read()
    record, created = services.ingestion.ingest(file.filename, data)
    response.status_code = 201 if created else 200
    return DocumentResponse.from_record(record, already_indexed=not created)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(services: Services = Depends(get_services)) -> DocumentListResponse:
    records = services.ingestion.list_documents()
    return DocumentListResponse(
        documents=[DocumentResponse.from_record(record) for record in records]
    )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    services: Services = Depends(get_services),
) -> DocumentDetailResponse:
    record = services.ingestion.get_document(document_id)
    chunks = [
        ChunkResponse.model_validate(chunk)
        for chunk in services.ingestion.chunks_for(document_id)
    ]
    return DocumentDetailResponse(
        **DocumentResponse.from_record(record).model_dump(),
        chunks=chunks,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    services: Services = Depends(get_services),
) -> None:
    services.ingestion.delete_document(document_id)

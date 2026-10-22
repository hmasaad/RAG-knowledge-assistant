"""Upload, list, inspect, update, and delete indexed documents."""

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile
from fastapi.responses import Response as RawResponse

from app.api.dependencies import get_services, get_user_id
from app.api.schemas import (
    ChunkResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentResponse,
    VersionListResponse,
    VersionResponse,
)
from app.wiring import Services

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=DocumentResponse)
def upload_document(
    response: Response,
    file: UploadFile = File(...),
    visibility: str = Form("private"),
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> DocumentResponse:
    data = file.file.read()
    record, created = services.ingestion.ingest(
        file.filename,
        data,
        owner=user_id,
        visibility=visibility,
    )
    response.status_code = _status_code(record.status, created=created)
    return DocumentResponse.from_record(record, already_indexed=not created)


@router.put("/documents/{document_id}", response_model=DocumentResponse)
def update_document(
    document_id: str,
    response: Response,
    file: UploadFile = File(...),
    visibility: str = Form(""),
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> DocumentResponse:
    data = file.file.read()
    record, changed = services.ingestion.update_document(
        document_id,
        file.filename,
        data,
        owner=user_id,
        visibility=visibility,
    )
    response.status_code = 202 if record.status == "queued" else 200
    return DocumentResponse.from_record(record, already_indexed=not changed)


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> DocumentListResponse:
    records = services.ingestion.list_documents(user_id=user_id)
    return DocumentListResponse(
        documents=[DocumentResponse.from_record(record) for record in records]
    )


@router.get("/documents/{document_id}/versions", response_model=VersionListResponse)
def list_versions(
    document_id: str,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> VersionListResponse:
    versions = services.ingestion.list_versions(document_id, user_id=user_id)
    return VersionListResponse(
        versions=[VersionResponse.from_record(version) for version in versions]
    )


@router.get("/documents/{document_id}/versions/{version}/file")
def download_version(
    document_id: str,
    version: int,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> RawResponse:
    stored, data = services.ingestion.read_version(document_id, version, user_id=user_id)
    filename = stored.filename.replace('"', "")
    return RawResponse(
        content=data,
        media_type=stored.content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
def get_document(
    document_id: str,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> DocumentDetailResponse:
    record = services.ingestion.get_document(document_id, user_id=user_id)
    chunks = [
        ChunkResponse.model_validate(chunk)
        for chunk in services.ingestion.chunks_for(document_id, user_id=user_id)
    ]
    return DocumentDetailResponse(
        **DocumentResponse.from_record(record).model_dump(),
        chunks=chunks,
    )


@router.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    services: Services = Depends(get_services),
    user_id: str = Depends(get_user_id),
) -> None:
    services.ingestion.delete_document(document_id, user_id=user_id)


def _status_code(status: str, *, created: bool) -> int:
    if status == "queued":
        return 202
    if created:
        return 201
    return 200

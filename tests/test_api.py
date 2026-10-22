from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["chunks"] == 0


def test_upload_search_and_delete(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    created = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    duplicate = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
    )
    listed = client.get("/api/documents")
    document_id = created.json()["id"]
    detail = client.get(f"/api/documents/{document_id}")
    found = client.post(
        "/api/search",
        json={"query": "which file implements the retry logic", "top_k": 2},
    )
    removed = client.delete(f"/api/documents/{document_id}")
    missing = client.get(f"/api/documents/{document_id}")

    assert created.status_code == 201
    assert created.json()["already_indexed"] is False
    assert created.json()["status"] == "indexed"
    assert duplicate.status_code == 200
    assert duplicate.json()["already_indexed"] is True
    assert duplicate.json()["id"] == document_id
    assert listed.json()["documents"][0]["id"] == document_id
    assert any("app/http/retry.py" in chunk["text"] for chunk in detail.json()["chunks"])
    assert found.status_code == 200
    assert "app/http/retry.py" in found.json()["hits"][0]["text"]
    assert removed.status_code == 204
    assert missing.status_code == 404


def test_upload_rejects_unsupported_and_blank_queries(client: TestClient) -> None:
    rejected = client.post(
        "/api/documents",
        files={"file": ("notes.docx", b"hello", "application/octet-stream")},
    )
    blank = client.post("/api/search", json={"query": "   "})

    assert rejected.status_code == 415
    assert blank.status_code == 422

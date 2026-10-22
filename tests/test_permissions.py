"""Document visibility: private rows stay with their owner, shared rows are readable."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.security.prompts import REFUSAL_TEXT

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {"question": "which file implements the retry logic", "top_k": 3}
QUERY = {"query": "which file implements the retry logic", "top_k": 3}


def _upload(
    client: TestClient,
    user_id: str,
    *,
    visibility: str = "private",
    name: str = "api-errors.md",
) -> dict:
    payload = (ROOT / "samples" / name).read_bytes()
    response = client.post(
        "/api/documents",
        files={"file": (name, payload, "text/markdown")},
        data={"visibility": visibility},
        headers={"X-User-Id": user_id},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_private_documents_are_invisible_to_other_users(client: TestClient) -> None:
    created = _upload(client, "ada")
    document_id = created["id"]

    hidden = client.get(f"/api/documents/{document_id}", headers={"X-User-Id": "grace"})
    listed = client.get("/api/documents", headers={"X-User-Id": "grace"})
    found = client.post("/api/search", json=QUERY, headers={"X-User-Id": "grace"})
    retrieved = client.post("/api/retrieve", json=QUERY, headers={"X-User-Id": "grace"})
    answered = client.post("/api/answer", json=QUESTION, headers={"X-User-Id": "grace"})
    removed = client.delete(f"/api/documents/{document_id}", headers={"X-User-Id": "grace"})
    owner = client.post("/api/answer", json=QUESTION, headers={"X-User-Id": "ada"})

    assert created["owner"] == "ada"
    assert created["visibility"] == "private"
    assert hidden.status_code == 404
    assert listed.json()["documents"] == []
    assert found.json()["hits"] == []
    assert retrieved.json()["passages"] == []
    assert "No documents are visible" in retrieved.json()["inspection"]
    assert answered.json()["answer"] == REFUSAL_TEXT
    assert removed.status_code == 404
    assert "app/http/retry.py" in owner.json()["answer"]


def test_shared_documents_can_be_read_but_only_the_owner_deletes(client: TestClient) -> None:
    created = _upload(client, "ada", visibility="shared")
    document_id = created["id"]

    visible = client.get(f"/api/documents/{document_id}", headers={"X-User-Id": "grace"})
    answered = client.post("/api/answer", json=QUESTION, headers={"X-User-Id": "grace"})
    denied = client.delete(f"/api/documents/{document_id}", headers={"X-User-Id": "grace"})
    removed = client.delete(f"/api/documents/{document_id}", headers={"X-User-Id": "ada"})

    assert created["visibility"] == "shared"
    assert visible.status_code == 200
    assert "app/http/retry.py" in answered.json()["answer"]
    assert denied.status_code == 404
    assert removed.status_code == 204


def test_identical_bytes_are_indexed_once_per_owner(client: TestClient) -> None:
    ada = _upload(client, "ada")
    grace = _upload(client, "grace")

    assert ada["id"] != grace["id"]
    assert ada["checksum"] == grace["checksum"]


def test_invalid_user_and_visibility_are_rejected(client: TestClient) -> None:
    payload = (ROOT / "samples" / "api-errors.md").read_bytes()
    bad_user = client.get("/api/documents", headers={"X-User-Id": "bad user"})
    bad_visibility = client.post(
        "/api/documents",
        files={"file": ("api-errors.md", payload, "text/markdown")},
        data={"visibility": "public"},
        headers={"X-User-Id": "ada"},
    )

    assert bad_user.status_code == 400
    assert bad_visibility.status_code == 400

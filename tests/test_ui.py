"""The browser UI is served from the API process."""

from fastapi.testclient import TestClient


def test_home_page_is_the_assistant_ui(client: TestClient) -> None:
    page = client.get("/")
    styles = client.get("/ui/styles.css")
    script = client.get("/ui/app.js")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Research agent" in page.text
    assert "Stop after the first committed step" in page.text
    assert "Library" in page.text
    assert styles.status_code == 200
    assert script.status_code == 200

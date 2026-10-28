"""Errors returned by the API."""


class KnowledgeError(Exception):
    """A request could not be completed.

    ``status_code`` is the HTTP status the API should return.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class IngestionError(KnowledgeError):
    """A document could not be accepted or indexed."""


def index_failure_message(exc: Exception) -> str:
    """Turn an indexing exception into a message the UI can show."""
    if isinstance(exc, IngestionError):
        return exc.message
    text = str(exc).strip()
    lowered = text.casefold()
    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "Embedding failed. Set OPENAI_API_KEY, or use EMBEDDING_PROVIDER=local."
    if "dimension" in lowered or "expecting embedding with dimension" in lowered:
        return (
            "The embedding size does not match the existing index. "
            "Use the same EMBEDDING_PROVIDER or a new CHROMA_PATH."
        )
    if text and len(text) < 240:
        return f"Failed to index the document: {text}"
    return "Failed to index the document."

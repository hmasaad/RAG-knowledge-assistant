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

"""Errors raised while accepting or indexing a document."""


class IngestionError(Exception):
    """A document could not be accepted or indexed.

    ``status_code`` is the HTTP status the API should return.
    """

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code

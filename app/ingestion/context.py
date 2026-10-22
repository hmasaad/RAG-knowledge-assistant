"""Situate a chunk in its document before embedding it."""


def situate(filename: str, heading: str | None) -> str:
    """A short prefix that names the document and section."""
    name = filename.rsplit("/", 1)[-1]
    if heading:
        return f"Document: {name}. Section: {heading}."
    return f"Document: {name}."


def embedding_text(text: str, context: str = "") -> str:
    """Text the embedder should see. The stored chunk stays the original text."""
    if not context:
        return text
    return f"{context}\n\n{text}"

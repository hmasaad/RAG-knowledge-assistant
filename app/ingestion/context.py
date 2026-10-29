"""Situate a chunk in its document before embedding it."""


def situate(filename: str, heading: str | None, *, section_title: str = "") -> str:
    """A short prefix that names the document and section."""
    name = filename.rsplit("/", 1)[-1]
    section = heading or section_title
    parent = section_title if section_title and heading and section_title not in heading else ""
    parts = [f"Document: {name}."]
    if section:
        parts.append(f"Section: {section}.")
    if parent:
        parts.append(f"Parent heading: {parent}.")
    return " ".join(parts)


def embedding_text(text: str, context: str = "") -> str:
    """Text the embedder should see. The stored chunk stays the original text."""
    if not context:
        return text
    return f"{context}\n\n{text}"

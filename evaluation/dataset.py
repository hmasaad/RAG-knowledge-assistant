"""Load the retrieval cases used to check that indexed documents are searchable."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetrievalCase:
    id: str
    query: str
    expected_substring: str
    document: str


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[RetrievalCase] = []
    for item in payload:
        cases.append(
            RetrievalCase(
                id=item["id"],
                query=item["query"],
                expected_substring=item["expected_substring"],
                document=item["document"],
            )
        )
    return cases

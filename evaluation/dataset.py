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
    return [
        RetrievalCase(
            id=item["id"],
            query=item["query"],
            expected_substring=item["expected_substring"],
            document=item["document"],
        )
        for item in payload
    ]


@dataclass(frozen=True)
class AnswerCase:
    id: str
    question: str
    expect_evidence: bool
    expected_substring: str
    document: str


def load_answer_cases(path: Path) -> list[AnswerCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        AnswerCase(
            id=item["id"],
            question=item["question"],
            expect_evidence=item["expect_evidence"],
            expected_substring=item.get("expected_substring", ""),
            document=item.get("document", ""),
        )
        for item in payload
    ]

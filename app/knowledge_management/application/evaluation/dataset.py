import json
from pathlib import Path
from typing import Any

from ...domain.evaluation import EvaluationExample


def parse_examples(raw_examples: list[dict[str, Any]]) -> list[EvaluationExample]:
    """Mapuje surowe rekordy JSON na `EvaluationExample`, walidując wymagane pola."""
    examples: list[EvaluationExample] = []
    for index, record in enumerate(raw_examples):
        question = record.get("question")
        relevant = record.get("relevant_document_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Przykład #{index}: brak niepustego pola 'question'.")
        if not isinstance(relevant, list) or not all(isinstance(item, str) for item in relevant):
            raise ValueError(f"Przykład #{index}: 'relevant_document_ids' musi być listą tekstów.")
        reference_answer = record.get("reference_answer")
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise ValueError(f"Przykład #{index}: 'reference_answer' musi być tekstem lub null.")
        examples.append(
            EvaluationExample(
                question=question,
                relevant_document_ids=relevant,
                reference_answer=reference_answer,
            )
        )
    return examples


def load_examples(path: str | Path) -> list[EvaluationExample]:
    """Wczytuje golden set z pliku JSON (lista obiektów) na listę `EvaluationExample`."""
    content = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("Golden set musi być listą obiektów JSON.")
    return parse_examples(parsed)

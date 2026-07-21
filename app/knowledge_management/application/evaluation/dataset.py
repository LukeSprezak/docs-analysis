import json
from pathlib import Path
from typing import Any

from ...domain.evaluation import EvaluationExample


def parse_examples(raw_examples: list[dict[str, Any]]) -> list[EvaluationExample]:
    """Maps raw JSON records onto `EvaluationExample`, validating the required fields."""
    examples: list[EvaluationExample] = []
    for index, record in enumerate(raw_examples):
        question = record.get("question")
        relevant = record.get("relevant_document_ids")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Example #{index}: missing a non-empty 'question' field.")
        if not isinstance(relevant, list) or not all(isinstance(item, str) for item in relevant):
            raise ValueError(
                f"Example #{index}: 'relevant_document_ids' must be a list of strings."
            )
        reference_answer = record.get("reference_answer")
        if reference_answer is not None and not isinstance(reference_answer, str):
            raise ValueError(f"Example #{index}: 'reference_answer' must be a string or null.")
        examples.append(
            EvaluationExample(
                question=question,
                relevant_document_ids=relevant,
                reference_answer=reference_answer,
            )
        )
    return examples


def load_examples(path: str | Path) -> list[EvaluationExample]:
    """Loads the golden set from a JSON file (a list of objects) into `EvaluationExample`s."""
    content = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(content)
    if not isinstance(parsed, list):
        raise ValueError("Golden set must be a list of JSON objects.")
    return parse_examples(parsed)

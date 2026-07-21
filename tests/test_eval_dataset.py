import json

import pytest

from app.knowledge_management.application.evaluation.dataset import load_examples, parse_examples


def test_parse_examples_maps_fields():
    examples = parse_examples(
        [
            {
                "question": "Q1",
                "relevant_document_ids": ["a.pdf"],
                "reference_answer": "ref",
            },
            {"question": "Q2", "relevant_document_ids": []},
        ]
    )
    assert len(examples) == 2
    assert examples[0].question == "Q1"
    assert examples[0].relevant_document_ids == ["a.pdf"]
    assert examples[0].reference_answer == "ref"
    assert examples[1].reference_answer is None


def test_parse_examples_rejects_missing_question():
    with pytest.raises(ValueError, match="question"):
        parse_examples([{"relevant_document_ids": ["a.pdf"]}])


def test_parse_examples_rejects_non_list_relevant_ids():
    with pytest.raises(ValueError, match="relevant_document_ids"):
        parse_examples([{"question": "Q", "relevant_document_ids": "a.pdf"}])


def test_load_examples_reads_json_file(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps([{"question": "Q", "relevant_document_ids": ["a.pdf"]}]),
        encoding="utf-8",
    )
    examples = load_examples(path)
    assert len(examples) == 1
    assert examples[0].question == "Q"


def test_load_examples_rejects_non_list_root(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"question": "Q"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        load_examples(path)

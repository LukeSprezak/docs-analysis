from app.knowledge_management.domain.models import Document
from app.knowledge_management.ui.api.sources import format_sources


def test_format_sources_uses_filename_and_page():
    docs = [
        Document(id="x", content="...", metadata={"filename": "raport.pdf", "page": 3}),
    ]
    assert format_sources(docs) == ["raport.pdf (page 3)"]


def test_format_sources_deduplicates_chunks_from_same_location():
    docs = [
        Document(id="x", content="a", metadata={"filename": "raport.pdf", "page": 3}),
        Document(id="x", content="b", metadata={"filename": "raport.pdf", "page": 3}),
        Document(id="x", content="c", metadata={"filename": "raport.pdf", "page": 5}),
    ]
    assert format_sources(docs) == ["raport.pdf (page 3)", "raport.pdf (page 5)"]


def test_format_sources_without_page():
    docs = [Document(id="notes.txt", content="...", metadata={"filename": "notes.txt"})]
    assert format_sources(docs) == ["notes.txt"]


def test_format_sources_falls_back_to_doc_id_then_id():
    docs = [
        Document(id="fallback-id", content="...", metadata={"doc_id": "from-meta"}),
        Document(id="only-id", content="...", metadata={}),
    ]
    assert format_sources(docs) == ["from-meta", "only-id"]


def test_format_sources_empty():
    assert format_sources([]) == []

from app.knowledge_management.domain.models import Document
from app.knowledge_management.infrastructure.text.text_chunker import TextChunker


def test_chunker_splits_long_document_into_multiple_chunks():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    long_text = "Test sentence. " * 100
    doc = Document(id="doc.txt", content=long_text, metadata={"filename": "doc.txt"})

    chunks = chunker.chunk_many([doc])

    assert len(chunks) > 1
    assert all(len(c.content) <= 120 for c in chunks)


def test_chunker_preserves_metadata_and_assigns_unique_ids():
    chunker = TextChunker(chunk_size=50, chunk_overlap=0)
    doc = Document(
        id="report.pdf",
        content="First paragraph.\n\nThe second paragraph is a little longer than the first.",
        metadata={"filename": "report.pdf", "page": 1},
    )

    chunks = chunker.chunk_many([doc])

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))  # unique id
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["filename"] == "report.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["doc_id"] == "report.pdf"
        assert chunk.metadata["chunk_index"] == index


def test_chunker_keeps_global_index_across_pages():
    chunker = TextChunker(chunk_size=40, chunk_overlap=0)
    pages = [
        Document(id="f.pdf", content="Page one has its own text here.", metadata={"page": 1}),
        Document(id="f.pdf", content="Page two has different text to be divided.", metadata={"page": 2}),
    ]

    chunks = chunker.chunk_many(pages)

    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))  # No ID conflicts between pages
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunker_skips_empty_documents():
    chunker = TextChunker()
    chunks = chunker.chunk_many([Document(id="empty", content="   \n  ", metadata={})])
    assert chunks == []

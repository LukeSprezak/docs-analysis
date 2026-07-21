from langchain_core.embeddings import DeterministicFakeEmbedding

from app.knowledge_management.domain.models import Document
from app.knowledge_management.infrastructure.persistence.faiss_vectorstore_repo import (
    FaissVectorStoreRepo,
)
from app.knowledge_management.infrastructure.text.text_chunker import TextChunker


def _repo() -> FaissVectorStoreRepo:
    # Small, deterministic embeddings (no network); large chunks = 1 fragment per document.
    embeddings = DeterministicFakeEmbedding(size=16)
    return FaissVectorStoreRepo(embeddings=embeddings, chunker=TextChunker(chunk_size=10_000))


async def test_search_on_empty_store_returns_empty():
    assert await _repo().search("anything", owner_id="u1") == []


async def test_add_then_search_returns_real_document_id_from_metadata():
    repo = _repo()
    await repo.add_documents(
        [
            Document(
                id="report.pdf",
                content="Content about quicksort",
                metadata={"filename": "report.pdf", "page": 1},
            )
        ],
        owner_id="u1",
    )

    results = await repo.search("quicksort", owner_id="u1", top_k=4)

    assert len(results) == 1
    assert results[0].id == "report.pdf"  # not "unknown"
    assert results[0].metadata["page"] == 1


async def test_add_documents_chunks_long_content():
    repo = FaissVectorStoreRepo(
        embeddings=DeterministicFakeEmbedding(size=16),
        chunker=TextChunker(chunk_size=100, chunk_overlap=0),
    )
    long_text = "A sentence about data structures. " * 50
    await repo.add_documents(
        [Document(id="d.txt", content=long_text, metadata={"filename": "d.txt"})], owner_id="u1"
    )

    # More than one result -> chunking actually happened.
    results = await repo.search("data structures", owner_id="u1", top_k=100)
    assert len(results) > 1


async def test_delete_by_document_id_removes_only_that_document():
    repo = _repo()
    await repo.add_documents(
        [Document(id="a.txt", content="alpha", metadata={"filename": "a.txt"})], owner_id="u1"
    )
    await repo.add_documents(
        [Document(id="b.txt", content="beta", metadata={"filename": "b.txt"})], owner_id="u1"
    )

    await repo.delete_by_document_id("a.txt", owner_id="u1")

    remaining_ids = {doc.id for doc in await repo.search("anything", owner_id="u1", top_k=100)}
    assert "a.txt" not in remaining_ids
    assert "b.txt" in remaining_ids


async def test_delete_on_empty_store_is_noop():
    repo = _repo()
    await repo.delete_by_document_id("does-not-exist", owner_id="u1")  # does not raise


async def test_reupload_replaces_chunks_without_orphans():
    # A re-upload with fewer fragments must not leave the old (orphaned) ones behind.
    repo = FaissVectorStoreRepo(
        embeddings=DeterministicFakeEmbedding(size=16),
        chunker=TextChunker(chunk_size=100, chunk_overlap=0),
    )
    long_text = "A sentence about data structures. " * 50
    await repo.add_documents(
        [Document(id="d.txt", content=long_text, metadata={"filename": "d.txt"})], owner_id="u1"
    )
    chunk_count_before = len(await repo.search("structures", owner_id="u1", top_k=1000))
    assert chunk_count_before > 1

    # Re-uploading the same document with shorter content → 1 fragment.
    await repo.add_documents(
        [Document(id="d.txt", content="short", metadata={"filename": "d.txt"})], owner_id="u1"
    )

    remaining = await repo.search("short structures", owner_id="u1", top_k=1000)
    assert len(remaining) == 1
    assert remaining[0].content == "short"


async def test_search_isolates_documents_by_owner():
    # Retrieval isolation: a user's query never returns someone else's fragments.
    repo = _repo()
    await repo.add_documents(
        [
            Document(
                id="secret.pdf",
                content="alice's secret about quicksort",
                metadata={"filename": "secret.pdf"},
            )
        ],
        owner_id="alice",
    )
    await repo.add_documents(
        [
            Document(
                id="bob.pdf",
                content="bob's notes about quicksort",
                metadata={"filename": "bob.pdf"},
            )
        ],
        owner_id="bob",
    )

    alice_results = {doc.id for doc in await repo.search("quicksort", owner_id="alice", top_k=100)}
    assert alice_results == {"secret.pdf"}
    assert "bob.pdf" not in alice_results

from app.retrieval.indexing import IndexDocumentUseCase, RemoveFromIndexUseCase
from app.retrieval.models import Entity, GraphFragment
from app.retrieval.ports import EntityExtractor, KnowledgeGraphRepo
from app.shared.kernel.document import Document
from tests.fakes import StubVectorStoreRepo


class RecordingVectorRepo(StubVectorStoreRepo):
    def __init__(self, calls: list[tuple[str, str, str]]) -> None:
        self.calls = calls
        self.added: list[tuple[list[Document], str]] = []

    async def add_documents(self, documents: list[Document], owner_id: str) -> None:
        self.added.append((documents, owner_id))

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        self.calls.append(("vectors", doc_id, owner_id))


class RecordingGraphRepo(KnowledgeGraphRepo):
    def __init__(self, calls: list[tuple[str, str, str]]) -> None:
        self.calls = calls
        self.fragments: list[tuple[GraphFragment, str]] = []

    async def add_fragment(self, fragment: GraphFragment, owner_id: str) -> None:
        self.fragments.append((fragment, owner_id))

    async def search_related(self, query: str, owner_id: str, top_k: int = 4) -> list[Document]:
        raise NotImplementedError

    async def delete_by_document_id(self, doc_id: str, owner_id: str) -> None:
        self.calls.append(("graph", doc_id, owner_id))


class RecordingExtractor(EntityExtractor):
    def __init__(self) -> None:
        self.seen: list[Document] = []

    async def extract(self, document: Document) -> GraphFragment:
        self.seen.append(document)
        return GraphFragment(
            doc_id=document.id, entities=[Entity(name="quicksort", type="Algorithm")], relations=[]
        )


async def test_index_extracts_graph_facts_from_the_whole_document_not_the_pages():
    calls: list[tuple[str, str, str]] = []
    vector_repo = RecordingVectorRepo(calls)
    graph_repo = RecordingGraphRepo(calls)
    extractor = RecordingExtractor()
    document = Document(id="o1::r.pdf", content="page 1\n\npage 2", metadata={"filename": "r.pdf"})
    pages = [
        Document(id="ignored", content="page 1", metadata={"page": 1}),
        Document(id="ignored", content="page 2", metadata={"page": 2}),
    ]

    await IndexDocumentUseCase(vector_repo, graph_repo, extractor).execute(document, "o1", pages)

    added_documents, added_owner = vector_repo.added[0]
    assert added_owner == "o1"
    assert [page.metadata["page"] for page in added_documents] == [1, 2]
    # Relations span page boundaries, so the extractor sees the whole document.
    assert extractor.seen == [document]
    fragment, fragment_owner = graph_repo.fragments[0]
    assert fragment.doc_id == "o1::r.pdf"
    assert fragment_owner == "o1"


async def test_remove_from_index_retracts_vectors_then_graph_facts():
    calls: list[tuple[str, str, str]] = []

    await RemoveFromIndexUseCase(RecordingVectorRepo(calls), RecordingGraphRepo(calls)).execute(
        "o1::a.txt", "o1"
    )

    assert calls == [("vectors", "o1::a.txt", "o1"), ("graph", "o1::a.txt", "o1")]

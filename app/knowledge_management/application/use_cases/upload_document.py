from ...domain.repositories import DocumentRepo, VectorStoreRepo
from ...domain.models import Document

class UploadDocumentUseCase:
    def __init__(self, doc_repo: DocumentRepo, vector_repo: VectorStoreRepo):
        self.doc_repo = doc_repo
        self.vector_repo = vector_repo

    def execute(self, doc_id: str, content: str, metadata: dict) -> Document:
        document = Document(id=doc_id, content=content, metadata=metadata)
        self.doc_repo.save(document)
        self.vector_repo.add_documents([document])
        return document

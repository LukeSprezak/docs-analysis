from typing import List, Optional, Dict
from ...domain.repositories import DocumentRepo
from ...domain.models import Document

class FilesystemDocumentRepo(DocumentRepo):
    def __init__(self):
        self._storage: Dict[str, Document] = {}

    def save(self, document: Document) -> None:
        self._storage[document.id] = document

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        return self._storage.get(doc_id)

    def list_all(self) -> List[Document]:
        return list(self._storage.values())

from typing import List
from ...domain.repositories import VectorStoreRepo
from ...domain.models import Document

class FaissVectorStoreRepo(VectorStoreRepo):
    def __init__(self):
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)

    def search(self, query: str, top_k: int = 4) -> List[Document]:
        return self.documents[:top_k]

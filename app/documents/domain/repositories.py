from abc import ABC, abstractmethod

from app.shared.kernel.document import Document


class DocumentRepo(ABC):
    @abstractmethod
    async def save(self, document: Document, owner_id: str) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, doc_id: str, owner_id: str) -> Document | None:
        pass

    @abstractmethod
    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Document]:
        pass

    @abstractmethod
    async def delete(self, doc_id: str, owner_id: str) -> None:
        pass

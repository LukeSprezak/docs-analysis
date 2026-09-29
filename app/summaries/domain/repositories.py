from abc import ABC, abstractmethod

from app.shared.kernel.document import Document
from app.summaries.domain.models import Summary


class SummarizerService(ABC):
    @abstractmethod
    async def summarize(self, documents: list[Document]) -> str:
        pass


class SummaryRepo(ABC):
    @abstractmethod
    async def save(self, text: str, document_ids: list[str], owner_id: str) -> Summary:
        """Stores a new summary and returns it, with the id and timestamp the store assigned.

        Takes the content rather than a `Summary`: the identity is the store's to mint, so
        there is no half-built summary for a caller to hold on to."""
        pass

    @abstractmethod
    async def get_by_id(self, summary_id: str, owner_id: str) -> Summary | None:
        pass

    @abstractmethod
    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Summary]:
        pass

    @abstractmethod
    async def delete(self, summary_id: str, owner_id: str) -> None:
        pass

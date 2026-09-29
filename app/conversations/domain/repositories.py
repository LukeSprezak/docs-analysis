from abc import ABC, abstractmethod

from app.conversations.domain.models import Conversation


class ConversationRepo(ABC):
    @abstractmethod
    async def save(self, conversation: Conversation, owner_id: str) -> None:
        pass

    @abstractmethod
    async def get_by_id(self, conversation_id: str, owner_id: str) -> Conversation | None:
        pass

    @abstractmethod
    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]:
        pass

    @abstractmethod
    async def delete(self, conversation_id: str, owner_id: str) -> None:
        pass

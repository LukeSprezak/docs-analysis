from ...domain.models import Conversation
from ...domain.repositories import ConversationRepo


class ListConversationsUseCase:
    def __init__(self, conversation_repo: ConversationRepo):
        self.conversation_repo = conversation_repo

    async def execute(
        self, owner_id: str, limit: int = 50, offset: int = 0
    ) -> list[Conversation]:
        return await self.conversation_repo.list_all(owner_id, limit=limit, offset=offset)

class GetConversationUseCase:
    def __init__(self, conversation_repo: ConversationRepo):
        self.conversation_repo = conversation_repo

    async def execute(self, conversation_id: str, owner_id: str) -> Conversation | None:
        return await self.conversation_repo.get_by_id(conversation_id, owner_id)

class DeleteConversationUseCase:
    def __init__(self, conversation_repo: ConversationRepo):
        self.conversation_repo = conversation_repo

    async def execute(self, conversation_id: str, owner_id: str) -> None:
        await self.conversation_repo.delete(conversation_id, owner_id)

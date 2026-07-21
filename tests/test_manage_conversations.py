from app.knowledge_management.application.use_cases.manage_conversations import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
)
from app.knowledge_management.domain.models import Conversation
from tests.fakes import StubConversationRepo


class FakeConversationRepo(StubConversationRepo):
    def __init__(
        self,
        conversations: list[Conversation] | None = None,
        single: Conversation | None = None,
    ) -> None:
        self._conversations = conversations or []
        self._single = single
        self.list_args: tuple[str, int, int] | None = None
        self.get_args: tuple[str, str] | None = None
        self.deleted: tuple[str, str] | None = None

    async def list_all(self, owner_id: str, limit: int = 50, offset: int = 0) -> list[Conversation]:
        self.list_args = (owner_id, limit, offset)
        return self._conversations

    async def get_by_id(self, conversation_id: str, owner_id: str) -> Conversation | None:
        self.get_args = (conversation_id, owner_id)
        return self._single

    async def delete(self, conversation_id: str, owner_id: str) -> None:
        self.deleted = (conversation_id, owner_id)


async def test_list_passes_owner_and_pagination_to_repo():
    conversations = [Conversation(id="c1", title="t", messages=[])]
    repo = FakeConversationRepo(conversations=conversations)

    result = await ListConversationsUseCase(repo).execute("o1", limit=10, offset=5)

    assert result == conversations
    assert repo.list_args == ("o1", 10, 5)


async def test_get_filters_by_owner():
    conversation = Conversation(id="c1", title="t", messages=[])
    repo = FakeConversationRepo(single=conversation)

    result = await GetConversationUseCase(repo).execute("c1", "o1")

    assert result is conversation
    assert repo.get_args == ("c1", "o1")


async def test_delete_passes_owner_to_repo():
    repo = FakeConversationRepo()

    await DeleteConversationUseCase(repo).execute("c1", "o1")

    assert repo.deleted == ("c1", "o1")

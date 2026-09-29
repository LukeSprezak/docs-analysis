from app.conversations.domain.repositories import ConversationRepo
from app.conversations.infrastructure.postgres_conversation_repo import PostgresConversationRepo
from app.shared.config import settings
from app.shared.enums import PersistenceProvider


def create_conversation_repo() -> ConversationRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresConversationRepo()
    raise NotImplementedError(
        f"No ConversationRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'"
    )

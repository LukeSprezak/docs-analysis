"""Selects the persistence adapter for the Identity context.

The counterpart of `knowledge_management/infrastructure/persistence/factory.py`: the only
place naming a concrete `UserRepo` implementation, driven by the same global
`PERSISTENCE_PROVIDER` switch so both contexts move to a new backing store together.
"""

from app.identity.domain.repositories import UserRepo
from app.identity.infrastructure.postgres_user_repo import PostgresUserRepo
from app.shared.config import settings
from app.shared.enums import PersistenceProvider


def create_user_repo() -> UserRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresUserRepo()
    raise NotImplementedError(f"No UserRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'")

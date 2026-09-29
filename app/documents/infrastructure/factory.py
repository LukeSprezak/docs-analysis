from app.documents.domain.repositories import DocumentRepo
from app.documents.infrastructure.postgres_document_repo import PostgresDocumentRepo
from app.shared.config import settings
from app.shared.enums import PersistenceProvider


def create_document_repo() -> DocumentRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresDocumentRepo()
    raise NotImplementedError(
        f"No DocumentRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'"
    )

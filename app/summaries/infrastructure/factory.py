from app.shared.config import settings
from app.shared.enums import PersistenceProvider
from app.summaries.domain.repositories import SummaryRepo
from app.summaries.infrastructure.postgres_summary_repo import PostgresSummaryRepo


def create_summary_repo() -> SummaryRepo:
    if settings.PERSISTENCE_PROVIDER == PersistenceProvider.POSTGRES:
        return PostgresSummaryRepo()
    raise NotImplementedError(
        f"No SummaryRepo adapter for provider '{settings.PERSISTENCE_PROVIDER}'"
    )

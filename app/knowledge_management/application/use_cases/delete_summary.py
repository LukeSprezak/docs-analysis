from ...domain.repositories import SummaryRepo


class DeleteSummaryUseCase:
    def __init__(self, summary_repo: SummaryRepo):
        self.summary_repo = summary_repo

    async def execute(self, summary_id: str, owner_id: str) -> None:
        await self.summary_repo.delete(summary_id, owner_id)

from app.shared.exceptions import ValidationException

from ...domain.models import Summary
from ...domain.repositories import DocumentRepo, SummarizerService, SummaryRepo


class SummarizeDocsUseCase:
    def __init__(
        self, doc_repo: DocumentRepo, summarizer: SummarizerService, summary_repo: SummaryRepo
    ):
        self.doc_repo = doc_repo
        self.summarizer = summarizer
        self.summary_repo = summary_repo

    async def execute(self, doc_ids: list[str], owner_id: str) -> Summary:
        documents = []
        for doc_id in doc_ids:
            # get_by_id filters by owner_id — you can only summarize your own documents.
            doc = await self.doc_repo.get_by_id(doc_id, owner_id)
            if doc:
                documents.append(doc)

        # Every requested id was missing or belongs to someone else. Summarizing nothing is
        # a paid LLM call on an empty prompt, and the summary it produces would be stored
        # claiming documents it never saw — a stale list in the UI is enough to get here.
        if not documents:
            raise ValidationException("None of the requested documents were found")

        summary_text = await self.summarizer.summarize(documents)
        # The ids actually summarized, not the ones asked for: the two differ whenever the
        # loop above skipped something, and the stored summary has to describe its own input.
        return await self.summary_repo.save(
            summary_text, [document.id for document in documents], owner_id
        )

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
            # get_by_id filtruje po owner_id — można streszczać tylko własne dokumenty.
            doc = await self.doc_repo.get_by_id(doc_id, owner_id)
            if doc:
                documents.append(doc)

        summary_text = await self.summarizer.summarize(documents)
        summary = Summary(text=summary_text, document_ids=doc_ids)
        await self.summary_repo.save(summary, owner_id)
        return summary

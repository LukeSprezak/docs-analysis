from typing import List
from ...domain.repositories import DocumentRepo, SummarizerService
from ...domain.models import Summary


class SummarizeDocsUseCase:
    def __init__(self, doc_repo: DocumentRepo, summarizer: SummarizerService):
        self.doc_repo = doc_repo
        self.summarizer = summarizer

    def execute(self, doc_ids: List[str]) -> Summary:
        documents = []
        for doc_id in doc_ids:
            doc = self.doc_repo.get_by_id(doc_id)
            if doc:
                documents.append(doc)

        summary_text = self.summarizer.summarize(documents)
        return Summary(text=summary_text, document_ids=doc_ids)

from typing import List
from ...domain.repositories import SummarizerService
from ...domain.models import Document

class LangChainSummarizer(SummarizerService):
    def summarize(self, documents: List[Document]) -> str:
        count = len(documents)
        return f"Podsumowanie {count} dokumentów."

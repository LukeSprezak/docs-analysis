from app.shared.llm.llm_factory import LLMFactory
from app.summaries.repo import PostgresSummaryRepo, SummaryRepo
from app.summaries.summarizer import LangChainSummarizer, SummarizerService


def get_summary_repo() -> SummaryRepo:
    return PostgresSummaryRepo()


def get_summarizer() -> SummarizerService:
    return LangChainSummarizer(llm=LLMFactory.get_llm())

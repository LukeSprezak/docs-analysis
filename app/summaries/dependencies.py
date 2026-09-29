import threading
from typing import Annotated

from fastapi import Depends

from app.documents.api import DocumentRepo, get_doc_repo
from app.shared.llm.llm_factory import LLMFactory
from app.summaries.application.delete_summary import DeleteSummaryUseCase
from app.summaries.application.summarize_docs import SummarizeDocsUseCase
from app.summaries.domain.repositories import SummaryRepo
from app.summaries.infrastructure import factory
from app.summaries.infrastructure.langchain_summarizer import LangChainSummarizer

_singleton_lock = threading.Lock()

_summary_repo: SummaryRepo | None = None


def get_summary_repo() -> SummaryRepo:
    global _summary_repo
    with _singleton_lock:
        if _summary_repo is None:
            _summary_repo = factory.create_summary_repo()
        return _summary_repo


def init() -> None:
    get_summary_repo()


async def shutdown() -> None:
    # Nothing to close: the repository borrows the shared pool that `dispose_engine()` closes.
    global _summary_repo
    _summary_repo = None


def get_summarizer() -> LangChainSummarizer:
    return LangChainSummarizer(llm=LLMFactory.get_llm())


def get_summarize_docs_use_case(
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    summarizer: Annotated[LangChainSummarizer, Depends(get_summarizer)],
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
) -> SummarizeDocsUseCase:
    return SummarizeDocsUseCase(doc_repo, summarizer, summary_repo)


def get_delete_summary_use_case(
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
) -> DeleteSummaryUseCase:
    return DeleteSummaryUseCase(summary_repo)

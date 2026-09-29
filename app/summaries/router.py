from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.documents.api import DocumentRepo, get_doc_repo
from app.identity.dependencies import get_current_user
from app.identity.models import User
from app.shared.config import settings
from app.shared.rate_limit import limiter
from app.summaries.dependencies import get_summarizer, get_summary_repo
from app.summaries.repo import SummaryRepo
from app.summaries.service import summarize_docs
from app.summaries.summarizer import SummarizerService

router = APIRouter(prefix="/summarize", tags=["summarize"])


class SummarizeRequest(BaseModel):
    document_ids: list[str]


class SummarizeResponse(BaseModel):
    # Not optional: a summary only ever leaves through here after `SummaryRepo.save`, which
    # mints the id and the timestamp (pinned by the contract suite). Declaring them nullable
    # pushed the same lie into the client's types, where nothing would have caught an adapter
    # that stopped filling them in — here pydantic fails the response instead.
    summary: str
    document_ids: list[str]
    id: str
    created_at: str


# 201: the request creates a summary — it is stored and comes back with its own id.
@router.post("/", response_model=SummarizeResponse, status_code=201)
@limiter.limit(settings.RATE_LIMIT_LLM)
async def create_summary(
    request: Request,
    summarize_request: SummarizeRequest,
    doc_repo: Annotated[DocumentRepo, Depends(get_doc_repo)],
    summarizer: Annotated[SummarizerService, Depends(get_summarizer)],
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> SummarizeResponse:
    result = await summarize_docs(
        doc_repo, summarizer, summary_repo, summarize_request.document_ids, current_user.id
    )
    return SummarizeResponse(
        summary=result.text,
        document_ids=result.document_ids,
        id=result.id,
        created_at=result.created_at,
    )


@router.get("/", response_model=list[SummarizeResponse])
async def list_summaries(
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=settings.LIST_MAX_LIMIT)] = settings.LIST_DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[SummarizeResponse]:
    summaries = await summary_repo.list_all(current_user.id, limit=limit, offset=offset)
    return [
        SummarizeResponse(
            summary=s.text, document_ids=s.document_ids, id=s.id, created_at=s.created_at
        )
        for s in summaries
    ]


@router.delete("/{summary_id}")
async def delete_summary(
    summary_id: UUID,
    summary_repo: Annotated[SummaryRepo, Depends(get_summary_repo)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    await summary_repo.delete(str(summary_id), current_user.id)
    return {"status": "success"}

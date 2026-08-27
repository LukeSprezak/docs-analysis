from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.knowledge_management.application.use_cases.delete_summary import DeleteSummaryUseCase
from app.knowledge_management.application.use_cases.summarize_docs import SummarizeDocsUseCase
from app.knowledge_management.domain.repositories import SummaryRepo
from app.shared.config import settings
from app.shared.dependencies import (
    get_delete_summary_use_case,
    get_summarize_docs_use_case,
    get_summary_repo,
)
from app.shared.rate_limit import limiter

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
async def summarize_docs(
    request: Request,
    summarize_request: SummarizeRequest,
    use_case: Annotated[SummarizeDocsUseCase, Depends(get_summarize_docs_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> SummarizeResponse:
    result = await use_case.execute(summarize_request.document_ids, current_user.id)
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
    use_case: Annotated[DeleteSummaryUseCase, Depends(get_delete_summary_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    await use_case.execute(str(summary_id), current_user.id)
    return {"status": "success"}

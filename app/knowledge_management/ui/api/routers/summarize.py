from typing import Annotated

from fastapi import APIRouter, Depends, Query
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

router = APIRouter(prefix="/summarize", tags=["summarize"])


class SummarizeRequest(BaseModel):
    document_ids: list[str]


class SummarizeResponse(BaseModel):
    summary: str
    document_ids: list[str]
    id: str | None = None
    created_at: str | None = None


@router.post("/", response_model=SummarizeResponse)
async def summarize_docs(
    request: SummarizeRequest,
    use_case: Annotated[SummarizeDocsUseCase, Depends(get_summarize_docs_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> SummarizeResponse:
    result = await use_case.execute(request.document_ids, current_user.id)
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
    summary_id: str,
    use_case: Annotated[DeleteSummaryUseCase, Depends(get_delete_summary_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    await use_case.execute(summary_id, current_user.id)
    return {"status": "success"}

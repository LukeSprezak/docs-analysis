from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.identity.dependencies import get_current_user
from app.identity.models import User
from app.retrieval.dependencies import get_rag_service, get_retrieval_pipeline
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.ports import RAGService
from app.shared.config import settings
from app.shared.kernel.document_identity import citation_label
from app.shared.rate_limit import limiter

router = APIRouter(prefix="/qa", tags=["qa"])


class AskQuestionCommand(BaseModel):
    question: str


class SourceResponse(BaseModel):
    source: str
    page: int | None
    # Vector similarity (0-1); None for keyword-only hits and knowledge-graph facts.
    score: float | None
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


@router.post("/ask", response_model=AnswerResponse)
@limiter.limit(settings.RATE_LIMIT_LLM)
async def ask_question(
    request: Request,
    command: AskQuestionCommand,
    retrieval_pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
    rag_service: Annotated[RAGService, Depends(get_rag_service)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> AnswerResponse:
    # Retrieval is limited to the asker's own documents (owner_id).
    sources = await retrieval_pipeline.retrieve(command.question, current_user.id)
    answer = await rag_service.answer_question(command.question, sources)
    return AnswerResponse(
        answer=answer,
        sources=[
            SourceResponse(
                source=citation_label(document),
                page=document.metadata.get("page"),
                score=document.metadata.get("score"),
                snippet=document.content[:200],
            )
            for document in sources
        ],
    )

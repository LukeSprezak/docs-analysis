from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.knowledge_management.application.use_cases.ask_question import AskQuestionUseCase
from app.knowledge_management.ui.api.sources import format_sources
from app.shared.config import settings
from app.shared.dependencies import get_ask_question_use_case
from app.shared.rate_limit import limiter

router = APIRouter(prefix="/qa", tags=["qa"])


class AskQuestionCommand(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/ask", response_model=AnswerResponse)
@limiter.limit(settings.RATE_LIMIT_LLM)
async def ask_question(
    request: Request,
    command: AskQuestionCommand,
    use_case: Annotated[AskQuestionUseCase, Depends(get_ask_question_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> AnswerResponse:
    result = await use_case.execute(command.question, current_user.id)
    return AnswerResponse(answer=result.text, sources=format_sources(result.sources))

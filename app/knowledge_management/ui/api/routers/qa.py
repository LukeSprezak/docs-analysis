from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/qa", tags=["qa"])

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    return AnswerResponse(
        answer=f"To jest przykładowa odpowiedź na: {request.question}",
        sources=["doc1.pdf"]
    )

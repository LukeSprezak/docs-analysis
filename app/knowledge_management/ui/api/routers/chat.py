from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return ChatResponse(
        answer=f"Odpowiedź na czacie: {request.message}",
        sources=[]
    )

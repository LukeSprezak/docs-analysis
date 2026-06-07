from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/summarize", tags=["summarize"])

class SummarizeRequest(BaseModel):
    document_ids: List[str]

class SummarizeResponse(BaseModel):
    summary: str
    document_ids: List[str]

@router.post("/", response_model=SummarizeResponse)
async def summarize_docs(request: SummarizeRequest):
    return SummarizeResponse(
        summary="To jest podsumowanie wybranych dokumentów.",
        document_ids=request.document_ids
    )

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.identity.dependencies import get_current_user
from app.identity.domain.models import User
from app.knowledge_management.application.use_cases.chat_with_docs import ChatWithDocsUseCase
from app.knowledge_management.application.use_cases.manage_conversations import (
    DeleteConversationUseCase,
    GetConversationUseCase,
    ListConversationsUseCase,
)
from app.knowledge_management.ui.api.sources import format_sources
from app.shared.config import settings
from app.shared.dependencies import (
    get_chat_with_docs_use_case,
    get_delete_conversation_use_case,
    get_get_conversation_use_case,
    get_list_conversations_use_case,
)
from app.shared.exceptions import EntityNotFoundException
from app.shared.rate_limit import limiter

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessageSchema(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    conversation_id: str


class ConversationSchema(BaseModel):
    id: str
    title: str
    messages: list[ChatMessageSchema]
    created_at: str | None = None


@router.post("/", response_model=ChatResponse)
@limiter.limit(settings.RATE_LIMIT_LLM)
async def chat(
    request: Request,
    chat_request: ChatRequest,
    use_case: Annotated[ChatWithDocsUseCase, Depends(get_chat_with_docs_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> ChatResponse:
    result, conv_id = await use_case.execute(
        chat_request.message, current_user.id, chat_request.conversation_id
    )
    return ChatResponse(
        answer=result.text, sources=format_sources(result.sources), conversation_id=conv_id
    )


@router.post("/stream")
@limiter.limit(settings.RATE_LIMIT_LLM)
async def chat_stream(
    request: Request,
    chat_request: ChatRequest,
    use_case: Annotated[ChatWithDocsUseCase, Depends(get_chat_with_docs_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> StreamingResponse:
    """Streams the answer as NDJSON: {"type":"token"...} lines, then a final
    {"type":"done","conversation_id":...,"sources":[...]}."""
    owner_id = current_user.id

    def encode(event: dict[str, Any]) -> str:
        if event["type"] == "done":
            payload: dict[str, Any] = {
                "type": "done",
                "conversation_id": event["conversation_id"],
                "sources": format_sources(event["sources"]),
            }
        else:
            payload = event
        return json.dumps(payload, ensure_ascii=False) + "\n"

    stream = use_case.execute_stream(chat_request.message, owner_id, chat_request.conversation_id)
    # The first event is pulled here, outside the response body. Everything that can still
    # fail with a status code — an unknown conversation id above all — happens on that first
    # step, and once StreamingResponse starts iterating the 200 headers are already sent:
    # an exception raised in there cannot become a 404, it only truncates the stream.
    first_event = await anext(stream)

    async def generate() -> AsyncIterator[str]:
        yield encode(first_event)
        async for event in stream:
            yield encode(event)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/conversations", response_model=list[ConversationSchema])
async def list_conversations(
    use_case: Annotated[ListConversationsUseCase, Depends(get_list_conversations_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
    limit: Annotated[int, Query(ge=1, le=settings.LIST_MAX_LIMIT)] = settings.LIST_DEFAULT_LIMIT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ConversationSchema]:
    conversations = await use_case.execute(current_user.id, limit=limit, offset=offset)
    return [
        ConversationSchema(
            id=c.id,
            title=c.title,
            messages=[
                ChatMessageSchema(role=m.role, content=m.content, timestamp=m.timestamp)
                for m in c.messages
            ],
            created_at=c.created_at,
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationSchema)
async def get_conversation(
    conversation_id: str,
    use_case: Annotated[GetConversationUseCase, Depends(get_get_conversation_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> ConversationSchema:
    conversation = await use_case.execute(conversation_id, current_user.id)
    if not conversation:
        raise EntityNotFoundException(entity="Conversation", identifier=conversation_id)
    return ConversationSchema(
        id=conversation.id,
        title=conversation.title,
        messages=[
            ChatMessageSchema(role=m.role, content=m.content, timestamp=m.timestamp)
            for m in conversation.messages
        ],
        created_at=conversation.created_at,
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    use_case: Annotated[DeleteConversationUseCase, Depends(get_delete_conversation_use_case)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str]:
    await use_case.execute(conversation_id, current_user.id)
    return {"status": "success"}

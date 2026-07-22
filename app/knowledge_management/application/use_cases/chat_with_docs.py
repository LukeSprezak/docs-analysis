import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from ...domain.models import Answer, ChatMessage, Conversation, Document
from ...domain.null_knowledge_graph_repo import NullKnowledgeGraphRepo
from ...domain.repositories import (
    ConversationRepo,
    KnowledgeGraphRepo,
    RAGService,
    RerankerService,
    VectorStoreRepo,
)
from ..retrieval.candidate_retrieval import CandidateRetriever


class ChatWithDocsUseCase:
    def __init__(
        self,
        vector_repo: VectorStoreRepo,
        rag_service: RAGService,
        conversation_repo: ConversationRepo,
        reranker: RerankerService,
        candidate_count: int = 20,
        top_k: int = 4,
        graph_repo: KnowledgeGraphRepo | None = None,
    ):
        # See AskQuestionUseCase: the null graph keeps callers that ignore the feature simple.
        self.retriever = CandidateRetriever(vector_repo, graph_repo or NullKnowledgeGraphRepo())
        self.rag_service = rag_service
        self.conversation_repo = conversation_repo
        self.reranker = reranker
        self.candidate_count = candidate_count
        self.top_k = top_k

    async def _prepare_context(
        self,
        message: str,
        owner_id: str,
        history: list[dict[str, Any]] | None,
        conversation_id: str | None,
    ) -> tuple[Conversation, list[dict[str, str]], list[Document]]:
        """Loads the conversation, builds the history, rephrases the question and searches.

        The step shared by the plain and the streaming variant.
        """
        conversation = (
            await self.conversation_repo.get_by_id(conversation_id, owner_id)
            if conversation_id
            else None
        )
        if conversation is None:
            conversation_id = conversation_id or str(uuid.uuid4())
            conversation = Conversation(id=conversation_id, title=message[:50], messages=[])

        prior_messages = [{"role": m.role, "content": m.content} for m in conversation.messages]
        if not prior_messages and history:
            prior_messages = [
                {"role": item["role"], "content": item["content"]}
                for item in history
                if "role" in item and "content" in item
            ]

        # A context-dependent question ("and what about that?") becomes standalone, otherwise
        # retrieval searches for a meaningless phrase.
        search_query = (
            await self.rag_service.condense_question(message, prior_messages)
            if prior_messages
            else message
        )
        candidates = await self.retriever.retrieve(
            search_query, owner_id, candidate_count=self.candidate_count
        )
        relevant_documents = await self.reranker.rerank(search_query, candidates, top_k=self.top_k)
        return conversation, prior_messages, relevant_documents

    async def _persist_turn(
        self, conversation: Conversation, owner_id: str, message: str, answer_text: str
    ) -> None:
        timestamp = datetime.now().isoformat()
        conversation.messages.append(ChatMessage(role="user", content=message, timestamp=timestamp))
        conversation.messages.append(
            ChatMessage(role="assistant", content=answer_text, timestamp=timestamp)
        )
        await self.conversation_repo.save(conversation, owner_id)

    async def execute(
        self,
        message: str,
        owner_id: str,
        history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> tuple[Answer, str]:
        conversation, prior_messages, relevant_documents = await self._prepare_context(
            message, owner_id, history, conversation_id
        )
        answer_text = await self.rag_service.answer_question(
            message, relevant_documents, history=prior_messages
        )
        await self._persist_turn(conversation, owner_id, message, answer_text)

        answer = Answer(text=answer_text, sources=relevant_documents)
        return answer, conversation.id

    async def execute_stream(
        self,
        message: str,
        owner_id: str,
        history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streams events: {"type":"token",...} one after another, then {"type":"done",...}.

        Repo/condense/retrieval are async (connection pool + ainvoke), and the answer itself
        is streamed token by token from the LLM — nothing blocks the event loop.
        """
        conversation, prior_messages, relevant_documents = await self._prepare_context(
            message, owner_id, history, conversation_id
        )

        collected_tokens: list[str] = []
        async for token in self.rag_service.astream_answer(
            message, relevant_documents, history=prior_messages
        ):
            collected_tokens.append(token)
            yield {"type": "token", "content": token}

        answer_text = "".join(collected_tokens)
        await self._persist_turn(conversation, owner_id, message, answer_text)
        yield {
            "type": "done",
            "conversation_id": conversation.id,
            "sources": relevant_documents,
        }

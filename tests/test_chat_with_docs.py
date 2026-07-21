import asyncio
from unittest.mock import AsyncMock, MagicMock

from app.knowledge_management.application.use_cases.chat_with_docs import ChatWithDocsUseCase
from app.knowledge_management.domain.models import ChatMessage, Conversation, Document


def _vec(docs=None):
    vec = MagicMock()
    vec.search = AsyncMock(return_value=docs or [])
    return vec


def _passthrough_reranker():
    reranker = MagicMock()
    reranker.rerank = AsyncMock(side_effect=lambda query, documents, top_k=4: documents)
    return reranker


def test_chat_uses_persisted_history_for_condense_and_answer():
    conv = Conversation(
        id="c1",
        title="t",
        messages=[
            ChatMessage(role="user", content="What is quick sort?"),
            ChatMessage(role="assistant", content="This is a sorting algorithm."),
        ],
    )
    conv_repo = MagicMock()
    conv_repo.get_by_id = AsyncMock(return_value=conv)
    conv_repo.save = AsyncMock()
    rag = MagicMock()
    rag.condense_question = AsyncMock(return_value="What is the complexity of quicksort?")
    rag.answer_question = AsyncMock(return_value="O(n log n)")
    vec = _vec([Document(id="d", content="...", metadata={})])

    uc = ChatWithDocsUseCase(vec, rag, conv_repo, _passthrough_reranker())
    answer, cid = asyncio.run(
        uc.execute("And what's the complexity?", "owner1", history=None, conversation_id="c1")
    )

    # Condense receives a direct question + a conversation history
    rag.condense_question.assert_called_once()
    q_arg, hist_arg = rag.condense_question.call_args.args
    assert q_arg == "And what's the complexity?"
    assert hist_arg == [
        {"role": "user", "content": "What is the complexity of quicksort?"},
        {"role": "assistant", "content": "This is a sorting algorithm."},
    ]
    # retrieval after the REVISED question, not after the original one
    vec.search.assert_called_once()
    assert vec.search.call_args.args[0] == "Jaka jest złożoność quicksort?"
    # response generated with history
    assert rag.answer_question.call_args.kwargs["history"] == hist_arg
    assert answer.text == "O(n log n)"
    assert cid == "c1"
    conv_repo.save.assert_called_once()


def test_chat_first_turn_skips_condense():
    conv_repo = MagicMock()
    conv_repo.get_by_id = AsyncMock(return_value=None)
    conv_repo.save = AsyncMock()
    rag = MagicMock()
    rag.condense_question = AsyncMock()
    rag.answer_question = AsyncMock(return_value="ans")
    vec = _vec()

    uc = ChatWithDocsUseCase(vec, rag, conv_repo, _passthrough_reranker())
    _, cid = asyncio.run(
        uc.execute("First question", "owner1", history=None, conversation_id=None)
    )

    rag.condense_question.assert_not_called()
    vec.search.assert_called_once()
    assert vec.search.call_args.args[0] == "First question"
    assert rag.answer_question.call_args.kwargs["history"] == []
    assert cid  # generated UUID


def test_chat_falls_back_to_history_param_when_no_persisted_conversation():
    conv_repo = MagicMock()
    conv_repo.get_by_id = AsyncMock(return_value=None)
    conv_repo.save = AsyncMock()
    rag = MagicMock()
    rag.condense_question = AsyncMock(return_value="standalone")
    rag.answer_question = AsyncMock(return_value="a")
    vec = _vec()

    uc = ChatWithDocsUseCase(vec, rag, conv_repo, _passthrough_reranker())
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    asyncio.run(uc.execute("followup", "owner1", history=history, conversation_id=None))

    rag.condense_question.assert_called_once()
    assert rag.condense_question.call_args.args[1] == history
    vec.search.assert_called_once()
    assert vec.search.call_args.args[0] == "standalone"


def test_chat_reranks_candidates_before_answering():
    conv_repo = MagicMock()
    conv_repo.get_by_id = AsyncMock(return_value=None)
    conv_repo.save = AsyncMock()
    rag = MagicMock()
    rag.answer_question = AsyncMock(return_value="ans")
    candidates = [Document(id=str(i), content=f"c{i}", metadata={}) for i in range(5)]
    vec = _vec(candidates)
    reranker = MagicMock()
    # reranker flips and crops to 2
    reranker.rerank = AsyncMock(
        side_effect=lambda query, documents, top_k=4: list(reversed(documents))[:2]
    )

    uc = ChatWithDocsUseCase(vec, rag, conv_repo, reranker, candidate_count=5, top_k=2)
    answer, _ = asyncio.run(uc.execute("q", "owner1", conversation_id=None))

    # The reranker receives candidates from the vector search
    reranker.rerank.assert_called_once()
    assert reranker.rerank.call_args.args[1] == candidates
    # These sorted excerpts are included in the responses and serve as sources
    assert [document.id for document in answer.sources] == ["4", "3"]
    assert rag.answer_question.call_args.args[1] == [candidates[4], candidates[3]]


def test_execute_stream_yields_tokens_then_done_and_persists_full_answer():
    conv_repo = MagicMock()
    conv_repo.get_by_id = AsyncMock(return_value=None)
    conv_repo.save = AsyncMock()
    rag = MagicMock()
    rag.condense_question = AsyncMock(return_value="q")

    async def fake_astream(message, documents, history=None):
        for token in ["Hel", "lo"]:
            yield token

    rag.astream_answer = fake_astream
    vec = _vec([Document(id="d", content="c", metadata={"filename": "d"})])

    uc = ChatWithDocsUseCase(vec, rag, conv_repo, _passthrough_reranker())

    async def collect() -> list[dict]:
        return [event async for event in uc.execute_stream("question", "owner1", conversation_id=None)]

    events = asyncio.run(collect())

    token_events = [e["content"] for e in events if e["type"] == "token"]
    assert token_events == ["Hel", "lo"]

    done = events[-1]
    assert done["type"] == "done"
    assert done["conversation_id"]
    assert len(done["sources"]) == 1
    assert done["sources"][0].id == "d"

    # The complete, compiled answer is saved
    conv_repo.save.assert_called_once()
    saved_conversation = conv_repo.save.call_args.args[0]
    assert saved_conversation.messages[-1].role == "assistant"
    assert saved_conversation.messages[-1].content == "Hello"
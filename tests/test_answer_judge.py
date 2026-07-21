import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.knowledge_management.domain.models import Document
from app.knowledge_management.infrastructure.llm.answer_judge import LLMAnswerJudge


def _judge(content: str) -> LLMAnswerJudge:
    return LLMAnswerJudge(GenericFakeChatModel(messages=iter([AIMessage(content=content)])))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.8", 0.8),
        ("1.0", 1.0),
        ("0", 0.0),
        ("Score: 0,75", 0.75),  # comma as the decimal separator
        ("no idea", 0.0),  # unparsable → 0.0
        ("1.5", 1.0),  # clamped to [0, 1]
    ],
)
def test_parse_score(raw, expected):
    assert LLMAnswerJudge._parse_score(raw) == expected


async def test_score_faithfulness_parses_model_output():
    judge = _judge("0.9")
    context = [Document(id="d", content="content", metadata={})]
    assert await judge.score_faithfulness("answer", context) == 0.9


async def test_score_answer_relevance_parses_model_output():
    judge = _judge("0.4")
    assert await judge.score_answer_relevance("question", "answer") == 0.4
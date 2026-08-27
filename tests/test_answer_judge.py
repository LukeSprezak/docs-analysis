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
        ("0,75", 0.75),  # comma as the decimal separator
        (" 0.6\n", 0.6),  # surrounding whitespace
        ("1.5", 1.0),  # clamped to [0, 1]
        ("no idea", None),  # unparsable → no score, which is not a score of zero
        ("Score: 0.75", None),  # the prompt asks for the bare number and nothing else
        ("7/10", None),  # ERR-03: a loose search would read the `1` of `10` as a perfect score
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


async def test_unparsable_answer_scores_as_none_not_zero():
    """ERR-03: a failed measurement must not enter the mean as the worst possible result."""
    judge = _judge("I cannot rate this")
    assert await judge.score_answer_relevance("question", "answer") is None

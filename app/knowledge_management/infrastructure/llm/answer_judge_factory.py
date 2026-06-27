from app.shared.config import settings
from app.shared.enums import EvalJudgeProvider

from ...domain.repositories import AnswerJudge
from .answer_judge import LLMAnswerJudge
from .llm_factory import LLMFactory


class AnswerJudgeFactory:
    @staticmethod
    def get_judge() -> AnswerJudge | None:
        """Returns a judge according to `EVAL_JUDGE_PROVIDER` or ``None``, when generation evaluation
        is disabled (by default). ``None`` = means that eval calculates only retrieval metrics, without
        any LLM calls or an API key required"""

        if settings.EVAL_JUDGE_PROVIDER == EvalJudgeProvider.LLM:
            return LLMAnswerJudge(llm=LLMFactory.get_llm())
        return None

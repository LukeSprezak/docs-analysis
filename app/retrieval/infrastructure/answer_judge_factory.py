from app.retrieval.domain.repositories import AnswerJudge
from app.retrieval.infrastructure.answer_judge import LLMAnswerJudge
from app.shared.config import settings
from app.shared.enums import EvalJudgeProvider
from app.shared.llm.llm_factory import LLMFactory


class AnswerJudgeFactory:
    @staticmethod
    def get_judge() -> AnswerJudge | None:
        """Returns a judge per `EVAL_JUDGE_PROVIDER`, or ``None`` when generation scoring is
        disabled (the default). ``None`` means the eval computes retrieval metrics only,
        with no LLM call and no API key required."""
        if settings.EVAL_JUDGE_PROVIDER == EvalJudgeProvider.LLM:
            return LLMAnswerJudge(llm=LLMFactory.get_llm())
        return None

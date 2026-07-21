from app.shared.config import settings
from app.shared.enums import EvalJudgeProvider

from ...domain.repositories import AnswerJudge
from .answer_judge import LLMAnswerJudge
from .llm_factory import LLMFactory


class AnswerJudgeFactory:
    @staticmethod
    def get_judge() -> AnswerJudge | None:
        """Zwraca sędziego wg `EVAL_JUDGE_PROVIDER` albo ``None``, gdy ocena generacji
        jest wyłączona (domyślnie). ``None`` = eval liczy tylko metryki retrievalu, bez
        żadnego wywołania LLM ani wymaganego klucza API."""
        if settings.EVAL_JUDGE_PROVIDER == EvalJudgeProvider.LLM:
            return LLMAnswerJudge(llm=LLMFactory.get_llm())
        return None
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ...domain.models import Document
from ...domain.repositories import AnswerJudge

FAITHFULNESS_SYSTEM_PROMPT = """
Oceniasz, na ile ODPOWIEDŹ jest ugruntowana wyłącznie w dostarczonym KONTEKŚCIE.
1.0 = każde twierdzenie odpowiedzi wynika z kontekstu; 0.0 = odpowiedź zmyślona /
sprzeczna z kontekstem / spoza niego. Zwróć WYŁĄCZNIE liczbę dziesiętną od 0.0 do 1.0,
bez komentarza.
"""

ANSWER_RELEVANCE_SYSTEM_PROMPT = """
Oceniasz, na ile ODPOWIEDŹ faktycznie odpowiada na PYTANIE (pomijając jej prawdziwość).
1.0 = wprost i całościowo adresuje pytanie; 0.0 = nie na temat / wymijająca. Zwróć
WYŁĄCZNIE liczbę dziesiętną od 0.0 do 1.0, bez komentarza.
"""


class LLMAnswerJudge(AnswerJudge):
    """Answer quality judge backed by the configured LLM (LLM-as-judge).

    RAGAS-style metrics without the heavy `ragas` dependency. A response the parser cannot
    read scores as `None` — a missing measurement, not a zero — and a readable one is clamped
    to [0, 1].
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @staticmethod
    def _parse_score(raw: str) -> float | None:
        """The score the model returned, or None when it did not return one.

        `None` rather than 0.0, because the two are different facts: folding a failed parse
        into the mean as the worst possible score drags the metric down with no sign that
        anything failed. The match is anchored for the same reason — a loose search finds the
        `1` inside `7/10` and reports a perfect score.
        """
        text = raw.strip()
        if not re.fullmatch(r"[01](?:[.,]\d+)?", text):
            return None
        return max(0.0, min(1.0, float(text.replace(",", "."))))

    async def _score(
        self, system_prompt: str, human_template: str, values: dict[str, str]
    ) -> float | None:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_template)]
        )
        chain = prompt | self.llm
        response = await chain.ainvoke(values)
        return self._parse_score(str(response.content))

    async def score_faithfulness(self, answer: str, context: list[Document]) -> float | None:
        context_text = "\n".join(document.content for document in context)
        return await self._score(
            FAITHFULNESS_SYSTEM_PROMPT,
            "Kontekst:\n{context}\n\nOdpowiedź:\n{answer}",
            {"context": context_text, "answer": answer},
        )

    async def score_answer_relevance(self, question: str, answer: str) -> float | None:
        return await self._score(
            ANSWER_RELEVANCE_SYSTEM_PROMPT,
            "Pytanie:\n{question}\n\nOdpowiedź:\n{answer}",
            {"question": question, "answer": answer},
        )

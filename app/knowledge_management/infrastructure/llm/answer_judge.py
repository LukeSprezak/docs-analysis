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
    """Sędzia jakości odpowiedzi oparty o skonfigurowany LLM (LLM-as-judge).

    Metryki w stylu RAGAS bez ciężkiej zależności `ragas`. Tolerancyjny parsing wyniku:
    przy nieparsowalnej odpowiedzi degraduje się do 0.0; wynik przycinany do [0, 1].
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @staticmethod
    def _parse_score(raw: str) -> float:
        match = re.search(r"[01](?:[.,]\d+)?", raw)
        if not match:
            return 0.0
        try:
            value = float(match.group(0).replace(",", "."))
        except ValueError:
            return 0.0
        return max(0.0, min(1.0, value))

    async def _score(
        self, system_prompt: str, human_template: str, values: dict[str, str]
    ) -> float:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_template)]
        )
        chain = prompt | self.llm
        response = await chain.ainvoke(values)
        return self._parse_score(str(response.content))

    async def score_faithfulness(self, answer: str, context: list[Document]) -> float:
        context_text = "\n".join(document.content for document in context)
        return await self._score(
            FAITHFULNESS_SYSTEM_PROMPT,
            "Kontekst:\n{context}\n\nOdpowiedź:\n{answer}",
            {"context": context_text, "answer": answer},
        )

    async def score_answer_relevance(self, question: str, answer: str) -> float:
        return await self._score(
            ANSWER_RELEVANCE_SYSTEM_PROMPT,
            "Pytanie:\n{question}\n\nOdpowiedź:\n{answer}",
            {"question": question, "answer": answer},
        )
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from ...domain.models import Document
from ...domain.repositories import SummarizerService
from .spotlighting import (
    CONTEXT_END_DELIMITER,
    CONTEXT_START_DELIMITER,
    SECURITY_PROMPT_SECTION,
    strip_delimiters,
)

SUMMARIZE_SYSTEM_PROMPT = f"""
Tworzysz zwięzłe i przejrzyste podsumowania dokumentów.
Użyj formatowania Markdown, takiego jak listy punktowane i pogrubienia, aby podkreślić
kluczowe informacje.
{SECURITY_PROMPT_SECTION}"""


class LangChainSummarizer(SummarizerService):
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        return "\n".join(strip_delimiters(document.content) for document in documents)

    async def summarize(self, documents: list[Document]) -> str:
        text_to_summarize = self._format_documents(documents)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SUMMARIZE_SYSTEM_PROMPT),
                (
                    "human",
                    "Podsumuj dokumenty:\n"
                    f"{CONTEXT_START_DELIMITER}\n"
                    "{text}\n"
                    f"{CONTEXT_END_DELIMITER}",
                ),
            ]
        )

        chain = prompt | self.llm
        response = await chain.ainvoke({"text": text_to_summarize})

        return str(response.content)

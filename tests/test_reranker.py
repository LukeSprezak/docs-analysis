from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.knowledge_management.domain.models import Document
from app.knowledge_management.infrastructure.llm.reranker import (
    CohereReranker,
    CohereRerankResponse,
    LLMReranker,
    LocalCrossEncoderReranker,
    NoOpReranker,
)
from app.knowledge_management.infrastructure.llm.reranker_factory import RerankerFactory
from app.shared.config import settings
from app.shared.enums import RerankerProvider


def _docs(count: int) -> list[Document]:
    return [Document(id=str(i), content=f"passage {i}", metadata={}) for i in range(count)]


def _llm(content: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=content)]))


async def test_noop_reranker_returns_first_top_k():
    documents = _docs(5)
    assert await NoOpReranker().rerank("q", documents, top_k=2) == documents[:2]


async def test_llm_reranker_reorders_by_model_indices():
    reranker = LLMReranker(_llm("[3, 0, 1]"))
    result = await reranker.rerank("q", _docs(5), top_k=3)
    assert [document.id for document in result] == ["3", "0", "1"]


async def test_llm_reranker_falls_back_to_input_order_on_unparseable_output():
    reranker = LLMReranker(_llm("sorry, I don't know"))
    documents = _docs(4)
    assert await reranker.rerank("q", documents, top_k=2) == documents[:2]


async def test_llm_reranker_appends_indices_missing_from_model_output():
    reranker = LLMReranker(_llm("[2]"))
    result = await reranker.rerank("q", _docs(4), top_k=3)
    # after the indicated [2] we append the rest in their original order
    assert [document.id for document in result] == ["2", "0", "1"]


async def test_llm_reranker_ignores_out_of_range_indices():
    reranker = LLMReranker(_llm("[9, 1, 0]"))
    result = await reranker.rerank("q", _docs(3), top_k=3)
    assert [document.id for document in result] == ["1", "0", "2"]


async def test_llm_reranker_skips_llm_for_single_document():
    # Empty iterator: if the LLM were called, it would raise StopIteration.
    reranker = LLMReranker(GenericFakeChatModel(messages=iter([])))
    documents = _docs(1)
    assert await reranker.rerank("q", documents, top_k=4) == documents


@dataclass
class _FakeRerankItem:
    index: int


@dataclass
class _FakeRerankResponse:
    results: list[_FakeRerankItem]


class _FakeCohereClient:
    """Stands in for cohere.ClientV2 — returns a predetermined ordering and records the call."""

    def __init__(self, ordered_indices: list[int]):
        self._ordered_indices = ordered_indices
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> CohereRerankResponse:
        self.calls.append(
            {"model": model, "query": query, "documents": list(documents), "top_n": top_n}
        )
        return _FakeRerankResponse(
            results=[_FakeRerankItem(index=i) for i in self._ordered_indices[:top_n]]
        )


async def test_cohere_reranker_maps_result_indices_to_documents():
    client = _FakeCohereClient(ordered_indices=[3, 1, 0, 2, 4])
    reranker = CohereReranker(client=client, model="rerank-v3.5")

    result = await reranker.rerank("question", _docs(5), top_k=2)

    assert [document.id for document in result] == ["3", "1"]
    # we send the fragment contents and the correct top_n
    call = client.calls[0]
    assert call["documents"] == [f"passage {i}" for i in range(5)]
    assert call["top_n"] == 2
    assert call["model"] == "rerank-v3.5"


async def test_cohere_reranker_skips_api_for_single_document():
    client = _FakeCohereClient(ordered_indices=[0])
    reranker = CohereReranker(client=client)

    documents = _docs(1)
    assert await reranker.rerank("q", documents, top_k=4) == documents
    assert client.calls == []  # no API call for <=1 document


class _FakeCrossEncoder:
    """Stands in for the sentence-transformers CrossEncoder — returns predetermined scores."""

    def __init__(self, scores_by_passage: dict[str, float]):
        self._scores_by_passage = scores_by_passage
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        self.calls.append(sentence_pairs)
        return [self._scores_by_passage[passage] for _, passage in sentence_pairs]


async def test_local_cross_encoder_sorts_by_score_descending():
    documents = _docs(4)
    # passage 2 is the most relevant, then 0, then 3, then 1
    scorer = _FakeCrossEncoder(
        {"passage 0": 0.7, "passage 1": 0.1, "passage 2": 0.9, "passage 3": 0.5}
    )
    reranker = LocalCrossEncoderReranker(scorer=scorer)

    result = await reranker.rerank("question", documents, top_k=2)

    assert [document.id for document in result] == ["2", "0"]
    # the scorer receives (query, fragment content) pairs
    assert scorer.calls[0] == [("question", f"passage {i}") for i in range(4)]


async def test_local_cross_encoder_skips_scoring_for_single_document():
    scorer = _FakeCrossEncoder({"passage 0": 0.5})
    reranker = LocalCrossEncoderReranker(scorer=scorer)

    documents = _docs(1)
    assert await reranker.rerank("q", documents, top_k=4) == documents
    assert scorer.calls == []  # no scoring for <=1 document


def test_factory_returns_noop_when_disabled():
    with patch.object(settings, "RERANKER_PROVIDER", RerankerProvider.NONE):
        assert isinstance(RerankerFactory.get_reranker(), NoOpReranker)


def test_factory_cohere_requires_api_key():
    with (
        patch.object(settings, "RERANKER_PROVIDER", RerankerProvider.COHERE),
        patch.object(settings, "COHERE_API_KEY", None),
        pytest.raises(ValueError, match="COHERE_API_KEY"),
    ):
        RerankerFactory.get_reranker()

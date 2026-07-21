from dataclasses import dataclass, field


@dataclass
class EvaluationExample:
    """Jeden przykład referencyjny z golden setu (zbioru pytań kontrolnych).

    ``relevant_document_ids`` to identyfikatory dokumentów, które POWINNY trafić do
    kontekstu dla danego pytania (zwykle nazwy plików). ``reference_answer`` jest
    opcjonalna — wykorzystują ją wyłącznie metryki generacji (LLM-as-judge).
    """

    question: str
    relevant_document_ids: list[str]
    reference_answer: str | None = None


@dataclass
class RetrievalExampleResult:
    """Wynik retrievalu dla pojedynczego pytania (do wglądu/diagnostyki)."""

    question: str
    retrieved_document_ids: list[str]
    relevant_document_ids: list[str]
    is_hit: bool
    reciprocal_rank: float
    precision_at_k: float
    recall_at_k: float


@dataclass
class RetrievalMetrics:
    """Metryki retrievalu zagregowane po całym golden secie.

    - ``hit_rate`` — odsetek pytań, dla których w top_k znalazł się ≥1 trafny dokument.
    - ``mean_reciprocal_rank`` — średnia z 1/(pozycja pierwszego trafienia).
    - ``mean_precision_at_k`` — średni odsetek trafnych wśród zwróconych top_k.
    - ``mean_recall_at_k`` — średni odsetek pokrytych trafnych dokumentów.
    """

    example_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    mean_precision_at_k: float
    mean_recall_at_k: float


@dataclass
class GenerationExampleResult:
    """Wynik oceny wygenerowanej odpowiedzi dla pojedynczego pytania."""

    question: str
    answer: str
    faithfulness: float
    answer_relevance: float


@dataclass
class GenerationMetrics:
    """Metryki jakości generacji (LLM-as-judge) zagregowane po golden secie.

    - ``mean_faithfulness`` — na ile odpowiedzi są ugruntowane w kontekście (0-1).
    - ``mean_answer_relevance`` — na ile odpowiedzi adresują pytanie (0-1).
    """

    example_count: int
    mean_faithfulness: float
    mean_answer_relevance: float


@dataclass
class EvaluationReport:
    """Pełny raport ewaluacji. Sekcja generacji jest opcjonalna — pojawia się tylko,
    gdy skonfigurowano sędziego (``EVAL_JUDGE_PROVIDER=llm``)."""

    retrieval: RetrievalMetrics
    retrieval_details: list[RetrievalExampleResult] = field(default_factory=list)
    generation: GenerationMetrics | None = None
    generation_details: list[GenerationExampleResult] = field(default_factory=list)
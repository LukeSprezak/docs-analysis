from dataclasses import dataclass, field


@dataclass
class EvaluationExample:
    """One example from the golden set (a collection of test questions)."""

    question: str
    relevant_document_ids: list[str]
    reference_answer: str | None = None


@dataclass
class RetrievalExampleResult:
    """Retrieval result for a single question (for review/diagnostic purposes)."""

    question: str
    retrieved_document_ids: list[str]
    relevant_document_ids: list[str]
    is_hit: bool
    reciprocal_rank: float
    precision_at_k: float
    recall_at_k: float


@dataclass
class RetrievalMetrics:
    """Retrieval metrics aggregated across the entire golden set."""

    example_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    mean_precision_at_k: float
    mean_recall_at_k: float


@dataclass
class GenerationExampleResult:
    """The score for the generated response to a single question."""

    question: str
    answer: str
    faithfulness: float
    answer_relevance: float


@dataclass
class GenerationMetrics:
    """Generation quality metrics (LLM-as-judge) aggregated by golden set."""

    example_count: int
    mean_faithfulness: float
    mean_answer_relevance: float


@dataclass
class EvaluationReport:
    """Full evaluation report. The generation section is optional—it appears only when a judge has been configured
    (``EVAL_JUDGE_PROVIDER=llm``)."""

    retrieval: RetrievalMetrics
    retrieval_details: list[RetrievalExampleResult] = field(default_factory=list)
    generation: GenerationMetrics | None = None
    generation_details: list[GenerationExampleResult] = field(default_factory=list)

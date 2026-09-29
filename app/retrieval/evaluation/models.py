from dataclasses import dataclass, field


@dataclass
class EvaluationExample:
    """A single reference example from the golden set (the control question suite).

    ``relevant_document_ids`` are the ids of documents that SHOULD end up in the context
    for the given question (usually file names). ``reference_answer`` is optional — only
    the generation metrics (LLM-as-judge) use it.
    """

    question: str
    relevant_document_ids: list[str]
    reference_answer: str | None = None
    category: str | None = None


# The two question shapes worth measuring separately. A single average over both hides the
# effect being measured: adding a knowledge graph helps CROSS_DOCUMENT questions and can only
# dilute precision on SINGLE_PASSAGE ones, so a mixed set can show ~zero delta while the graph
# is in fact winning badly on one half and losing on the other.
CATEGORY_SINGLE_PASSAGE = "single_passage"
CATEGORY_CROSS_DOCUMENT = "cross_document"
UNCATEGORIZED = "uncategorized"


@dataclass
class RetrievalExampleResult:
    """Retrieval result for a single question (for inspection/diagnostics)."""

    question: str
    retrieved_document_ids: list[str]
    relevant_document_ids: list[str]
    is_hit: bool
    reciprocal_rank: float
    precision_at_k: float
    recall_at_k: float
    category: str | None = None


@dataclass
class RetrievalMetrics:
    """Retrieval metrics aggregated over the whole golden set.

    - ``hit_rate`` — share of questions where top_k contained >=1 relevant document.
    - ``mean_reciprocal_rank`` — mean of 1/(position of the first hit).
    - ``mean_precision_at_k`` — mean share of relevant documents among the returned top_k.
    - ``mean_recall_at_k`` — mean share of relevant documents that were covered.
    """

    example_count: int
    hit_rate: float
    mean_reciprocal_rank: float
    mean_precision_at_k: float
    mean_recall_at_k: float


@dataclass
class GenerationExampleResult:
    """The judge's verdict on one question's generated answer."""

    question: str
    answer: str
    faithfulness: float | None
    answer_relevance: float | None


@dataclass
class GenerationMetrics:
    """Generation quality metrics (LLM-as-judge) aggregated over the golden set.

    - ``mean_faithfulness`` — how well the answers are grounded in the context (0-1).
    - ``mean_answer_relevance`` — how well the answers address the question (0-1).

    The means cover the answers the judge actually scored. The two ``scored_*`` counts say how
    many that was: a judge that fails to return a readable score contributes nothing to the
    mean instead of a zero, so without the counts a metric computed from three answers out of
    twenty would look exactly like one computed from all twenty.
    """

    example_count: int
    mean_faithfulness: float
    mean_answer_relevance: float
    scored_faithfulness_count: int
    scored_answer_relevance_count: int


@dataclass
class EvaluationReport:
    """The full evaluation report. The generation section is optional — it appears only when
    a judge is configured (``EVAL_JUDGE_PROVIDER=llm``)."""

    retrieval: RetrievalMetrics
    retrieval_details: list[RetrievalExampleResult] = field(default_factory=list)
    generation: GenerationMetrics | None = None
    generation_details: list[GenerationExampleResult] = field(default_factory=list)

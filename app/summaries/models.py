from dataclasses import dataclass


@dataclass
class Summary:
    """A stored summary. The store mints `id` and `created_at`, so a `Summary` only exists
    once it has been saved — see `SummaryRepo.save`, which takes the content and returns
    this. Optional fields here meant every reader downstream had to handle an identity that
    in practice was always there, and the API contract inherited the same lie."""

    text: str
    document_ids: list[str]
    id: str
    created_at: str

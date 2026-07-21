from ...domain.models import Document


def format_sources(documents: list[Document]) -> list[str]:
    """Builds a readable, deduplicated list of the answer's sources.

    Turns the chunks returned by retrieval into labels like ``"report.pdf (page 3)"``, so the
    user can see where the answer came from instead of a useless ``"unknown"``. Several
    chunks from the same place are merged, preserving the order in which they were hit.
    """
    labels: list[str] = []
    for doc in documents:
        filename = doc.metadata.get("filename") or doc.metadata.get("doc_id") or doc.id
        page = doc.metadata.get("page")
        label = f"{filename} (page {page})" if page is not None else str(filename)
        if label not in labels:
            labels.append(label)
    return labels

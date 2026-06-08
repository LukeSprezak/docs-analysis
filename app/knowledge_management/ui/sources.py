from ...domain.models import Document


def format_sources(documents: list[Document]) -> list[str]:
    labels: list[str] = []
    for doc in documents:
        filename = doc.metadata.get("filename") or doc.metadata.get("doc_id") or doc.id
        page = doc.metadata.get("page")
        label = f"{filename} (str. {page})" if page is not None else str(filename)
        if label not in labels:
            labels.append(label)
    return labels

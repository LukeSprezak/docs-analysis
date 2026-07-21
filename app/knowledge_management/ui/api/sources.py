from ...domain.models import Document


def format_sources(documents: list[Document]) -> list[str]:
    """Buduje czytelną, zdeduplikowaną listę źródeł odpowiedzi.

    Z fragmentów (chunków) zwróconych przez retrieval robi etykiety w stylu
    ``"raport.pdf (str. 3)"``. Dzięki temu użytkownik widzi, skąd pochodzi
    odpowiedź, zamiast bezużytecznego ``"unknown"``. Wiele fragmentów z tego
    samego miejsca jest scalanych, z zachowaniem kolejności trafień.
    """
    labels: list[str] = []
    for doc in documents:
        filename = doc.metadata.get("filename") or doc.metadata.get("doc_id") or doc.id
        page = doc.metadata.get("page")
        label = f"{filename} (str. {page})" if page is not None else str(filename)
        if label not in labels:
            labels.append(label)
    return labels

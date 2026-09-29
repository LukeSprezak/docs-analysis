from app.documents.repo import DocumentRepo, PostgresDocumentRepo


def get_doc_repo() -> DocumentRepo:
    return PostgresDocumentRepo()

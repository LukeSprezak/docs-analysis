from .models import KnowledgeBase, Document


class KnowledgeManagementService:

    def validate_knowledge_base(self, kb: KnowledgeBase) -> bool:
        return len(kb.name) > 0 and len(kb.documents) > 0

    def process_document_content(self, content: str) -> str:
        return content.strip()

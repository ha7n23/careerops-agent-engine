"""Application service for trusted career-document extraction."""

from careerops_agent_engine.application.ports.document_extractor import (
    DocumentExtractor,
)
from careerops_agent_engine.application.ports.document_storage import (
    DocumentStorage,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ExtractedDocumentText,
)


class CVDocumentExtractionService:
    """Load user-owned bytes and perform native text extraction."""

    def __init__(
        self,
        *,
        storage: DocumentStorage,
        extractor: DocumentExtractor,
    ) -> None:
        """Store trusted document-processing dependencies."""

        self._storage = storage
        self._extractor = extractor

    def extract(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> ExtractedDocumentText:
        """Extract one validated document inside its user boundary."""

        data = self._storage.read(
            user_id=user_id,
            storage_key=document.storage_key,
        )

        return self._extractor.extract(
            document_id=document.document_id,
            document_format=document.document_format,
            data=data,
        )

"""Application port for extracting one stored career document."""

from typing import Protocol

from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ExtractedDocumentText,
)


class CareerDocumentExtraction(Protocol):
    """Extract trusted text from one user-owned stored document."""

    def extract(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> ExtractedDocumentText:
        """Return bounded extracted text for the document."""

        ...

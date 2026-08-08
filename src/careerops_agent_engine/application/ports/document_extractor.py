"""Application port for native career-document text extraction."""

from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)
from careerops_agent_engine.domain.models.document import (
    ExtractedDocumentText,
)


class DocumentExtractor(Protocol):
    """Extract native text from validated document bytes."""

    def extract(
        self,
        *,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> ExtractedDocumentText:
        """Return deterministic native document text."""

        ...

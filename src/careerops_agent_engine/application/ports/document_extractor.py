"""Application port for native career-source text extraction."""

from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)
from careerops_agent_engine.domain.models.document import (
    ExtractedDocumentText,
)


class DocumentExtractor(Protocol):
    """Extract text from validated source bytes."""

    def extract(
        self,
        *,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> ExtractedDocumentText:
        """Return deterministic native document text."""

        ...

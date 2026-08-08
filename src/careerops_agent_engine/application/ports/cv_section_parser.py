"""Application port for deterministic CV section parsing."""

from typing import Protocol

from careerops_agent_engine.domain.models.document import (
    ExtractedDocumentText,
    ParsedCVDocument,
)


class CVSectionParser(Protocol):
    """Convert extracted CV text into recognised ordered sections."""

    def parse(
        self,
        *,
        document: ExtractedDocumentText,
    ) -> ParsedCVDocument:
        """Return deterministic structural CV sections."""

        ...

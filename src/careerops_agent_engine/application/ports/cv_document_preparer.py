"""Application port for preparing uploaded CV documents."""

from typing import Protocol

from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ParsedCVDocument,
)


class CVDocumentPreparer(Protocol):
    """Extract and deterministically parse one career document."""

    def prepare(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> ParsedCVDocument:
        """Return the recognised CV structure for one user document."""

        ...

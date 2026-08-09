"""Application port for persistent career-document metadata."""

from typing import Protocol

from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)


class CareerDocumentRepository(Protocol):
    """Persist user-owned career-document metadata."""

    def save(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> None:
        """Create or update trusted document metadata."""

        ...

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CareerDocument | None:
        """Retrieve one document within its user boundary."""

        ...

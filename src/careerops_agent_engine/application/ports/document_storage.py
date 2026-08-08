"""Application port for career-document byte storage."""

from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)


class DocumentStorage(Protocol):
    """Store document bytes behind an infrastructure boundary."""

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Persist bytes and return an opaque storage key."""

        ...

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Read previously stored bytes inside the user boundary."""

        ...

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Delete previously stored document bytes."""

        ...

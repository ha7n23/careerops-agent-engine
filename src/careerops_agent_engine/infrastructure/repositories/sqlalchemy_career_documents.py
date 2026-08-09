"""SQLAlchemy repository for persistent career-document metadata."""

from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.infrastructure.database.models.cv_evidence import (
    CareerDocumentRecord,
)


class SqlAlchemyCareerDocumentRepository(CareerDocumentRepository):
    """Persist user-owned CV document metadata."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Store the configured SQLAlchemy session factory."""

        self._session_factory = session_factory

    def save(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> None:
        """Create metadata or update only its lifecycle status."""

        with self._session_factory.begin() as session:
            record = session.get(
                CareerDocumentRecord,
                (
                    user_id,
                    document.document_id,
                ),
            )

            if record is None:
                session.add(
                    document_to_record(
                        user_id=user_id,
                        document=document,
                    )
                )
                return

            validate_immutable_document_metadata(
                record=record,
                document=document,
            )

            record.status = document.status.value

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CareerDocument | None:
        """Retrieve one document inside the authenticated user scope."""

        with self._session_factory() as session:
            record = session.get(
                CareerDocumentRecord,
                (
                    user_id,
                    document_id,
                ),
            )

        if record is None:
            return None

        return document_record_to_domain(record)


def document_to_record(
    *,
    user_id: str,
    document: CareerDocument,
) -> CareerDocumentRecord:
    """Convert trusted document metadata to persistence form."""

    return CareerDocumentRecord(
        user_id=user_id,
        document_id=document.document_id,
        original_filename=document.original_filename,
        document_format=document.document_format.value,
        media_type=document.media_type,
        size_bytes=document.size_bytes,
        sha256_hex=document.sha256_hex,
        storage_key=document.storage_key,
        status=document.status.value,
    )


def document_record_to_domain(
    record: CareerDocumentRecord,
) -> CareerDocument:
    """Convert persistent metadata to the domain model."""

    return CareerDocument(
        document_id=record.document_id,
        original_filename=record.original_filename,
        document_format=CareerDocumentFormat(record.document_format),
        media_type=record.media_type,
        size_bytes=record.size_bytes,
        sha256_hex=record.sha256_hex,
        storage_key=record.storage_key,
        status=CareerDocumentStatus(record.status),
    )


def validate_immutable_document_metadata(
    *,
    record: CareerDocumentRecord,
    document: CareerDocument,
) -> None:
    """Prevent an existing document identity from being rewritten."""

    persisted_identity = (
        record.original_filename,
        record.document_format,
        record.media_type,
        record.size_bytes,
        record.sha256_hex,
        record.storage_key,
    )

    incoming_identity = (
        document.original_filename,
        document.document_format.value,
        document.media_type,
        document.size_bytes,
        document.sha256_hex,
        document.storage_key,
    )

    if persisted_identity != incoming_identity:
        raise ValueError(
            "A career document cannot change its stored identity metadata."
        )

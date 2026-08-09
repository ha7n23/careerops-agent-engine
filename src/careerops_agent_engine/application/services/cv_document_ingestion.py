"""Persistent orchestration for validated CV uploads."""

from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.application.ports.document_storage import (
    DocumentStorage,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)


class CVDocumentIngestionService:
    """Store validated bytes and persist trusted document metadata."""

    def __init__(
        self,
        *,
        upload_service: CVDocumentUploadService,
        repository: CareerDocumentRepository,
        storage: DocumentStorage,
    ) -> None:
        """Store upload and persistence dependencies."""

        self._upload_service = upload_service
        self._repository = repository
        self._storage = storage

    @property
    def max_upload_bytes(self) -> int:
        """Expose the HTTP-safe upload bound."""

        return self._upload_service.max_upload_bytes

    def ingest(
        self,
        *,
        user_id: str,
        original_filename: str,
        declared_media_type: str | None,
        data: bytes,
    ) -> CareerDocument:
        """Store one validated CV and persist its trusted metadata."""

        document = self._upload_service.upload(
            user_id=user_id,
            original_filename=original_filename,
            declared_media_type=declared_media_type,
            data=data,
        )

        try:
            self._repository.save(
                user_id=user_id,
                document=document,
            )
        except Exception:
            self._storage.delete(
                user_id=user_id,
                storage_key=document.storage_key,
            )
            raise

        return document

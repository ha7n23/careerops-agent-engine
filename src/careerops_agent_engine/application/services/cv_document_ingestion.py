"""Persistent orchestration for validated career-document sources."""

from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    DocumentUploadValidationError,
)
from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.application.ports.document_storage import (
    DocumentStorage,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
    build_document_id,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)

DEFAULT_MAX_TEXT_CHARACTERS = 30_000
MAX_TEXT_TITLE_CHARACTERS = 200
TEXT_MEDIA_TYPE = "text/plain; charset=utf-8"


class CVDocumentIngestionService:
    """Store validated sources and persist trusted document metadata."""

    def __init__(
        self,
        *,
        upload_service: CVDocumentUploadService,
        repository: CareerDocumentRepository,
        storage: DocumentStorage,
        max_text_characters: int = DEFAULT_MAX_TEXT_CHARACTERS,
    ) -> None:
        """Store ingestion dependencies and the pasted-text bound."""

        if max_text_characters < 1:
            raise ValueError("Maximum text character count must be positive.")

        self._upload_service = upload_service
        self._repository = repository
        self._storage = storage
        self._max_text_characters = max_text_characters

    @property
    def max_upload_bytes(self) -> int:
        """Expose the HTTP-safe upload bound."""

        return self._upload_service.max_upload_bytes

    @property
    def max_text_characters(self) -> int:
        """Expose the public pasted-text character bound."""

        return self._max_text_characters

    def ingest(
        self,
        *,
        user_id: str,
        original_filename: str,
        declared_media_type: str | None,
        data: bytes,
    ) -> CareerDocument:
        """Store one validated PDF/DOCX source and persist its metadata."""

        document = self._upload_service.upload(
            user_id=user_id,
            original_filename=original_filename,
            declared_media_type=declared_media_type,
            data=data,
        )

        return self._persist_document(
            user_id=user_id,
            document=document,
        )

    def ingest_text(
        self,
        *,
        user_id: str,
        title: str,
        content: str,
    ) -> CareerDocument:
        """Store one validated pasted-text evidence source."""

        normalised_title = normalise_text_title(title)

        normalised_content = normalise_text_content(
            content,
            max_characters=self._max_text_characters,
        )

        data = normalised_content.encode("utf-8")

        document_id = build_document_id()

        storage_key = self._storage.save(
            user_id=user_id,
            document_id=document_id,
            document_format=CareerDocumentFormat.TEXT,
            data=data,
        )

        document = CareerDocument(
            document_id=document_id,
            original_filename=f"{normalised_title}.txt",
            document_format=CareerDocumentFormat.TEXT,
            media_type=TEXT_MEDIA_TYPE,
            size_bytes=len(data),
            sha256_hex=sha256(data).hexdigest(),
            storage_key=storage_key,
            status=CareerDocumentStatus.UPLOADED,
        )

        return self._persist_document(
            user_id=user_id,
            document=document,
        )

    def _persist_document(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> CareerDocument:
        """Persist metadata and remove stored bytes if persistence fails."""

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


def normalise_text_title(title: str) -> str:
    """Validate and normalise a frontend-visible text-source title."""

    normalised = " ".join(title.split())

    if not normalised:
        raise DocumentUploadValidationError(
            "The text evidence source requires a title."
        )

    if len(normalised) > MAX_TEXT_TITLE_CHARACTERS:
        raise DocumentUploadValidationError(
            "The text evidence source title is too long."
        )

    return normalised


def normalise_text_content(
    content: str,
    *,
    max_characters: int,
) -> str:
    """Normalise line endings and enforce the pasted-text boundary."""

    normalised = content.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not normalised:
        raise DocumentUploadValidationError("The text evidence source cannot be empty.")

    if len(normalised) > max_characters:
        raise DocumentUploadValidationError(
            "The text evidence source exceeds the maximum allowed length."
        )

    return normalised

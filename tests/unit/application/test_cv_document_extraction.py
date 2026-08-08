"""Tests for career-document extraction orchestration."""

from careerops_agent_engine.application.services.cv_document_extraction import (
    CVDocumentExtractionService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ExtractedDocumentText,
)


class FakeStorage:
    """Return controlled document bytes."""

    def __init__(
        self,
        data: bytes,
    ) -> None:
        """Store deterministic test bytes."""

        self.data = data
        self.read_user_id: str | None = None
        self.read_storage_key: str | None = None

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Unused storage operation."""

        del (
            user_id,
            document_id,
            document_format,
            data,
        )

        raise NotImplementedError

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Capture ownership parameters and return bytes."""

        self.read_user_id = user_id
        self.read_storage_key = storage_key

        return self.data

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Unused storage operation."""

        del user_id, storage_key

        raise NotImplementedError


class FakeExtractor:
    """Capture trusted extraction metadata."""

    def __init__(self) -> None:
        """Create empty captured arguments."""

        self.document_id: str | None = None
        self.document_format: CareerDocumentFormat | None = None
        self.data: bytes | None = None

    def extract(
        self,
        *,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> ExtractedDocumentText:
        """Return deterministic extracted text."""

        self.document_id = document_id
        self.document_format = document_format
        self.data = data

        return ExtractedDocumentText(
            document_id=document_id,
            text="Python Engineer",
            page_count=1,
            paragraph_count=None,
            warnings=[],
        )


def test_extraction_uses_trusted_document_metadata() -> None:
    """The service should load and extract the owned stored document."""

    storage = FakeStorage(b"trusted bytes")

    extractor = FakeExtractor()

    service = CVDocumentExtractionService(
        storage=storage,
        extractor=extractor,
    )

    document = CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=13,
        sha256_hex="a" * 64,
        storage_key=("documents/usr-test/DOC-001.pdf"),
        status=CareerDocumentStatus.UPLOADED,
    )

    result = service.extract(
        user_id="USER-001",
        document=document,
    )

    assert result.text == "Python Engineer"

    assert storage.read_user_id == "USER-001"

    assert storage.read_storage_key == document.storage_key

    assert extractor.document_id == document.document_id

    assert extractor.document_format is CareerDocumentFormat.PDF

    assert extractor.data == b"trusted bytes"

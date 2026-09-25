"""Tests for secure career-document uploads."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from careerops_agent_engine.application.exceptions import (
    DocumentUploadValidationError,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)


class FakeDocumentStorage:
    """Capture validated bytes without filesystem access."""

    def __init__(self) -> None:
        """Create empty fake storage."""

        self.saved_data: bytes | None = None

    def save(
        self,
        *,
        user_id: str,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> str:
        """Capture stored data and return an opaque key."""

        self.saved_data = data

        return f"users/{user_id}/{document_id}.{document_format.value}"

    def read(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> bytes:
        """Return captured bytes."""

        del user_id, storage_key

        if self.saved_data is None:
            raise FileNotFoundError

        return self.saved_data

    def delete(
        self,
        *,
        user_id: str,
        storage_key: str,
    ) -> None:
        """Delete captured bytes."""

        del user_id, storage_key
        self.saved_data = None


def build_docx_bytes() -> bytes:
    """Create a minimal DOCX-shaped ZIP package."""

    output = BytesIO()

    with ZipFile(
        output,
        mode="w",
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types />",
        )
        archive.writestr(
            "word/document.xml",
            "<document />",
        )

    return output.getvalue()


def build_service(
    storage: FakeDocumentStorage,
    *,
    max_upload_bytes: int = 1024 * 1024,
) -> CVDocumentUploadService:
    """Create an upload service for tests."""

    return CVDocumentUploadService(
        storage=storage,
        max_upload_bytes=max_upload_bytes,
    )


def test_valid_pdf_is_accepted_and_hashed() -> None:
    """A real PDF signature should cross the upload boundary."""

    storage = FakeDocumentStorage()
    service = build_service(storage)

    data = b"%PDF-1.7\nCareerOps test document"

    document = service.upload(
        user_id="USER-001",
        original_filename="cv.pdf",
        declared_media_type="application/pdf",
        data=data,
    )

    assert document.document_format is CareerDocumentFormat.PDF
    assert document.status is CareerDocumentStatus.UPLOADED
    assert document.original_filename == "cv.pdf"
    assert document.size_bytes == len(data)
    assert len(document.sha256_hex) == 64
    assert storage.saved_data == data


def test_valid_docx_package_is_accepted() -> None:
    """A DOCX package should be detected from its archive structure."""

    storage = FakeDocumentStorage()
    service = build_service(storage)

    document = service.upload(
        user_id="USER-001",
        original_filename="cv.docx",
        declared_media_type=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        data=build_docx_bytes(),
    )

    assert document.document_format is CareerDocumentFormat.DOCX


def test_arbitrary_zip_is_not_accepted_as_docx() -> None:
    """A generic ZIP file must not cross the DOCX boundary."""

    output = BytesIO()

    with ZipFile(
        output,
        mode="w",
    ) as archive:
        archive.writestr(
            "notes.txt",
            "not a Word document",
        )

    service = build_service(FakeDocumentStorage())

    with pytest.raises(
        DocumentUploadValidationError,
        match="Only valid PDF and DOCX",
    ):
        service.upload(
            user_id="USER-001",
            original_filename="fake.docx",
            declared_media_type=("application/octet-stream"),
            data=output.getvalue(),
        )


def test_extension_must_match_detected_bytes() -> None:
    """Client filename must agree with the actual document format."""

    service = build_service(FakeDocumentStorage())

    with pytest.raises(
        DocumentUploadValidationError,
        match="extension does not match",
    ):
        service.upload(
            user_id="USER-001",
            original_filename="cv.docx",
            declared_media_type=("application/octet-stream"),
            data=b"%PDF-1.7\nPDF bytes",
        )


def test_conflicting_media_type_is_rejected() -> None:
    """Client MIME metadata cannot contradict detected bytes."""

    service = build_service(FakeDocumentStorage())

    with pytest.raises(
        DocumentUploadValidationError,
        match="media type does not match",
    ):
        service.upload(
            user_id="USER-001",
            original_filename="cv.pdf",
            declared_media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            data=b"%PDF-1.7\nPDF bytes",
        )


def test_oversized_document_is_rejected_before_storage() -> None:
    """Configured size limits must be enforced before persistence."""

    storage = FakeDocumentStorage()

    service = build_service(
        storage,
        max_upload_bytes=10,
    )

    with pytest.raises(
        DocumentUploadValidationError,
        match="maximum allowed size",
    ):
        service.upload(
            user_id="USER-001",
            original_filename="cv.pdf",
            declared_media_type="application/pdf",
            data=b"%PDF-1.7\nthis is too large",
        )

    assert storage.saved_data is None


def test_client_path_is_not_preserved_as_filename() -> None:
    """User-controlled path components must be discarded."""

    storage = FakeDocumentStorage()
    service = build_service(storage)

    document = service.upload(
        user_id="USER-001",
        original_filename=(r"C:\Users\candidate\Documents\cv.pdf"),
        declared_media_type="application/pdf",
        data=b"%PDF-1.7\nPDF bytes",
    )

    assert document.original_filename == "cv.pdf"


def test_text_file_is_not_accepted_by_document_upload() -> None:
    """Pasted text must use its dedicated trusted endpoint later."""

    service = build_service(FakeDocumentStorage())

    with pytest.raises(
        DocumentUploadValidationError,
        match="Only valid PDF and DOCX",
    ):
        service.upload(
            user_id="USER-001",
            original_filename="evidence.txt",
            declared_media_type="text/plain",
            data=b"Built CareerOps using Python and FastAPI.",
        )


def test_highly_compressed_oversized_docx_is_rejected() -> None:
    """Compressed uploads must respect the uncompressed archive bound."""

    output = BytesIO()

    with ZipFile(
        output,
        mode="w",
        compression=ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "[Content_Types].xml",
            "<Types />",
        )
        archive.writestr(
            "word/document.xml",
            "<document />",
        )
        archive.writestr(
            "word/media/oversized.bin",
            b"A" * (10 * 1024 * 1024 + 1),
        )

    compressed_data = output.getvalue()

    assert len(compressed_data) < 1024 * 1024

    service = build_service(FakeDocumentStorage())

    with pytest.raises(
        DocumentUploadValidationError,
        match="Only valid PDF and DOCX",
    ):
        service.upload(
            user_id="USER-001",
            original_filename="oversized.docx",
            declared_media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            data=compressed_data,
        )

"""Secure validation and storage of uploaded CV documents."""

from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from careerops_agent_engine.application.exceptions import (
    DocumentUploadValidationError,
)
from careerops_agent_engine.application.ports.document_storage import (
    DocumentStorage,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)

PDF_MEDIA_TYPE = "application/pdf"

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

GENERIC_MEDIA_TYPES = {
    "",
    "application/octet-stream",
}

DOCX_GENERIC_MEDIA_TYPES = {
    "application/zip",
}

EXPECTED_EXTENSIONS = {
    CareerDocumentFormat.PDF: ".pdf",
    CareerDocumentFormat.DOCX: ".docx",
}

CANONICAL_MEDIA_TYPES = {
    CareerDocumentFormat.PDF: PDF_MEDIA_TYPE,
    CareerDocumentFormat.DOCX: DOCX_MEDIA_TYPE,
}

MAX_DOCX_ARCHIVE_ENTRIES = 2_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


class CVDocumentUploadService:
    """Validate untrusted CV uploads before trusted storage."""

    def __init__(
        self,
        *,
        storage: DocumentStorage,
        max_upload_bytes: int,
    ) -> None:
        """Store validation and storage dependencies."""

        if max_upload_bytes < 1:
            raise ValueError("Maximum upload size must be positive.")

        self._storage = storage
        self._max_upload_bytes = max_upload_bytes

    @property
    def max_upload_bytes(self) -> int:
        """Return the configured HTTP-safe upload limit."""

        return self._max_upload_bytes

    def upload(
        self,
        *,
        user_id: str,
        original_filename: str,
        declared_media_type: str | None,
        data: bytes,
    ) -> CareerDocument:
        """Validate and securely store one PDF or DOCX CV."""

        safe_filename = normalise_filename(original_filename)

        validate_upload_size(
            data=data,
            max_upload_bytes=self._max_upload_bytes,
        )

        document_format = detect_document_format(data)

        validate_filename_extension(
            filename=safe_filename,
            document_format=document_format,
        )

        validate_declared_media_type(
            declared_media_type=declared_media_type,
            document_format=document_format,
        )

        document_id = build_document_id()

        storage_key = self._storage.save(
            user_id=user_id,
            document_id=document_id,
            document_format=document_format,
            data=data,
        )

        return CareerDocument(
            document_id=document_id,
            original_filename=safe_filename,
            document_format=document_format,
            media_type=CANONICAL_MEDIA_TYPES[document_format],
            size_bytes=len(data),
            sha256_hex=sha256(data).hexdigest(),
            storage_key=storage_key,
            status=CareerDocumentStatus.UPLOADED,
        )


def build_document_id() -> str:
    """Create an opaque document identifier."""

    return f"DOC-{uuid4().hex.upper()}"


def normalise_filename(
    filename: str,
) -> str:
    """Reduce an untrusted client filename to its basename."""

    normalised = filename.replace(
        "\\",
        "/",
    ).strip()

    safe_name = PurePosixPath(normalised).name.strip()

    if not safe_name or safe_name in {
        ".",
        "..",
    }:
        raise DocumentUploadValidationError(
            "The uploaded document requires a valid filename."
        )

    if "\x00" in safe_name:
        raise DocumentUploadValidationError(
            "The uploaded document filename contains invalid characters."
        )

    if len(safe_name) > 255:
        raise DocumentUploadValidationError(
            "The uploaded document filename is too long."
        )

    return safe_name


def validate_upload_size(
    *,
    data: bytes,
    max_upload_bytes: int,
) -> None:
    """Reject empty and oversized documents."""

    if not data:
        raise DocumentUploadValidationError("The uploaded document is empty.")

    if len(data) > max_upload_bytes:
        raise DocumentUploadValidationError(
            "The uploaded document exceeds the maximum allowed size."
        )


def detect_document_format(
    data: bytes,
) -> CareerDocumentFormat:
    """Detect PDF or DOCX from bytes rather than client metadata."""

    if data.startswith(b"%PDF-"):
        return CareerDocumentFormat.PDF

    if is_valid_docx_archive(data):
        return CareerDocumentFormat.DOCX

    raise DocumentUploadValidationError(
        "Only valid PDF and DOCX documents are supported."
    )


def is_valid_docx_archive(
    data: bytes,
) -> bool:
    """Recognise a DOCX package and reject suspicious ZIP structure."""

    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()

            if len(entries) > MAX_DOCX_ARCHIVE_ENTRIES:
                return False

            total_uncompressed_bytes = sum(entry.file_size for entry in entries)

            if total_uncompressed_bytes > MAX_DOCX_UNCOMPRESSED_BYTES:
                return False

            names = {entry.filename for entry in entries}

            required_entries = {
                "[Content_Types].xml",
                "word/document.xml",
            }

            return required_entries <= names

    except (
        BadZipFile,
        OSError,
    ):
        return False


def validate_filename_extension(
    *,
    filename: str,
    document_format: CareerDocumentFormat,
) -> None:
    """Ensure the visible extension agrees with detected bytes."""

    suffix = PurePosixPath(filename).suffix.casefold()

    expected = EXPECTED_EXTENSIONS[document_format]

    if suffix != expected:
        raise DocumentUploadValidationError(
            "The uploaded document extension does not match its actual file format."
        )


def validate_declared_media_type(
    *,
    declared_media_type: str | None,
    document_format: CareerDocumentFormat,
) -> None:
    """Reject conflicting client MIME metadata."""

    media_type = (declared_media_type or "").strip().casefold()

    if media_type in GENERIC_MEDIA_TYPES:
        return

    allowed_media_types = {CANONICAL_MEDIA_TYPES[document_format]}

    if document_format is CareerDocumentFormat.DOCX:
        allowed_media_types |= DOCX_GENERIC_MEDIA_TYPES

    if media_type not in allowed_media_types:
        raise DocumentUploadValidationError(
            "The uploaded document media type does not match its actual file format."
        )

"""Application port for DOCX-to-PDF conversion."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ConvertedPDFDocument:
    """PDF bytes produced from one verified DOCX artifact."""

    filename: str
    media_type: str
    data: bytes

    def __post_init__(self) -> None:
        """Reject unusable converter output."""

        if not self.filename:
            raise ValueError("Converted PDF filename cannot be blank.")

        if self.media_type != "application/pdf":
            raise ValueError("Converted document must use PDF media type.")

        if not self.data:
            raise ValueError("Converted PDF bytes cannot be empty.")


class PDFConverter(Protocol):
    """Convert trusted editable CV documents into PDF."""

    def convert_docx(
        self,
        *,
        docx_data: bytes,
        source_filename: str,
    ) -> ConvertedPDFDocument:
        """Convert verified DOCX bytes into a PDF document."""

        ...

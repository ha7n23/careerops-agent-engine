"""Native PDF, DOCX and UTF-8 text extraction."""

from io import BytesIO
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from careerops_agent_engine.application.exceptions import (
    DocumentExtractionError,
    DocumentTextUnavailableError,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
)
from careerops_agent_engine.domain.models.document import (
    ExtractedDocumentText,
)

DEFAULT_MAX_PDF_PAGES = 15
DEFAULT_MAX_EXTRACTED_CHARACTERS = 30_000


class NativeDocumentExtractor:
    """Extract bounded text from validated PDF, DOCX and UTF-8 bytes."""

    def __init__(
        self,
        *,
        max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
        max_extracted_characters: int = DEFAULT_MAX_EXTRACTED_CHARACTERS,
    ) -> None:
        """Store deterministic document-complexity limits."""

        if max_pdf_pages < 1:
            raise ValueError("Maximum PDF page count must be positive.")

        if max_extracted_characters < 1:
            raise ValueError("Maximum extracted character count must be positive.")

        self._max_pdf_pages = max_pdf_pages
        self._max_extracted_characters = max_extracted_characters

    def extract(
        self,
        *,
        document_id: str,
        document_format: CareerDocumentFormat,
        data: bytes,
    ) -> ExtractedDocumentText:
        """Extract text using the bounded format-specific parser."""

        if document_format is CareerDocumentFormat.PDF:
            return extract_pdf(
                document_id=document_id,
                data=data,
                max_pages=self._max_pdf_pages,
                max_characters=self._max_extracted_characters,
            )

        if document_format is CareerDocumentFormat.DOCX:
            return extract_docx(
                document_id=document_id,
                data=data,
                max_characters=self._max_extracted_characters,
            )

        if document_format is CareerDocumentFormat.TEXT:
            return extract_text(
                document_id=document_id,
                data=data,
                max_characters=self._max_extracted_characters,
            )

        raise DocumentExtractionError("Unsupported career-document format.")


def extract_pdf(
    *,
    document_id: str,
    data: bytes,
    max_pages: int,
    max_characters: int,
) -> ExtractedDocumentText:
    """Extract bounded native text from a validated PDF."""

    try:
        reader = PdfReader(
            BytesIO(data),
            strict=False,
        )
    except (
        PdfReadError,
        OSError,
        ValueError,
    ) as exc:
        raise DocumentExtractionError("The PDF could not be parsed.") from exc

    page_count = len(reader.pages)

    if page_count > max_pages:
        raise DocumentExtractionError(
            "The PDF exceeds the maximum supported page count."
        )

    page_texts: list[str] = []
    warnings: list[str] = []
    character_count = 0

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        try:
            raw_text = page.extract_text()
        except Exception as exc:
            raise DocumentExtractionError(
                f"PDF text extraction failed on page {page_number}."
            ) from exc

        cleaned = normalise_text(raw_text or "")

        if not cleaned:
            warnings.append(f"Page {page_number} contained no extractable native text.")
            continue

        character_count = append_bounded_text_block(
            blocks=page_texts,
            text=cleaned,
            current_characters=character_count,
            max_characters=max_characters,
            separator_length=2,
        )

    text = "\n\n".join(page_texts).strip()

    if not text:
        raise DocumentTextUnavailableError(
            "The PDF contains no extractable native text."
        )

    return ExtractedDocumentText(
        document_id=document_id,
        text=text,
        page_count=page_count,
        paragraph_count=None,
        warnings=warnings,
    )


def extract_docx(
    *,
    document_id: str,
    data: bytes,
    max_characters: int,
) -> ExtractedDocumentText:
    """Extract bounded paragraph and table text from a DOCX package."""

    try:
        document = Document(BytesIO(data))
    except (
        BadZipFile,
        PackageNotFoundError,
        OSError,
        ValueError,
        KeyError,
    ) as exc:
        raise DocumentExtractionError("The DOCX document could not be parsed.") from exc

    text_blocks: list[str] = []
    paragraph_count = 0
    character_count = 0

    for paragraph in document.paragraphs:
        cleaned = normalise_text(paragraph.text)

        if not cleaned:
            continue

        character_count = append_bounded_text_block(
            blocks=text_blocks,
            text=cleaned,
            current_characters=character_count,
            max_characters=max_characters,
            separator_length=1,
        )

        paragraph_count += 1

    for table in document.tables:
        for row in table.rows:
            cells = [normalise_text(cell.text) for cell in row.cells]

            populated_cells = [cell for cell in cells if cell]

            if not populated_cells:
                continue

            character_count = append_bounded_text_block(
                blocks=text_blocks,
                text=" | ".join(populated_cells),
                current_characters=character_count,
                max_characters=max_characters,
                separator_length=1,
            )

    text = "\n".join(text_blocks).strip()

    if not text:
        raise DocumentTextUnavailableError(
            "The DOCX document contains no extractable native text."
        )

    return ExtractedDocumentText(
        document_id=document_id,
        text=text,
        page_count=None,
        paragraph_count=paragraph_count,
        warnings=[],
    )


def extract_text(
    *,
    document_id: str,
    data: bytes,
    max_characters: int,
) -> ExtractedDocumentText:
    """Extract and normalise one bounded trusted UTF-8 text source."""

    try:
        raw_text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentExtractionError("The text source is not valid UTF-8.") from exc

    if len(raw_text) > max_characters:
        raise DocumentExtractionError(
            "The document exceeds the maximum extracted text length."
        )

    text = normalise_text(raw_text)

    if not text:
        raise DocumentTextUnavailableError("The text source contains no usable text.")

    return ExtractedDocumentText(
        document_id=document_id,
        text=text,
        page_count=None,
        paragraph_count=len(text.splitlines()),
        warnings=[],
    )


def append_bounded_text_block(
    *,
    blocks: list[str],
    text: str,
    current_characters: int,
    max_characters: int,
    separator_length: int,
) -> int:
    """Append one text block without crossing the extraction bound."""

    added_characters = len(text)

    if blocks:
        added_characters += separator_length

    updated_characters = current_characters + added_characters

    if updated_characters > max_characters:
        raise DocumentExtractionError(
            "The document exceeds the maximum extracted text length."
        )

    blocks.append(text)

    return updated_characters


def normalise_text(
    text: str,
) -> str:
    """Normalise extracted whitespace without rewriting content."""

    lines = [" ".join(line.split()) for line in text.splitlines()]

    populated_lines = [line for line in lines if line]

    return "\n".join(populated_lines)

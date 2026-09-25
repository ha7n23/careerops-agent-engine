"""Tests for bounded career-document processing settings."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.core.config import Settings


def test_document_processing_limits_have_practical_cv_defaults() -> None:
    """Defaults should support large realistic CVs without oversized LLM calls."""

    settings = Settings()

    assert settings.document_upload_max_bytes == 5 * 1024 * 1024
    assert settings.document_pdf_max_pages == 15
    assert settings.document_extraction_max_characters == 30_000
    assert settings.document_max_sections == 20


def test_pdf_page_limit_must_be_positive() -> None:
    """A disabled page bound must not reach the PDF parser."""

    with pytest.raises(ValidationError):
        Settings(document_pdf_max_pages=0)


def test_extracted_character_limit_has_safe_minimum() -> None:
    """Runtime configuration must retain a useful bounded text allowance."""

    with pytest.raises(ValidationError):
        Settings(document_extraction_max_characters=4_999)


def test_section_limit_must_be_positive() -> None:
    """Section parsing must never become unbounded through configuration."""

    with pytest.raises(ValidationError):
        Settings(document_max_sections=0)

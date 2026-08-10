"""Tests for isolated LibreOffice DOCX-to-PDF conversion."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from careerops_agent_engine.application.exceptions import (
    CVPDFConversionError,
)
from careerops_agent_engine.infrastructure.documents.libreoffice_pdf_converter import (
    LibreOfficePDFConverter,
    build_libreoffice_command,
    build_safe_source_stem,
)


def build_test_pdf() -> bytes:
    """Create one structurally valid one-page PDF."""

    writer = PdfWriter()

    writer.add_blank_page(
        width=612,
        height=792,
    )

    from io import BytesIO

    stream = BytesIO()

    writer.write(stream)

    return stream.getvalue()


def test_converter_returns_valid_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful LibreOffice output should cross the adapter boundary."""

    pdf_data = build_test_pdf()

    def fake_run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
    ) -> SimpleNamespace:
        """Simulate successful LibreOffice conversion."""

        assert capture_output is True
        assert text is True
        assert timeout == 30.0
        assert check is False

        output_directory = Path(command[command.index("--outdir") + 1])

        output_directory.joinpath("CVV-001.pdf").write_bytes(pdf_data)

        return SimpleNamespace(
            returncode=0,
            stdout="converted",
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    converter = LibreOfficePDFConverter(
        executable="soffice",
        timeout_seconds=30.0,
    )

    result = converter.convert_docx(
        docx_data=b"trusted-docx-bytes",
        source_filename="CVV-001.docx",
    )

    assert result.filename == ("CVV-001.pdf")

    assert result.media_type == ("application/pdf")

    assert result.data == pdf_data


def test_command_uses_isolated_headless_profile(
    tmp_path: Path,
) -> None:
    """Conversion command should not depend on the user's office profile."""

    input_path = tmp_path / "input.docx"

    output_directory = tmp_path / "output"

    profile_directory = tmp_path / "profile"

    command = build_libreoffice_command(
        executable="soffice",
        input_path=input_path,
        output_directory=output_directory,
        profile_directory=profile_directory,
    )

    assert command[0] == "soffice"

    assert "--headless" in command
    assert "--convert-to" in command
    assert "pdf" in command

    assert any(
        argument.startswith("-env:UserInstallation=file:") for argument in command
    )

    assert str(input_path) == command[-1]


def test_converter_rejects_failed_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero LibreOffice exit cannot produce accepted output."""

    def fake_run(
        *args: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        """Simulate process failure."""

        del args
        del kwargs

        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="failure",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    converter = LibreOfficePDFConverter(
        executable="soffice",
        timeout_seconds=30.0,
    )

    with pytest.raises(
        CVPDFConversionError,
        match="conversion failed",
    ):
        converter.convert_docx(
            docx_data=b"docx",
            source_filename="CVV-001.docx",
        )


def test_converter_rejects_missing_pdf_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exit code zero is insufficient without the expected output file."""

    def fake_run(
        *args: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        """Simulate misleading successful process exit."""

        del args
        del kwargs

        return SimpleNamespace(
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        subprocess,
        "run",
        fake_run,
    )

    converter = LibreOfficePDFConverter(
        executable="soffice",
        timeout_seconds=30.0,
    )

    with pytest.raises(
        CVPDFConversionError,
        match="expected PDF output",
    ):
        converter.convert_docx(
            docx_data=b"docx",
            source_filename="CVV-001.docx",
        )


def test_safe_source_stem_removes_path_influence() -> None:
    """User filenames should not control converter filesystem structure."""

    assert build_safe_source_stem("../../My CV (final)!") == "MyCVfinal"

    assert build_safe_source_stem("***") == "careerops-cv"

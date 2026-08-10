"""LibreOffice-backed conversion of verified DOCX files to PDF."""

import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from careerops_agent_engine.application.exceptions import (
    CVPDFConversionError,
)
from careerops_agent_engine.application.ports.pdf_converter import (
    ConvertedPDFDocument,
    PDFConverter,
)

PDF_MEDIA_TYPE = "application/pdf"


class LibreOfficePDFConverter(PDFConverter):
    """Convert DOCX bytes using an isolated LibreOffice process."""

    def __init__(
        self,
        *,
        executable: str,
        timeout_seconds: float,
    ) -> None:
        """Store process configuration."""

        if not executable.strip():
            raise ValueError("LibreOffice executable cannot be blank.")

        if timeout_seconds <= 0:
            raise ValueError("PDF conversion timeout must be positive.")

        self._executable = executable
        self._timeout_seconds = timeout_seconds

    def convert_docx(
        self,
        *,
        docx_data: bytes,
        source_filename: str,
    ) -> ConvertedPDFDocument:
        """Convert one trusted DOCX to structurally valid PDF bytes."""

        if not docx_data:
            raise CVPDFConversionError("DOCX conversion input cannot be empty.")

        source_path = Path(source_filename)

        if source_path.suffix.lower() != ".docx":
            raise CVPDFConversionError(
                "PDF conversion requires a DOCX source filename."
            )

        safe_stem = build_safe_source_stem(source_path.stem)

        with TemporaryDirectory(prefix="careerops-pdf-") as temporary_directory:
            root = Path(temporary_directory)

            input_directory = root / "input"

            output_directory = root / "output"

            profile_directory = root / "libreoffice-profile"

            input_directory.mkdir()
            output_directory.mkdir()
            profile_directory.mkdir()

            input_path = input_directory / f"{safe_stem}.docx"

            input_path.write_bytes(docx_data)

            command = build_libreoffice_command(
                executable=self._executable,
                input_path=input_path,
                output_directory=output_directory,
                profile_directory=profile_directory,
            )

            self._run_conversion(command)

            output_path = output_directory / f"{safe_stem}.pdf"

            if not output_path.is_file():
                raise CVPDFConversionError(
                    "LibreOffice completed without producing the expected PDF output."
                )

            pdf_data = output_path.read_bytes()

        validate_converted_pdf(pdf_data)

        return ConvertedPDFDocument(
            filename=f"{safe_stem}.pdf",
            media_type=PDF_MEDIA_TYPE,
            data=pdf_data,
        )

    def _run_conversion(
        self,
        command: list[str],
    ) -> None:
        """Run LibreOffice without shell interpretation."""

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )

        except FileNotFoundError as exc:
            raise CVPDFConversionError(
                "LibreOffice executable is unavailable."
            ) from exc

        except subprocess.TimeoutExpired as exc:
            raise CVPDFConversionError("LibreOffice PDF conversion timed out.") from exc

        if result.returncode != 0:
            raise CVPDFConversionError("LibreOffice PDF conversion failed.")


def build_libreoffice_command(
    *,
    executable: str,
    input_path: Path,
    output_directory: Path,
    profile_directory: Path,
) -> list[str]:
    """Build one isolated headless LibreOffice command."""

    profile_uri = profile_directory.resolve().as_uri()

    return [
        executable,
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        "--nolockcheck",
        (f"-env:UserInstallation={profile_uri}"),
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_directory),
        str(input_path),
    ]


def build_safe_source_stem(
    stem: str,
) -> str:
    """Create a converter-local filename independent of user input."""

    cleaned = "".join(
        character
        for character in stem
        if (
            character.isalnum()
            or character
            in {
                "-",
                "_",
            }
        )
    )

    if not cleaned:
        return "careerops-cv"

    return cleaned[:120]


def validate_converted_pdf(
    data: bytes,
) -> None:
    """Require basic structural validity before accepting conversion."""

    if not data.startswith(b"%PDF-"):
        raise CVPDFConversionError("LibreOffice output is not a PDF document.")

    try:
        reader = PdfReader(BytesIO(data))

    except (
        PdfReadError,
        ValueError,
    ) as exc:
        raise CVPDFConversionError("Converted PDF structure is invalid.") from exc

    if len(reader.pages) < 1:
        raise CVPDFConversionError("Converted PDF contains no pages.")

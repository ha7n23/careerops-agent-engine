"""Smoke-test CareerOps DOCX and PDF generation inside its runtime container."""

from io import BytesIO

from pypdf import PdfReader

from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVChangeApplicationMode,
    CVSection,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    CVVersionProvenance,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_renderer import (
    CareerOpsStandardDocxRenderer,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_verifier import (
    CareerOpsStandardDocxVerifier,
)
from careerops_agent_engine.infrastructure.documents.careerops_pdf_verifier import (
    CareerOpsStandardPDFVerifier,
)
from careerops_agent_engine.infrastructure.documents.libreoffice_pdf_converter import (
    LibreOfficePDFConverter,
)


def main() -> None:
    """Prove rendering, verification, conversion, and PDF parsing."""

    settings = get_settings()

    original_text = "Built CareerOps using Python."

    final_text = "Built CareerOps using Python, FastAPI, and Docker."

    version = CVVersion(
        cv_version_id="CVV-CONTAINER-SMOKE",
        version_number=1,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=StructuredCV(
            cv_id="CV-CONTAINER-SMOKE",
            source_document_id="DOC-CONTAINER-SMOKE",
            preamble_text=("CareerOps Container Test Candidate\ncandidate@example.com"),
            sections=[
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    free_text=(f"CareerOps Agent Engine\n{final_text}"),
                )
            ],
        ),
        applied_changes=[
            AppliedCVChange(
                change_id="CHG-CONTAINER-SMOKE",
                proposal_id="CVP-CONTAINER-SMOKE",
                section=CVSection.PROJECTS,
                application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                source_anchor=original_text,
                original_text=original_text,
                applied_text=final_text,
                anchor_evidence_ids=["EVD-CONTAINER-SMOKE"],
                requirement_ids=["REQ-CONTAINER-SMOKE"],
                supporting_evidence_ids=["EVD-CONTAINER-SMOKE"],
            )
        ],
        provenance=CVVersionProvenance(
            job_id="JOB-CONTAINER-SMOKE",
            thread_id="THR-CONTAINER-SMOKE",
            source_document_id="DOC-CONTAINER-SMOKE",
            requirement_ids=["REQ-CONTAINER-SMOKE"],
            supporting_evidence_ids=["EVD-CONTAINER-SMOKE"],
            final_proposal_ids=["CVP-CONTAINER-SMOKE"],
            review_status=(ApprovalStatus.APPROVED),
            template_id="careerops-standard",
            template_version="1.0.0",
            workflow_version="container-smoke-v1",
        ),
        artifacts=[],
    )

    # 1. Render the real CareerOps DOCX.
    docx = CareerOpsStandardDocxRenderer().render(version=version)

    # 2. Verify the actual DOCX bytes.
    docx_verification = CareerOpsStandardDocxVerifier().verify(
        version=version,
        data=docx.data,
    )

    if not docx_verification.passed:
        raise RuntimeError(
            f"Container DOCX verification failed: {docx_verification.notes}"
        )

    # 3. Convert using the LibreOffice executable configured
    #    inside the Linux container.
    pdf = LibreOfficePDFConverter(
        executable=(settings.libreoffice_executable),
        timeout_seconds=(settings.pdf_conversion_timeout_seconds),
    ).convert_docx(
        docx_data=docx.data,
        source_filename=docx.filename,
    )

    # 4. Verify CareerOps-visible PDF content.
    pdf_verification = CareerOpsStandardPDFVerifier().verify(
        version=version,
        data=pdf.data,
    )

    if not pdf_verification.passed:
        raise RuntimeError(
            f"Container PDF verification failed: {pdf_verification.notes}"
        )

    # 5. Independently reopen the real generated PDF.
    reader = PdfReader(BytesIO(pdf.data))

    if len(reader.pages) < 1:
        raise RuntimeError("Container-generated PDF contains no pages.")

    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    normalized_text = " ".join(extracted_text.split())

    if final_text not in normalized_text:
        raise RuntimeError("Container-generated PDF lost expected CV content.")

    print("CareerOps container document pipeline successful.")

    print(f"LibreOffice executable: {settings.libreoffice_executable}")

    print(f"DOCX bytes: {len(docx.data)}")

    print(f"PDF bytes: {len(pdf.data)}")

    print(f"PDF pages: {len(reader.pages)}")

    print("DOCX verification: passed")

    print("PDF verification: passed")

    print("REAL CONTAINER DOCX -> PDF PIPELINE PASSED")


if __name__ == "__main__":
    main()

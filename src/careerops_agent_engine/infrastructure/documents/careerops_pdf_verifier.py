"""Deterministic verification of CareerOps generated PDF output."""

import re
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from careerops_agent_engine.application.ports.cv_artifact_verifier import (
    CVArtifactVerificationResult,
    CVArtifactVerifier,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
)
from careerops_agent_engine.domain.models.structured_cv import (
    CVEntry,
    StructuredCVSection,
)


class CareerOpsStandardPDFVerifier(CVArtifactVerifier):
    """Verify generated PDF content against its immutable CV version."""

    @property
    def artifact_format(self) -> CVArtifactFormat:
        """Return the supported generated-artifact format."""

        return CVArtifactFormat.PDF

    def verify(
        self,
        *,
        version: CVVersion,
        data: bytes,
    ) -> CVArtifactVerificationResult:
        """Verify PDF structure, text content and provenance privacy."""

        if not data:
            return failed("Stored PDF bytes are empty.")

        if not data.startswith(b"%PDF-"):
            return failed("Stored PDF is not a PDF document.")

        try:
            reader = PdfReader(BytesIO(data))

        except (
            PdfReadError,
            ValueError,
        ):
            return failed("Stored PDF structure is invalid.")

        if len(reader.pages) < 1:
            return failed("Stored PDF contains no pages.")

        try:
            extracted_text = "\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        except Exception:
            return failed("Stored PDF text could not be extracted.")

        normalized_text = normalize_pdf_text(extracted_text)

        if not normalized_text:
            return failed("Stored PDF contains no extractable CV text.")

        expected_fragments = build_expected_visible_fragments(version)

        if not fragments_appear_in_order(
            text=normalized_text,
            fragments=expected_fragments,
        ):
            return failed(
                "PDF visible content does not match the structured CV version."
            )

        internal_ids = collect_internal_provenance_ids(version)

        if any(
            normalize_pdf_text(identifier) in normalized_text
            for identifier in internal_ids
        ):
            return failed("PDF exposes internal CareerOps provenance identifiers.")

        return CVArtifactVerificationResult(passed=True)


def build_expected_visible_fragments(
    version: CVVersion,
) -> list[str]:
    """Build ordered visible-text fragments expected in the standard PDF."""

    fragments: list[str] = []

    preamble = version.structured_cv.preamble_text

    if preamble is not None:
        fragments.extend(preamble.split("\n"))

    for section in version.structured_cv.sections:
        fragments.extend(expected_section_fragments(section))

    return [fragment for fragment in fragments if normalize_pdf_text(fragment)]


def expected_section_fragments(
    section: StructuredCVSection,
) -> list[str]:
    """Build expected visible fragments for one CV section."""

    fragments = [section.heading.upper()]

    if section.free_text is not None:
        fragments.extend(section.free_text.split("\n"))

    for entry in section.entries:
        fragments.extend(expected_entry_fragments(entry))

    return fragments


def expected_entry_fragments(
    entry: CVEntry,
) -> list[str]:
    """Build expected visible fragments for one structured entry."""

    fragments = [entry.title]

    metadata = [
        value
        for value in (
            entry.subtitle,
            entry.location,
            entry.date_text,
        )
        if value is not None
    ]

    if metadata:
        fragments.append(" | ".join(metadata))

    fragments.extend(bullet.text for bullet in entry.bullets)

    return fragments


def fragments_appear_in_order(
    *,
    text: str,
    fragments: list[str],
) -> bool:
    """Require every expected fragment to occur in source order."""

    cursor = 0

    for fragment in fragments:
        normalized_fragment = normalize_pdf_text(fragment)

        position = text.find(
            normalized_fragment,
            cursor,
        )

        if position < 0:
            return False

        cursor = position + len(normalized_fragment)

    return True


def normalize_pdf_text(
    text: str,
) -> str:
    """Normalize extraction whitespace without changing words."""

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def collect_internal_provenance_ids(
    version: CVVersion,
) -> set[str]:
    """Collect internal identifiers that must remain outside visible PDF."""

    identifiers = {
        version.cv_version_id,
        version.provenance.job_id,
        version.provenance.thread_id,
        *version.provenance.requirement_ids,
        *version.provenance.supporting_evidence_ids,
        *version.provenance.final_proposal_ids,
    }

    for change in version.applied_changes:
        identifiers.add(change.change_id)

    return {identifier for identifier in identifiers if identifier}


def failed(
    note: str,
) -> CVArtifactVerificationResult:
    """Create one deterministic failed verification result."""

    return CVArtifactVerificationResult(
        passed=False,
        notes=(note,),
    )

"""Deterministic parsing of extracted CV text into sections."""

from collections.abc import Mapping

from careerops_agent_engine.application.ports.cv_section_parser import (
    CVSectionParser,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
)
from careerops_agent_engine.domain.models.document import (
    ExtractedDocumentText,
    ParsedCVDocument,
    ParsedCVSection,
)

HEADING_ALIASES: Mapping[
    CVSection,
    frozenset[str],
] = {
    CVSection.PROFILE: frozenset(
        {
            "profile",
            "personal profile",
            "professional profile",
            "professional summary",
            "career profile",
            "summary",
            "career summary",
            "objective",
            "career objective",
        }
    ),
    CVSection.SKILLS: frozenset(
        {
            "skills",
            "technical skills",
            "key skills",
            "core skills",
            "core competencies",
            "technical competencies",
            "technologies",
            "tools and technologies",
        }
    ),
    CVSection.EXPERIENCE: frozenset(
        {
            "experience",
            "work experience",
            "professional experience",
            "employment",
            "employment history",
            "work history",
            "career history",
        }
    ),
    CVSection.PROJECTS: frozenset(
        {
            "projects",
            "selected projects",
            "key projects",
            "technical projects",
            "personal projects",
            "academic projects",
            "project experience",
        }
    ),
    CVSection.EDUCATION: frozenset(
        {
            "education",
            "academic background",
            "academic history",
            "education and qualifications",
            "qualifications",
        }
    ),
    CVSection.CERTIFICATIONS: frozenset(
        {
            "certifications",
            "certification",
            "certificates",
            "professional certifications",
            "licenses and certifications",
            "certifications and licenses",
        }
    ),
}


class DeterministicCVSectionParser(CVSectionParser):
    """Identify supported CV sections using exact heading aliases."""

    def parse(
        self,
        *,
        document: ExtractedDocumentText,
    ) -> ParsedCVDocument:
        """Parse extracted text while preserving source ordering."""

        preamble_lines: list[str] = []
        sections: list[ParsedCVSection] = []
        warnings: list[str] = []

        current_section: CVSection | None = None
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush_current_section() -> None:
            """Store the currently accumulated recognised section."""

            nonlocal current_section
            nonlocal current_heading
            nonlocal current_lines

            if current_section is None or current_heading is None:
                return

            text = "\n".join(current_lines).strip()

            if text:
                sections.append(
                    ParsedCVSection(
                        section=current_section,
                        heading=current_heading,
                        text=text,
                        order_index=len(sections),
                    )
                )
            else:
                warnings.append(
                    "Recognised CV heading contained "
                    "no section content: "
                    f"{current_heading}"
                )

            current_section = None
            current_heading = None
            current_lines = []

        for raw_line in document.text.splitlines():
            line = normalise_line(raw_line)

            if not line:
                continue

            detected_section = detect_section_heading(line)

            if detected_section is not None:
                flush_current_section()

                current_section = detected_section
                current_heading = clean_display_heading(line)
                current_lines = []

                continue

            if current_section is None:
                preamble_lines.append(line)
            else:
                current_lines.append(line)

        flush_current_section()

        if not sections:
            warnings.append("No recognised CV section headings were found.")

        preamble = "\n".join(preamble_lines).strip()

        return ParsedCVDocument(
            document_id=document.document_id,
            preamble_text=(preamble if preamble else None),
            sections=sections,
            warnings=warnings,
        )


def detect_section_heading(
    line: str,
) -> CVSection | None:
    """Match a whole line against supported CV heading aliases."""

    candidate = normalise_heading(line)

    for section, aliases in HEADING_ALIASES.items():
        if candidate in aliases:
            return section

    return None


def normalise_heading(
    heading: str,
) -> str:
    """Normalise a possible heading for exact alias comparison."""

    candidate = normalise_line(heading)

    candidate = candidate.rstrip(":").strip()

    return candidate.casefold()


def clean_display_heading(
    heading: str,
) -> str:
    """Preserve a readable version of the detected source heading."""

    return normalise_line(heading).rstrip(":").strip()


def normalise_line(
    line: str,
) -> str:
    """Normalise whitespace without rewriting CV content."""

    return " ".join(line.split())

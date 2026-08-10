"""Deterministic assembly of parsed source CV content."""

from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    StructuredCVAssemblyError,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
)
from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)


class BaseStructuredCVAssembler:
    """Convert parsed source CV content without rewriting it."""

    def assemble(
        self,
        *,
        document: ParsedCVDocument,
    ) -> StructuredCV:
        """Build a render-independent base CV from parsed source text."""

        if not document.sections:
            raise StructuredCVAssemblyError(
                "A structured CV requires at least one recognised source section."
            )

        section_order: list[CVSection] = []
        section_headings: dict[
            CVSection,
            str,
        ] = {}
        section_texts: dict[
            CVSection,
            list[str],
        ] = {}

        for source_section in document.sections:
            section = source_section.section

            if section not in section_texts:
                section_order.append(section)

                section_headings[section] = source_section.heading

                section_texts[section] = []

            section_texts[section].append(source_section.text)

        assembled_sections = [
            StructuredCVSection(
                section=section,
                heading=section_headings[section],
                free_text="\n".join(section_texts[section]),
            )
            for section in section_order
        ]

        return StructuredCV(
            cv_id=build_structured_cv_id(document_id=document.document_id),
            source_document_id=(document.document_id),
            preamble_text=(document.preamble_text),
            sections=assembled_sections,
        )


def build_structured_cv_id(
    *,
    document_id: str,
) -> str:
    """Build a stable CV-family identifier from its source document."""

    digest = sha256(document_id.encode("utf-8")).hexdigest()[:16].upper()

    return f"CV-{digest}"

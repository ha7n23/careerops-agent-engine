"""Application service for grounded CV evidence proposals."""

import re
from hashlib import sha256

from careerops_agent_engine.application.exceptions import (
    CVEvidenceProposalValidationError,
)
from careerops_agent_engine.application.ports.cv_evidence_extractor import (
    CVEvidenceExtractor,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
    ParsedCVSection,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceCandidate,
    CareerEvidenceProposal,
    SourceReference,
)

ALLOWED_CATEGORIES_BY_SECTION: dict[
    CVSection,
    set[EvidenceCategory],
] = {
    CVSection.PROFILE: {
        EvidenceCategory.SKILL,
        EvidenceCategory.ACHIEVEMENT,
    },
    CVSection.SKILLS: {
        EvidenceCategory.SKILL,
    },
    CVSection.EXPERIENCE: {
        EvidenceCategory.EMPLOYMENT,
        EvidenceCategory.ACHIEVEMENT,
    },
    CVSection.PROJECTS: {
        EvidenceCategory.PROJECT,
        EvidenceCategory.ACHIEVEMENT,
    },
    CVSection.EDUCATION: {
        EvidenceCategory.EDUCATION,
        EvidenceCategory.ACHIEVEMENT,
    },
    CVSection.CERTIFICATIONS: {
        EvidenceCategory.CERTIFICATION,
    },
}


class CVEvidenceProposalService:
    """Create pending evidence proposals from model candidates."""

    def __init__(
        self,
        *,
        extractor: CVEvidenceExtractor,
    ) -> None:
        """Store the model-backed candidate extractor."""

        self._extractor = extractor

    def generate(
        self,
        *,
        document: ParsedCVDocument,
    ) -> list[CareerEvidenceProposal]:
        """Extract, validate, and attach trusted provenance."""

        candidates = self._extractor.extract(document=document)

        sections_by_index = build_section_index(document.sections)

        proposals: list[CareerEvidenceProposal] = []

        seen_proposal_ids: set[str] = set()

        for candidate in candidates:
            section = sections_by_index.get(candidate.source_section_order_index)

            if section is None:
                raise (
                    CVEvidenceProposalValidationError(
                        "Evidence candidate references an unknown CV section."
                    )
                )

            validate_candidate(
                candidate=candidate,
                section=section,
            )

            proposal_id = build_evidence_proposal_id(
                document_id=(document.document_id),
                candidate=candidate,
            )

            if proposal_id in seen_proposal_ids:
                raise (
                    CVEvidenceProposalValidationError(
                        "Duplicate CV evidence proposals were generated."
                    )
                )

            seen_proposal_ids.add(proposal_id)

            proposals.append(
                CareerEvidenceProposal(
                    proposal_id=proposal_id,
                    category=candidate.category,
                    title=candidate.title,
                    source_section=(section.section),
                    source_section_order_index=(section.order_index),
                    technologies=list(candidate.technologies),
                    capabilities=list(candidate.capabilities),
                    claims=list(candidate.claims),
                    source_references=[
                        SourceReference(
                            source_type=(EvidenceSourceType.UPLOADED_CV),
                            source_id=(document.document_id),
                            source_excerpt=(candidate.source_excerpt),
                        )
                    ],
                    warnings=list(candidate.warnings),
                )
            )

        return proposals


def build_section_index(
    sections: list[ParsedCVSection],
) -> dict[int, ParsedCVSection]:
    """Build a unique source-order lookup."""

    indexed = {section.order_index: section for section in sections}

    if len(indexed) != len(sections):
        raise CVEvidenceProposalValidationError(
            "Parsed CV contains duplicate section order indexes."
        )

    return indexed


def validate_candidate(
    *,
    candidate: CareerEvidenceCandidate,
    section: ParsedCVSection,
) -> None:
    """Apply deterministic provenance and category checks."""

    allowed_categories = ALLOWED_CATEGORIES_BY_SECTION[section.section]

    if candidate.category not in allowed_categories:
        raise CVEvidenceProposalValidationError(
            "Evidence candidate category is not valid for its source CV section."
        )

    source_text = normalise_provenance_text(section.text)

    excerpt = normalise_provenance_text(candidate.source_excerpt)

    if not excerpt or excerpt not in source_text:
        raise CVEvidenceProposalValidationError(
            "Evidence candidate source excerpt "
            "is not present in the referenced CV section."
        )

    validate_string_list(
        values=candidate.technologies,
        field_name="technologies",
    )

    validate_string_list(
        values=candidate.capabilities,
        field_name="capabilities",
    )

    validate_string_list(
        values=candidate.claims,
        field_name="claims",
    )

    validate_claim_grounding(
        candidate=candidate,
    )

    validate_technology_grounding(
        candidate=candidate,
    )


def validate_string_list(
    *,
    values: list[str],
    field_name: str,
) -> None:
    """Reject blank or duplicate model-generated list values."""

    normalised = [value.strip() for value in values]

    if any(not value for value in normalised):
        raise CVEvidenceProposalValidationError(
            f"Evidence candidate contains a blank {field_name} value."
        )

    folded = [value.casefold() for value in normalised]

    if len(set(folded)) != len(folded):
        raise CVEvidenceProposalValidationError(
            f"Evidence candidate contains duplicate {field_name} values."
        )


def validate_claim_grounding(
    *,
    candidate: CareerEvidenceCandidate,
) -> None:
    """Require every factual claim to be extractively grounded."""

    excerpt = normalise_provenance_text(candidate.source_excerpt).casefold()

    for claim in candidate.claims:
        normalised_claim = normalise_provenance_text(claim).casefold()

        if normalised_claim not in excerpt:
            raise CVEvidenceProposalValidationError(
                "Evidence candidate claim is not "
                "an extractive span of its source excerpt."
            )


def validate_technology_grounding(
    *,
    candidate: CareerEvidenceCandidate,
) -> None:
    """Require technologies to be explicitly named in the excerpt."""

    for technology in candidate.technologies:
        if not contains_explicit_text(
            source=candidate.source_excerpt,
            value=technology,
        ):
            raise CVEvidenceProposalValidationError(
                "Evidence candidate technology is not "
                "explicitly present in its source excerpt."
            )


def contains_explicit_text(
    *,
    source: str,
    value: str,
) -> bool:
    """Match explicit text with alphanumeric boundaries."""

    normalised_source = normalise_provenance_text(source)

    normalised_value = normalise_provenance_text(value)

    pattern = r"(?<![A-Za-z0-9])" + re.escape(normalised_value) + r"(?![A-Za-z0-9])"

    return (
        re.search(
            pattern,
            normalised_source,
            flags=re.IGNORECASE,
        )
        is not None
    )


def normalise_provenance_text(
    text: str,
) -> str:
    """Normalise whitespace for deterministic excerpt matching."""

    return " ".join(text.split())


def build_evidence_proposal_id(
    *,
    document_id: str,
    candidate: CareerEvidenceCandidate,
) -> str:
    """Build a stable content-derived evidence proposal identifier."""

    fingerprint = "|".join(
        [
            document_id,
            str(candidate.source_section_order_index),
            candidate.category.value,
            normalise_provenance_text(candidate.source_excerpt),
        ]
    )

    digest = sha256(fingerprint.encode("utf-8")).hexdigest()[:16].upper()

    return f"EVP-{digest}"

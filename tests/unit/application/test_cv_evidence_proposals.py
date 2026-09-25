"""Tests for grounded CV evidence proposal construction."""

import pytest

from careerops_agent_engine.application.exceptions import (
    CVEvidenceProposalValidationError,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    CVEvidenceProposalService,
    build_evidence_proposal_id,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
    ParsedCVSection,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceCandidate,
)


class FakeCVEvidenceExtractor:
    """Return controlled model evidence candidates."""

    def __init__(
        self,
        candidates: list[CareerEvidenceCandidate],
    ) -> None:
        """Store deterministic extraction output."""

        self.candidates = candidates

    def extract(
        self,
        *,
        document: ParsedCVDocument,
    ) -> list[CareerEvidenceCandidate]:
        """Return configured candidates."""

        del document

        return list(self.candidates)


def build_document() -> ParsedCVDocument:
    """Create a parsed CV with trusted source sections."""

    return ParsedCVDocument(
        document_id="DOC-001",
        preamble_text=("Candidate Name\ncandidate@example.com"),
        sections=[
            ParsedCVSection(
                section=CVSection.EXPERIENCE,
                heading="Experience",
                text=(
                    "Software Engineer at Example Ltd.\n"
                    "Built Python APIs using FastAPI "
                    "and Docker."
                ),
                order_index=0,
            ),
            ParsedCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                text=("CareerOps\nBuilt a stateful LangGraph workflow."),
                order_index=1,
            ),
            ParsedCVSection(
                section=CVSection.SKILLS,
                heading="Skills",
                text="Python, FastAPI, Docker",
                order_index=2,
            ),
        ],
        warnings=[],
    )


def build_valid_candidate() -> CareerEvidenceCandidate:
    """Create one grounded employment candidate."""

    return CareerEvidenceCandidate(
        category=EvidenceCategory.EMPLOYMENT,
        title="Software Engineer at Example Ltd.",
        source_section_order_index=0,
        source_excerpt=(
            "Software Engineer at Example Ltd. "
            "Built Python APIs using FastAPI "
            "and Docker."
        ),
        technologies=[
            "Python",
            "FastAPI",
            "Docker",
        ],
        capabilities=[
            "API development",
            "Application containerisation",
        ],
        claims=[
            "Software Engineer at Example Ltd.",
            "Built Python APIs using FastAPI and Docker.",
        ],
        warnings=[],
    )


def test_valid_candidate_becomes_pending_proposal() -> None:
    """Grounded model output should remain pending human review."""

    service = CVEvidenceProposalService(
        extractor=FakeCVEvidenceExtractor([build_valid_candidate()])
    )

    proposals = service.generate(document=build_document())

    assert len(proposals) == 1

    proposal = proposals[0]

    assert proposal.proposal_id.startswith("EVP-")

    assert proposal.verification_status is VerificationStatus.PENDING

    assert proposal.source_section is CVSection.EXPERIENCE

    assert proposal.claims

    source = proposal.source_references[0]

    assert source.source_type is EvidenceSourceType.UPLOADED_CV

    assert source.source_id == "DOC-001"

    assert source.source_excerpt == (build_valid_candidate().source_excerpt)


def test_invented_source_excerpt_is_rejected() -> None:
    """Gemini cannot create provenance absent from the CV."""

    candidate = build_valid_candidate().model_copy(
        update={
            "source_excerpt": ("Built and operated Kubernetes clusters in production.")
        }
    )

    service = CVEvidenceProposalService(extractor=FakeCVEvidenceExtractor([candidate]))

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match="source excerpt is not present",
    ):
        service.generate(document=build_document())


def test_unknown_section_index_is_rejected() -> None:
    """A candidate cannot reference a section that does not exist."""

    candidate = build_valid_candidate().model_copy(
        update={"source_section_order_index": 99}
    )

    service = CVEvidenceProposalService(extractor=FakeCVEvidenceExtractor([candidate]))

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match="unknown CV section",
    ):
        service.generate(document=build_document())


def test_category_must_match_source_section() -> None:
    """Skills text cannot become invented project evidence."""

    candidate = CareerEvidenceCandidate(
        category=EvidenceCategory.PROJECT,
        title="Python Skills",
        source_section_order_index=2,
        source_excerpt=("Python, FastAPI, Docker"),
        technologies=[
            "Python",
            "FastAPI",
            "Docker",
        ],
        capabilities=[],
        claims=["Lists Python, FastAPI and Docker."],
        warnings=[],
    )

    service = CVEvidenceProposalService(extractor=FakeCVEvidenceExtractor([candidate]))

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match="not valid for its source CV section",
    ):
        service.generate(document=build_document())


def test_duplicate_candidates_are_rejected() -> None:
    """Identical model candidates should not produce duplicate proposals."""

    candidate = build_valid_candidate()

    service = CVEvidenceProposalService(
        extractor=FakeCVEvidenceExtractor(
            [
                candidate,
                candidate,
            ]
        )
    )

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match="Duplicate CV evidence proposals",
    ):
        service.generate(document=build_document())


def test_empty_extraction_is_valid() -> None:
    """A CV with no extractable evidence may return no proposals."""

    service = CVEvidenceProposalService(extractor=FakeCVEvidenceExtractor([]))

    assert service.generate(document=build_document()) == []


def test_evidence_proposal_id_is_stable() -> None:
    """The same grounded candidate should retain its proposal ID."""

    candidate = build_valid_candidate()

    first = build_evidence_proposal_id(
        document_id="DOC-001",
        candidate=candidate,
    )

    second = build_evidence_proposal_id(
        document_id="DOC-001",
        candidate=candidate,
    )

    assert first == second
    assert first.startswith("EVP-")
    assert len(first) <= 64


def test_paraphrased_claim_is_rejected() -> None:
    """Model claims must preserve the CV's factual wording."""

    candidate = build_valid_candidate().model_copy(
        update={"claims": [("Worked as a Software Engineer at Example Ltd.")]}
    )

    service = CVEvidenceProposalService(extractor=FakeCVEvidenceExtractor([candidate]))

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match="claim is not an extractive span",
    ):
        service.generate(document=build_document())


def test_unmentioned_technology_is_rejected() -> None:
    """Technologies must be explicitly present in source evidence."""

    candidate = build_valid_candidate().model_copy(
        update={
            "technologies": [
                "Python",
                "FastAPI",
                "Docker",
                "Kubernetes",
            ]
        }
    )

    service = CVEvidenceProposalService(extractor=FakeCVEvidenceExtractor([candidate]))

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match=("technology is not explicitly present"),
    ):
        service.generate(document=build_document())


def test_evidence_proposal_id_ignores_generated_title() -> None:
    """Model wording changes must not change source evidence identity."""

    candidate = build_valid_candidate()

    renamed_candidate = candidate.model_copy(
        update={"title": "Alternative Model Generated Title"}
    )

    original_id = build_evidence_proposal_id(
        document_id="DOC-001",
        candidate=candidate,
    )

    renamed_id = build_evidence_proposal_id(
        document_id="DOC-001",
        candidate=renamed_candidate,
    )

    assert original_id == renamed_id


def test_proposal_preserves_manual_text_provenance() -> None:
    """Text-derived proposals must not be labelled as uploaded CV evidence."""

    service = CVEvidenceProposalService(
        extractor=FakeCVEvidenceExtractor([build_valid_candidate()])
    )

    text_document = build_document().model_copy(
        update={
            "document_id": "DOC-TEXT-001",
            "source_type": EvidenceSourceType.MANUAL_ENTRY,
        }
    )

    proposals = service.generate(document=text_document)

    source = proposals[0].source_references[0]

    assert source.source_type is EvidenceSourceType.MANUAL_ENTRY
    assert source.source_id == "DOC-TEXT-001"


def test_excessive_candidate_count_is_rejected() -> None:
    """One extraction cannot create an unbounded review queue."""

    service = CVEvidenceProposalService(
        extractor=FakeCVEvidenceExtractor([build_valid_candidate() for _ in range(31)])
    )

    with pytest.raises(
        CVEvidenceProposalValidationError,
        match="maximum supported candidate count",
    ):
        service.generate(document=build_document())

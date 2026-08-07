"""Tests for grounded CV proposal generation."""

from collections.abc import Sequence

import pytest

from careerops_agent_engine.application.exceptions import (
    CVProposalValidationError,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
    build_proposal_id,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_requirement() -> JobRequirement:
    """Create one requirement for proposal tests."""

    return JobRequirement(
        requirement_id="REQ-001",
        name="Python Engineering",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected=("Practical Python software-engineering experience."),
        importance_score=5,
        source_text=("Strong Python software-engineering experience is required."),
    )


def build_evidence() -> CareerEvidence:
    """Create one approved project record."""

    return CareerEvidence(
        evidence_id="EVD-PYTHON",
        category=EvidenceCategory.PROJECT,
        title="AI API",
        verification_status=VerificationStatus.APPROVED,
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=[
            "API development",
        ],
        approved_claims=["Built a FastAPI application using Python."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id="SRC-PYTHON",
            )
        ],
    )


class FakeCVProposalGenerator:
    """Predictable proposal generator for service tests."""

    def __init__(
        self,
        *,
        supporting_evidence_ids: list[str] | None = None,
    ) -> None:
        """Configure optional model-selected evidence IDs."""

        self.supporting_evidence_ids = supporting_evidence_ids
        self.call_count = 0

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Return a deterministic CV proposal."""

        del job_id, evidence_match

        self.call_count += 1

        selected_ids = (
            self.supporting_evidence_ids
            if self.supporting_evidence_ids is not None
            else [evidence.evidence_id for evidence in approved_evidence]
        )

        return CVChangeProposal(
            proposal_id=proposal_id,
            section=CVSection.PROJECTS,
            proposed_text=("Built a FastAPI application using Python."),
            requirement_ids=[requirement.requirement_id],
            supporting_evidence_ids=selected_ids,
            confidence_score=0.95,
            warnings=[],
        )


def build_service(
    generator: FakeCVProposalGenerator,
) -> CVProposalGenerationService:
    """Create an isolated proposal service."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence(),
            ]
        }
    )

    return CVProposalGenerationService(
        repository=repository,
        generator=generator,
    )


def test_strong_match_generates_grounded_proposal() -> None:
    """Strong direct evidence should produce a proposal."""

    requirement = build_requirement()
    generator = FakeCVProposalGenerator()
    service = build_service(generator)

    match = EvidenceMatch(
        requirement_id="REQ-001",
        match_strength=MatchStrength.STRONG,
        direct_evidence_ids=["EVD-PYTHON"],
        related_evidence_ids=[],
        explanation="Direct Python evidence exists.",
        gap=False,
    )

    proposal = service.generate_for_requirement(
        job_id="JOB-001",
        user_id="USER-001",
        requirement=requirement,
        evidence_match=match,
    )

    assert proposal is not None
    assert generator.call_count == 1
    assert proposal.requirement_ids == ["REQ-001"]
    assert proposal.supporting_evidence_ids == ["EVD-PYTHON"]
    assert proposal.requires_human_approval is True


def test_related_match_does_not_generate_cv_claim() -> None:
    """Related evidence must remain a gap, not become a CV claim."""

    requirement = build_requirement()
    generator = FakeCVProposalGenerator()
    service = build_service(generator)

    match = EvidenceMatch(
        requirement_id="REQ-001",
        match_strength=MatchStrength.RELATED,
        direct_evidence_ids=[],
        related_evidence_ids=["EVD-PYTHON"],
        explanation="Only adjacent experience exists.",
        gap=True,
    )

    proposal = service.generate_for_requirement(
        job_id="JOB-001",
        user_id="USER-001",
        requirement=requirement,
        evidence_match=match,
    )

    assert proposal is None
    assert generator.call_count == 0


def test_cross_user_direct_evidence_is_rejected() -> None:
    """The generator cannot use inaccessible evidence."""

    requirement = build_requirement()
    generator = FakeCVProposalGenerator()
    service = build_service(generator)

    match = EvidenceMatch(
        requirement_id="REQ-001",
        match_strength=MatchStrength.STRONG,
        direct_evidence_ids=["EVD-PYTHON"],
        related_evidence_ids=[],
        explanation="Direct evidence was claimed.",
        gap=False,
    )

    with pytest.raises(
        CVProposalValidationError,
        match="no longer approved or accessible",
    ):
        service.generate_for_requirement(
            job_id="JOB-001",
            user_id="USER-OTHER",
            requirement=requirement,
            evidence_match=match,
        )

    assert generator.call_count == 0


def test_generator_cannot_cite_unapproved_context() -> None:
    """Generated evidence IDs must belong to direct context."""

    requirement = build_requirement()

    generator = FakeCVProposalGenerator(supporting_evidence_ids=["EVD-INVENTED"])
    service = build_service(generator)

    match = EvidenceMatch(
        requirement_id="REQ-001",
        match_strength=MatchStrength.STRONG,
        direct_evidence_ids=["EVD-PYTHON"],
        related_evidence_ids=[],
        explanation="Direct Python evidence exists.",
        gap=False,
    )

    with pytest.raises(
        CVProposalValidationError,
        match="outside the validated direct-evidence set",
    ):
        service.generate_for_requirement(
            job_id="JOB-001",
            user_id="USER-001",
            requirement=requirement,
            evidence_match=match,
        )


def test_proposal_id_is_stable_for_same_inputs() -> None:
    """Retries should resolve to the same proposal identifier."""

    first = build_proposal_id(
        job_id="JOB-001",
        requirement_id="REQ-001",
    )
    second = build_proposal_id(
        job_id="JOB-001",
        requirement_id="REQ-001",
    )

    assert first == second
    assert first.startswith("CVP-")
    assert len(first) <= 64

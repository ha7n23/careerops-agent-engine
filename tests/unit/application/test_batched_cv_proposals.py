"""Tests for batched grounded CV proposal generation."""

from collections.abc import Sequence

import pytest

from careerops_agent_engine.application.exceptions import (
    CVProposalValidationError,
)
from careerops_agent_engine.application.ports.cv_proposal_generator import (
    CVProposalGenerationRequest,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
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


class FakeBatchGenerator:
    """Return deterministic proposals while counting batch calls."""

    def __init__(
        self,
        *,
        omit_last: bool = False,
    ) -> None:
        """Configure the fake provider response."""

        self.omit_last = omit_last
        self.batch_call_count = 0

    def generate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVChangeProposal:
        """Reject accidental use of single proposal generation."""

        del (
            proposal_id,
            job_id,
            requirement,
            evidence_match,
            approved_evidence,
        )

        raise AssertionError("Batched generation must not use the single-call path.")

    def generate_batch(
        self,
        *,
        job_id: str,
        requests: Sequence[CVProposalGenerationRequest],
    ) -> list[CVChangeProposal]:
        """Return one proposal for each supplied request."""

        del job_id

        self.batch_call_count += 1
        selected_requests = list(requests)

        if self.omit_last:
            selected_requests = selected_requests[:-1]

        return [
            CVChangeProposal(
                proposal_id=request.proposal_id,
                section=CVSection.PROJECTS,
                proposed_text=(
                    f"Built an API for {request.requirement.requirement_id}."
                ),
                requirement_ids=[request.requirement.requirement_id],
                supporting_evidence_ids=[
                    evidence.evidence_id for evidence in request.approved_evidence
                ],
                confidence_score=0.95,
                warnings=[],
            )
            for request in selected_requests
        ]

    def regenerate(
        self,
        *,
        proposal_id: str,
        job_id: str,
        requirement: JobRequirement,
        evidence_match: EvidenceMatch,
        approved_evidence: Sequence[CareerEvidence],
        previous_proposal: CVChangeProposal,
        reviewer_feedback: str,
    ) -> CVChangeProposal:
        """Reject regeneration in initial-batch tests."""

        del (
            proposal_id,
            job_id,
            requirement,
            evidence_match,
            approved_evidence,
            previous_proposal,
            reviewer_feedback,
        )

        raise AssertionError("Initial batch generation must not use regeneration.")


def build_requirement(
    requirement_id: str,
) -> JobRequirement:
    """Create one requirement."""

    return JobRequirement(
        requirement_id=requirement_id,
        name=f"Requirement {requirement_id}",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected="Practical engineering experience.",
        importance_score=5,
        source_text="Practical engineering experience is required.",
    )


def build_evidence(
    evidence_id: str,
) -> CareerEvidence:
    """Create one approved evidence record."""

    return CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title=f"Project {evidence_id}",
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python"],
        capabilities=["API development"],
        approved_claims=["Built a Python API."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id=f"SRC-{evidence_id}",
            )
        ],
    )


def build_match(
    *,
    requirement_id: str,
    evidence_id: str,
    strength: MatchStrength = MatchStrength.STRONG,
) -> EvidenceMatch:
    """Create one evidence match."""

    eligible = strength in {
        MatchStrength.STRONG,
        MatchStrength.PARTIAL,
    }

    return EvidenceMatch(
        requirement_id=requirement_id,
        match_strength=strength,
        direct_evidence_ids=([evidence_id] if eligible else []),
        related_evidence_ids=([] if eligible else [evidence_id]),
        explanation="Evidence match for testing.",
        gap=not eligible,
    )


def build_service(
    generator: FakeBatchGenerator,
) -> CVProposalGenerationService:
    """Create a service with two approved evidence records."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence("EVD-001"),
                build_evidence("EVD-002"),
            ]
        }
    )

    return CVProposalGenerationService(
        repository=repository,
        generator=generator,
    )


def test_multiple_eligible_requirements_use_one_batch_call() -> None:
    """Eligible requirements should share one provider invocation."""

    generator = FakeBatchGenerator()
    service = build_service(generator)

    proposals = service.generate_for_requirements(
        job_id="JOB-001",
        user_id="USER-001",
        requirements=[
            build_requirement("REQ-001"),
            build_requirement("REQ-002"),
        ],
        evidence_matches=[
            build_match(
                requirement_id="REQ-001",
                evidence_id="EVD-001",
            ),
            build_match(
                requirement_id="REQ-002",
                evidence_id="EVD-002",
            ),
        ],
    )

    assert generator.batch_call_count == 1
    assert [proposal.requirement_ids for proposal in proposals] == [
        ["REQ-001"],
        ["REQ-002"],
    ]
    assert [proposal.supporting_evidence_ids for proposal in proposals] == [
        ["EVD-001"],
        ["EVD-002"],
    ]


def test_no_eligible_requirements_skips_batch_call() -> None:
    """Related evidence should not trigger proposal generation."""

    generator = FakeBatchGenerator()
    service = build_service(generator)

    proposals = service.generate_for_requirements(
        job_id="JOB-001",
        user_id="USER-001",
        requirements=[
            build_requirement("REQ-001"),
        ],
        evidence_matches=[
            build_match(
                requirement_id="REQ-001",
                evidence_id="EVD-001",
                strength=MatchStrength.RELATED,
            )
        ],
    )

    assert proposals == []
    assert generator.batch_call_count == 0


def test_incomplete_batch_response_is_rejected() -> None:
    """The provider cannot silently omit an eligible requirement."""

    generator = FakeBatchGenerator(omit_last=True)
    service = build_service(generator)

    with pytest.raises(
        CVProposalValidationError,
        match="exactly one proposal per eligible requirement",
    ):
        service.generate_for_requirements(
            job_id="JOB-001",
            user_id="USER-001",
            requirements=[
                build_requirement("REQ-001"),
                build_requirement("REQ-002"),
            ],
            evidence_matches=[
                build_match(
                    requirement_id="REQ-001",
                    evidence_id="EVD-001",
                ),
                build_match(
                    requirement_id="REQ-002",
                    evidence_id="EVD-002",
                ),
            ],
        )

    assert generator.batch_call_count == 1

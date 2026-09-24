"""Tests for deterministic CV claim-verification controls."""

from collections.abc import Sequence

import pytest

from careerops_agent_engine.application.exceptions import (
    CVClaimVerificationValidationError,
)
from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerificationRequest,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)
from careerops_agent_engine.domain.models.verification import (
    ClaimAssessment,
    CVClaimVerificationReport,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_evidence() -> CareerEvidence:
    """Create approved Python evidence."""

    return CareerEvidence(
        evidence_id="EVD-PYTHON",
        category=EvidenceCategory.PROJECT,
        title="AI API",
        verification_status=VerificationStatus.APPROVED,
        technologies=[
            "Python",
            "FastAPI",
            "Docker",
        ],
        capabilities=[
            "API development",
            "Containerisation",
        ],
        approved_claims=[
            "Built a FastAPI application using Python.",
            "Containerised the application using Docker.",
        ],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id="SRC-PYTHON",
            )
        ],
    )


def build_proposal() -> CVChangeProposal:
    """Create one proposal backed by approved evidence."""

    return CVChangeProposal(
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        proposed_text=("Built a FastAPI application using Python."),
        requirement_ids=["REQ-001"],
        supporting_evidence_ids=["EVD-PYTHON"],
        confidence_score=0.95,
        warnings=[],
    )


class FakeClaimVerifier:
    """Return a preconfigured verification report."""

    def __init__(
        self,
        report: CVClaimVerificationReport,
    ) -> None:
        """Store the report returned by the fake."""

        self.report = report
        self.call_count = 0

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Return the configured result."""

        del proposal, approved_evidence

        self.call_count += 1
        return self.report

    def verify_batch(
        self,
        *,
        requests: Sequence[CVClaimVerificationRequest],
    ) -> list[CVClaimVerificationReport]:
        """Reuse deterministic single verification for test requests."""

        return [
            self.verify(
                proposal=request.proposal,
                approved_evidence=request.approved_evidence,
            )
            for request in requests
        ]


def build_service(
    report: CVClaimVerificationReport,
) -> tuple[
    CVClaimVerificationService,
    FakeClaimVerifier,
]:
    """Create a verification service with approved evidence."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence(),
            ]
        }
    )

    verifier = FakeClaimVerifier(report)

    service = CVClaimVerificationService(
        repository=repository,
        verifier=verifier,
    )

    return service, verifier


def test_fully_supported_report_is_accepted() -> None:
    """A grounded report should pass deterministic validation."""

    report = CVClaimVerificationReport(
        proposal_id="CVP-001",
        claims=[
            ClaimAssessment(
                claim_text=("Built a FastAPI application using Python."),
                supported=True,
                supporting_evidence_ids=["EVD-PYTHON"],
                explanation=("The approved project claim states this directly."),
            )
        ],
        coverage_complete=True,
        coverage_notes=[],
        fully_supported=True,
        unsupported_claims=[],
    )

    service, verifier = build_service(report)

    result = service.verify_proposal(
        user_id="USER-001",
        proposal=build_proposal(),
    )

    assert result.fully_supported is True
    assert result.unsupported_claims == []
    assert verifier.call_count == 1


def test_unsupported_claim_remains_blocking() -> None:
    """Unsupported content must not become review-ready."""

    unsupported_text = "Reduced API latency by 40 percent."

    report = CVClaimVerificationReport(
        proposal_id="CVP-001",
        claims=[
            ClaimAssessment(
                claim_text=unsupported_text,
                supported=False,
                supporting_evidence_ids=[],
                explanation=("No approved evidence contains this metric."),
            )
        ],
        coverage_complete=True,
        coverage_notes=[],
        fully_supported=False,
        unsupported_claims=[unsupported_text],
    )

    service, _ = build_service(report)

    result = service.verify_proposal(
        user_id="USER-001",
        proposal=build_proposal(),
    )

    assert result.fully_supported is False
    assert result.unsupported_claims == [unsupported_text]


def test_incomplete_coverage_is_not_fully_supported() -> None:
    """Unverified portions of the proposal must block approval."""

    report = CVClaimVerificationReport(
        proposal_id="CVP-001",
        claims=[
            ClaimAssessment(
                claim_text=("Built a FastAPI application using Python."),
                supported=True,
                supporting_evidence_ids=["EVD-PYTHON"],
                explanation="Directly supported.",
            )
        ],
        coverage_complete=False,
        coverage_notes=["One additional assertion could not be assessed."],
        fully_supported=False,
        unsupported_claims=[],
    )

    service, _ = build_service(report)

    result = service.verify_proposal(
        user_id="USER-001",
        proposal=build_proposal(),
    )

    assert result.fully_supported is False
    assert result.coverage_complete is False


def test_verifier_cannot_cite_unapproved_evidence() -> None:
    """Verifier evidence IDs must remain inside proposal context."""

    report = CVClaimVerificationReport(
        proposal_id="CVP-001",
        claims=[
            ClaimAssessment(
                claim_text=("Built a FastAPI application using Python."),
                supported=True,
                supporting_evidence_ids=["EVD-INVENTED"],
                explanation="Claimed support.",
            )
        ],
        coverage_complete=True,
        coverage_notes=[],
        fully_supported=True,
        unsupported_claims=[],
    )

    service, _ = build_service(report)

    with pytest.raises(
        CVClaimVerificationValidationError,
        match="outside the proposal's approved evidence set",
    ):
        service.verify_proposal(
            user_id="USER-001",
            proposal=build_proposal(),
        )


def test_cross_user_evidence_is_rejected_before_verification() -> None:
    """Authenticated user isolation applies before the model call."""

    report = CVClaimVerificationReport(
        proposal_id="CVP-001",
        claims=[
            ClaimAssessment(
                claim_text=("Built a FastAPI application using Python."),
                supported=True,
                supporting_evidence_ids=["EVD-PYTHON"],
                explanation="Directly supported.",
            )
        ],
        coverage_complete=True,
        coverage_notes=[],
        fully_supported=True,
        unsupported_claims=[],
    )

    service, verifier = build_service(report)

    with pytest.raises(
        CVClaimVerificationValidationError,
        match="no longer approved or accessible",
    ):
        service.verify_proposal(
            user_id="USER-OTHER",
            proposal=build_proposal(),
        )

    assert verifier.call_count == 0

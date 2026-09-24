"""Tests for batched CV claim-verification controls."""

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


class FakeBatchClaimVerifier:
    """Return deterministic reports while counting batch calls."""

    def __init__(
        self,
        *,
        omit_last: bool = False,
        cross_cite: bool = False,
    ) -> None:
        """Configure the fake provider response."""

        self.omit_last = omit_last
        self.cross_cite = cross_cite
        self.batch_call_count = 0

    def verify(
        self,
        *,
        proposal: CVChangeProposal,
        approved_evidence: Sequence[CareerEvidence],
    ) -> CVClaimVerificationReport:
        """Reject accidental use of single verification."""

        del proposal, approved_evidence

        raise AssertionError("Initial verification must use the batch path.")

    def verify_batch(
        self,
        *,
        requests: Sequence[CVClaimVerificationRequest],
    ) -> list[CVClaimVerificationReport]:
        """Return one supported report per selected request."""

        self.batch_call_count += 1
        selected_requests = list(requests)

        if self.omit_last:
            selected_requests = selected_requests[:-1]

        reports: list[CVClaimVerificationReport] = []

        for index, request in enumerate(selected_requests):
            evidence_id = (
                "EVD-001"
                if self.cross_cite and index == 1
                else request.approved_evidence[0].evidence_id
            )
            claim_text = request.proposal.proposed_text

            reports.append(
                CVClaimVerificationReport(
                    proposal_id=(request.proposal.proposal_id),
                    claims=[
                        ClaimAssessment(
                            claim_text=claim_text,
                            supported=True,
                            supporting_evidence_ids=[evidence_id],
                            explanation=("Approved evidence supports the claim."),
                        )
                    ],
                    coverage_complete=True,
                    coverage_notes=[],
                    fully_supported=True,
                    unsupported_claims=[],
                )
            )

        return reports


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
        approved_claims=[f"Built a Python API using {evidence_id}."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id=f"SRC-{evidence_id}",
            )
        ],
    )


def build_proposal(
    *,
    proposal_id: str,
    requirement_id: str,
    evidence_id: str,
) -> CVChangeProposal:
    """Create one proposal with scoped evidence."""

    return CVChangeProposal(
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        proposed_text=(f"Built a Python API using {evidence_id}."),
        requirement_ids=[requirement_id],
        supporting_evidence_ids=[evidence_id],
        confidence_score=0.95,
        warnings=[],
    )


def build_service(
    verifier: FakeBatchClaimVerifier,
) -> CVClaimVerificationService:
    """Create a service with two approved records."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence("EVD-001"),
                build_evidence("EVD-002"),
            ]
        }
    )

    return CVClaimVerificationService(
        repository=repository,
        verifier=verifier,
    )


def build_proposals() -> list[CVChangeProposal]:
    """Create two independently grounded proposals."""

    return [
        build_proposal(
            proposal_id="CVP-001",
            requirement_id="REQ-001",
            evidence_id="EVD-001",
        ),
        build_proposal(
            proposal_id="CVP-002",
            requirement_id="REQ-002",
            evidence_id="EVD-002",
        ),
    ]


def test_multiple_proposals_use_one_batch_call() -> None:
    """Initial reports should share one verifier invocation."""

    verifier = FakeBatchClaimVerifier()
    service = build_service(verifier)

    reports = service.verify_proposals(
        user_id="USER-001",
        proposals=build_proposals(),
    )

    assert verifier.batch_call_count == 1
    assert [report.proposal_id for report in reports] == [
        "CVP-001",
        "CVP-002",
    ]
    assert all(report.fully_supported for report in reports)


def test_empty_proposals_skip_batch_call() -> None:
    """No proposals should require no verifier call."""

    verifier = FakeBatchClaimVerifier()
    service = build_service(verifier)

    reports = service.verify_proposals(
        user_id="USER-001",
        proposals=[],
    )

    assert reports == []
    assert verifier.batch_call_count == 0


def test_incomplete_batch_response_is_rejected() -> None:
    """The provider cannot omit a proposal report."""

    verifier = FakeBatchClaimVerifier(omit_last=True)
    service = build_service(verifier)

    with pytest.raises(
        CVClaimVerificationValidationError,
        match="exactly one report per proposal",
    ):
        service.verify_proposals(
            user_id="USER-001",
            proposals=build_proposals(),
        )

    assert verifier.batch_call_count == 1


def test_cross_proposal_evidence_is_rejected() -> None:
    """One proposal cannot borrow another proposal's evidence."""

    verifier = FakeBatchClaimVerifier(cross_cite=True)
    service = build_service(verifier)

    with pytest.raises(
        CVClaimVerificationValidationError,
        match="outside the proposal's approved evidence set",
    ):
        service.verify_proposals(
            user_id="USER-001",
            proposals=build_proposals(),
        )

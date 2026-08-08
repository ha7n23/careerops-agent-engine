"""Tests for deterministic CV evidence overlap detection."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.application.services.cv_evidence_duplicates import (
    CVEvidenceDuplicateDetector,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceOverlapScope,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceOverlapFinding,
    CareerEvidenceProposal,
    SourceReference,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_proposal(
    *,
    proposal_id: str,
    claim: str,
    excerpt: str,
    order_index: int,
) -> CareerEvidenceProposal:
    """Create one pending CV evidence proposal."""

    return CareerEvidenceProposal(
        proposal_id=proposal_id,
        category=EvidenceCategory.PROJECT,
        title=f"Project {order_index}",
        source_section=CVSection.PROJECTS,
        source_section_order_index=order_index,
        technologies=["Python"],
        capabilities=[],
        claims=[claim],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=excerpt,
            )
        ],
        warnings=[],
    )


def build_approved_evidence(
    *,
    evidence_id: str,
    claim: str,
    status: VerificationStatus = (VerificationStatus.APPROVED),
) -> CareerEvidence:
    """Create one existing career evidence record."""

    return CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title="Existing Project",
        verification_status=status,
        technologies=["Python"],
        capabilities=[],
        approved_claims=[claim],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.MANUAL_ENTRY),
                source_id=f"SRC-{evidence_id}",
            )
        ],
    )


def test_normalised_claim_overlap_is_detected_within_document() -> None:
    """Case, punctuation and whitespace should not hide duplicates."""

    first = build_proposal(
        proposal_id="EVP-001",
        claim="Built Python APIs using FastAPI.",
        excerpt="Project A. Built Python APIs using FastAPI.",
        order_index=0,
    )

    second = build_proposal(
        proposal_id="EVP-002",
        claim="  BUILT Python APIs using FastAPI  ",
        excerpt="Project B. BUILT Python APIs using FastAPI.",
        order_index=1,
    )

    detector = CVEvidenceDuplicateDetector(repository=InMemoryEvidenceRepository())

    findings = detector.detect(
        user_id="USER-001",
        proposals=[
            first,
            second,
        ],
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.scope is EvidenceOverlapScope.WITHIN_DOCUMENT

    assert finding.proposal_id == "EVP-002"
    assert finding.matching_proposal_id == "EVP-001"
    assert finding.matched_claims == ["BUILT Python APIs using FastAPI"]


def test_same_source_excerpt_is_detected_within_document() -> None:
    """Two proposals anchored to the same source should be surfaced."""

    first = build_proposal(
        proposal_id="EVP-001",
        claim="CareerOps Agent Engine",
        excerpt=("CareerOps Agent Engine. Built a stateful workflow."),
        order_index=0,
    )

    second = build_proposal(
        proposal_id="EVP-002",
        claim="Built a stateful workflow.",
        excerpt=("CareerOps Agent Engine. Built a stateful workflow."),
        order_index=1,
    )

    detector = CVEvidenceDuplicateDetector(repository=InMemoryEvidenceRepository())

    findings = detector.detect(
        user_id="USER-001",
        proposals=[
            first,
            second,
        ],
    )

    assert len(findings) == 1
    assert findings[0].same_source_excerpt is True


def test_overlap_with_approved_evidence_is_detected() -> None:
    """A pending claim already approved for the user should be flagged."""

    proposal = build_proposal(
        proposal_id="EVP-001",
        claim="Built a stateful LangGraph workflow.",
        excerpt="Built a stateful LangGraph workflow.",
        order_index=0,
    )

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_approved_evidence(
                    evidence_id="EVD-001",
                    claim=("Built a stateful LangGraph workflow."),
                )
            ]
        }
    )

    detector = CVEvidenceDuplicateDetector(repository=repository)

    findings = detector.detect(
        user_id="USER-001",
        proposals=[proposal],
    )

    assert len(findings) == 1

    finding = findings[0]

    assert finding.scope is EvidenceOverlapScope.APPROVED_EVIDENCE

    assert finding.matching_evidence_id == "EVD-001"


def test_rejected_and_cross_user_evidence_are_not_compared() -> None:
    """The existing approved-evidence security boundary must remain intact."""

    claim = "Built a stateful LangGraph workflow."

    proposal = build_proposal(
        proposal_id="EVP-001",
        claim=claim,
        excerpt=claim,
        order_index=0,
    )

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_approved_evidence(
                    evidence_id="EVD-REJECTED",
                    claim=claim,
                    status=(VerificationStatus.REJECTED),
                )
            ],
            "USER-002": [
                build_approved_evidence(
                    evidence_id="EVD-OTHER-USER",
                    claim=claim,
                )
            ],
        }
    )

    detector = CVEvidenceDuplicateDetector(repository=repository)

    assert (
        detector.detect(
            user_id="USER-001",
            proposals=[proposal],
        )
        == []
    )


def test_distinct_evidence_produces_no_overlap_findings() -> None:
    """Unrelated proposals should pass without duplicate warnings."""

    proposal = build_proposal(
        proposal_id="EVP-001",
        claim="Built Python APIs using FastAPI.",
        excerpt="Built Python APIs using FastAPI.",
        order_index=0,
    )

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_approved_evidence(
                    evidence_id="EVD-001",
                    claim="Built a computer vision classifier.",
                )
            ]
        }
    )

    detector = CVEvidenceDuplicateDetector(repository=repository)

    assert (
        detector.detect(
            user_id="USER-001",
            proposals=[proposal],
        )
        == []
    )


def test_overlap_finding_requires_deterministic_matching_signal() -> None:
    """An overlap finding cannot exist without evidence of overlap."""

    with pytest.raises(
        ValidationError,
        match="deterministic matching signal",
    ):
        CareerEvidenceOverlapFinding(
            proposal_id="EVP-002",
            scope=EvidenceOverlapScope.WITHIN_DOCUMENT,
            matching_proposal_id="EVP-001",
            matched_claims=[],
            same_source_excerpt=False,
        )

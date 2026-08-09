"""Tests for persistent CV evidence-review domain models."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import (
    CVEvidenceReviewRunStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceProposal,
    SourceReference,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
    EvidenceReviewResult,
)


def build_proposal() -> CareerEvidenceProposal:
    """Create one pending proposal for audit tests."""

    return CareerEvidenceProposal(
        proposal_id="EVP-001",
        category=EvidenceCategory.PROJECT,
        title="CareerOps",
        source_section=CVSection.PROJECTS,
        source_section_order_index=0,
        technologies=["Python"],
        capabilities=[],
        claims=["Built CareerOps using Python."],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=("Built CareerOps using Python."),
            )
        ],
        warnings=[],
    )


def test_awaiting_review_requires_proposals() -> None:
    """An empty run cannot claim to await human review."""

    with pytest.raises(
        ValidationError,
        match="requires pending proposals",
    ):
        CVEvidenceReviewRunSnapshot(
            review_run_id="EVR-001",
            user_id="USER-001",
            document_id="DOC-001",
            status=(CVEvidenceReviewRunStatus.AWAITING_REVIEW),
        )


def test_completed_run_requires_review_result() -> None:
    """Completion must correspond to a human-review result."""

    with pytest.raises(
        ValidationError,
        match="requires a human-review result",
    ):
        CVEvidenceReviewRunSnapshot(
            review_run_id="EVR-001",
            user_id="USER-001",
            document_id="DOC-001",
            status=(CVEvidenceReviewRunStatus.COMPLETED),
            proposals=[build_proposal()],
        )


def test_awaiting_run_cannot_contain_review_result() -> None:
    """A result cannot appear before the trust boundary is crossed."""

    result = EvidenceReviewResult(rejected_proposal_ids=["EVP-001"])

    with pytest.raises(
        ValidationError,
        match="Only a completed",
    ):
        CVEvidenceReviewRunSnapshot(
            review_run_id="EVR-001",
            user_id="USER-001",
            document_id="DOC-001",
            status=(CVEvidenceReviewRunStatus.AWAITING_REVIEW),
            proposals=[build_proposal()],
            review_result=result,
        )


def test_audit_result_must_match_human_decision() -> None:
    """Audit history cannot misrepresent the applied decision."""

    decision = EvidenceReviewDecision(rejected_proposal_ids=["EVP-001"])

    inconsistent_result = EvidenceReviewResult(rejected_proposal_ids=["EVP-002"])

    with pytest.raises(
        ValidationError,
        match="rejection result",
    ):
        CVEvidenceReviewAuditEntry(
            review_id="EVR-AUDIT-001",
            review_run_id="EVR-001",
            sequence_number=1,
            decision=decision,
            result=inconsistent_result,
        )

"""Tests for human evidence-review domain models."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceProposalEdit,
    EvidenceReviewDecision,
)


def test_evidence_edit_requires_actual_change() -> None:
    """An empty edit must not masquerade as human correction."""

    with pytest.raises(
        ValidationError,
        match="must change at least one field",
    ):
        EvidenceProposalEdit(proposal_id="EVP-001")


def test_review_requires_at_least_one_decision() -> None:
    """An empty human review cannot cross the trust boundary."""

    with pytest.raises(
        ValidationError,
        match="requires at least one proposal decision",
    ):
        EvidenceReviewDecision()


def test_proposal_cannot_be_approved_and_rejected() -> None:
    """One evidence proposal cannot receive conflicting outcomes."""

    with pytest.raises(
        ValidationError,
        match="both approved and rejected",
    ):
        EvidenceReviewDecision(
            approved_proposal_ids=["EVP-001"],
            rejected_proposal_ids=["EVP-001"],
        )


def test_mixed_review_decision_is_valid() -> None:
    """One batch may approve, edit and reject different evidence."""

    decision = EvidenceReviewDecision(
        approved_proposal_ids=["EVP-001"],
        rejected_proposal_ids=["EVP-003"],
        edits=[
            EvidenceProposalEdit(
                proposal_id="EVP-002",
                capabilities=["API development"],
            )
        ],
    )

    assert decision.approved_proposal_ids == ["EVP-001"]
    assert decision.edits[0].proposal_id == ("EVP-002")

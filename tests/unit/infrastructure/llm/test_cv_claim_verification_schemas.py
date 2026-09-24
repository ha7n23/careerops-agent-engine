"""Tests for provider-facing claim-verification schemas."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.infrastructure.llm.schemas import (
    GeneratedClaimAssessment,
    GeneratedClaimVerificationBatch,
    GeneratedClaimVerificationBatchItem,
)


def build_batch_item(
    proposal_id: str = "CVP-001",
) -> GeneratedClaimVerificationBatchItem:
    """Create one valid generated verification."""

    return GeneratedClaimVerificationBatchItem(
        proposal_id=proposal_id,
        claims=[
            GeneratedClaimAssessment(
                claim_text="Built a Python API.",
                supported=True,
                supporting_evidence_ids=["EVD-PYTHON"],
                explanation="Approved evidence supports the claim.",
            )
        ],
        coverage_complete=True,
        coverage_notes=[],
    )


def test_batch_preserves_proposal_mapping() -> None:
    """Each verification should identify its proposal."""

    batch = GeneratedClaimVerificationBatch(
        verifications=[
            build_batch_item("CVP-001"),
            build_batch_item("CVP-002"),
        ]
    )

    assert [verification.proposal_id for verification in batch.verifications] == [
        "CVP-001",
        "CVP-002",
    ]


def test_batch_requires_at_least_one_verification() -> None:
    """An empty provider result should fail validation."""

    with pytest.raises(ValidationError):
        GeneratedClaimVerificationBatch(verifications=[])

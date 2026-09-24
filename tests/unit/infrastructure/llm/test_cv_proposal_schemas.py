"""Tests for provider-facing CV proposal schemas."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import CVSection
from careerops_agent_engine.infrastructure.llm.schemas import (
    GeneratedCVProposalBatch,
    GeneratedCVProposalBatchItem,
)


def build_batch_item(
    requirement_id: str = "REQ-001",
) -> GeneratedCVProposalBatchItem:
    """Create one valid generated batch item."""

    return GeneratedCVProposalBatchItem(
        requirement_id=requirement_id,
        section=CVSection.PROJECTS,
        proposed_text="Built a Python API using FastAPI.",
        supporting_evidence_ids=["EVD-PYTHON"],
        confidence_score=0.95,
        warnings=[],
    )


def test_batch_preserves_requirement_mapping() -> None:
    """Each generated proposal should identify its requirement."""

    batch = GeneratedCVProposalBatch(
        proposals=[
            build_batch_item("REQ-001"),
            build_batch_item("REQ-002"),
        ]
    )

    assert [proposal.requirement_id for proposal in batch.proposals] == [
        "REQ-001",
        "REQ-002",
    ]


def test_batch_requires_at_least_one_proposal() -> None:
    """An empty provider result should fail schema validation."""

    with pytest.raises(ValidationError):
        GeneratedCVProposalBatch(proposals=[])

"""Tests for CV proposal domain models."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import CVSection
from careerops_agent_engine.domain.models.cv import CVChangeProposal


def build_proposal() -> CVChangeProposal:
    """Create a valid evidence-grounded CV proposal."""

    return CVChangeProposal(
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        target_entry_id="PROJECT-001",
        current_text="Built an AI workflow.",
        proposed_text=(
            "Built a stateful LangGraph workflow with checkpointing and human approval."
        ),
        requirement_ids=["REQ-001"],
        supporting_evidence_ids=["EVD-001"],
        confidence_score=0.94,
    )


def test_cv_proposal_accepts_grounded_change() -> None:
    """A proposal with requirements and evidence should be valid."""

    proposal = build_proposal()

    assert proposal.requires_human_approval is True
    assert proposal.supporting_evidence_ids == ["EVD-001"]


def test_cv_proposal_requires_supporting_evidence() -> None:
    """A CV change cannot be proposed without evidence references."""

    proposal_data = build_proposal().model_dump()
    proposal_data["supporting_evidence_ids"] = []

    with pytest.raises(ValidationError):
        CVChangeProposal.model_validate(proposal_data)


def test_cv_proposal_rejects_duplicate_evidence_ids() -> None:
    """Duplicate references must not inflate apparent evidence support."""

    proposal_data = build_proposal().model_dump()
    proposal_data["supporting_evidence_ids"] = [
        "EVD-001",
        "EVD-001",
    ]

    with pytest.raises(
        ValidationError,
        match="Supporting evidence identifiers must be unique",
    ):
        CVChangeProposal.model_validate(proposal_data)


def test_cv_proposal_must_change_existing_text() -> None:
    """A replacement proposal must differ from the current wording."""

    proposal_data = build_proposal().model_dump()
    proposal_data["proposed_text"] = proposal_data["current_text"]

    with pytest.raises(
        ValidationError,
        match="Proposed text must differ",
    ):
        CVChangeProposal.model_validate(proposal_data)

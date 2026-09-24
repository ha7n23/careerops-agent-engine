"""Tests for batched CV proposal generation."""

from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.application.ports.cv_proposal_generator import (
    CVProposalGenerationRequest,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.job import JobRequirement
from careerops_agent_engine.infrastructure.llm.cv_proposal_generator import (
    LangChainCVProposalGenerator,
)


def build_request(
    *,
    requirement_id: str,
    evidence_id: str,
) -> CVProposalGenerationRequest:
    """Create one valid grounded batch request."""

    requirement = JobRequirement(
        requirement_id=requirement_id,
        name=f"Requirement {requirement_id}",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected="Practical engineering experience.",
        importance_score=5,
        source_text="Practical engineering experience is required.",
    )
    evidence = CareerEvidence(
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
    match = EvidenceMatch(
        requirement_id=requirement_id,
        match_strength=MatchStrength.STRONG,
        direct_evidence_ids=[evidence_id],
        related_evidence_ids=[],
        explanation="Direct approved evidence exists.",
        gap=False,
    )

    return CVProposalGenerationRequest(
        proposal_id=f"CVP-{requirement_id}",
        requirement=requirement,
        evidence_match=match,
        approved_evidence=(evidence,),
    )


def build_generator() -> tuple[
    LangChainCVProposalGenerator,
    Mock,
    Mock,
]:
    """Create a generator with isolated structured-model mocks."""

    model = Mock(spec=BaseChatModel)
    single_structured_model = Mock()
    batch_structured_model = Mock()

    model.with_structured_output.side_effect = [
        single_structured_model,
        batch_structured_model,
    ]

    generator = LangChainCVProposalGenerator(
        model=model,
        model_name="quality-model",
    )

    return (
        generator,
        single_structured_model,
        batch_structured_model,
    )


def test_batch_uses_one_model_invocation_and_preserves_request_order() -> None:
    """Multiple requests should use one call and deterministic ordering."""

    generator, single_model, batch_model = build_generator()

    first_request = build_request(
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )
    second_request = build_request(
        requirement_id="REQ-002",
        evidence_id="EVD-002",
    )

    batch_model.invoke.return_value = {
        "proposals": [
            {
                "requirement_id": "REQ-002",
                "section": CVSection.PROJECTS,
                "proposed_text": "Built the second Python API.",
                "supporting_evidence_ids": ["EVD-002"],
                "confidence_score": 0.92,
                "warnings": [],
            },
            {
                "requirement_id": "REQ-001",
                "section": CVSection.PROJECTS,
                "proposed_text": "Built the first Python API.",
                "supporting_evidence_ids": ["EVD-001"],
                "confidence_score": 0.95,
                "warnings": [],
            },
        ]
    }

    proposals = generator.generate_batch(
        job_id="JOB-001",
        requests=[
            first_request,
            second_request,
        ],
    )

    single_model.invoke.assert_not_called()
    batch_model.invoke.assert_called_once()

    assert [proposal.requirement_ids for proposal in proposals] == [
        ["REQ-001"],
        ["REQ-002"],
    ]
    assert [proposal.proposal_id for proposal in proposals] == [
        "CVP-REQ-001",
        "CVP-REQ-002",
    ]
    assert [proposal.supporting_evidence_ids for proposal in proposals] == [
        ["EVD-001"],
        ["EVD-002"],
    ]


def test_empty_batch_does_not_invoke_model() -> None:
    """No eligible requests should require no provider call."""

    generator, single_model, batch_model = build_generator()

    proposals = generator.generate_batch(
        job_id="JOB-001",
        requests=[],
    )

    assert proposals == []
    single_model.invoke.assert_not_called()
    batch_model.invoke.assert_not_called()


def test_batch_rejects_missing_requirement_result() -> None:
    """The provider must return one proposal for every request."""

    generator, _, batch_model = build_generator()

    first_request = build_request(
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )
    second_request = build_request(
        requirement_id="REQ-002",
        evidence_id="EVD-002",
    )

    batch_model.invoke.return_value = {
        "proposals": [
            {
                "requirement_id": "REQ-001",
                "section": CVSection.PROJECTS,
                "proposed_text": "Built the first Python API.",
                "supporting_evidence_ids": ["EVD-001"],
                "confidence_score": 0.95,
                "warnings": [],
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match="exactly one proposal per requested requirement",
    ):
        generator.generate_batch(
            job_id="JOB-001",
            requests=[
                first_request,
                second_request,
            ],
        )


def test_batch_rejects_duplicate_requirement_result() -> None:
    """Duplicate provider results must not silently replace each other."""

    generator, _, batch_model = build_generator()

    request = build_request(
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )

    generated_proposal = {
        "requirement_id": "REQ-001",
        "section": CVSection.PROJECTS,
        "proposed_text": "Built a Python API.",
        "supporting_evidence_ids": ["EVD-001"],
        "confidence_score": 0.95,
        "warnings": [],
    }

    batch_model.invoke.return_value = {
        "proposals": [
            generated_proposal,
            generated_proposal,
        ]
    }

    with pytest.raises(
        ValueError,
        match="duplicate requirement identifiers",
    ):
        generator.generate_batch(
            job_id="JOB-001",
            requests=[request],
        )

"""Tests for batched CV claim verification."""

from unittest.mock import Mock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from careerops_agent_engine.application.ports.cv_claim_verifier import (
    CVClaimVerificationRequest,
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
from careerops_agent_engine.infrastructure.llm.cv_claim_verifier import (
    LangChainCVClaimVerifier,
)


def build_request(
    *,
    proposal_id: str,
    requirement_id: str,
    evidence_id: str,
) -> CVClaimVerificationRequest:
    """Create one grounded verification request."""

    proposal = CVChangeProposal(
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        proposed_text=f"Built a Python API for {requirement_id}.",
        requirement_ids=[requirement_id],
        supporting_evidence_ids=[evidence_id],
        confidence_score=0.95,
        warnings=[],
    )
    evidence = CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title=f"Project {evidence_id}",
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python"],
        capabilities=["API development"],
        approved_claims=[f"Built a Python API for {requirement_id}."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.MANUAL_ENTRY,
                source_id=f"SRC-{evidence_id}",
            )
        ],
    )

    return CVClaimVerificationRequest(
        proposal=proposal,
        approved_evidence=(evidence,),
    )


def build_verifier() -> tuple[
    LangChainCVClaimVerifier,
    Mock,
    Mock,
]:
    """Create a verifier with isolated structured-model mocks."""

    model = Mock(spec=BaseChatModel)
    single_structured_model = Mock()
    batch_structured_model = Mock()

    model.with_structured_output.side_effect = [
        single_structured_model,
        batch_structured_model,
    ]

    verifier = LangChainCVClaimVerifier(
        model=model,
        model_name="quality-model",
    )

    return (
        verifier,
        single_structured_model,
        batch_structured_model,
    )


def build_generated_verification(
    *,
    proposal_id: str,
    requirement_id: str,
    evidence_id: str,
) -> dict[str, object]:
    """Create one valid provider verification payload."""

    return {
        "proposal_id": proposal_id,
        "claims": [
            {
                "claim_text": (f"Built a Python API for {requirement_id}."),
                "supported": True,
                "supporting_evidence_ids": [evidence_id],
                "explanation": ("The approved evidence directly supports the claim."),
            }
        ],
        "coverage_complete": True,
        "coverage_notes": [],
    }


def test_batch_uses_one_invocation_and_preserves_request_order() -> None:
    """Multiple proposals should use one ordered verification call."""

    verifier, single_model, batch_model = build_verifier()

    first_request = build_request(
        proposal_id="CVP-001",
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )
    second_request = build_request(
        proposal_id="CVP-002",
        requirement_id="REQ-002",
        evidence_id="EVD-002",
    )

    batch_model.invoke.return_value = {
        "verifications": [
            build_generated_verification(
                proposal_id="CVP-002",
                requirement_id="REQ-002",
                evidence_id="EVD-002",
            ),
            build_generated_verification(
                proposal_id="CVP-001",
                requirement_id="REQ-001",
                evidence_id="EVD-001",
            ),
        ]
    }

    reports = verifier.verify_batch(
        requests=[
            first_request,
            second_request,
        ]
    )

    single_model.invoke.assert_not_called()
    batch_model.invoke.assert_called_once()

    assert [report.proposal_id for report in reports] == [
        "CVP-001",
        "CVP-002",
    ]
    assert all(report.fully_supported for report in reports)
    assert [report.claims[0].supporting_evidence_ids for report in reports] == [
        ["EVD-001"],
        ["EVD-002"],
    ]


def test_empty_batch_does_not_invoke_model() -> None:
    """No proposals should require no provider call."""

    verifier, single_model, batch_model = build_verifier()

    reports = verifier.verify_batch(requests=[])

    assert reports == []
    single_model.invoke.assert_not_called()
    batch_model.invoke.assert_not_called()


def test_batch_rejects_missing_proposal_result() -> None:
    """Every requested proposal must receive a report."""

    verifier, _, batch_model = build_verifier()

    first_request = build_request(
        proposal_id="CVP-001",
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )
    second_request = build_request(
        proposal_id="CVP-002",
        requirement_id="REQ-002",
        evidence_id="EVD-002",
    )

    batch_model.invoke.return_value = {
        "verifications": [
            build_generated_verification(
                proposal_id="CVP-001",
                requirement_id="REQ-001",
                evidence_id="EVD-001",
            )
        ]
    }

    with pytest.raises(
        ValueError,
        match="exactly one verification per requested proposal",
    ):
        verifier.verify_batch(
            requests=[
                first_request,
                second_request,
            ]
        )


def test_batch_rejects_duplicate_proposal_result() -> None:
    """Duplicate reports must not silently replace each other."""

    verifier, _, batch_model = build_verifier()

    request = build_request(
        proposal_id="CVP-001",
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )
    generated = build_generated_verification(
        proposal_id="CVP-001",
        requirement_id="REQ-001",
        evidence_id="EVD-001",
    )

    batch_model.invoke.return_value = {
        "verifications": [
            generated,
            generated,
        ]
    }

    with pytest.raises(
        ValueError,
        match="duplicate proposal identifiers",
    ):
        verifier.verify_batch(requests=[request])

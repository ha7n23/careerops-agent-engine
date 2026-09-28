"""Tests for the human CV-evidence approval boundary."""

import pytest

from careerops_agent_engine.application.exceptions import (
    CVEvidenceReviewValidationError,
)
from careerops_agent_engine.application.services.cv_evidence_review import (
    CVEvidenceReviewService,
    build_approved_evidence_id,
)
from careerops_agent_engine.domain.enums import (
    CVSection,
    EvidenceCategory,
    EvidenceDuplicateResolutionAction,
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
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceDuplicateResolution,
    EvidenceProposalEdit,
    EvidenceReviewDecision,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)


def build_proposal(
    *,
    proposal_id: str,
    excerpt: str,
    claims: list[str],
    technologies: list[str],
) -> CareerEvidenceProposal:
    """Create one grounded pending evidence proposal."""

    return CareerEvidenceProposal(
        proposal_id=proposal_id,
        category=EvidenceCategory.PROJECT,
        title="CareerOps Agent Engine",
        source_section=CVSection.PROJECTS,
        source_section_order_index=0,
        technologies=technologies,
        capabilities=["AI application engineering"],
        claims=claims,
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=excerpt,
            )
        ],
        warnings=[],
    )


def standard_proposal() -> CareerEvidenceProposal:
    """Create the standard valid review proposal."""

    return build_proposal(
        proposal_id="EVP-001",
        excerpt=("CareerOps Agent Engine. Built Python APIs using FastAPI and Docker."),
        claims=[
            "CareerOps Agent Engine.",
            ("Built Python APIs using FastAPI and Docker."),
        ],
        technologies=[
            "Python",
            "FastAPI",
            "Docker",
        ],
    )


def existing_evidence() -> CareerEvidence:
    """Create the existing approved overlap target."""

    return CareerEvidence(
        evidence_id="EVD-EXISTING",
        category=EvidenceCategory.PROJECT,
        title="Existing CareerOps evidence",
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python"],
        capabilities=["API development"],
        approved_claims=["Built an earlier CareerOps API."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.UPLOADED_CV,
                source_id="DOC-OLD",
                source_excerpt="Built an earlier CareerOps API.",
            )
        ],
    )


def review_service() -> CVEvidenceReviewService:
    """Create a review service with one user-owned target."""

    return CVEvidenceReviewService(
        repository=InMemoryEvidenceRepository({"USER-001": [existing_evidence()]})
    )


def approved_resolution(
    *,
    action: EvidenceDuplicateResolutionAction,
) -> EvidenceDuplicateResolution:
    """Create a resolution for the standard approved-evidence finding."""

    return EvidenceDuplicateResolution(
        proposal_id="EVP-001",
        scope=EvidenceOverlapScope.APPROVED_EVIDENCE,
        matching_evidence_id="EVD-EXISTING",
        action=action,
    )


def test_approve_creates_only_approved_evidence() -> None:
    """Explicit approval should cross the evidence trust boundary."""

    proposal = standard_proposal()

    result = CVEvidenceReviewService().review(
        proposals=[proposal],
        overlap_findings=[],
        decision=EvidenceReviewDecision(approved_proposal_ids=[proposal.proposal_id]),
    )

    assert len(result.approved_evidence) == 1

    evidence = result.approved_evidence[0]

    assert evidence.verification_status is VerificationStatus.APPROVED

    assert evidence.approved_claims == (proposal.claims)

    assert evidence.source_references == (proposal.source_references)


def test_mixed_review_is_supported() -> None:
    """A batch may approve, edit and reject different proposals."""

    first = standard_proposal()

    second = build_proposal(
        proposal_id="EVP-002",
        excerpt=("Built a stateful LangGraph workflow."),
        claims=["Built a stateful LangGraph workflow."],
        technologies=["LangGraph"],
    )

    third = build_proposal(
        proposal_id="EVP-003",
        excerpt="Built a computer vision classifier.",
        claims=["Built a computer vision classifier."],
        technologies=[],
    )

    result = CVEvidenceReviewService().review(
        proposals=[
            first,
            second,
            third,
        ],
        overlap_findings=[],
        decision=EvidenceReviewDecision(
            approved_proposal_ids=["EVP-001"],
            rejected_proposal_ids=["EVP-003"],
            edits=[
                EvidenceProposalEdit(
                    proposal_id="EVP-002",
                    capabilities=["Stateful workflow engineering"],
                )
            ],
        ),
    )

    assert result.approved_proposal_ids == ["EVP-001"]

    assert result.edited_proposal_ids == ["EVP-002"]

    assert result.rejected_proposal_ids == ["EVP-003"]

    assert len(result.approved_evidence) == 2


def test_every_proposal_requires_human_decision() -> None:
    """A batch cannot silently leave evidence undecided."""

    first = standard_proposal()

    second = first.model_copy(update={"proposal_id": "EVP-002"})

    with pytest.raises(
        CVEvidenceReviewValidationError,
        match="Every evidence proposal",
    ):
        CVEvidenceReviewService().review(
            proposals=[
                first,
                second,
            ],
            overlap_findings=[],
            decision=EvidenceReviewDecision(approved_proposal_ids=["EVP-001"]),
        )


def test_unknown_proposal_is_rejected() -> None:
    """Review input cannot reference evidence outside the batch."""

    with pytest.raises(
        CVEvidenceReviewValidationError,
        match="outside the current review set",
    ):
        CVEvidenceReviewService().review(
            proposals=[standard_proposal()],
            overlap_findings=[],
            decision=EvidenceReviewDecision(approved_proposal_ids=["EVP-UNKNOWN"]),
        )


def test_paraphrased_human_claim_is_rejected() -> None:
    """Edited factual claims must remain grounded in the CV source."""

    proposal = standard_proposal()

    with pytest.raises(
        CVEvidenceReviewValidationError,
        match="claim is not an extractive span",
    ):
        CVEvidenceReviewService().review(
            proposals=[proposal],
            overlap_findings=[],
            decision=EvidenceReviewDecision(
                edits=[
                    EvidenceProposalEdit(
                        proposal_id=(proposal.proposal_id),
                        claims=[("Developed production-ready FastAPI services.")],
                    )
                ]
            ),
        )


def test_unmentioned_human_technology_is_rejected() -> None:
    """CV provenance cannot be expanded with an unsupported technology."""

    proposal = standard_proposal()

    with pytest.raises(
        CVEvidenceReviewValidationError,
        match="technology is not explicitly present",
    ):
        CVEvidenceReviewService().review(
            proposals=[proposal],
            overlap_findings=[],
            decision=EvidenceReviewDecision(
                edits=[
                    EvidenceProposalEdit(
                        proposal_id=(proposal.proposal_id),
                        technologies=[
                            "Python",
                            "FastAPI",
                            "Docker",
                            "Kubernetes",
                        ],
                    )
                ]
            ),
        )


def test_overlap_requires_explicit_resolution() -> None:
    """A duplicate warning cannot be silently ignored on approval."""

    proposal = standard_proposal()

    finding = CareerEvidenceOverlapFinding(
        proposal_id=proposal.proposal_id,
        scope=(EvidenceOverlapScope.APPROVED_EVIDENCE),
        matching_evidence_id="EVD-EXISTING",
        matched_claims=["CareerOps Agent Engine."],
        same_source_excerpt=False,
    )

    with pytest.raises(
        CVEvidenceReviewValidationError,
        match="explicit duplicate resolution",
    ):
        CVEvidenceReviewService().review(
            proposals=[proposal],
            overlap_findings=[finding],
            decision=EvidenceReviewDecision(
                approved_proposal_ids=[proposal.proposal_id]
            ),
        )


def test_overlap_may_be_accepted_as_separate() -> None:
    """A human may deliberately accept overlapping evidence separately."""

    proposal = standard_proposal()

    finding = CareerEvidenceOverlapFinding(
        proposal_id=proposal.proposal_id,
        scope=(EvidenceOverlapScope.APPROVED_EVIDENCE),
        matching_evidence_id="EVD-EXISTING",
        matched_claims=["CareerOps Agent Engine."],
        same_source_excerpt=False,
    )

    result = review_service().review(
        user_id="USER-001",
        proposals=[proposal],
        overlap_findings=[finding],
        decision=EvidenceReviewDecision(
            approved_proposal_ids=[proposal.proposal_id],
            duplicate_resolutions=[
                approved_resolution(
                    action=EvidenceDuplicateResolutionAction.ACCEPT_SEPARATE
                )
            ],
        ),
    )

    assert len(result.approved_evidence) == 1

    assert result.duplicate_resolutions[0].action is (
        EvidenceDuplicateResolutionAction.ACCEPT_SEPARATE
    )


def test_rejected_overlap_keeps_existing_evidence() -> None:
    """Rejecting a duplicate must explicitly retain existing evidence."""

    proposal = standard_proposal()

    finding = CareerEvidenceOverlapFinding(
        proposal_id=proposal.proposal_id,
        scope=(EvidenceOverlapScope.APPROVED_EVIDENCE),
        matching_evidence_id="EVD-EXISTING",
        matched_claims=["CareerOps Agent Engine."],
    )

    result = review_service().review(
        user_id="USER-001",
        proposals=[proposal],
        overlap_findings=[finding],
        decision=EvidenceReviewDecision(
            rejected_proposal_ids=[proposal.proposal_id],
            duplicate_resolutions=[
                approved_resolution(
                    action=EvidenceDuplicateResolutionAction.KEEP_EXISTING
                )
            ],
        ),
    )

    assert result.approved_evidence == []

    assert result.rejected_proposal_ids == [proposal.proposal_id]


@pytest.mark.parametrize(
    ("action", "expected_title", "expected_claims"),
    [
        (
            EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
            "CareerOps Agent Engine",
            [
                "CareerOps Agent Engine.",
                "Built Python APIs using FastAPI and Docker.",
            ],
        ),
        (
            EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
            "Existing CareerOps evidence",
            [
                "Built an earlier CareerOps API.",
                "CareerOps Agent Engine.",
                "Built Python APIs using FastAPI and Docker.",
            ],
        ),
    ],
)
def test_overlap_can_replace_or_merge_existing_evidence(
    action: EvidenceDuplicateResolutionAction,
    expected_title: str,
    expected_claims: list[str],
) -> None:
    """Mutation actions preserve the existing ID and grounded provenance."""

    proposal = standard_proposal()
    finding = CareerEvidenceOverlapFinding(
        proposal_id=proposal.proposal_id,
        scope=EvidenceOverlapScope.APPROVED_EVIDENCE,
        matching_evidence_id="EVD-EXISTING",
        matched_claims=["CareerOps Agent Engine."],
    )

    result = review_service().review(
        user_id="USER-001",
        proposals=[proposal],
        overlap_findings=[finding],
        decision=EvidenceReviewDecision(
            approved_proposal_ids=[proposal.proposal_id],
            duplicate_resolutions=[approved_resolution(action=action)],
        ),
    )

    assert result.approved_evidence == []
    assert len(result.evidence_updates) == 1
    update = result.evidence_updates[0]
    assert update.before == existing_evidence()
    assert update.after.evidence_id == "EVD-EXISTING"
    assert update.after.title == expected_title
    assert update.after.approved_claims == expected_claims
    assert len(update.after.source_references) == 2


def test_resolution_cannot_target_another_users_evidence() -> None:
    """A persisted finding cannot bypass the authenticated user boundary."""

    proposal = standard_proposal()
    finding = CareerEvidenceOverlapFinding(
        proposal_id=proposal.proposal_id,
        scope=EvidenceOverlapScope.APPROVED_EVIDENCE,
        matching_evidence_id="EVD-EXISTING",
        matched_claims=["CareerOps Agent Engine."],
    )

    with pytest.raises(
        CVEvidenceReviewValidationError,
        match="unavailable for this user",
    ):
        review_service().review(
            user_id="USER-OTHER",
            proposals=[proposal],
            overlap_findings=[finding],
            decision=EvidenceReviewDecision(
                approved_proposal_ids=[proposal.proposal_id],
                duplicate_resolutions=[
                    approved_resolution(
                        action=EvidenceDuplicateResolutionAction.ACCEPT_SEPARATE
                    )
                ],
            ),
        )


def test_approved_evidence_id_is_stable() -> None:
    """Retrying approval of one proposal retains its evidence ID."""

    first = build_approved_evidence_id("EVP-001")

    second = build_approved_evidence_id("EVP-001")

    assert first == second
    assert first.startswith("EVD-")

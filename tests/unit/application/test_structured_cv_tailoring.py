"""Tests for deterministic application of approved CV proposals."""

import pytest

from careerops_agent_engine.application.exceptions import (
    StructuredCVAssemblyError,
)
from careerops_agent_engine.application.services.structured_cv_tailoring import (
    StructuredCVProposalApplier,
)
from careerops_agent_engine.domain.enums import (
    CVChangeApplicationMode,
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
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)

SOURCE_TEXT = (
    "CareerOps\n"
    "Built CareerOps using Python and FastAPI.\n"
    "Another project remained unchanged."
)


def build_cv(
    *,
    projects_text: str = SOURCE_TEXT,
) -> StructuredCV:
    """Create one preserved base CV."""

    return StructuredCV(
        cv_id="CV-001",
        source_document_id="DOC-001",
        preamble_text="Example Candidate",
        sections=[
            StructuredCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                free_text=projects_text,
            )
        ],
    )


def build_evidence(
    *,
    evidence_id: str = "EVD-001",
    source_excerpt: str = ("Built CareerOps using Python and FastAPI."),
    source_id: str = "DOC-001",
    verification_status: VerificationStatus = (VerificationStatus.APPROVED),
) -> CareerEvidence:
    """Create approved uploaded-CV evidence."""

    return CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title="CareerOps",
        verification_status=verification_status,
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=[
            "Application development",
        ],
        approved_claims=[
            source_excerpt,
        ],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id=source_id,
                source_excerpt=source_excerpt,
            )
        ],
    )


def build_proposal(
    *,
    proposal_id: str = "CVP-001",
    proposed_text: str = ("Built CareerOps using Python, FastAPI and PostgreSQL."),
    current_text: str | None = None,
    evidence_id: str = "EVD-001",
) -> CVChangeProposal:
    """Create one human-approved final proposal."""

    return CVChangeProposal(
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        current_text=current_text,
        proposed_text=proposed_text,
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            evidence_id,
        ],
        confidence_score=1.0,
    )


def test_applier_replaces_only_grounded_source_span() -> None:
    """Untouched source content must remain unchanged."""

    result = StructuredCVProposalApplier().apply(
        base_cv=build_cv(),
        proposals=[build_proposal()],
        approved_evidence=[build_evidence()],
    )

    projects = result.structured_cv.sections[0]

    assert projects.free_text == (
        "CareerOps\n"
        "Built CareerOps using Python, FastAPI "
        "and PostgreSQL.\n"
        "Another project remained unchanged."
    )

    assert len(result.applied_changes) == 1

    change = result.applied_changes[0]

    assert change.application_mode is CVChangeApplicationMode.ANCHORED_REPLACEMENT

    assert change.proposal_id == "CVP-001"

    assert change.anchor_evidence_ids == ["EVD-001"]

    assert change.original_text == ("Built CareerOps using Python and FastAPI.")


def test_applier_matches_anchor_across_whitespace() -> None:
    """Line wrapping must not destroy a valid source anchor."""

    cv = build_cv(
        projects_text=(
            "CareerOps\n"
            "Built CareerOps using\n"
            "Python and FastAPI.\n"
            "Another project remained unchanged."
        )
    )

    result = StructuredCVProposalApplier().apply(
        base_cv=cv,
        proposals=[build_proposal()],
        approved_evidence=[build_evidence()],
    )

    assert result.structured_cv.sections[0].free_text == (
        "CareerOps\n"
        "Built CareerOps using Python, FastAPI "
        "and PostgreSQL.\n"
        "Another project remained unchanged."
    )


def test_current_text_is_preferred_as_explicit_anchor() -> None:
    """Proposal current text may identify the exact replacement span."""

    result = StructuredCVProposalApplier().apply(
        base_cv=build_cv(),
        proposals=[
            build_proposal(
                current_text=("Another project remained unchanged."),
                proposed_text=("Another project used tested Python APIs."),
            )
        ],
        approved_evidence=[build_evidence()],
    )

    assert result.applied_changes[0].anchor_evidence_ids == []

    assert result.structured_cv.sections[0].free_text == (
        "CareerOps\n"
        "Built CareerOps using Python and FastAPI.\n"
        "Another project used tested Python APIs."
    )


def test_applier_rejects_ambiguous_source_anchor() -> None:
    """Repeated source text must not be replaced by guesswork."""

    repeated = (
        "Built CareerOps using Python and FastAPI.\n"
        "Built CareerOps using Python and FastAPI."
    )

    with pytest.raises(
        StructuredCVAssemblyError,
        match="match exactly once",
    ):
        StructuredCVProposalApplier().apply(
            base_cv=build_cv(projects_text=repeated),
            proposals=[build_proposal()],
            approved_evidence=[build_evidence()],
        )


def test_applier_rejects_unapproved_evidence() -> None:
    """Pending or rejected evidence cannot alter final CV text."""

    with pytest.raises(
        StructuredCVAssemblyError,
        match="only approved evidence",
    ):
        StructuredCVProposalApplier().apply(
            base_cv=build_cv(),
            proposals=[build_proposal()],
            approved_evidence=[
                build_evidence(verification_status=(VerificationStatus.PENDING))
            ],
        )


def test_applier_rejects_evidence_from_other_document() -> None:
    """Evidence from another source CV cannot anchor this document."""

    with pytest.raises(
        StructuredCVAssemblyError,
        match="exactly one grounded source anchor",
    ):
        StructuredCVProposalApplier().apply(
            base_cv=build_cv(),
            proposals=[build_proposal()],
            approved_evidence=[build_evidence(source_id="DOC-OTHER")],
        )


def test_applier_rejects_overlapping_changes() -> None:
    """Two proposals cannot mutate overlapping source content."""

    broad_evidence = build_evidence(
        evidence_id="EVD-BROAD",
        source_excerpt=("CareerOps\nBuilt CareerOps using Python and FastAPI."),
    )

    narrow_evidence = build_evidence(
        evidence_id="EVD-NARROW",
        source_excerpt=("Built CareerOps using Python and FastAPI."),
    )

    broad_proposal = build_proposal(
        proposal_id="CVP-BROAD",
        proposed_text="CareerOps platform project.",
        evidence_id="EVD-BROAD",
    )

    narrow_proposal = build_proposal(
        proposal_id="CVP-NARROW",
        proposed_text=("Built CareerOps using Python and FastAPI."),
        evidence_id="EVD-NARROW",
    )

    with pytest.raises(
        StructuredCVAssemblyError,
        match="overlapping source spans",
    ):
        StructuredCVProposalApplier().apply(
            base_cv=build_cv(),
            proposals=[
                broad_proposal,
                narrow_proposal,
            ],
            approved_evidence=[
                broad_evidence,
                narrow_evidence,
            ],
        )

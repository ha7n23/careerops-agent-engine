"""Tests for immutable CV versions and rendered artifacts."""

import pytest
from pydantic import ValidationError

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVChangeApplicationMode,
    CVSection,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    CVVersionProvenance,
    LLMProvenanceReference,
    RenderedCVArtifact,
)
from careerops_agent_engine.domain.models.structured_cv import (
    CVBullet,
    CVEntry,
    StructuredCV,
    StructuredCVSection,
)


def build_structured_cv(
    *,
    source_document_id: str = "DOC-001",
    evidence_id: str = "EVD-001",
    requirement_id: str = "REQ-001",
) -> StructuredCV:
    """Create one small grounded structured CV."""

    return StructuredCV(
        cv_id="CV-001",
        source_document_id=source_document_id,
        sections=[
            StructuredCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                entries=[
                    CVEntry(
                        entry_id="ENT-001",
                        section=CVSection.PROJECTS,
                        title="CareerOps",
                        bullets=[
                            CVBullet(
                                bullet_id="BLT-001",
                                text=("Built a Python API using FastAPI."),
                                supporting_evidence_ids=[
                                    evidence_id,
                                ],
                                requirement_ids=[
                                    requirement_id,
                                ],
                            )
                        ],
                    )
                ],
            )
        ],
    )


def build_applied_change(
    *,
    proposal_id: str = "CVP-001",
    evidence_id: str = "EVD-001",
    requirement_id: str = "REQ-001",
) -> AppliedCVChange:
    """Create one deterministic applied tailoring change."""

    return AppliedCVChange(
        change_id="CHG-001",
        proposal_id=proposal_id,
        section=CVSection.PROJECTS,
        application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
        source_anchor=("Built a Python API using FastAPI."),
        original_text=("Built a Python API using FastAPI."),
        applied_text=("Built a production-style Python API using FastAPI."),
        anchor_evidence_ids=[
            evidence_id,
        ],
        requirement_ids=[
            requirement_id,
        ],
        supporting_evidence_ids=[
            evidence_id,
        ],
    )


def build_provenance(
    *,
    source_document_id: str = "DOC-001",
    evidence_ids: list[str] | None = None,
    requirement_ids: list[str] | None = None,
    review_status: ApprovalStatus = (ApprovalStatus.APPROVED),
) -> CVVersionProvenance:
    """Create trusted CV-version provenance."""

    return CVVersionProvenance(
        job_id="JOB-001",
        thread_id="THR-001",
        source_document_id=source_document_id,
        requirement_ids=(
            requirement_ids if requirement_ids is not None else ["REQ-001"]
        ),
        supporting_evidence_ids=(
            evidence_ids if evidence_ids is not None else ["EVD-001"]
        ),
        final_proposal_ids=[
            "CVP-001",
        ],
        review_status=review_status,
        template_id="careerops-standard",
        template_version="1.0.0",
        workflow_version="1.0.0",
        llm_references=[
            LLMProvenanceReference(
                component="cv_proposal_generation",
                provider="google",
                model="gemini-3.5-flash",
                prompt_version="cv-proposal-v1",
            )
        ],
    )


def build_artifact(
    *,
    artifact_id: str,
    artifact_format: CVArtifactFormat,
    verification_status: (
        CVArtifactVerificationStatus
    ) = CVArtifactVerificationStatus.VERIFIED,
) -> RenderedCVArtifact:
    """Create one stored rendered artifact."""

    return RenderedCVArtifact(
        artifact_id=artifact_id,
        artifact_format=artifact_format,
        storage_key=(f"cv-versions/{artifact_id}.{artifact_format.value}"),
        sha256_hex="a" * 64,
        size_bytes=1_024,
        verification_status=verification_status,
    )


def test_assembled_cv_version_accepts_grounded_provenance() -> None:
    """Structured content may exist before any rendering occurs."""

    version = CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=build_structured_cv(),
        provenance=build_provenance(),
        applied_changes=[build_applied_change()],
    )

    assert version.artifacts == []
    assert version.version_number == 1


def test_version_requires_accepted_human_review() -> None:
    """Unapproved proposals cannot become a final CV version."""

    with pytest.raises(
        ValidationError,
        match="approved or edited",
    ):
        build_provenance(review_status=ApprovalStatus.PENDING)


def test_version_rejects_source_document_mismatch() -> None:
    """Structured content must retain its original document lineage."""

    with pytest.raises(
        ValidationError,
        match="same source document",
    ):
        CVVersion(
            cv_version_id="CVV-001",
            version_number=1,
            status=CVVersionStatus.ASSEMBLED,
            structured_cv=build_structured_cv(source_document_id="DOC-001"),
            provenance=build_provenance(source_document_id="DOC-OTHER"),
            applied_changes=[build_applied_change()],
        )


def test_version_rejects_undeclared_evidence_reference() -> None:
    """Every bullet evidence ID must appear in version provenance."""

    with pytest.raises(
        ValidationError,
        match="evidence reference",
    ):
        CVVersion(
            cv_version_id="CVV-001",
            version_number=1,
            status=CVVersionStatus.ASSEMBLED,
            structured_cv=build_structured_cv(evidence_id="EVD-MISSING"),
            provenance=build_provenance(),
            applied_changes=[build_applied_change(evidence_id="EVD-MISSING")],
        )


def test_version_rejects_undeclared_requirement_reference() -> None:
    """Every bullet requirement ID must appear in version provenance."""

    with pytest.raises(
        ValidationError,
        match="requirement reference",
    ):
        CVVersion(
            cv_version_id="CVV-001",
            version_number=1,
            status=CVVersionStatus.ASSEMBLED,
            structured_cv=build_structured_cv(requirement_id="REQ-MISSING"),
            provenance=build_provenance(),
            applied_changes=[build_applied_change(requirement_id="REQ-MISSING")],
        )


def test_later_version_requires_parent() -> None:
    """Version lineage must be explicit after version one."""

    with pytest.raises(
        ValidationError,
        match="require a parent",
    ):
        CVVersion(
            cv_version_id="CVV-002",
            version_number=2,
            status=CVVersionStatus.ASSEMBLED,
            structured_cv=build_structured_cv(),
            provenance=build_provenance(),
            applied_changes=[build_applied_change()],
        )


def test_rendered_version_requires_artifact() -> None:
    """Rendered lifecycle state cannot exist without a file."""

    with pytest.raises(
        ValidationError,
        match="at least one rendered artifact",
    ):
        CVVersion(
            cv_version_id="CVV-001",
            version_number=1,
            status=CVVersionStatus.RENDERED,
            structured_cv=build_structured_cv(),
            provenance=build_provenance(),
            applied_changes=[build_applied_change()],
        )


def test_verified_version_requires_verified_docx_and_pdf() -> None:
    """Final verified state requires both required output formats."""

    version = CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=CVVersionStatus.VERIFIED,
        structured_cv=build_structured_cv(),
        provenance=build_provenance(),
        artifacts=[
            build_artifact(
                artifact_id="ART-DOCX",
                artifact_format=(CVArtifactFormat.DOCX),
            ),
            build_artifact(
                artifact_id="ART-PDF",
                artifact_format=(CVArtifactFormat.PDF),
            ),
        ],
        applied_changes=[build_applied_change()],
    )

    assert version.status is CVVersionStatus.VERIFIED

    assert len(version.artifacts) == 2


def test_failed_artifact_requires_verification_notes() -> None:
    """A failed rendered file must explain its failed check."""

    with pytest.raises(
        ValidationError,
        match="verification notes",
    ):
        build_artifact(
            artifact_id="ART-FAILED",
            artifact_format=CVArtifactFormat.PDF,
            verification_status=(CVArtifactVerificationStatus.FAILED),
        )


def test_version_requires_exact_final_proposal_coverage() -> None:
    """Version provenance cannot claim an unapplied final proposal."""

    provenance = build_provenance().model_copy(
        update={"final_proposal_ids": ["CVP-OTHER"]}
    )

    with pytest.raises(
        ValidationError,
        match="correspond exactly",
    ):
        CVVersion(
            cv_version_id="CVV-001",
            version_number=1,
            status=CVVersionStatus.ASSEMBLED,
            structured_cv=build_structured_cv(),
            applied_changes=[build_applied_change()],
            provenance=provenance,
        )

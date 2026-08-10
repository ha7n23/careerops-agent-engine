"""Domain models for versioned and rendered CV artifacts."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.base import (
    DomainModel,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
)


class LLMProvenanceReference(DomainModel):
    """Version information for one LLM-backed workflow component."""

    component: str = Field(
        min_length=1,
        max_length=120,
    )

    provider: str = Field(
        min_length=1,
        max_length=120,
    )

    model: str = Field(
        min_length=1,
        max_length=250,
    )

    prompt_version: str = Field(
        min_length=1,
        max_length=120,
    )


class CVVersionProvenance(DomainModel):
    """Traceability information for one generated CV version."""

    job_id: str = Field(
        min_length=1,
        max_length=64,
    )

    thread_id: str = Field(
        min_length=1,
        max_length=64,
    )

    source_document_id: str = Field(
        min_length=1,
        max_length=64,
    )

    requirement_ids: list[str] = Field(
        min_length=1,
    )

    supporting_evidence_ids: list[str] = Field(
        min_length=1,
    )

    final_proposal_ids: list[str] = Field(
        min_length=1,
    )

    review_status: ApprovalStatus

    template_id: str = Field(
        min_length=1,
        max_length=120,
    )

    template_version: str = Field(
        min_length=1,
        max_length=120,
    )

    workflow_version: str = Field(
        min_length=1,
        max_length=120,
    )

    llm_references: list[LLMProvenanceReference] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_provenance(
        self,
    ) -> Self:
        """Require accepted review state and unique references."""

        if self.review_status not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.EDITED,
        }:
            raise ValueError(
                "A CV version requires an approved or edited human-review outcome."
            )

        identifier_groups = (
            (
                "Requirement identifiers",
                self.requirement_ids,
            ),
            (
                "Supporting evidence identifiers",
                self.supporting_evidence_ids,
            ),
            (
                "Final proposal identifiers",
                self.final_proposal_ids,
            ),
        )

        for label, identifiers in identifier_groups:
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{label} must be unique.")

        return self


class RenderedCVArtifact(DomainModel):
    """Metadata for one stored rendered CV file."""

    artifact_id: str = Field(
        min_length=1,
        max_length=64,
    )

    artifact_format: CVArtifactFormat

    storage_key: str = Field(
        min_length=1,
        max_length=512,
    )

    sha256_hex: str = Field(
        pattern=r"^[0-9a-fA-F]{64}$",
    )

    size_bytes: int = Field(
        gt=0,
    )

    verification_status: CVArtifactVerificationStatus = (
        CVArtifactVerificationStatus.PENDING
    )

    verification_notes: list[str] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_verification(
        self,
    ) -> Self:
        """Require an explanation when verification fails."""

        if (
            self.verification_status is CVArtifactVerificationStatus.FAILED
            and not self.verification_notes
        ):
            raise ValueError(
                "A failed artifact verification requires verification notes."
            )

        return self


class CVVersion(DomainModel):
    """Immutable structured CV version with provenance and artifacts."""

    cv_version_id: str = Field(
        min_length=1,
        max_length=64,
    )

    version_number: int = Field(
        ge=1,
    )

    parent_version_id: str | None = Field(
        default=None,
        max_length=64,
    )

    status: CVVersionStatus

    structured_cv: StructuredCV

    applied_changes: list[AppliedCVChange] = Field(
        min_length=1,
    )

    provenance: CVVersionProvenance

    artifacts: list[RenderedCVArtifact] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_version(
        self,
    ) -> Self:
        """Enforce version lineage, provenance and artifact lifecycle."""

        self._validate_lineage()
        self._validate_content_provenance()
        self._validate_artifacts()

        return self

    def _validate_lineage(
        self,
    ) -> None:
        """Require unambiguous version ancestry."""

        if self.parent_version_id == self.cv_version_id:
            raise ValueError("A CV version cannot be its own parent.")

        if self.version_number == 1 and self.parent_version_id is not None:
            raise ValueError("The first CV version cannot have a parent.")

        if self.version_number > 1 and self.parent_version_id is None:
            raise ValueError("Later CV versions require a parent version.")

    def _validate_content_provenance(
        self,
    ) -> None:
        """Ensure final content and applied changes are fully declared."""

        if self.structured_cv.source_document_id != self.provenance.source_document_id:
            raise ValueError(
                "Structured CV and version provenance "
                "must reference the same source document."
            )

        declared_evidence = set(self.provenance.supporting_evidence_ids)

        declared_requirements = set(self.provenance.requirement_ids)

        declared_proposals = set(self.provenance.final_proposal_ids)

        bullet_evidence = {
            evidence_id
            for section in self.structured_cv.sections
            for entry in section.entries
            for bullet in entry.bullets
            for evidence_id in bullet.supporting_evidence_ids
        }

        bullet_requirements = {
            requirement_id
            for section in self.structured_cv.sections
            for entry in section.entries
            for bullet in entry.bullets
            for requirement_id in bullet.requirement_ids
        }

        change_evidence = {
            evidence_id
            for change in self.applied_changes
            for evidence_id in change.supporting_evidence_ids
        }

        change_requirements = {
            requirement_id
            for change in self.applied_changes
            for requirement_id in change.requirement_ids
        }

        change_proposal_ids = [change.proposal_id for change in self.applied_changes]

        if len(change_proposal_ids) != len(set(change_proposal_ids)):
            raise ValueError("Each final CV proposal may be applied only once.")

        if set(change_proposal_ids) != declared_proposals:
            raise ValueError(
                "Applied CV changes must correspond exactly "
                "to final proposal provenance."
            )

        if not (bullet_evidence | change_evidence) <= declared_evidence:
            raise ValueError(
                "Every CV evidence reference must exist in version provenance."
            )

        if not (bullet_requirements | change_requirements) <= declared_requirements:
            raise ValueError(
                "Every CV requirement reference must exist in version provenance."
            )

        available_sections = {
            section.section for section in self.structured_cv.sections
        }

        if any(
            change.section not in available_sections for change in self.applied_changes
        ):
            raise ValueError(
                "Every applied CV change must target a "
                "section present in the structured CV."
            )

    def _validate_artifacts(
        self,
    ) -> None:
        """Enforce unique artifacts and lifecycle consistency."""

        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]

        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("Rendered artifact identifiers must be unique.")

        artifact_formats = [artifact.artifact_format for artifact in self.artifacts]

        if len(artifact_formats) != len(set(artifact_formats)):
            raise ValueError(
                "A CV version may contain only one artifact of each format."
            )

        if self.status is CVVersionStatus.ASSEMBLED and self.artifacts:
            raise ValueError(
                "An assembled CV version cannot already contain rendered artifacts."
            )

        if self.status is CVVersionStatus.RENDERED and not self.artifacts:
            raise ValueError(
                "A rendered CV version requires at least one rendered artifact."
            )

        if self.status is CVVersionStatus.VERIFIED:
            required_formats = {
                CVArtifactFormat.DOCX,
                CVArtifactFormat.PDF,
            }

            if set(artifact_formats) != required_formats:
                raise ValueError(
                    "A verified CV version requires both DOCX and PDF artifacts."
                )

            if any(
                artifact.verification_status
                is not CVArtifactVerificationStatus.VERIFIED
                for artifact in self.artifacts
            ):
                raise ValueError(
                    "Every artifact in a verified CV version must be verified."
                )

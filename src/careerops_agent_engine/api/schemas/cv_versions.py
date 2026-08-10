"""Safe API schemas for generated CV versions."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from careerops_agent_engine.application.services.final_cv_generation import (
    FinalCVGenerationExecutionResult,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    RenderedCVArtifact,
)


class FinalCVGenerationRequest(BaseModel):
    """Request trusted final-CV generation from an accepted workflow."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(
        min_length=1,
        max_length=64,
    )

    source_document_id: str = Field(
        min_length=1,
        max_length=64,
    )


class CVArtifactResponse(BaseModel):
    """Safe metadata for one generated CV file."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    artifact_format: CVArtifactFormat
    size_bytes: int
    sha256_hex: str
    verification_status: CVArtifactVerificationStatus

    @classmethod
    def from_domain(
        cls,
        artifact: RenderedCVArtifact,
    ) -> "CVArtifactResponse":
        """Build public metadata without exposing storage keys."""

        return cls(
            artifact_id=artifact.artifact_id,
            artifact_format=(artifact.artifact_format),
            size_bytes=artifact.size_bytes,
            sha256_hex=artifact.sha256_hex,
            verification_status=(artifact.verification_status),
        )


class CVVersionResponse(BaseModel):
    """Safe public metadata for one generated CV version."""

    model_config = ConfigDict(extra="forbid")

    cv_version_id: str
    cv_id: str

    version_number: int
    parent_version_id: str | None

    status: CVVersionStatus

    source_document_id: str
    job_id: str
    thread_id: str

    review_status: ApprovalStatus

    template_id: str
    template_version: str
    workflow_version: str

    artifacts: list[CVArtifactResponse]

    @classmethod
    def from_domain(
        cls,
        version: CVVersion,
    ) -> "CVVersionResponse":
        """Build safe public metadata from one domain version."""

        return cls(
            cv_version_id=(version.cv_version_id),
            cv_id=(version.structured_cv.cv_id),
            version_number=(version.version_number),
            parent_version_id=(version.parent_version_id),
            status=version.status,
            source_document_id=(version.provenance.source_document_id),
            job_id=(version.provenance.job_id),
            thread_id=(version.provenance.thread_id),
            review_status=(version.provenance.review_status),
            template_id=(version.provenance.template_id),
            template_version=(version.provenance.template_version),
            workflow_version=(version.provenance.workflow_version),
            artifacts=[
                CVArtifactResponse.from_domain(artifact)
                for artifact in version.artifacts
            ],
        )


class FinalCVVersionResponse(BaseModel):
    """Safe public state of one generated CV version."""

    model_config = ConfigDict(extra="forbid")

    cv_version_id: str
    cv_id: str

    version_number: int
    parent_version_id: str | None

    status: CVVersionStatus

    source_document_id: str
    job_id: str
    thread_id: str

    review_status: ApprovalStatus

    template_id: str
    template_version: str
    workflow_version: str

    artifacts: list[CVArtifactResponse]

    reused_existing_version: bool

    @classmethod
    def from_execution(
        cls,
        execution: FinalCVGenerationExecutionResult,
    ) -> "FinalCVVersionResponse":
        """Build a safe response from authoritative generation state."""

        version = execution.version

        return cls(
            cv_version_id=(version.cv_version_id),
            cv_id=(version.structured_cv.cv_id),
            version_number=(version.version_number),
            parent_version_id=(version.parent_version_id),
            status=version.status,
            source_document_id=(version.provenance.source_document_id),
            job_id=(version.provenance.job_id),
            thread_id=(version.provenance.thread_id),
            review_status=(version.provenance.review_status),
            template_id=(version.provenance.template_id),
            template_version=(version.provenance.template_version),
            workflow_version=(version.provenance.workflow_version),
            artifacts=[
                CVArtifactResponse.from_domain(artifact)
                for artifact in version.artifacts
            ],
            reused_existing_version=(execution.reused_existing_version),
        )

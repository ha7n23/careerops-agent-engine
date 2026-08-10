"""Complete an approved CareerOps tailoring workflow into verified artifacts."""

from dataclasses import dataclass
from typing import Protocol

from careerops_agent_engine.application.exceptions import (
    FinalCVGenerationError,
)
from careerops_agent_engine.application.ports.cv_version_repository import (
    CVVersionRepository,
)
from careerops_agent_engine.application.services.cv_artifact_rendering import (
    CVArtifactRenderingResult,
)
from careerops_agent_engine.application.services.cv_artifact_verification import (
    CVArtifactVerificationExecutionResult,
)
from careerops_agent_engine.application.services.cv_pdf_conversion import (
    CVPDFConversionExecutionResult,
)
from careerops_agent_engine.application.services.cv_version_builder import (
    CVVersionBuilder,
)
from careerops_agent_engine.application.services.final_cv_assembly import (
    FinalCVAssemblyExecutionResult,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVVersionStatus,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    LLMProvenanceReference,
    RenderedCVArtifact,
)

FINAL_CV_WORKFLOW_VERSION = "final-cv-v1"


class FinalCVAssemblyRunner(Protocol):
    """Assemble trusted final CV content."""

    def assemble(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_document_id: str,
    ) -> FinalCVAssemblyExecutionResult:
        """Return deterministic tailored CV content."""

        ...


class CVArtifactRenderingRunner(Protocol):
    """Render and persist one generated artifact."""

    def render_and_store(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVArtifactRenderingResult:
        """Render and persist an artifact."""

        ...


class CVArtifactVerificationRunner(Protocol):
    """Verify one persisted artifact."""

    def verify(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVArtifactVerificationExecutionResult:
        """Verify and persist one artifact result."""

        ...


class CVPDFConversionRunner(Protocol):
    """Convert a verified DOCX into a stored PDF."""

    def convert_and_store(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVPDFConversionExecutionResult:
        """Convert and persist the PDF artifact."""

        ...


@dataclass(frozen=True)
class FinalCVGenerationExecutionResult:
    """Final authoritative CV-version generation result."""

    version: CVVersion
    reused_existing_version: bool


class FinalCVGenerationService:
    """Create or resume one verified final CV version."""

    def __init__(
        self,
        *,
        assembly_service: FinalCVAssemblyRunner,
        version_builder: CVVersionBuilder,
        version_repository: CVVersionRepository,
        docx_rendering_service: CVArtifactRenderingRunner,
        docx_verification_service: CVArtifactVerificationRunner,
        pdf_conversion_service: CVPDFConversionRunner,
        pdf_verification_service: CVArtifactVerificationRunner,
        template_id: str,
        template_version: str,
        workflow_version: str,
        llm_references: list[LLMProvenanceReference],
    ) -> None:
        """Store final CV workflow dependencies and provenance."""

        if not template_id.strip():
            raise ValueError("Final CV template identifier cannot be blank.")

        if not template_version.strip():
            raise ValueError("Final CV template version cannot be blank.")

        if not workflow_version.strip():
            raise ValueError("Final CV workflow version cannot be blank.")

        self._assembly_service = assembly_service
        self._version_builder = version_builder
        self._version_repository = version_repository

        self._docx_rendering_service = docx_rendering_service

        self._docx_verification_service = docx_verification_service

        self._pdf_conversion_service = pdf_conversion_service

        self._pdf_verification_service = pdf_verification_service

        self._template_id = template_id
        self._template_version = template_version
        self._workflow_version = workflow_version

        self._llm_references = tuple(llm_references)

    def generate(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_document_id: str,
    ) -> FinalCVGenerationExecutionResult:
        """Create or resume one complete verified CV version."""

        assembly = self._assembly_service.assemble(
            user_id=user_id,
            thread_id=thread_id,
            source_document_id=source_document_id,
        )

        family_versions = self._version_repository.list_for_cv(
            user_id=user_id,
            cv_id=(assembly.tailoring_result.structured_cv.cv_id),
        )

        existing = find_existing_generation(
            versions=family_versions,
            thread_id=thread_id,
            source_document_id=source_document_id,
        )

        reused_existing_version = existing is not None

        if existing is None:
            version = self._create_version(
                user_id=user_id,
                assembly=assembly,
                family_versions=family_versions,
            )

        else:
            validate_existing_generation(
                version=existing,
                assembly=assembly,
            )

            version = existing

        if version.status is CVVersionStatus.VERIFIED:
            return FinalCVGenerationExecutionResult(
                version=version,
                reused_existing_version=(reused_existing_version),
            )

        ensure_no_failed_artifacts(version)

        version = self._ensure_verified_docx(
            user_id=user_id,
            version=version,
        )

        version = self._ensure_verified_pdf(
            user_id=user_id,
            version=version,
        )

        final_version = self._version_repository.get(
            user_id=user_id,
            cv_version_id=(version.cv_version_id),
        )

        if final_version is None:
            raise RuntimeError("Final CV version became unavailable.")

        if final_version.status is not CVVersionStatus.VERIFIED:
            raise FinalCVGenerationError(
                "Final CV generation completed without a verified CV version."
            )

        return FinalCVGenerationExecutionResult(
            version=final_version,
            reused_existing_version=(reused_existing_version),
        )

    def _create_version(
        self,
        *,
        user_id: str,
        assembly: FinalCVAssemblyExecutionResult,
        family_versions: list[CVVersion],
    ) -> CVVersion:
        """Create the next immutable version in one CV family."""

        if family_versions:
            parent = family_versions[-1]

            version_number = parent.version_number + 1

            parent_version_id = parent.cv_version_id

        else:
            version_number = 1
            parent_version_id = None

        version = self._version_builder.build(
            assembly=assembly,
            version_number=version_number,
            parent_version_id=parent_version_id,
            template_id=self._template_id,
            template_version=(self._template_version),
            workflow_version=(self._workflow_version),
            llm_references=list(self._llm_references),
        )

        self._version_repository.save(
            user_id=user_id,
            version=version,
        )

        persisted = self._version_repository.get(
            user_id=user_id,
            cv_version_id=(version.cv_version_id),
        )

        if persisted is None:
            raise RuntimeError("Persisted CV version became unavailable.")

        return persisted

    def _ensure_verified_docx(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> CVVersion:
        """Render and verify DOCX only when required."""

        docx = find_version_artifact(
            version=version,
            artifact_format=(CVArtifactFormat.DOCX),
        )

        if docx is None:
            rendered = self._docx_rendering_service.render_and_store(
                user_id=user_id,
                cv_version_id=(version.cv_version_id),
            )

            if rendered.artifact.artifact_format is not CVArtifactFormat.DOCX:
                raise RuntimeError(
                    "DOCX rendering service returned the wrong artifact format."
                )

            version = rendered.version

            docx = rendered.artifact

        if docx.verification_status is CVArtifactVerificationStatus.FAILED:
            raise FinalCVGenerationError(
                "The persisted DOCX artifact failed verification."
            )

        if docx.verification_status is CVArtifactVerificationStatus.PENDING:
            verification = self._docx_verification_service.verify(
                user_id=user_id,
                cv_version_id=(version.cv_version_id),
            )

            if verification.artifact.artifact_format is not CVArtifactFormat.DOCX:
                raise RuntimeError(
                    "DOCX verification service returned the wrong artifact format."
                )

            if not verification.passed:
                raise FinalCVGenerationError(
                    "Generated DOCX failed deterministic verification."
                )

            version = verification.version

        return version

    def _ensure_verified_pdf(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> CVVersion:
        """Convert and verify PDF only when required."""

        pdf = find_version_artifact(
            version=version,
            artifact_format=(CVArtifactFormat.PDF),
        )

        if pdf is None:
            conversion = self._pdf_conversion_service.convert_and_store(
                user_id=user_id,
                cv_version_id=(version.cv_version_id),
            )

            if conversion.artifact.artifact_format is not CVArtifactFormat.PDF:
                raise RuntimeError(
                    "PDF conversion service returned the wrong artifact format."
                )

            version = conversion.version

            pdf = conversion.artifact

        if pdf.verification_status is CVArtifactVerificationStatus.FAILED:
            raise FinalCVGenerationError(
                "The persisted PDF artifact failed verification."
            )

        if pdf.verification_status is CVArtifactVerificationStatus.PENDING:
            verification = self._pdf_verification_service.verify(
                user_id=user_id,
                cv_version_id=(version.cv_version_id),
            )

            if verification.artifact.artifact_format is not CVArtifactFormat.PDF:
                raise RuntimeError(
                    "PDF verification service returned the wrong artifact format."
                )

            if not verification.passed:
                raise FinalCVGenerationError(
                    "Generated PDF failed deterministic verification."
                )

            version = verification.version

        return version


def find_existing_generation(
    *,
    versions: list[CVVersion],
    thread_id: str,
    source_document_id: str,
) -> CVVersion | None:
    """Find the one version belonging to this workflow/document pair."""

    matches = [
        version
        for version in versions
        if (
            version.provenance.thread_id == thread_id
            and version.provenance.source_document_id == source_document_id
        )
    ]

    if len(matches) > 1:
        raise FinalCVGenerationError(
            "Multiple CV versions exist for the same "
            "job-analysis run and source document."
        )

    if not matches:
        return None

    return matches[0]


def validate_existing_generation(
    *,
    version: CVVersion,
    assembly: FinalCVAssemblyExecutionResult,
) -> None:
    """Ensure a retry still refers to the same accepted business result."""

    expected_proposal_ids = [
        proposal.proposal_id for proposal in assembly.job_run.final_cv_proposals
    ]

    if (
        version.provenance.job_id != assembly.job_run.job_id
        or version.provenance.final_proposal_ids != expected_proposal_ids
    ):
        raise FinalCVGenerationError(
            "Existing CV version no longer matches the accepted job-analysis result."
        )


def ensure_no_failed_artifacts(
    version: CVVersion,
) -> None:
    """Prevent immutable failed artifacts from being silently replaced."""

    if any(
        artifact.verification_status is CVArtifactVerificationStatus.FAILED
        for artifact in version.artifacts
    ):
        raise FinalCVGenerationError(
            "The CV version contains a failed immutable artifact."
        )


def find_version_artifact(
    *,
    version: CVVersion,
    artifact_format: CVArtifactFormat,
) -> RenderedCVArtifact | None:
    """Return one artifact format from a CV version."""

    return next(
        (
            artifact
            for artifact in version.artifacts
            if artifact.artifact_format is artifact_format
        ),
        None,
    )

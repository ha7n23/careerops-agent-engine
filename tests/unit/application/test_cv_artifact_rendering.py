"""Tests for generated CV rendering and persistence orchestration."""

from hashlib import sha256
from pathlib import Path

import pytest

from careerops_agent_engine.application.exceptions import (
    CVRenderingError,
)
from careerops_agent_engine.application.ports.cv_renderer import (
    RenderedCVDocument,
)
from careerops_agent_engine.application.services.cv_artifact_rendering import (
    CVArtifactRenderingService,
    build_rendered_artifact_id,
)
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
    RenderedCVArtifact,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.storage.local_artifact_storage import (
    LocalArtifactStorage,
)


def build_version() -> CVVersion:
    """Create one valid assembled CV version."""

    return CVVersion(
        cv_version_id="CVV-001",
        version_number=1,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=StructuredCV(
            cv_id="CV-001",
            source_document_id="DOC-001",
            sections=[
                StructuredCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    free_text=("Built CareerOps using Python and FastAPI."),
                )
            ],
        ),
        applied_changes=[
            AppliedCVChange(
                change_id="CHG-001",
                proposal_id="CVP-001",
                section=CVSection.PROJECTS,
                application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
                source_anchor=("Built CareerOps using Python."),
                original_text=("Built CareerOps using Python."),
                applied_text=("Built CareerOps using Python and FastAPI."),
                anchor_evidence_ids=["EVD-001"],
                requirement_ids=["REQ-001"],
                supporting_evidence_ids=["EVD-001"],
            )
        ],
        provenance=CVVersionProvenance(
            job_id="JOB-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
            requirement_ids=["REQ-001"],
            supporting_evidence_ids=["EVD-001"],
            final_proposal_ids=["CVP-001"],
            review_status=(ApprovalStatus.APPROVED),
            template_id="careerops-standard",
            template_version="1.0.0",
            workflow_version="1.0.0",
        ),
        artifacts=[],
    )


class FakeRenderer:
    """Return deterministic DOCX-like bytes."""

    @property
    def template_id(self) -> str:
        """Return fake template ID."""

        return "careerops-standard"

    @property
    def template_version(self) -> str:
        """Return fake template version."""

        return "1.0.0"

    def render(
        self,
        *,
        version: CVVersion,
    ) -> RenderedCVDocument:
        """Return deterministic bytes for the requested version."""

        return RenderedCVDocument(
            artifact_format=(CVArtifactFormat.DOCX),
            media_type=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            filename=(f"{version.cv_version_id}.docx"),
            data=b"deterministic-docx-bytes",
        )


class FakeVersionRepository:
    """Small in-memory repository for rendering orchestration."""

    def __init__(
        self,
        version: CVVersion | None,
        *,
        fail_attach: bool = False,
    ) -> None:
        self.version = version
        self.fail_attach = fail_attach

    def get(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion | None:
        """Return one owned version."""

        if (
            user_id != "USER-001"
            or self.version is None
            or self.version.cv_version_id != cv_version_id
        ):
            return None

        return self.version

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Attach one artifact or simulate persistence failure."""

        if self.fail_attach:
            raise ValueError("Simulated metadata failure.")

        if (
            user_id != "USER-001"
            or self.version is None
            or self.version.cv_version_id != cv_version_id
        ):
            raise ValueError("CV version is unavailable.")

        existing = next(
            (
                item
                for item in self.version.artifacts
                if item.artifact_format is artifact.artifact_format
            ),
            None,
        )

        if existing is None:
            self.version = self.version.model_copy(
                update={
                    "status": (CVVersionStatus.RENDERED),
                    "artifacts": [
                        *self.version.artifacts,
                        artifact,
                    ],
                }
            )

        return self.version

    def save(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Persisting versions is outside this test fake's responsibility."""

        del user_id
        del version

        raise AssertionError("Rendering orchestration must not save CV versions.")

    def list_for_cv(
        self,
        *,
        user_id: str,
        cv_id: str,
    ) -> list[CVVersion]:
        """Listing version history is outside this test boundary."""

        del user_id
        del cv_id

        raise AssertionError("Rendering orchestration must not list CV versions.")

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Artifact verification belongs to the later verification workflow."""

        del user_id
        del cv_version_id
        del artifact_id
        del verification_status
        del verification_notes

        raise AssertionError("Rendering orchestration must not verify CV artifacts.")


def test_rendering_service_stores_and_attaches_artifact(
    tmp_path: Path,
) -> None:
    """Rendered bytes should become stored pending artifact metadata."""

    repository = FakeVersionRepository(build_version())

    storage = LocalArtifactStorage(tmp_path)

    service = CVArtifactRenderingService(
        version_repository=repository,
        artifact_storage=storage,
        renderer=FakeRenderer(),
    )

    result = service.render_and_store(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert result.version.status is CVVersionStatus.RENDERED

    assert result.artifact.artifact_format is CVArtifactFormat.DOCX

    assert result.artifact.sha256_hex == (
        sha256(b"deterministic-docx-bytes").hexdigest()
    )

    assert (
        storage.read(
            user_id="USER-001",
            storage_key=(result.artifact.storage_key),
        )
        == b"deterministic-docx-bytes"
    )


def test_exact_render_retry_reuses_same_artifact(
    tmp_path: Path,
) -> None:
    """Deterministic rendering should preserve artifact identity."""

    repository = FakeVersionRepository(build_version())

    service = CVArtifactRenderingService(
        version_repository=repository,
        artifact_storage=(LocalArtifactStorage(tmp_path)),
        renderer=FakeRenderer(),
    )

    first = service.render_and_store(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    second = service.render_and_store(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert first.artifact.artifact_id == second.artifact.artifact_id

    assert first.artifact.sha256_hex == second.artifact.sha256_hex


def test_metadata_failure_removes_newly_created_file(
    tmp_path: Path,
) -> None:
    """A failed DB attachment should not leave a new orphan file."""

    repository = FakeVersionRepository(
        build_version(),
        fail_attach=True,
    )

    storage = LocalArtifactStorage(tmp_path)

    service = CVArtifactRenderingService(
        version_repository=repository,
        artifact_storage=storage,
        renderer=FakeRenderer(),
    )

    digest = sha256(b"deterministic-docx-bytes").hexdigest()

    artifact_id = build_rendered_artifact_id(
        cv_version_id="CVV-001",
        artifact_format=(CVArtifactFormat.DOCX),
        sha256_hex=digest,
    )

    with pytest.raises(
        ValueError,
        match="Simulated metadata failure",
    ):
        service.render_and_store(
            user_id="USER-001",
            cv_version_id="CVV-001",
        )

    expected_key = (
        f"usr-{sha256(b'USER-001').hexdigest()[:32]}/CVV-001/{artifact_id}.docx"
    )

    with pytest.raises(
        FileNotFoundError,
        match="unavailable",
    ):
        storage.read(
            user_id="USER-001",
            storage_key=expected_key,
        )


def test_missing_version_fails_before_rendering(
    tmp_path: Path,
) -> None:
    """Unknown or cross-user version IDs must not produce files."""

    service = CVArtifactRenderingService(
        version_repository=(FakeVersionRepository(None)),
        artifact_storage=(LocalArtifactStorage(tmp_path)),
        renderer=FakeRenderer(),
    )

    with pytest.raises(
        CVRenderingError,
        match="unavailable",
    ):
        service.render_and_store(
            user_id="USER-001",
            cv_version_id="CVV-MISSING",
        )

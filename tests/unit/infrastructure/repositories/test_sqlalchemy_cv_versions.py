"""Tests for immutable SQLAlchemy CV-version persistence."""

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVChangeApplicationMode,
    CVSection,
    CVVersionStatus,
    JobAnalysisRunStatus,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.cv_application import (
    AppliedCVChange,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
    CVVersionProvenance,
    RenderedCVArtifact,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.structured_cv import (
    StructuredCV,
    StructuredCVSection,
)
from careerops_agent_engine.infrastructure.database.base import (
    Base,
)
from careerops_agent_engine.infrastructure.database.models.job_analysis import (
    JobAnalysisRunRecord,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_career_documents import (  # noqa: E501
    SqlAlchemyCareerDocumentRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_cv_versions import (  # noqa: E501
    SqlAlchemyCVVersionRepository,
)


def build_proposal() -> CVChangeProposal:
    """Create one accepted persisted final proposal."""

    return CVChangeProposal(
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        proposed_text=("Built CareerOps using Python and FastAPI."),
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            "EVD-001",
        ],
        confidence_score=1.0,
    )


def build_document() -> CareerDocument:
    """Create one persisted source CV."""

    return CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key=("documents/user-001/DOC-001.pdf"),
        status=CareerDocumentStatus.EXTRACTED,
    )


def build_version(
    *,
    cv_version_id: str = "CVV-001",
    cv_id: str = "CV-001",
    version_number: int = 1,
    parent_version_id: str | None = None,
    template_version: str = "1.0.0",
) -> CVVersion:
    """Create one assembled immutable CV version."""

    change = AppliedCVChange(
        change_id=(f"CHG-{version_number:03d}"),
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        application_mode=(CVChangeApplicationMode.ANCHORED_REPLACEMENT),
        source_anchor=("Built CareerOps using Python."),
        original_text=("Built CareerOps using Python."),
        applied_text=("Built CareerOps using Python and FastAPI."),
        anchor_evidence_ids=[
            "EVD-001",
        ],
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            "EVD-001",
        ],
    )

    structured_cv = StructuredCV(
        cv_id=cv_id,
        source_document_id="DOC-001",
        preamble_text="Example Candidate",
        sections=[
            StructuredCVSection(
                section=CVSection.PROJECTS,
                heading="Projects",
                free_text=("CareerOps\nBuilt CareerOps using Python and FastAPI."),
            )
        ],
    )

    provenance = CVVersionProvenance(
        job_id="JOB-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            "EVD-001",
        ],
        final_proposal_ids=[
            "CVP-001",
        ],
        review_status=(ApprovalStatus.APPROVED),
        template_id="careerops-standard",
        template_version=template_version,
        workflow_version="1.0.0",
    )

    return CVVersion(
        cv_version_id=cv_version_id,
        version_number=version_number,
        parent_version_id=parent_version_id,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=structured_cv,
        applied_changes=[change],
        provenance=provenance,
        artifacts=[],
    )


def build_artifact(
    *,
    artifact_format: CVArtifactFormat = (CVArtifactFormat.DOCX),
    artifact_id: str = "ART-DOCX",
    storage_key: str = ("cv-versions/CVV-001/cv.docx"),
    sha256_hex: str = "b" * 64,
) -> RenderedCVArtifact:
    """Create one newly rendered artifact."""

    return RenderedCVArtifact(
        artifact_id=artifact_id,
        artifact_format=artifact_format,
        storage_key=storage_key,
        sha256_hex=sha256_hex,
        size_bytes=1_024,
        verification_status=(CVArtifactVerificationStatus.PENDING),
        verification_notes=[],
    )


@pytest.fixture
def repository() -> Iterator[SqlAlchemyCVVersionRepository]:
    """Create a repository with trusted parent workflow rows."""

    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = create_session_factory(engine)

    document_repository = SqlAlchemyCareerDocumentRepository(session_factory)

    document_repository.save(
        user_id="USER-001",
        document=build_document(),
    )

    proposal = build_proposal()

    with session_factory.begin() as session:
        session.add(
            JobAnalysisRunRecord(
                thread_id="THR-001",
                user_id="USER-001",
                job_id="JOB-001",
                status=(JobAnalysisRunStatus.COMPLETED.value),
                role_title=("Junior AI Engineer"),
                fit_score=100.0,
                review_status=(ApprovalStatus.APPROVED.value),
                cv_proposals=[proposal.model_dump(mode="json")],
                claim_verification_reports=[],
                reviewable_proposal_ids=["CVP-001"],
                blocked_proposal_ids=[],
                final_cv_proposals=[proposal.model_dump(mode="json")],
            )
        )

    try:
        yield SqlAlchemyCVVersionRepository(session_factory)
    finally:
        engine.dispose()


def test_save_and_get_assembled_version(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Immutable structured content should round-trip exactly."""

    version = build_version()

    repository.save(
        user_id="USER-001",
        version=version,
    )

    assert (
        repository.get(
            user_id="USER-001",
            cv_version_id="CVV-001",
        )
        == version
    )


def test_exact_version_retry_is_idempotent(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Retrying identical content should not create another version."""

    version = build_version()

    repository.save(
        user_id="USER-001",
        version=version,
    )

    repository.save(
        user_id="USER-001",
        version=version,
    )

    assert repository.list_for_cv(
        user_id="USER-001",
        cv_id="CV-001",
    ) == [version]


def test_get_enforces_user_boundary(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Another user cannot retrieve persisted CV content."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    assert (
        repository.get(
            user_id="USER-OTHER",
            cv_version_id="CVV-001",
        )
        is None
    )


def test_same_version_id_cannot_change_immutable_content(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """One immutable version identity cannot later change provenance."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    changed = build_version(template_version="2.0.0")

    with pytest.raises(
        ValueError,
        match="different persisted content",
    ):
        repository.save(
            user_id="USER-001",
            version=changed,
        )


def test_family_version_number_cannot_have_two_identities(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """One CV family/version number maps to exactly one version ID."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    collision = build_version(cv_version_id="CVV-OTHER")

    with pytest.raises(
        ValueError,
        match="family version number",
    ):
        repository.save(
            user_id="USER-001",
            version=collision,
        )


def test_later_version_requires_persisted_preceding_parent(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Version lineage cannot point to an unavailable parent."""

    version_two = build_version(
        cv_version_id="CVV-002",
        version_number=2,
        parent_version_id="CVV-MISSING",
    )

    with pytest.raises(
        ValueError,
        match="preceding version",
    ):
        repository.save(
            user_id="USER-001",
            version=version_two,
        )


def test_versions_are_listed_in_version_order(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """A CV family should expose deterministic version history."""

    version_one = build_version()

    repository.save(
        user_id="USER-001",
        version=version_one,
    )

    version_two = build_version(
        cv_version_id="CVV-002",
        version_number=2,
        parent_version_id="CVV-001",
    )

    repository.save(
        user_id="USER-001",
        version=version_two,
    )

    assert repository.list_for_cv(
        user_id="USER-001",
        cv_id="CV-001",
    ) == [
        version_one,
        version_two,
    ]


def test_job_proposal_provenance_must_match_persisted_run(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Persistence must reject fabricated final-proposal provenance."""

    version = build_version()

    changed_provenance = version.provenance.model_copy(
        update={"final_proposal_ids": ["CVP-FABRICATED"]}
    )

    fabricated = CVVersion(
        cv_version_id="CVV-FABRICATED",
        version_number=1,
        parent_version_id=None,
        status=CVVersionStatus.ASSEMBLED,
        structured_cv=(version.structured_cv),
        applied_changes=[
            version.applied_changes[0].model_copy(
                update={"proposal_id": ("CVP-FABRICATED")}
            )
        ],
        provenance=changed_provenance,
        artifacts=[],
    )

    with pytest.raises(
        ValueError,
        match="proposal provenance",
    ):
        repository.save(
            user_id="USER-001",
            version=fabricated,
        )


def test_attach_artifact_moves_version_to_rendered(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """First rendered file should advance the lifecycle."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    artifact = build_artifact()

    result = repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=artifact,
    )

    assert result.status is CVVersionStatus.RENDERED

    assert result.artifacts == [artifact]


def test_artifact_attachment_retry_is_idempotent(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Retrying the same rendered file should not duplicate it."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    artifact = build_artifact()

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=artifact,
    )

    result = repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=artifact,
    )

    assert len(result.artifacts) == 1


def test_same_format_cannot_be_replaced(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """An immutable DOCX/PDF slot cannot silently change bytes."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=build_artifact(),
    )

    replacement = build_artifact(
        artifact_id="ART-DIFFERENT",
        storage_key=("cv-versions/CVV-001/other.docx"),
        sha256_hex="c" * 64,
    )

    with pytest.raises(
        ValueError,
        match="different artifact",
    ):
        repository.attach_artifact(
            user_id="USER-001",
            cv_version_id="CVV-001",
            artifact=replacement,
        )


def test_artifact_attachment_enforces_user_boundary(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Another user cannot attach files to a CV version."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    with pytest.raises(
        ValueError,
        match="unavailable",
    ):
        repository.attach_artifact(
            user_id="USER-OTHER",
            cv_version_id="CVV-001",
            artifact=build_artifact(),
        )


def test_verification_updates_pending_artifact(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """A pending artifact may become verified exactly once."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=build_artifact(),
    )

    result = repository.set_artifact_verification(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-DOCX",
        verification_status=(CVArtifactVerificationStatus.VERIFIED),
        verification_notes=[],
    )

    assert result.status is CVVersionStatus.RENDERED

    assert (
        result.artifacts[0].verification_status is CVArtifactVerificationStatus.VERIFIED
    )


def test_failed_verification_is_terminal(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """A failed immutable artifact cannot later be relabelled verified."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=build_artifact(),
    )

    repository.set_artifact_verification(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-DOCX",
        verification_status=(CVArtifactVerificationStatus.FAILED),
        verification_notes=["DOCX content verification failed."],
    )

    with pytest.raises(
        ValueError,
        match="terminal",
    ):
        repository.set_artifact_verification(
            user_id="USER-001",
            cv_version_id="CVV-001",
            artifact_id="ART-DOCX",
            verification_status=(CVArtifactVerificationStatus.VERIFIED),
            verification_notes=[],
        )


def test_both_verified_artifacts_promote_version_to_verified(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """A CV version becomes verified only after DOCX and PDF pass."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    docx = build_artifact()

    pdf = build_artifact(
        artifact_format=CVArtifactFormat.PDF,
        artifact_id="ART-PDF",
        storage_key=("cv-versions/CVV-001/cv.pdf"),
        sha256_hex="c" * 64,
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=docx,
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=pdf,
    )

    repository.set_artifact_verification(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-DOCX",
        verification_status=(CVArtifactVerificationStatus.VERIFIED),
        verification_notes=[],
    )

    result = repository.set_artifact_verification(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-PDF",
        verification_status=(CVArtifactVerificationStatus.VERIFIED),
        verification_notes=[],
    )

    assert result.status is CVVersionStatus.VERIFIED

    assert {artifact.artifact_format for artifact in result.artifacts} == {
        CVArtifactFormat.DOCX,
        CVArtifactFormat.PDF,
    }

    assert all(
        artifact.verification_status is CVArtifactVerificationStatus.VERIFIED
        for artifact in result.artifacts
    )


def test_verification_retry_is_idempotent(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """An identical terminal verification retry should succeed."""

    repository.save(
        user_id="USER-001",
        version=build_version(),
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=build_artifact(),
    )

    repository.set_artifact_verification(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-DOCX",
        verification_status=(CVArtifactVerificationStatus.VERIFIED),
        verification_notes=[],
    )

    result = repository.set_artifact_verification(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact_id="ART-DOCX",
        verification_status=(CVArtifactVerificationStatus.VERIFIED),
        verification_notes=[],
    )

    assert (
        result.artifacts[0].verification_status is CVArtifactVerificationStatus.VERIFIED
    )


def test_original_version_save_retry_survives_lifecycle_progress(
    repository: SqlAlchemyCVVersionRepository,
) -> None:
    """Rendering must not make an immutable version-save retry fail."""

    assembled = build_version()

    repository.save(
        user_id="USER-001",
        version=assembled,
    )

    repository.attach_artifact(
        user_id="USER-001",
        cv_version_id="CVV-001",
        artifact=build_artifact(),
    )

    repository.save(
        user_id="USER-001",
        version=assembled,
    )

    persisted = repository.get(
        user_id="USER-001",
        cv_version_id="CVV-001",
    )

    assert persisted is not None

    assert persisted.status is CVVersionStatus.RENDERED

    assert len(persisted.artifacts) == 1

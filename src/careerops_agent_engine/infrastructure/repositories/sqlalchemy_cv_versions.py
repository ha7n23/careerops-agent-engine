"""SQLAlchemy persistence for immutable versioned CV content."""

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.application.ports.cv_version_repository import (
    CVVersionRepository,
)
from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
    CVArtifactVerificationStatus,
    CVVersionStatus,
    JobAnalysisRunStatus,
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
)
from careerops_agent_engine.infrastructure.database.models.cv_evidence import (
    CareerDocumentRecord,
)
from careerops_agent_engine.infrastructure.database.models.cv_version import (
    CVArtifactRecord,
    CVVersionRecord,
)
from careerops_agent_engine.infrastructure.database.models.job_analysis import (
    JobAnalysisRunRecord,
)


class SqlAlchemyCVVersionRepository(CVVersionRepository):
    """Persist CV versions while protecting immutable provenance."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
    ) -> None:
        """Store the configured SQLAlchemy session factory."""

        self._session_factory = session_factory

    def save(
        self,
        *,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Persist an assembled CV version idempotently."""

        if version.status is not CVVersionStatus.ASSEMBLED or version.artifacts:
            raise ValueError(
                "Initial CV-version persistence requires "
                "an assembled version without artifacts."
            )

        with self._session_factory.begin() as session:
            existing = session.get(
                CVVersionRecord,
                version.cv_version_id,
            )

            if existing is not None:
                self._validate_existing_retry(
                    session=session,
                    user_id=user_id,
                    version=version,
                    existing=existing,
                )
                return

            self._validate_source_document(
                session=session,
                user_id=user_id,
                version=version,
            )

            self._validate_job_run(
                session=session,
                user_id=user_id,
                version=version,
            )

            self._validate_family_version_identity(
                session=session,
                user_id=user_id,
                version=version,
            )

            self._validate_parent_version(
                session=session,
                user_id=user_id,
                version=version,
            )

            session.add(
                CVVersionRecord(
                    cv_version_id=(version.cv_version_id),
                    user_id=user_id,
                    cv_id=(version.structured_cv.cv_id),
                    version_number=(version.version_number),
                    parent_version_id=(version.parent_version_id),
                    source_document_id=(version.provenance.source_document_id),
                    job_id=(version.provenance.job_id),
                    thread_id=(version.provenance.thread_id),
                    status=version.status.value,
                    structured_cv=(version.structured_cv.model_dump(mode="json")),
                    applied_changes=[
                        change.model_dump(mode="json")
                        for change in version.applied_changes
                    ],
                    provenance=(version.provenance.model_dump(mode="json")),
                )
            )

    def get(
        self,
        *,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersion | None:
        """Retrieve one version inside its authenticated boundary."""

        statement = select(CVVersionRecord).where(
            CVVersionRecord.cv_version_id == cv_version_id,
            CVVersionRecord.user_id == user_id,
        )

        with self._session_factory() as session:
            record = session.execute(statement).scalar_one_or_none()

            if record is None:
                return None

            artifact_records = self._load_artifacts(
                session=session,
                cv_version_id=(record.cv_version_id),
            )

            return cv_version_record_to_domain(
                record=record,
                artifact_records=artifact_records,
            )

    def list_for_cv(
        self,
        *,
        user_id: str,
        cv_id: str,
    ) -> list[CVVersion]:
        """Return one CV family's versions in ascending order."""

        statement = (
            select(CVVersionRecord)
            .where(
                CVVersionRecord.user_id == user_id,
                CVVersionRecord.cv_id == cv_id,
            )
            .order_by(
                CVVersionRecord.version_number,
                CVVersionRecord.cv_version_id,
            )
        )

        with self._session_factory() as session:
            records = session.execute(statement).scalars().all()

            return [
                cv_version_record_to_domain(
                    record=record,
                    artifact_records=(
                        self._load_artifacts(
                            session=session,
                            cv_version_id=(record.cv_version_id),
                        )
                    ),
                )
                for record in records
            ]

    def attach_artifact(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact: RenderedCVArtifact,
    ) -> CVVersion:
        """Attach one immutable rendered artifact."""

        if artifact.verification_status is not CVArtifactVerificationStatus.PENDING:
            raise ValueError("New CV artifacts must begin pending verification.")

        if artifact.verification_notes:
            raise ValueError(
                "A newly rendered artifact cannot already contain verification notes."
            )

        with self._session_factory.begin() as session:
            version_record = self._get_owned_version_record(
                session=session,
                user_id=user_id,
                cv_version_id=cv_version_id,
            )

            if version_record is None:
                raise ValueError("CV version is unavailable.")

            existing_format = session.execute(
                select(CVArtifactRecord).where(
                    CVArtifactRecord.cv_version_id == cv_version_id,
                    CVArtifactRecord.artifact_format == artifact.artifact_format.value,
                )
            ).scalar_one_or_none()

            if existing_format is not None:
                if not same_artifact_identity(
                    record=existing_format,
                    artifact=artifact,
                ):
                    raise ValueError(
                        "This CV version already contains "
                        "a different artifact for that format."
                    )

                # Exact rendering retry. Do not downgrade an
                # artifact that may already have been verified.

            else:
                if version_record.status == CVVersionStatus.VERIFIED.value:
                    raise ValueError(
                        "A verified CV version cannot accept new rendered artifacts."
                    )

                existing_id = session.get(
                    CVArtifactRecord,
                    artifact.artifact_id,
                )

                if existing_id is not None:
                    raise ValueError("CV artifact identifier already exists.")

                existing_storage = session.execute(
                    select(CVArtifactRecord).where(
                        CVArtifactRecord.storage_key == artifact.storage_key
                    )
                ).scalar_one_or_none()

                if existing_storage is not None:
                    raise ValueError("CV artifact storage key already exists.")

                session.add(
                    CVArtifactRecord(
                        artifact_id=(artifact.artifact_id),
                        cv_version_id=cv_version_id,
                        artifact_format=(artifact.artifact_format.value),
                        storage_key=(artifact.storage_key),
                        sha256_hex=(artifact.sha256_hex),
                        size_bytes=(artifact.size_bytes),
                        verification_status=(artifact.verification_status.value),
                        verification_notes=list(artifact.verification_notes),
                    )
                )

                version_record.status = CVVersionStatus.RENDERED.value

        persisted = self.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if persisted is None:
            raise RuntimeError("Persisted CV version became unavailable.")

        return persisted

    def set_artifact_verification(
        self,
        *,
        user_id: str,
        cv_version_id: str,
        artifact_id: str,
        verification_status: CVArtifactVerificationStatus,
        verification_notes: list[str],
    ) -> CVVersion:
        """Apply one terminal artifact-verification result."""

        if verification_status not in {
            CVArtifactVerificationStatus.VERIFIED,
            CVArtifactVerificationStatus.FAILED,
        }:
            raise ValueError(
                "Artifact verification must resolve to verified or failed."
            )

        if (
            verification_status is CVArtifactVerificationStatus.FAILED
            and not verification_notes
        ):
            raise ValueError("Failed artifact verification requires notes.")

        with self._session_factory.begin() as session:
            version_record = self._get_owned_version_record(
                session=session,
                user_id=user_id,
                cv_version_id=cv_version_id,
            )

            if version_record is None:
                raise ValueError("CV version is unavailable.")

            artifact_record = session.get(
                CVArtifactRecord,
                artifact_id,
            )

            if (
                artifact_record is None
                or artifact_record.cv_version_id != cv_version_id
            ):
                raise ValueError("CV artifact is unavailable.")

            current_status = CVArtifactVerificationStatus(
                artifact_record.verification_status
            )

            if current_status is CVArtifactVerificationStatus.PENDING:
                artifact_record.verification_status = verification_status.value

                artifact_record.verification_notes = list(verification_notes)

            elif (
                current_status is verification_status
                and list(artifact_record.verification_notes) == verification_notes
            ):
                # Exact verification retry.
                pass

            else:
                raise ValueError(
                    "Artifact verification status is terminal and cannot be changed."
                )

            session.flush()

            artifact_records = self._load_artifacts(
                session=session,
                cv_version_id=cv_version_id,
            )

            required_formats = {
                CVArtifactFormat.DOCX,
                CVArtifactFormat.PDF,
            }

            persisted_formats = {
                CVArtifactFormat(record.artifact_format) for record in artifact_records
            }

            all_verified = all(
                CVArtifactVerificationStatus(record.verification_status)
                is CVArtifactVerificationStatus.VERIFIED
                for record in artifact_records
            )

            if persisted_formats == required_formats and all_verified:
                version_record.status = CVVersionStatus.VERIFIED.value
            else:
                version_record.status = CVVersionStatus.RENDERED.value

        persisted = self.get(
            user_id=user_id,
            cv_version_id=cv_version_id,
        )

        if persisted is None:
            raise RuntimeError("Persisted CV version became unavailable.")

        return persisted

    @staticmethod
    def _load_artifacts(
        *,
        session: Session,
        cv_version_id: str,
    ) -> list[CVArtifactRecord]:
        """Load persisted artifacts in deterministic format order."""

        statement = (
            select(CVArtifactRecord)
            .where(CVArtifactRecord.cv_version_id == cv_version_id)
            .order_by(CVArtifactRecord.artifact_format)
        )

        return list(session.execute(statement).scalars().all())

    def _validate_existing_retry(
        self,
        *,
        session: Session,
        user_id: str,
        version: CVVersion,
        existing: CVVersionRecord,
    ) -> None:
        """Allow only an exact same-user retry."""

        if existing.user_id != user_id:
            raise ValueError("CV version identifier already exists.")

        artifact_records = self._load_artifacts(
            session=session,
            cv_version_id=(existing.cv_version_id),
        )

        persisted = cv_version_record_to_domain(
            record=existing,
            artifact_records=artifact_records,
        )

        if not same_immutable_version_content(
            persisted=persisted,
            candidate=version,
        ):
            raise ValueError(
                "CV version identifier already exists with different persisted content."
            )

    @staticmethod
    def _validate_source_document(
        *,
        session: Session,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Require the persisted source document to belong to the user."""

        document = session.get(
            CareerDocumentRecord,
            (
                user_id,
                version.provenance.source_document_id,
            ),
        )

        if document is None:
            raise ValueError("Source career document is unavailable.")

    @staticmethod
    def _validate_job_run(
        *,
        session: Session,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Require provenance to match the persisted trusted job run."""

        job_run = session.get(
            JobAnalysisRunRecord,
            version.provenance.thread_id,
        )

        if (
            job_run is None
            or job_run.user_id != user_id
            or job_run.job_id != version.provenance.job_id
        ):
            raise ValueError("Job-analysis run is unavailable for this CV version.")

        if job_run.status != JobAnalysisRunStatus.COMPLETED.value:
            raise ValueError(
                "CV version persistence requires a completed job-analysis run."
            )

        if job_run.review_status != version.provenance.review_status.value:
            raise ValueError(
                "CV version review provenance does not "
                "match the persisted job-analysis run."
            )

        persisted_proposal_ids = [
            str(
                proposal.get(
                    "proposal_id",
                    "",
                )
            )
            for proposal in job_run.final_cv_proposals
        ]

        if persisted_proposal_ids != (version.provenance.final_proposal_ids):
            raise ValueError(
                "CV version proposal provenance does not "
                "match persisted final CV proposals."
            )

    @staticmethod
    def _validate_family_version_identity(
        *,
        session: Session,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Prevent two identities claiming one CV family/version number."""

        statement = select(CVVersionRecord).where(
            CVVersionRecord.user_id == user_id,
            CVVersionRecord.cv_id == version.structured_cv.cv_id,
            CVVersionRecord.version_number == version.version_number,
        )

        existing = session.execute(statement).scalar_one_or_none()

        if existing is not None:
            raise ValueError("This CV family version number already exists.")

    @staticmethod
    def _validate_parent_version(
        *,
        session: Session,
        user_id: str,
        version: CVVersion,
    ) -> None:
        """Ensure later versions continue the same user's CV lineage."""

        if version.parent_version_id is None:
            return

        parent = session.get(
            CVVersionRecord,
            version.parent_version_id,
        )

        if (
            parent is None
            or parent.user_id != user_id
            or parent.cv_id != version.structured_cv.cv_id
            or parent.version_number != version.version_number - 1
        ):
            raise ValueError(
                "Parent CV version must be the preceding "
                "version in the same user CV family."
            )

    @staticmethod
    def _get_owned_version_record(
        *,
        session: Session,
        user_id: str,
        cv_version_id: str,
    ) -> CVVersionRecord | None:
        """Load one CV version within its authenticated boundary."""

        statement = select(CVVersionRecord).where(
            CVVersionRecord.cv_version_id == cv_version_id,
            CVVersionRecord.user_id == user_id,
        )

        return session.execute(statement).scalar_one_or_none()


def cv_version_record_to_domain(
    *,
    record: CVVersionRecord,
    artifact_records: list[CVArtifactRecord],
) -> CVVersion:
    """Convert persisted version and artifact records to the domain."""

    artifacts = [
        RenderedCVArtifact(
            artifact_id=(artifact.artifact_id),
            artifact_format=(CVArtifactFormat(artifact.artifact_format)),
            storage_key=(artifact.storage_key),
            sha256_hex=(artifact.sha256_hex),
            size_bytes=(artifact.size_bytes),
            verification_status=(
                CVArtifactVerificationStatus(artifact.verification_status)
            ),
            verification_notes=list(artifact.verification_notes),
        )
        for artifact in artifact_records
    ]

    return CVVersion(
        cv_version_id=(record.cv_version_id),
        version_number=(record.version_number),
        parent_version_id=(record.parent_version_id),
        status=(CVVersionStatus(record.status)),
        structured_cv=(StructuredCV.model_validate(record.structured_cv)),
        applied_changes=[
            AppliedCVChange.model_validate(payload)
            for payload in record.applied_changes
        ],
        provenance=(CVVersionProvenance.model_validate(record.provenance)),
        artifacts=artifacts,
    )


def same_immutable_version_content(
    *,
    persisted: CVVersion,
    candidate: CVVersion,
) -> bool:
    """Compare immutable CV content while ignoring lifecycle progress."""

    return (
        persisted.cv_version_id == candidate.cv_version_id
        and persisted.version_number == candidate.version_number
        and persisted.parent_version_id == candidate.parent_version_id
        and persisted.structured_cv == candidate.structured_cv
        and persisted.applied_changes == candidate.applied_changes
        and persisted.provenance == candidate.provenance
    )


def same_artifact_identity(
    *,
    record: CVArtifactRecord,
    artifact: RenderedCVArtifact,
) -> bool:
    """Compare immutable rendered-file identity."""

    return (
        record.artifact_id == artifact.artifact_id
        and record.artifact_format == artifact.artifact_format.value
        and record.storage_key == artifact.storage_key
        and record.sha256_hex == artifact.sha256_hex
        and record.size_bytes == artifact.size_bytes
    )

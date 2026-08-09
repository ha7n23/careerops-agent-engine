"""FastAPI dependency providers."""

from functools import lru_cache
from typing import Annotated

from fastapi import Header, HTTPException, status
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.application.ports.cv_evidence_audit_repository import (
    CVEvidenceAuditRepository,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.ports.job_analysis_audit_repository import (
    JobAnalysisAuditRepository,
)
from careerops_agent_engine.application.services.cv_claim_verification import (
    CVClaimVerificationService,
)
from careerops_agent_engine.application.services.cv_document_extraction import (
    CVDocumentExtractionService,
)
from careerops_agent_engine.application.services.cv_document_ingestion import (
    CVDocumentIngestionService,
)
from careerops_agent_engine.application.services.cv_document_preparation import (
    CVDocumentPreparationService,
)
from careerops_agent_engine.application.services.cv_document_upload import (
    CVDocumentUploadService,
)
from careerops_agent_engine.application.services.cv_evidence_duplicates import (
    CVEvidenceDuplicateDetector,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    CVEvidenceProposalService,
)
from careerops_agent_engine.application.services.cv_evidence_review import (
    CVEvidenceReviewService,
)
from careerops_agent_engine.application.services.cv_evidence_workflow import (
    CVEvidenceWorkflowService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.documents.cv_section_parser import (
    DeterministicCVSectionParser,
)
from careerops_agent_engine.infrastructure.documents.native_document_extractor import (
    NativeDocumentExtractor,
)
from careerops_agent_engine.infrastructure.llm.factory import (
    create_cv_claim_verifier,
    create_cv_evidence_extractor,
    create_cv_proposal_generator,
    create_evidence_discovery_runner,
    create_requirement_extractor,
)
from careerops_agent_engine.infrastructure.repositories import (
    sqlalchemy_job_analysis_audit,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_career_documents import (  # noqa: E501
    SqlAlchemyCareerDocumentRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_cv_evidence_audit import (  # noqa: E501
    SqlAlchemyCVEvidenceAuditRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)
from careerops_agent_engine.infrastructure.storage.local_document_storage import (
    LocalDocumentStorage,
)


def get_authenticated_user_id(
    x_user_id: Annotated[
        str | None,
        Header(alias="X-User-ID"),
    ] = None,
) -> str:
    """Return the authenticated development user identifier."""

    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The X-User-ID header is required.",
        )

    return x_user_id.strip()


@lru_cache
def get_database_engine() -> Engine:
    """Create one reusable SQLAlchemy engine per process."""

    return create_database_engine()


@lru_cache
def get_database_session_factory() -> sessionmaker[Session]:
    """Create one reusable database session factory."""

    return create_session_factory(get_database_engine())


@lru_cache
def get_evidence_repository() -> EvidenceRepository:
    """Create the PostgreSQL approved-evidence repository."""

    return SqlAlchemyEvidenceRepository(get_database_session_factory())


@lru_cache
def get_job_analysis_audit_repository() -> JobAnalysisAuditRepository:
    """Create the PostgreSQL job-analysis audit repository."""

    return sqlalchemy_job_analysis_audit.SqlAlchemyJobAnalysisAuditRepository(
        get_database_session_factory()
    )


@lru_cache
def get_document_storage() -> LocalDocumentStorage:
    """Create the private local document-storage adapter."""

    settings = get_settings()

    return LocalDocumentStorage(settings.document_storage_root)


@lru_cache
def get_cv_document_upload_service() -> CVDocumentUploadService:
    """Create the validated CV upload service."""

    settings = get_settings()

    return CVDocumentUploadService(
        storage=get_document_storage(),
        max_upload_bytes=(settings.document_upload_max_bytes),
    )


@lru_cache
def get_career_document_repository() -> CareerDocumentRepository:
    """Create the PostgreSQL career-document repository."""

    return SqlAlchemyCareerDocumentRepository(get_database_session_factory())


@lru_cache
def get_cv_evidence_audit_repository() -> CVEvidenceAuditRepository:
    """Create the PostgreSQL CV evidence-review audit repository."""

    return SqlAlchemyCVEvidenceAuditRepository(get_database_session_factory())


@lru_cache
def get_cv_document_ingestion_service() -> CVDocumentIngestionService:
    """Create the persistent CV ingestion service."""

    return CVDocumentIngestionService(
        upload_service=get_cv_document_upload_service(),
        repository=get_career_document_repository(),
        storage=get_document_storage(),
    )


@lru_cache
def get_cv_document_preparation_service() -> CVDocumentPreparationService:
    """Create native extraction and deterministic CV preparation."""

    extraction_service = CVDocumentExtractionService(
        storage=get_document_storage(),
        extractor=NativeDocumentExtractor(),
    )

    return CVDocumentPreparationService(
        extraction_service=extraction_service,
        section_parser=DeterministicCVSectionParser(),
    )


@lru_cache
def get_cv_evidence_workflow_service() -> CVEvidenceWorkflowService:
    """Create the persistent CV evidence-review workflow."""

    evidence_repository = get_evidence_repository()

    return CVEvidenceWorkflowService(
        document_repository=get_career_document_repository(),
        audit_repository=get_cv_evidence_audit_repository(),
        preparation_service=get_cv_document_preparation_service(),
        proposal_service=CVEvidenceProposalService(
            extractor=create_cv_evidence_extractor()
        ),
        duplicate_detector=CVEvidenceDuplicateDetector(repository=evidence_repository),
        review_service=CVEvidenceReviewService(),
    )


@lru_cache
def get_job_analysis_service() -> JobAnalysisService:
    """Create one reusable job-analysis service per process."""

    repository = get_evidence_repository()

    cv_proposal_service = CVProposalGenerationService(
        repository=repository,
        generator=create_cv_proposal_generator(),
    )

    cv_claim_verification_service = CVClaimVerificationService(
        repository=repository,
        verifier=create_cv_claim_verifier(),
    )

    return JobAnalysisService(
        requirement_extractor=create_requirement_extractor(),
        evidence_discovery_runner=(create_evidence_discovery_runner(repository)),
        cv_proposal_service=cv_proposal_service,
        cv_claim_verification_service=(cv_claim_verification_service),
        checkpointer_factory=open_postgres_checkpointer,
        audit_repository=get_job_analysis_audit_repository(),
    )

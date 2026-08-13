"""FastAPI dependency providers."""

from functools import lru_cache

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from careerops_agent_engine.api.security import (
    get_authenticated_user_id as get_authenticated_user_id,
)
from careerops_agent_engine.application.ports.artifact_storage import (
    ArtifactStorage,
)
from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.application.ports.cv_evidence_audit_repository import (
    CVEvidenceAuditRepository,
)
from careerops_agent_engine.application.ports.cv_version_repository import (
    CVVersionRepository,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.ports.job_analysis_audit_repository import (
    JobAnalysisAuditRepository,
)
from careerops_agent_engine.application.services.cv_artifact_rendering import (
    CVArtifactRenderingService,
)
from careerops_agent_engine.application.services.cv_artifact_retrieval import (
    CVArtifactRetrievalService,
)
from careerops_agent_engine.application.services.cv_artifact_verification import (
    CVArtifactVerificationService,
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
from careerops_agent_engine.application.services.cv_pdf_conversion import (
    CVPDFConversionService,
)
from careerops_agent_engine.application.services.cv_proposals import (
    CVProposalGenerationService,
)
from careerops_agent_engine.application.services.cv_version_builder import (
    CVVersionBuilder,
)
from careerops_agent_engine.application.services.final_cv_assembly import (
    FinalCVAssemblyService,
)
from careerops_agent_engine.application.services.final_cv_generation import (
    FINAL_CV_WORKFLOW_VERSION,
    FinalCVGenerationService,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.application.services.structured_cv_assembly import (
    BaseStructuredCVAssembler,
)
from careerops_agent_engine.application.services.structured_cv_tailoring import (
    StructuredCVProposalApplier,
)
from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.infrastructure.database.checkpoint import (
    open_postgres_checkpointer,
)
from careerops_agent_engine.infrastructure.database.session import (
    create_database_engine,
    create_session_factory,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_renderer import (
    CareerOpsStandardDocxRenderer,
)
from careerops_agent_engine.infrastructure.documents.careerops_docx_verifier import (
    CareerOpsStandardDocxVerifier,
)
from careerops_agent_engine.infrastructure.documents.careerops_pdf_verifier import (
    CareerOpsStandardPDFVerifier,
)
from careerops_agent_engine.infrastructure.documents.cv_section_parser import (
    DeterministicCVSectionParser,
)
from careerops_agent_engine.infrastructure.documents.libreoffice_pdf_converter import (
    LibreOfficePDFConverter,
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
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_cv_versions import (
    SqlAlchemyCVVersionRepository,
)
from careerops_agent_engine.infrastructure.repositories.sqlalchemy_evidence import (
    SqlAlchemyEvidenceRepository,
)
from careerops_agent_engine.infrastructure.storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from careerops_agent_engine.infrastructure.storage.local_document_storage import (
    LocalDocumentStorage,
)


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


@lru_cache
def get_cv_version_repository() -> CVVersionRepository:
    """Create the PostgreSQL CV-version repository."""

    return SqlAlchemyCVVersionRepository(get_database_session_factory())


@lru_cache
def get_artifact_storage() -> ArtifactStorage:
    """Create private generated-artifact storage."""

    settings = get_settings()

    return LocalArtifactStorage(settings.artifact_storage_root)


@lru_cache
def get_final_cv_assembly_service() -> FinalCVAssemblyService:
    """Create deterministic final-CV assembly."""

    return FinalCVAssemblyService(
        job_audit_repository=(get_job_analysis_audit_repository()),
        document_repository=(get_career_document_repository()),
        evidence_repository=(get_evidence_repository()),
        document_preparer=(get_cv_document_preparation_service()),
        base_assembler=(BaseStructuredCVAssembler()),
        proposal_applier=(StructuredCVProposalApplier()),
    )


@lru_cache
def get_docx_rendering_service() -> CVArtifactRenderingService:
    """Create deterministic DOCX rendering orchestration."""

    return CVArtifactRenderingService(
        version_repository=(get_cv_version_repository()),
        artifact_storage=(get_artifact_storage()),
        renderer=(CareerOpsStandardDocxRenderer()),
    )


@lru_cache
def get_docx_verification_service() -> CVArtifactVerificationService:
    """Create deterministic DOCX verification orchestration."""

    return CVArtifactVerificationService(
        version_repository=(get_cv_version_repository()),
        artifact_storage=(get_artifact_storage()),
        verifier=(CareerOpsStandardDocxVerifier()),
    )


@lru_cache
def get_pdf_conversion_service() -> CVPDFConversionService:
    """Create verified-DOCX to PDF conversion orchestration."""

    settings = get_settings()

    return CVPDFConversionService(
        version_repository=(get_cv_version_repository()),
        artifact_storage=(get_artifact_storage()),
        converter=LibreOfficePDFConverter(
            executable=(settings.libreoffice_executable),
            timeout_seconds=(settings.pdf_conversion_timeout_seconds),
        ),
    )


@lru_cache
def get_pdf_verification_service() -> CVArtifactVerificationService:
    """Create deterministic PDF verification orchestration."""

    return CVArtifactVerificationService(
        version_repository=(get_cv_version_repository()),
        artifact_storage=(get_artifact_storage()),
        verifier=(CareerOpsStandardPDFVerifier()),
    )


@lru_cache
def get_final_cv_generation_service() -> FinalCVGenerationService:
    """Create the complete resumable final-CV workflow."""

    renderer = CareerOpsStandardDocxRenderer()

    return FinalCVGenerationService(
        assembly_service=(get_final_cv_assembly_service()),
        version_builder=CVVersionBuilder(),
        version_repository=(get_cv_version_repository()),
        docx_rendering_service=(get_docx_rendering_service()),
        docx_verification_service=(get_docx_verification_service()),
        pdf_conversion_service=(get_pdf_conversion_service()),
        pdf_verification_service=(get_pdf_verification_service()),
        template_id=renderer.template_id,
        template_version=(renderer.template_version),
        workflow_version=(FINAL_CV_WORKFLOW_VERSION),
        llm_references=[],
    )


@lru_cache
def get_cv_artifact_retrieval_service() -> CVArtifactRetrievalService:
    """Create secure generated-CV retrieval orchestration."""

    return CVArtifactRetrievalService(
        version_repository=(get_cv_version_repository()),
        artifact_storage=(get_artifact_storage()),
    )

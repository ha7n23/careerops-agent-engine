"""Integration proof from approved CV evidence into job analysis."""

from collections.abc import Iterator, Sequence

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from careerops_agent_engine.agents.nodes.job_analysis import (
    calculate_fit,
    create_discover_evidence_node,
)
from careerops_agent_engine.agents.states.job_analysis import (
    JobAnalysisState,
)
from careerops_agent_engine.domain.enums import (
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVEvidenceReviewRunStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    MatchStrength,
    RequirementCategory,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceProposal,
    EvidenceMatch,
    SourceReference,
)
from careerops_agent_engine.domain.models.evidence_audit import (
    CVEvidenceReviewAuditEntry,
    CVEvidenceReviewRunSnapshot,
)
from careerops_agent_engine.domain.models.evidence_review import (
    EvidenceReviewDecision,
    EvidenceReviewResult,
)
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
)
from careerops_agent_engine.infrastructure.database.base import Base
from careerops_agent_engine.infrastructure.database.session import (
    create_session_factory,
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


class RepositoryBackedEvidenceDiscoveryRunner:
    """Use the production evidence repository for deterministic discovery."""

    def __init__(
        self,
        repository: SqlAlchemyEvidenceRepository,
    ) -> None:
        """Store the approved-evidence repository."""

        self._repository = repository

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Classify an exact repository search as direct evidence."""

        evidence = self._repository.search_approved(
            user_id=user_id,
            query=requirement.name,
            limit=5,
        )

        if not evidence:
            return EvidenceMatch(
                requirement_id=requirement.requirement_id,
                match_strength=MatchStrength.NONE,
                direct_evidence_ids=[],
                related_evidence_ids=[],
                explanation=("No approved evidence matched the requirement."),
                gap=True,
            )

        return EvidenceMatch(
            requirement_id=requirement.requirement_id,
            match_strength=MatchStrength.STRONG,
            direct_evidence_ids=[record.evidence_id for record in evidence],
            related_evidence_ids=[],
            explanation=("Approved CV evidence directly matched the requirement."),
            gap=False,
        )

    def discover_for_requirements(
        self,
        requirements: Sequence[JobRequirement],
        *,
        user_id: str,
    ) -> list[EvidenceMatch]:
        """Discover repository-backed matches for all requirements."""

        return [
            self.discover(
                requirement,
                user_id=user_id,
            )
            for requirement in requirements
        ]


@pytest.fixture
def repositories() -> Iterator[
    tuple[
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyEvidenceRepository,
    ]
]:
    """Create repositories sharing one isolated relational database."""

    engine: Engine = create_engine("sqlite+pysqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = create_session_factory(engine)

    try:
        yield (
            SqlAlchemyCareerDocumentRepository(session_factory),
            SqlAlchemyCVEvidenceAuditRepository(session_factory),
            SqlAlchemyEvidenceRepository(session_factory),
        )
    finally:
        engine.dispose()


def build_document() -> CareerDocument:
    """Create trusted metadata for the uploaded CV."""

    return CareerDocument(
        document_id="DOC-BRIDGE-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key=("documents/user-bridge/DOC-BRIDGE-001.pdf"),
        status=CareerDocumentStatus.EXTRACTED,
    )


def build_proposal() -> CareerEvidenceProposal:
    """Create the pending proposal shown to the human reviewer."""

    source_excerpt = "Built a FastAPI application using Python."

    return CareerEvidenceProposal(
        proposal_id="EVP-BRIDGE-001",
        category=EvidenceCategory.PROJECT,
        title="Python API Project",
        source_section=CVSection.PROJECTS,
        source_section_order_index=0,
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=["API development"],
        claims=[source_excerpt],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-BRIDGE-001",
                source_excerpt=source_excerpt,
            )
        ],
        warnings=[],
    )


def build_approved_evidence() -> CareerEvidence:
    """Create the evidence promoted by explicit human approval."""

    source_excerpt = "Built a FastAPI application using Python."

    return CareerEvidence(
        evidence_id="EVD-BRIDGE-PYTHON",
        category=EvidenceCategory.PROJECT,
        title="Python API Project",
        verification_status=(VerificationStatus.APPROVED),
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=["API development"],
        approved_claims=[source_excerpt],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-BRIDGE-001",
                source_excerpt=source_excerpt,
            )
        ],
    )


def persist_human_approved_cv_evidence(
    *,
    document_repository: SqlAlchemyCareerDocumentRepository,
    audit_repository: SqlAlchemyCVEvidenceAuditRepository,
) -> None:
    """Cross the real CV human-review persistence boundary."""

    document_repository.save(
        user_id="USER-BRIDGE-001",
        document=build_document(),
    )

    proposal = build_proposal()
    approved_evidence = build_approved_evidence()

    result = EvidenceReviewResult(
        approved_proposal_ids=[proposal.proposal_id],
        approved_evidence=[approved_evidence],
    )

    snapshot = CVEvidenceReviewRunSnapshot(
        review_run_id="EVR-BRIDGE-001",
        user_id="USER-BRIDGE-001",
        document_id="DOC-BRIDGE-001",
        status=(CVEvidenceReviewRunStatus.COMPLETED),
        proposals=[proposal],
        overlap_findings=[],
        document_warnings=[],
        review_result=result,
    )

    decision = EvidenceReviewDecision(approved_proposal_ids=[proposal.proposal_id])

    review = CVEvidenceReviewAuditEntry(
        review_id="EVH-BRIDGE-001",
        review_run_id="EVR-BRIDGE-001",
        sequence_number=1,
        decision=decision,
        result=result,
    )

    audit_repository.save_review_result(
        snapshot=snapshot,
        review=review,
    )


def build_python_requirement() -> JobRequirement:
    """Create one requirement for approved Python evidence."""

    return JobRequirement(
        requirement_id="REQ-BRIDGE-PYTHON",
        name="Python",
        category=RequirementCategory.ESSENTIAL,
        evidence_expected=("Demonstrated Python development experience."),
        importance_score=5,
        source_text=("Strong Python development experience required."),
    )


def test_approved_cv_evidence_flows_into_job_analysis(
    repositories: tuple[
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """Approved CV evidence should directly affect job-analysis fit."""

    (
        document_repository,
        audit_repository,
        evidence_repository,
    ) = repositories

    persist_human_approved_cv_evidence(
        document_repository=document_repository,
        audit_repository=audit_repository,
    )

    requirement = build_python_requirement()

    runner = RepositoryBackedEvidenceDiscoveryRunner(evidence_repository)

    discover_evidence = create_discover_evidence_node(runner)

    state: JobAnalysisState = {
        "job_id": "JOB-BRIDGE-001",
        "user_id": "USER-BRIDGE-001",
        "job_description": ("Strong Python development experience required."),
        "requirements": [requirement.model_dump(mode="json")],
        "audit_events": [],
    }

    discovery_update = discover_evidence(state)

    match_payloads = discovery_update.get(
        "evidence_matches",
        [],
    )

    state["evidence_matches"] = match_payloads

    fit_update = calculate_fit(state)

    matches = [EvidenceMatch.model_validate(payload) for payload in match_payloads]

    assert len(matches) == 1

    match = matches[0]

    assert match.match_strength is MatchStrength.STRONG

    assert match.direct_evidence_ids == ["EVD-BRIDGE-PYTHON"]

    assert match.gap is False

    fit_score = fit_update.get("fit_score")

    assert fit_score is not None
    assert fit_score == 100.0


def test_cv_evidence_remains_user_isolated_from_job_analysis(
    repositories: tuple[
        SqlAlchemyCareerDocumentRepository,
        SqlAlchemyCVEvidenceAuditRepository,
        SqlAlchemyEvidenceRepository,
    ],
) -> None:
    """Another user must receive no credit for the approved CV evidence."""

    (
        document_repository,
        audit_repository,
        evidence_repository,
    ) = repositories

    persist_human_approved_cv_evidence(
        document_repository=document_repository,
        audit_repository=audit_repository,
    )

    requirement = build_python_requirement()

    runner = RepositoryBackedEvidenceDiscoveryRunner(evidence_repository)

    discover_evidence = create_discover_evidence_node(runner)

    state: JobAnalysisState = {
        "job_id": "JOB-BRIDGE-OTHER",
        "user_id": "USER-OTHER",
        "job_description": ("Strong Python development experience required."),
        "requirements": [requirement.model_dump(mode="json")],
        "audit_events": [],
    }

    discovery_update = discover_evidence(state)

    match_payloads = discovery_update.get(
        "evidence_matches",
        [],
    )

    state["evidence_matches"] = match_payloads

    fit_update = calculate_fit(state)

    matches = [EvidenceMatch.model_validate(payload) for payload in match_payloads]

    assert len(matches) == 1

    match = matches[0]

    assert match.match_strength is MatchStrength.NONE

    assert match.direct_evidence_ids == []
    assert match.related_evidence_ids == []
    assert match.gap is True

    fit_score = fit_update.get("fit_score")

    assert fit_score is not None
    assert fit_score == 0.0

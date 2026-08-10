"""Tests for persisted final-CV assembly orchestration."""

import pytest

from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    StructuredCVAssemblyError,
)
from careerops_agent_engine.application.services.final_cv_assembly import (
    FinalCVAssemblyService,
    collect_supporting_evidence_ids,
)
from careerops_agent_engine.application.services.structured_cv_assembly import (
    BaseStructuredCVAssembler,
)
from careerops_agent_engine.application.services.structured_cv_tailoring import (
    StructuredCVProposalApplier,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CareerDocumentFormat,
    CareerDocumentStatus,
    CVSection,
    EvidenceCategory,
    EvidenceSourceType,
    JobAnalysisRunStatus,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.audit import (
    CVReviewAuditEntry,
    JobAnalysisRunSnapshot,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
    ParsedCVDocument,
    ParsedCVSection,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)

SOURCE_TEXT = "CareerOps\nBuilt CareerOps using Python and FastAPI."


def build_document(
    *,
    status: CareerDocumentStatus = (CareerDocumentStatus.EXTRACTED),
) -> CareerDocument:
    """Create one trusted persisted source document."""

    return CareerDocument(
        document_id="DOC-001",
        original_filename="cv.pdf",
        document_format=CareerDocumentFormat.PDF,
        media_type="application/pdf",
        size_bytes=100,
        sha256_hex="a" * 64,
        storage_key=("documents/user-test/DOC-001.pdf"),
        status=status,
    )


def build_evidence() -> CareerEvidence:
    """Create source-grounded approved career evidence."""

    return CareerEvidence(
        evidence_id="EVD-001",
        category=EvidenceCategory.PROJECT,
        title="CareerOps",
        verification_status=(VerificationStatus.APPROVED),
        technologies=[
            "Python",
            "FastAPI",
        ],
        capabilities=[
            "API development",
        ],
        approved_claims=[("Built CareerOps using Python and FastAPI.")],
        source_references=[
            SourceReference(
                source_type=(EvidenceSourceType.UPLOADED_CV),
                source_id="DOC-001",
                source_excerpt=("Built CareerOps using Python and FastAPI."),
            )
        ],
    )


def build_proposal() -> CVChangeProposal:
    """Create one final human-approved CV proposal."""

    return CVChangeProposal(
        proposal_id="CVP-001",
        section=CVSection.PROJECTS,
        current_text=None,
        proposed_text=("Built a CareerOps application using Python and FastAPI."),
        requirement_ids=[
            "REQ-001",
        ],
        supporting_evidence_ids=[
            "EVD-001",
        ],
        confidence_score=1.0,
    )


def build_job_run(
    *,
    status: JobAnalysisRunStatus = (JobAnalysisRunStatus.COMPLETED),
    review_status: ApprovalStatus | None = (ApprovalStatus.APPROVED),
    include_final_proposal: bool = True,
) -> JobAnalysisRunSnapshot:
    """Create the persisted business result of job analysis."""

    return JobAnalysisRunSnapshot(
        thread_id="THR-001",
        user_id="USER-001",
        job_id="JOB-001",
        status=status,
        review_status=review_status,
        final_cv_proposals=([build_proposal()] if include_final_proposal else []),
    )


class FakeJobAnalysisAuditRepository:
    """Expose one user-scoped persisted job-analysis run."""

    def __init__(
        self,
        run: JobAnalysisRunSnapshot | None,
    ) -> None:
        self.run = run

    def save_run(
        self,
        snapshot: JobAnalysisRunSnapshot,
    ) -> None:
        """Writes are not used by final assembly."""

        del snapshot

        raise AssertionError("Final assembly must not mutate job-analysis state.")

    def save_review_result(
        self,
        *,
        snapshot: JobAnalysisRunSnapshot,
        review: CVReviewAuditEntry,
    ) -> None:
        """Writes are not used by final assembly."""

        del snapshot
        del review

        raise AssertionError("Final assembly must not write review history.")

    def get_run(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> JobAnalysisRunSnapshot | None:
        """Return the run only inside its user boundary."""

        if self.run is None:
            return None

        if self.run.user_id != user_id or self.run.thread_id != thread_id:
            return None

        return self.run

    def list_reviews(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> list[CVReviewAuditEntry]:
        """Review history is not needed for this boundary."""

        del user_id
        del thread_id

        return []


class FakeCareerDocumentRepository:
    """Expose one user-owned source document."""

    def __init__(
        self,
        document: CareerDocument | None,
    ) -> None:
        self.document = document

    def save(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> None:
        """Writes are not used by final assembly."""

        del user_id
        del document

        raise AssertionError("Final assembly must not rewrite source metadata.")

    def get(
        self,
        *,
        user_id: str,
        document_id: str,
    ) -> CareerDocument | None:
        """Return the source document inside its user boundary."""

        if self.document is None:
            return None

        if user_id != "USER-001" or self.document.document_id != document_id:
            return None

        return self.document


class FakeDocumentPreparer:
    """Return deterministic parsed source-CV content."""

    def __init__(self) -> None:
        self.call_count = 0

    def prepare(
        self,
        *,
        user_id: str,
        document: CareerDocument,
    ) -> ParsedCVDocument:
        """Return the preserved source Projects section."""

        self.call_count += 1

        assert user_id == "USER-001"
        assert document.document_id == "DOC-001"

        return ParsedCVDocument(
            document_id="DOC-001",
            preamble_text="Example Candidate",
            sections=[
                ParsedCVSection(
                    section=CVSection.PROJECTS,
                    heading="Projects",
                    text=SOURCE_TEXT,
                    order_index=0,
                )
            ],
            warnings=[],
        )


def build_service(
    *,
    run: JobAnalysisRunSnapshot | None = None,
    document: CareerDocument | None = None,
    include_evidence: bool = True,
) -> tuple[
    FinalCVAssemblyService,
    FakeDocumentPreparer,
]:
    """Create final assembly around deterministic collaborators."""

    preparer = FakeDocumentPreparer()

    evidence_repository = InMemoryEvidenceRepository(
        {"USER-001": ([build_evidence()] if include_evidence else [])}
    )

    service = FinalCVAssemblyService(
        job_audit_repository=(
            FakeJobAnalysisAuditRepository(run if run is not None else build_job_run())
        ),
        document_repository=(
            FakeCareerDocumentRepository(
                document if document is not None else build_document()
            )
        ),
        evidence_repository=evidence_repository,
        document_preparer=preparer,
        base_assembler=(BaseStructuredCVAssembler()),
        proposal_applier=(StructuredCVProposalApplier()),
    )

    return service, preparer


def test_final_assembly_uses_only_persisted_final_proposals() -> None:
    """Accepted persisted output should become deterministic tailored content."""

    service, preparer = build_service()

    result = service.assemble(
        user_id="USER-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
    )

    assert preparer.call_count == 1

    assert result.job_run.review_status is (ApprovalStatus.APPROVED)

    assert result.source_document.document_id == ("DOC-001")

    assert [evidence.evidence_id for evidence in result.approved_evidence] == [
        "EVD-001"
    ]

    projects = result.tailoring_result.structured_cv.sections[0]

    assert projects.free_text == (
        "CareerOps\nBuilt a CareerOps application using Python and FastAPI."
    )

    assert len(result.tailoring_result.applied_changes) == 1

    assert result.tailoring_result.applied_changes[0].proposal_id == "CVP-001"


def test_edited_review_outcome_is_also_accepted() -> None:
    """Human-edited verified output may become final CV content."""

    service, _ = build_service(run=build_job_run(review_status=ApprovalStatus.EDITED))

    result = service.assemble(
        user_id="USER-001",
        thread_id="THR-001",
        source_document_id="DOC-001",
    )

    assert result.job_run.review_status is (ApprovalStatus.EDITED)


def test_awaiting_review_run_cannot_be_rendered() -> None:
    """A proposal awaiting human review cannot cross this boundary."""

    service, preparer = build_service(
        run=build_job_run(
            status=(JobAnalysisRunStatus.AWAITING_REVIEW),
            review_status=None,
        )
    )

    with pytest.raises(
        StructuredCVAssemblyError,
        match="completed job-analysis run",
    ):
        service.assemble(
            user_id="USER-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
        )

    assert preparer.call_count == 0


def test_rejected_review_outcome_cannot_be_rendered() -> None:
    """Rejected proposals cannot become final CV content."""

    service, preparer = build_service(
        run=build_job_run(review_status=(ApprovalStatus.REJECTED))
    )

    with pytest.raises(
        StructuredCVAssemblyError,
        match="approved or edited",
    ):
        service.assemble(
            user_id="USER-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
        )

    assert preparer.call_count == 0


def test_run_without_final_proposals_cannot_be_assembled() -> None:
    """Generated-but-unaccepted proposals are insufficient."""

    service, preparer = build_service(run=build_job_run(include_final_proposal=False))

    with pytest.raises(
        StructuredCVAssemblyError,
        match="no final CV proposals",
    ):
        service.assemble(
            user_id="USER-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
        )

    assert preparer.call_count == 0


def test_missing_source_document_is_user_safe() -> None:
    """Unavailable source metadata must stop assembly."""

    service, preparer = build_service(document=None)

    # Replace the default created by build_service.
    service = FinalCVAssemblyService(
        job_audit_repository=(FakeJobAnalysisAuditRepository(build_job_run())),
        document_repository=(FakeCareerDocumentRepository(None)),
        evidence_repository=(
            InMemoryEvidenceRepository({"USER-001": [build_evidence()]})
        ),
        document_preparer=preparer,
        base_assembler=(BaseStructuredCVAssembler()),
        proposal_applier=(StructuredCVProposalApplier()),
    )

    with pytest.raises(
        CareerDocumentUnavailableError,
        match="unavailable",
    ):
        service.assemble(
            user_id="USER-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
        )

    assert preparer.call_count == 0


def test_missing_approved_evidence_stops_final_assembly() -> None:
    """Every persisted final proposal must still resolve trusted evidence."""

    service, preparer = build_service(include_evidence=False)

    with pytest.raises(
        StructuredCVAssemblyError,
        match="available approved evidence",
    ):
        service.assemble(
            user_id="USER-001",
            thread_id="THR-001",
            source_document_id="DOC-001",
        )

    assert preparer.call_count == 1


def test_supporting_evidence_collection_is_stable_and_unique() -> None:
    """Shared evidence should be loaded once in first-seen order."""

    first = build_proposal()

    second = CVChangeProposal(
        proposal_id="CVP-002",
        section=CVSection.PROFILE,
        proposed_text="Python API engineer.",
        requirement_ids=[
            "REQ-002",
        ],
        supporting_evidence_ids=[
            "EVD-002",
            "EVD-001",
        ],
        confidence_score=1.0,
    )

    assert collect_supporting_evidence_ids(
        [
            first,
            second,
        ]
    ) == [
        "EVD-001",
        "EVD-002",
    ]

"""Orchestration from persisted approved job analysis to tailored CV content."""

from dataclasses import dataclass

from careerops_agent_engine.application.exceptions import (
    CareerDocumentUnavailableError,
    StructuredCVAssemblyError,
)
from careerops_agent_engine.application.ports.career_document_repository import (
    CareerDocumentRepository,
)
from careerops_agent_engine.application.ports.cv_document_preparer import (
    CVDocumentPreparer,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.ports.job_analysis_audit_repository import (
    JobAnalysisAuditRepository,
)
from careerops_agent_engine.application.services.structured_cv_assembly import (
    BaseStructuredCVAssembler,
)
from careerops_agent_engine.application.services.structured_cv_tailoring import (
    StructuredCVProposalApplier,
)
from careerops_agent_engine.domain.enums import (
    ApprovalStatus,
    CareerDocumentStatus,
    JobAnalysisRunStatus,
)
from careerops_agent_engine.domain.models.audit import (
    JobAnalysisRunSnapshot,
)
from careerops_agent_engine.domain.models.cv import (
    CVChangeProposal,
)
from careerops_agent_engine.domain.models.cv_application import (
    StructuredCVTailoringResult,
)
from careerops_agent_engine.domain.models.document import (
    CareerDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)


@dataclass(frozen=True)
class FinalCVAssemblyExecutionResult:
    """Trusted inputs and deterministic result of final CV assembly."""

    job_run: JobAnalysisRunSnapshot
    source_document: CareerDocument
    approved_evidence: tuple[CareerEvidence, ...]
    tailoring_result: StructuredCVTailoringResult


class FinalCVAssemblyService:
    """Assemble a CV only from persisted human-approved workflow output."""

    def __init__(
        self,
        *,
        job_audit_repository: JobAnalysisAuditRepository,
        document_repository: CareerDocumentRepository,
        evidence_repository: EvidenceRepository,
        document_preparer: CVDocumentPreparer,
        base_assembler: BaseStructuredCVAssembler,
        proposal_applier: StructuredCVProposalApplier,
    ) -> None:
        """Store final-CV assembly dependencies."""

        self._job_audit_repository = job_audit_repository
        self._document_repository = document_repository
        self._evidence_repository = evidence_repository
        self._document_preparer = document_preparer
        self._base_assembler = base_assembler
        self._proposal_applier = proposal_applier

    def assemble(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_document_id: str,
    ) -> FinalCVAssemblyExecutionResult:
        """Build a tailored CV from persisted accepted workflow state."""

        job_run = self._load_accepted_job_run(
            user_id=user_id,
            thread_id=thread_id,
        )

        source_document = self._document_repository.get(
            user_id=user_id,
            document_id=source_document_id,
        )

        if (
            source_document is None
            or source_document.status is CareerDocumentStatus.QUARANTINED
        ):
            raise CareerDocumentUnavailableError("The career document is unavailable.")

        parsed_document = self._document_preparer.prepare(
            user_id=user_id,
            document=source_document,
        )

        base_cv = self._base_assembler.assemble(document=parsed_document)

        evidence_ids = collect_supporting_evidence_ids(job_run.final_cv_proposals)

        approved_evidence = self._load_approved_evidence(
            user_id=user_id,
            evidence_ids=evidence_ids,
        )

        tailoring_result = self._proposal_applier.apply(
            base_cv=base_cv,
            proposals=list(job_run.final_cv_proposals),
            approved_evidence=list(approved_evidence),
        )

        return FinalCVAssemblyExecutionResult(
            job_run=job_run,
            source_document=source_document,
            approved_evidence=approved_evidence,
            tailoring_result=tailoring_result,
        )

    def _load_accepted_job_run(
        self,
        *,
        user_id: str,
        thread_id: str,
    ) -> JobAnalysisRunSnapshot:
        """Load only a completed human-accepted job-analysis result."""

        job_run = self._job_audit_repository.get_run(
            user_id=user_id,
            thread_id=thread_id,
        )

        if job_run is None:
            raise StructuredCVAssemblyError(
                "The completed job-analysis run is unavailable."
            )

        if job_run.status is not JobAnalysisRunStatus.COMPLETED:
            raise StructuredCVAssemblyError(
                "A final CV requires a completed job-analysis run."
            )

        if job_run.review_status not in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.EDITED,
        }:
            raise StructuredCVAssemblyError(
                "A final CV requires an approved or edited human-review outcome."
            )

        if not job_run.final_cv_proposals:
            raise StructuredCVAssemblyError(
                "The completed job-analysis run contains no final CV proposals."
            )

        return job_run

    def _load_approved_evidence(
        self,
        *,
        user_id: str,
        evidence_ids: list[str],
    ) -> tuple[CareerEvidence, ...]:
        """Reload every supporting item through the trusted repository."""

        evidence_items: list[CareerEvidence] = []

        for evidence_id in evidence_ids:
            evidence = self._evidence_repository.get_approved(
                user_id=user_id,
                evidence_id=evidence_id,
            )

            if evidence is None:
                raise StructuredCVAssemblyError(
                    "Every final CV proposal must reference "
                    "available approved evidence."
                )

            evidence_items.append(evidence)

        return tuple(evidence_items)


def collect_supporting_evidence_ids(
    proposals: list[CVChangeProposal],
) -> list[str]:
    """Collect proposal evidence IDs once in deterministic source order."""

    evidence_ids: list[str] = []
    seen: set[str] = set()

    for proposal in proposals:
        for evidence_id in proposal.supporting_evidence_ids:
            if evidence_id in seen:
                continue

            seen.add(evidence_id)
            evidence_ids.append(evidence_id)

    return evidence_ids

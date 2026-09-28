"""Domain models for human review of extracted career evidence."""

from typing import Self

from pydantic import Field, model_validator

from careerops_agent_engine.domain.enums import (
    EvidenceDuplicateResolutionAction,
    EvidenceLifecycleStatus,
    EvidenceOverlapScope,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.base import DomainModel
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
)


class EvidenceProposalEdit(DomainModel):
    """Human corrections to one pending evidence proposal."""

    proposal_id: str = Field(
        min_length=1,
        max_length=64,
    )

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=250,
    )

    technologies: list[str] | None = None

    capabilities: list[str] | None = None

    claims: list[str] | None = Field(
        default=None,
        min_length=1,
    )

    @model_validator(mode="after")
    def require_actual_edit(self) -> Self:
        """Require at least one concrete replacement field."""

        if (
            self.title is None
            and self.technologies is None
            and self.capabilities is None
            and self.claims is None
        ):
            raise ValueError("An evidence edit must change at least one field.")

        return self


class EvidenceDuplicateResolution(DomainModel):
    """One explicit action for one reported overlap finding."""

    proposal_id: str = Field(min_length=1, max_length=64)
    scope: EvidenceOverlapScope
    action: EvidenceDuplicateResolutionAction
    matching_proposal_id: str | None = Field(default=None, max_length=64)
    matching_evidence_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_target_and_action(self) -> Self:
        """Bind each action to a valid target for its overlap scope."""

        if self.scope is EvidenceOverlapScope.WITHIN_DOCUMENT:
            if (
                self.matching_proposal_id is None
                or self.matching_evidence_id is not None
            ):
                raise ValueError(
                    "Within-document resolution requires one matching proposal."
                )

            if self.action not in {
                EvidenceDuplicateResolutionAction.KEEP_EXISTING,
                EvidenceDuplicateResolutionAction.ACCEPT_SEPARATE,
            }:
                raise ValueError(
                    "Within-document overlap supports only keep-existing "
                    "or accept-separate."
                )

        if self.scope is EvidenceOverlapScope.APPROVED_EVIDENCE and (
            self.matching_evidence_id is None or self.matching_proposal_id is not None
        ):
            raise ValueError(
                "Approved-evidence resolution requires one matching evidence item."
            )

        return self


class EvidenceDuplicateUpdate(DomainModel):
    """Auditable before/after state for one duplicate-driven update."""

    proposal_id: str = Field(min_length=1, max_length=64)
    action: EvidenceDuplicateResolutionAction
    before: CareerEvidence
    after: CareerEvidence

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        """Require one approved target and a supported mutation action."""

        if self.action not in {
            EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
            EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
        }:
            raise ValueError("Duplicate evidence update requires replace or merge.")

        if self.before.evidence_id != self.after.evidence_id:
            raise ValueError("Duplicate evidence update cannot change its identity.")

        if any(
            evidence.verification_status is not VerificationStatus.APPROVED
            for evidence in (self.before, self.after)
        ):
            raise ValueError("Duplicate evidence update requires approved evidence.")

        if any(
            evidence.lifecycle_status is not EvidenceLifecycleStatus.ACTIVE
            for evidence in (self.before, self.after)
        ):
            raise ValueError("Duplicate evidence update requires active evidence.")

        return self


class EvidenceReviewDecision(DomainModel):
    """One human decision covering a batch of evidence proposals."""

    approved_proposal_ids: list[str] = Field(default_factory=list)

    rejected_proposal_ids: list[str] = Field(default_factory=list)

    edits: list[EvidenceProposalEdit] = Field(default_factory=list)

    duplicate_resolutions: list[EvidenceDuplicateResolution] = Field(
        default_factory=list,
    )

    reviewer_comment: str | None = Field(
        default=None,
        max_length=1_000,
    )

    @model_validator(mode="after")
    def validate_decision_sets(self) -> Self:
        """Prevent duplicate and conflicting human decisions."""

        approved_ids = self.approved_proposal_ids
        rejected_ids = self.rejected_proposal_ids
        edited_ids = [edit.proposal_id for edit in self.edits]

        if len(approved_ids) != len(set(approved_ids)):
            raise ValueError("Approved evidence proposal identifiers must be unique.")

        if len(rejected_ids) != len(set(rejected_ids)):
            raise ValueError("Rejected evidence proposal identifiers must be unique.")

        if len(edited_ids) != len(set(edited_ids)):
            raise ValueError("Edited evidence proposal identifiers must be unique.")

        resolution_keys = [
            (
                resolution.proposal_id,
                resolution.scope,
                resolution.matching_proposal_id,
                resolution.matching_evidence_id,
            )
            for resolution in self.duplicate_resolutions
        ]

        if len(resolution_keys) != len(set(resolution_keys)):
            raise ValueError("Duplicate overlap resolutions must be unique.")

        approved_set = set(approved_ids)
        rejected_set = set(rejected_ids)
        edited_set = set(edited_ids)

        if approved_set & rejected_set:
            raise ValueError(
                "An evidence proposal cannot be both approved and rejected."
            )

        if approved_set & edited_set:
            raise ValueError("An evidence proposal cannot be both approved and edited.")

        if rejected_set & edited_set:
            raise ValueError("An evidence proposal cannot be both rejected and edited.")

        if not (approved_set or rejected_set or edited_set):
            raise ValueError("Evidence review requires at least one proposal decision.")

        return self


class EvidenceReviewResult(DomainModel):
    """Deterministic result of completed human evidence review."""

    approved_proposal_ids: list[str] = Field(default_factory=list)

    edited_proposal_ids: list[str] = Field(default_factory=list)

    rejected_proposal_ids: list[str] = Field(default_factory=list)

    duplicate_resolutions: list[EvidenceDuplicateResolution] = Field(
        default_factory=list,
    )

    approved_evidence: list[CareerEvidence] = Field(default_factory=list)

    evidence_updates: list[EvidenceDuplicateUpdate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Ensure only approved evidence appears in the result."""

        accepted_ids = {
            *self.approved_proposal_ids,
            *self.edited_proposal_ids,
        }

        mutation_resolutions = {
            resolution.proposal_id
            for resolution in self.duplicate_resolutions
            if resolution.action
            in {
                EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
                EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
            }
        }
        mutation_by_proposal = {
            resolution.proposal_id: resolution
            for resolution in self.duplicate_resolutions
            if resolution.action
            in {
                EvidenceDuplicateResolutionAction.REPLACE_EXISTING,
                EvidenceDuplicateResolutionAction.MERGE_INTO_EXISTING,
            }
        }

        update_proposal_ids = [update.proposal_id for update in self.evidence_updates]

        if len(update_proposal_ids) != len(set(update_proposal_ids)):
            raise ValueError("Duplicate evidence updates must have unique proposals.")

        if set(update_proposal_ids) != mutation_resolutions:
            raise ValueError(
                "Every merge or replacement must have one matching evidence update."
            )

        for update in self.evidence_updates:
            resolution = mutation_by_proposal[update.proposal_id]

            if (
                update.action is not resolution.action
                or update.after.evidence_id != resolution.matching_evidence_id
            ):
                raise ValueError(
                    "Duplicate evidence update must match its recorded resolution."
                )

        if len(self.approved_evidence) + len(self.evidence_updates) != len(
            accepted_ids
        ):
            raise ValueError(
                "Every accepted proposal must produce "
                "exactly one approved evidence record."
            )

        if any(
            evidence.verification_status is not VerificationStatus.APPROVED
            for evidence in [
                *self.approved_evidence,
                *(update.after for update in self.evidence_updates),
            ]
        ):
            raise ValueError(
                "Evidence review results may expose only approved evidence records."
            )

        evidence_ids = [evidence.evidence_id for evidence in self.approved_evidence]

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Approved evidence identifiers must be unique.")

        updated_ids = [update.after.evidence_id for update in self.evidence_updates]

        if len(updated_ids) != len(set(updated_ids)):
            raise ValueError("Updated evidence identifiers must be unique.")

        if set(evidence_ids) & set(updated_ids):
            raise ValueError(
                "Evidence cannot be both created and updated in one review."
            )

        return self

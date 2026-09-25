"""User-scoped access to the approved Evidence Registry."""

from careerops_agent_engine.application.exceptions import (
    CareerEvidenceEditValidationError,
    CareerEvidenceUnavailableError,
)
from careerops_agent_engine.application.ports.evidence_repository import (
    EvidenceRepository,
)
from careerops_agent_engine.application.services.cv_evidence_proposals import (
    contains_explicit_text,
    normalise_provenance_text,
)
from careerops_agent_engine.domain.enums import EvidenceLifecycleStatus
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceEdit,
)


class EvidenceRegistryService:
    """Read and manage approved evidence within one user boundary."""

    def __init__(
        self,
        repository: EvidenceRepository,
    ) -> None:
        """Store the approved-evidence repository."""

        self._repository = repository

    def list_approved(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[CareerEvidence]:
        """List only approved evidence belonging to the user."""

        return self._repository.list_approved(
            user_id=user_id,
            limit=limit,
        )

    def get_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence:
        """Retrieve one approved record without leaking ownership."""

        evidence = self._repository.get_approved_for_management(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        if evidence is None:
            raise CareerEvidenceUnavailableError(
                "The approved evidence record is unavailable."
            )

        return evidence

    def edit_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
        edit: CareerEvidenceEdit,
    ) -> CareerEvidence:
        """Validate and persist an edit without changing trusted provenance."""

        current = self.get_approved(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        validate_edit(
            current=current,
            edit=edit,
        )

        updated = self._repository.edit_approved(
            user_id=user_id,
            evidence_id=evidence_id,
            edit=edit,
        )

        if updated is None:
            raise unavailable_error()

        return updated

    def archive_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence:
        """Idempotently archive one approved record."""

        return self._set_lifecycle_status(
            user_id=user_id,
            evidence_id=evidence_id,
            lifecycle_status=EvidenceLifecycleStatus.ARCHIVED,
        )

    def restore_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence:
        """Idempotently restore one archived approved record."""

        return self._set_lifecycle_status(
            user_id=user_id,
            evidence_id=evidence_id,
            lifecycle_status=EvidenceLifecycleStatus.ACTIVE,
        )

    def _set_lifecycle_status(
        self,
        *,
        user_id: str,
        evidence_id: str,
        lifecycle_status: EvidenceLifecycleStatus,
    ) -> CareerEvidence:
        """Set lifecycle state without exposing record ownership."""

        evidence = self._repository.set_lifecycle_status(
            user_id=user_id,
            evidence_id=evidence_id,
            lifecycle_status=lifecycle_status,
        )

        if evidence is None:
            raise unavailable_error()

        return evidence


def validate_edit(
    *,
    current: CareerEvidence,
    edit: CareerEvidenceEdit,
) -> None:
    """Reject duplicate or newly ungrounded factual values."""

    for field_name, values in (
        ("technologies", edit.technologies),
        ("capabilities", edit.capabilities),
        ("approved claims", edit.approved_claims),
    ):
        if values is not None:
            validate_unique_values(
                field_name=field_name,
                values=values,
            )

    excerpts = [
        reference.source_excerpt
        for reference in current.source_references
        if reference.source_excerpt
    ]

    if edit.approved_claims is not None:
        existing_claims = {
            normalise_provenance_text(value).casefold()
            for value in current.approved_claims
        }

        for claim in edit.approved_claims:
            normalised_claim = normalise_provenance_text(claim).casefold()

            if normalised_claim in existing_claims:
                continue

            if not any(
                normalised_claim in normalise_provenance_text(excerpt).casefold()
                for excerpt in excerpts
            ):
                raise CareerEvidenceEditValidationError(
                    "New approved claims must be grounded in trusted source evidence."
                )

    if edit.technologies is not None:
        existing_technologies = {
            normalise_provenance_text(value).casefold()
            for value in current.technologies
        }

        for technology in edit.technologies:
            normalised_technology = normalise_provenance_text(technology).casefold()

            if normalised_technology in existing_technologies:
                continue

            if not any(
                contains_explicit_text(
                    source=excerpt,
                    value=technology,
                )
                for excerpt in excerpts
            ):
                raise CareerEvidenceEditValidationError(
                    "New technologies must be grounded in trusted source evidence."
                )


def validate_unique_values(
    *,
    field_name: str,
    values: list[str],
) -> None:
    """Reject duplicate user-managed list values."""

    folded = [value.casefold() for value in values]

    if len(folded) != len(set(folded)):
        raise CareerEvidenceEditValidationError(
            f"Evidence {field_name} must not contain duplicate values."
        )


def unavailable_error() -> CareerEvidenceUnavailableError:
    """Build the opaque record-availability error."""

    return CareerEvidenceUnavailableError(
        "The approved evidence record is unavailable."
    )

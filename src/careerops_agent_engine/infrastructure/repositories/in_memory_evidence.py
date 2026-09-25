"""In-memory evidence repository for development and testing."""

import re
from collections.abc import Mapping, Sequence

from careerops_agent_engine.domain.enums import (
    EvidenceLifecycleStatus,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    CareerEvidenceEdit,
)

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+#.-]*")


def tokenise(value: str) -> set[str]:
    """Convert searchable text into normalised tokens."""

    return set(TOKEN_PATTERN.findall(value.casefold()))


def build_searchable_text(evidence: CareerEvidence) -> str:
    """Combine approved evidence fields used by local search."""

    return " ".join(
        [
            evidence.title,
            *evidence.technologies,
            *evidence.capabilities,
            *evidence.approved_claims,
        ]
    )


class InMemoryEvidenceRepository:
    """Store isolated evidence collections for deterministic tests."""

    def __init__(
        self,
        records_by_user: Mapping[
            str,
            Sequence[CareerEvidence],
        ]
        | None = None,
    ) -> None:
        """Copy and validate the supplied user evidence collections."""

        self._records_by_user: dict[str, list[CareerEvidence]] = {}

        for user_id, records in (records_by_user or {}).items():
            evidence_ids = [evidence.evidence_id for evidence in records]

            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError("Evidence identifiers must be unique within a user.")

            self._records_by_user[user_id] = list(records)

    def search_approved(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[CareerEvidence]:
        """Search approved evidence using deterministic token overlap."""

        if limit < 1:
            raise ValueError("Search limit must be at least one.")

        query_tokens = tokenise(query)

        if not query_tokens:
            return []

        scored_records: list[tuple[int, CareerEvidence]] = []

        for evidence in self._approved_records(user_id):
            evidence_tokens = tokenise(build_searchable_text(evidence))
            score = len(query_tokens & evidence_tokens)

            if score > 0:
                scored_records.append((score, evidence))

        scored_records.sort(
            key=lambda item: (
                -item[0],
                item[1].title.casefold(),
                item[1].evidence_id,
            )
        )

        return [evidence for _, evidence in scored_records[:limit]]

    def get_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence | None:
        """Return one approved record inside the user boundary."""

        return next(
            (
                evidence
                for evidence in self._approved_records(user_id)
                if evidence.evidence_id == evidence_id
            ),
            None,
        )

    def list_approved(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[CareerEvidence]:
        """Return a bounded list of approved user evidence."""

        if limit < 1:
            raise ValueError("List limit must be at least one.")

        return self._approved_records(user_id)[:limit]

    def get_approved_for_management(
        self,
        *,
        user_id: str,
        evidence_id: str,
    ) -> CareerEvidence | None:
        """Return active or archived approved evidence for its owner."""

        return next(
            (
                evidence
                for evidence in self._records_by_user.get(user_id, [])
                if evidence.evidence_id == evidence_id
                and evidence.verification_status is VerificationStatus.APPROVED
            ),
            None,
        )

    def edit_approved(
        self,
        *,
        user_id: str,
        evidence_id: str,
        edit: CareerEvidenceEdit,
    ) -> CareerEvidence | None:
        """Apply an edit to one approved record."""

        evidence = self.get_approved_for_management(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        if evidence is None:
            return None

        updated = evidence.model_copy(
            update=edit.model_dump(exclude_none=True),
        )

        self._replace_record(
            user_id=user_id,
            evidence=updated,
        )

        return updated

    def set_lifecycle_status(
        self,
        *,
        user_id: str,
        evidence_id: str,
        lifecycle_status: EvidenceLifecycleStatus,
    ) -> CareerEvidence | None:
        """Idempotently change one approved record's lifecycle."""

        evidence = self.get_approved_for_management(
            user_id=user_id,
            evidence_id=evidence_id,
        )

        if evidence is None:
            return None

        if evidence.lifecycle_status is lifecycle_status:
            return evidence

        updated = evidence.model_copy(
            update={"lifecycle_status": lifecycle_status},
        )

        self._replace_record(
            user_id=user_id,
            evidence=updated,
        )

        return updated

    def _replace_record(
        self,
        *,
        user_id: str,
        evidence: CareerEvidence,
    ) -> None:
        """Replace one existing in-memory record."""

        records = self._records_by_user[user_id]

        for index, existing in enumerate(records):
            if existing.evidence_id == evidence.evidence_id:
                records[index] = evidence
                return

        raise ValueError("Evidence record is unavailable.")

    def _approved_records(
        self,
        user_id: str,
    ) -> list[CareerEvidence]:
        """Return only approved evidence for the specified user."""

        return [
            evidence
            for evidence in self._records_by_user.get(user_id, [])
            if (evidence.verification_status is VerificationStatus.APPROVED)
            and evidence.lifecycle_status is EvidenceLifecycleStatus.ACTIVE
        ]

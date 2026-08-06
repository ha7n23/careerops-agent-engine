"""In-memory evidence repository for development and testing."""

import re
from collections.abc import Mapping, Sequence

from careerops_agent_engine.domain.enums import VerificationStatus
from careerops_agent_engine.domain.models.evidence import CareerEvidence

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

    def _approved_records(
        self,
        user_id: str,
    ) -> list[CareerEvidence]:
        """Return only approved evidence for the specified user."""

        return [
            evidence
            for evidence in self._records_by_user.get(user_id, [])
            if (evidence.verification_status is VerificationStatus.APPROVED)
        ]

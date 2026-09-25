"""Tests for the approved Evidence Registry API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_evidence_registry_service,
)
from careerops_agent_engine.application.services.evidence_registry import (
    EvidenceRegistryService,
)
from careerops_agent_engine.domain.enums import (
    EvidenceCategory,
    EvidenceSourceType,
    VerificationStatus,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidence,
    SourceReference,
)
from careerops_agent_engine.infrastructure.repositories.in_memory_evidence import (
    InMemoryEvidenceRepository,
)
from careerops_agent_engine.main import app


def build_evidence(
    evidence_id: str,
    *,
    title: str,
) -> CareerEvidence:
    """Create one approved API evidence record."""

    return CareerEvidence(
        evidence_id=evidence_id,
        category=EvidenceCategory.PROJECT,
        title=title,
        verification_status=VerificationStatus.APPROVED,
        technologies=["Python", "FastAPI"],
        capabilities=["API development"],
        approved_claims=[f"Built {title} using Python and FastAPI."],
        source_references=[
            SourceReference(
                source_type=EvidenceSourceType.UPLOADED_CV,
                source_id="DOC-001",
                source_excerpt=(f"Built {title} using Python and FastAPI."),
            )
        ],
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create a client with isolated user-scoped evidence."""

    repository = InMemoryEvidenceRepository(
        {
            "USER-001": [
                build_evidence(
                    "EVD-001",
                    title="CareerOps",
                ),
                build_evidence(
                    "EVD-002",
                    title="Analytics API",
                ),
            ],
            "USER-002": [
                build_evidence(
                    "EVD-OTHER",
                    title="Private Project",
                )
            ],
        }
    )

    service = EvidenceRegistryService(repository)

    app.dependency_overrides[get_evidence_registry_service] = lambda: service

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(
            get_evidence_registry_service,
            None,
        )


def test_list_returns_only_authenticated_users_evidence(
    client: TestClient,
) -> None:
    """Registry listing must remain user scoped."""

    response = client.get(
        "/api/v1/evidence",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 2
    assert body["limit"] == 100

    assert [item["evidence_id"] for item in body["items"]] == [
        "EVD-001",
        "EVD-002",
    ]

    assert all("user_id" not in item for item in body["items"])


def test_list_honours_bounded_limit(
    client: TestClient,
) -> None:
    """The frontend may request a smaller bounded result set."""

    response = client.get(
        "/api/v1/evidence?limit=1",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["count"] == 1
    assert body["limit"] == 1
    assert len(body["items"]) == 1


def test_get_returns_frontend_safe_evidence(
    client: TestClient,
) -> None:
    """A user may retrieve their own approved evidence."""

    response = client.get(
        "/api/v1/evidence/EVD-001",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["evidence_id"] == "EVD-001"
    assert body["verification_status"] == "approved"
    assert body["lifecycle_status"] == "active"
    assert body["title"] == "CareerOps"
    assert body["technologies"] == ["Python", "FastAPI"]
    assert "user_id" not in body


@pytest.mark.parametrize(
    "evidence_id",
    [
        "EVD-OTHER",
        "EVD-MISSING",
    ],
)
def test_get_uses_opaque_not_found_response(
    client: TestClient,
    evidence_id: str,
) -> None:
    """Cross-user and unknown IDs must be indistinguishable."""

    response = client.get(
        f"/api/v1/evidence/{evidence_id}",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("The approved evidence record is unavailable.")


def test_list_rejects_unbounded_limit(
    client: TestClient,
) -> None:
    """The public API must enforce its maximum list size."""

    response = client.get(
        "/api/v1/evidence?limit=101",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 422


def test_edit_updates_safe_fields_and_preserves_provenance(
    client: TestClient,
) -> None:
    """The API may update grounded fields but not trusted metadata."""

    response = client.patch(
        "/api/v1/evidence/EVD-001",
        headers={"X-User-ID": "USER-001"},
        json={
            "title": "CareerOps Platform",
            "technologies": ["Python", "FastAPI"],
            "approved_claims": ["Built CareerOps using Python and FastAPI."],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "CareerOps Platform"
    assert body["verification_status"] == "approved"
    assert body["lifecycle_status"] == "active"
    assert body["source_references"][0]["source_id"] == "DOC-001"


def test_edit_rejects_trusted_field_impersonation(
    client: TestClient,
) -> None:
    """Ownership, status and provenance are not editable API fields."""

    response = client.patch(
        "/api/v1/evidence/EVD-001",
        headers={"X-User-ID": "USER-001"},
        json={
            "title": "Unsafe edit",
            "verification_status": "approved",
            "source_references": [],
            "user_id": "USER-002",
        },
    )

    assert response.status_code == 422


def test_archive_and_restore_are_idempotent_and_recoverable(
    client: TestClient,
) -> None:
    """Archived evidence leaves active lists but remains restorable."""

    for _ in range(2):
        response = client.post(
            "/api/v1/evidence/EVD-001/archive",
            headers={"X-User-ID": "USER-001"},
        )
        assert response.status_code == 200
        assert response.json()["lifecycle_status"] == "archived"

    listed = client.get(
        "/api/v1/evidence",
        headers={"X-User-ID": "USER-001"},
    )
    assert [item["evidence_id"] for item in listed.json()["items"]] == ["EVD-002"]

    archived = client.get(
        "/api/v1/evidence/EVD-001",
        headers={"X-User-ID": "USER-001"},
    )
    assert archived.status_code == 200
    assert archived.json()["lifecycle_status"] == "archived"

    for _ in range(2):
        response = client.post(
            "/api/v1/evidence/EVD-001/restore",
            headers={"X-User-ID": "USER-001"},
        )
        assert response.status_code == 200
        assert response.json()["lifecycle_status"] == "active"


@pytest.mark.parametrize(
    "evidence_id",
    ["EVD-OTHER", "EVD-MISSING"],
)
def test_archive_uses_opaque_not_found_response(
    client: TestClient,
    evidence_id: str,
) -> None:
    """Archive must not reveal whether another user owns an identifier."""

    response = client.post(
        f"/api/v1/evidence/{evidence_id}/archive",
        headers={"X-User-ID": "USER-001"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == ("The approved evidence record is unavailable.")

"""Tests for the evidence-grounded job-analysis API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_job_analysis_service,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.domain.enums import (
    MatchStrength,
    RequirementCategory,
)
from careerops_agent_engine.domain.models.evidence import EvidenceMatch
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)
from careerops_agent_engine.main import app


class FakeRequirementExtractor:
    """Deterministic extraction adapter for API tests."""

    def extract(
        self,
        job_description: str,
        *,
        job_id: str,
    ) -> JobRequirementExtraction:
        """Return predictable requirements without calling Gemini."""

        del job_description, job_id

        return JobRequirementExtraction(
            role_title="Junior AI Engineer",
            requirements=[
                JobRequirement(
                    requirement_id="REQ-PYTHON",
                    name="Python",
                    category=RequirementCategory.ESSENTIAL,
                    evidence_expected=(
                        "Practical Python software-engineering experience."
                    ),
                    importance_score=5,
                    source_text=("Strong Python development experience is required."),
                ),
                JobRequirement(
                    requirement_id="REQ-LANGGRAPH",
                    name="LangGraph",
                    category=RequirementCategory.ESSENTIAL,
                    evidence_expected=("Implementation of stateful agent workflows."),
                    importance_score=4,
                    source_text="Experience with LangGraph is required.",
                ),
            ],
        )


class FakeEvidenceDiscoveryRunner:
    """Deterministic evidence matcher used by API tests."""

    def discover(
        self,
        requirement: JobRequirement,
        *,
        user_id: str,
    ) -> EvidenceMatch:
        """Return a controlled evidence match."""

        assert user_id == "USER-API-001"

        if requirement.requirement_id == "REQ-PYTHON":
            return EvidenceMatch(
                requirement_id=requirement.requirement_id,
                match_strength=MatchStrength.STRONG,
                direct_evidence_ids=["EVD-PYTHON"],
                related_evidence_ids=[],
                explanation=("Approved evidence directly supports Python."),
                gap=False,
            )

        return EvidenceMatch(
            requirement_id=requirement.requirement_id,
            match_strength=MatchStrength.NONE,
            direct_evidence_ids=[],
            related_evidence_ids=[],
            explanation="No approved LangGraph evidence was found.",
            gap=True,
        )


def override_job_analysis_service() -> JobAnalysisService:
    """Return a service backed entirely by deterministic fakes."""

    return JobAnalysisService(
        requirement_extractor=FakeRequirementExtractor(),
        evidence_discovery_runner=FakeEvidenceDiscoveryRunner(),
    )


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide an API client with external services replaced."""

    app.dependency_overrides[get_job_analysis_service] = override_job_analysis_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_job_analysis_returns_evidence_grounded_result(
    client: TestClient,
) -> None:
    """A valid request should return requirements, matches and fit."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={"X-User-ID": "USER-API-001"},
        json={
            "job_id": "JOB-API-001",
            "job_description": (
                "We require strong Python and LangGraph workflow "
                "development experience."
            ),
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["job_id"] == "JOB-API-001"
    assert body["role_title"] == "Junior AI Engineer"
    assert body["fit_score"] == 55.56

    assert len(body["requirements"]) == 2
    assert len(body["evidence_matches"]) == 2

    assert body["evidence_matches"][0] == {
        "requirement_id": "REQ-PYTHON",
        "match_strength": "strong",
        "direct_evidence_ids": ["EVD-PYTHON"],
        "related_evidence_ids": [],
        "explanation": ("Approved evidence directly supports Python."),
        "gap": False,
    }

    assert body["evidence_matches"][1]["match_strength"] == "none"
    assert body["evidence_matches"][1]["gap"] is True

    assert [event["event"] for event in body["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "evidence_discovery_completed",
        "fit_score_calculated",
        "job_analysis_completed",
    ]


def test_short_job_description_returns_422(
    client: TestClient,
) -> None:
    """Invalid workflow input should become an API error."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={"X-User-ID": "USER-API-001"},
        json={
            "job_id": "JOB-API-002",
            "job_description": "Too short",
        },
    )

    assert response.status_code == 422
    assert "at least 20 characters" in response.json()["detail"]


def test_missing_user_header_returns_401(
    client: TestClient,
) -> None:
    """A request without trusted user context must be rejected."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-003",
            "job_description": ("We require strong Python and LangGraph experience."),
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == ("The X-User-ID header is required.")


def test_client_cannot_submit_matched_requirement_ids(
    client: TestClient,
) -> None:
    """The client must not control evidence-match classifications."""

    response = client.post(
        "/api/v1/job-analysis",
        headers={"X-User-ID": "USER-API-001"},
        json={
            "job_id": "JOB-API-004",
            "job_description": ("We require strong Python and LangGraph experience."),
            "matched_requirement_ids": ["REQ-PYTHON"],
        },
    )

    assert response.status_code == 422

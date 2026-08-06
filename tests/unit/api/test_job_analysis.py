"""Tests for the job-analysis API."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from careerops_agent_engine.api.dependencies import (
    get_job_analysis_service,
)
from careerops_agent_engine.application.services.job_analysis import (
    JobAnalysisService,
)
from careerops_agent_engine.domain.enums import RequirementCategory
from careerops_agent_engine.domain.models.job import (
    JobRequirement,
    JobRequirementExtraction,
)
from careerops_agent_engine.main import app


class FakeRequirementExtractor:
    """Deterministic extraction adapter used by API tests."""

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


def override_job_analysis_service() -> JobAnalysisService:
    """Return a job-analysis service backed by the fake extractor."""

    return JobAnalysisService(requirement_extractor=FakeRequirementExtractor())


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Provide an API client with external model calls replaced."""

    app.dependency_overrides[get_job_analysis_service] = override_job_analysis_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_job_analysis_returns_structured_result(
    client: TestClient,
) -> None:
    """A valid request should return requirements and weighted fit."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-001",
            "job_description": (
                "We require strong Python and LangGraph workflow "
                "development experience."
            ),
            "matched_requirement_ids": ["REQ-PYTHON"],
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "completed"
    assert body["job_id"] == "JOB-API-001"
    assert body["role_title"] == "Junior AI Engineer"
    assert body["fit_score"] == 55.56
    assert len(body["requirements"]) == 2

    assert [event["event"] for event in body["audit_events"]] == [
        "job_input_validated",
        "requirements_extracted",
        "fit_score_calculated",
        "job_analysis_completed",
    ]


def test_short_job_description_returns_422(
    client: TestClient,
) -> None:
    """Invalid workflow input should become an API validation error."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-002",
            "job_description": "Too short",
            "matched_requirement_ids": [],
        },
    )

    assert response.status_code == 422
    assert "at least 20 characters" in response.json()["detail"]


def test_unknown_match_identifier_returns_422(
    client: TestClient,
) -> None:
    """Unknown requirement IDs must not affect the calculated score."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-003",
            "job_description": (
                "We require strong Python and LangGraph workflow "
                "development experience."
            ),
            "matched_requirement_ids": ["REQ-NOT-REAL"],
        },
    )

    assert response.status_code == 422
    assert "Unknown matched requirement identifiers" in (response.json()["detail"])


def test_duplicate_match_identifiers_return_422(
    client: TestClient,
) -> None:
    """The API contract should reject duplicate match identifiers."""

    response = client.post(
        "/api/v1/job-analysis",
        json={
            "job_id": "JOB-API-004",
            "job_description": (
                "We require strong Python and LangGraph workflow "
                "development experience."
            ),
            "matched_requirement_ids": [
                "REQ-PYTHON",
                "REQ-PYTHON",
            ],
        },
    )

    assert response.status_code == 422

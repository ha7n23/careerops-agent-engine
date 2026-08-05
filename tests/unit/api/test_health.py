"""Tests for the service-health endpoint."""

from fastapi.testclient import TestClient

from careerops_agent_engine.main import app

client = TestClient(app)


def test_health_check_returns_service_information() -> None:
    """The health endpoint should return stable service metadata."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "CareerOps Agent Engine",
        "version": "0.1.0",
        "environment": "development",
    }

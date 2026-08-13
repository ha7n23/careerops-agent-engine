"""Tests for CareerOps HTTP security boundaries."""

from fastapi.testclient import TestClient
from pydantic import SecretStr

from careerops_agent_engine.core.config import (
    Settings,
    get_settings,
)
from careerops_agent_engine.main import (
    create_app,
)

TEST_SERVICE_KEY_VALUE = "careerops-http-security-key-1234567890"


def test_security_headers_are_added() -> None:
    """API responses should carry baseline security headers."""

    application = create_app()

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["X-Frame-Options"] == "DENY"

    assert response.headers["Referrer-Policy"] == "no-referrer"

    assert response.headers["Cache-Control"] == "no-store"


def test_untrusted_host_is_rejected() -> None:
    """Forged Host headers must not reach the application."""

    application = create_app()

    with TestClient(application) as client:
        response = client.get(
            "/health",
            headers={
                "Host": "attacker.example",
            },
        )

    assert response.status_code == 400


def test_unhandled_exception_is_sanitized() -> None:
    """Unexpected failures must not expose internals."""

    application = create_app()

    @application.get("/test-unhandled-error")
    def raise_unhandled_error() -> None:
        raise RuntimeError("sensitive internal implementation detail")

    with TestClient(
        application,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/test-unhandled-error")

    assert response.status_code == 500

    assert response.json() == {"detail": ("An internal server error occurred.")}

    assert "sensitive internal implementation detail" not in response.text


def test_error_response_keeps_security_headers() -> None:
    """Security headers should remain on failure responses."""

    application = create_app()

    @application.get("/test-security-error")
    def raise_unhandled_error() -> None:
        raise RuntimeError("must remain private")

    with TestClient(
        application,
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/test-security-error")

    assert response.status_code == 500

    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert response.headers["Cache-Control"] == "no-store"


def test_health_remains_public_in_service_key_mode() -> None:
    """Liveness must remain usable by infrastructure."""

    application = create_app()

    application.dependency_overrides[get_settings] = lambda: Settings(
        environment="development",
        auth_mode="service_key",
        service_api_key=SecretStr(TEST_SERVICE_KEY_VALUE),
    )

    with TestClient(application) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_business_endpoint_requires_service_key() -> None:
    """Business routes must enforce service authentication."""

    application = create_app()

    application.dependency_overrides[get_settings] = lambda: Settings(
        environment="development",
        auth_mode="service_key",
        service_api_key=SecretStr(TEST_SERVICE_KEY_VALUE),
    )

    with TestClient(application) as client:
        response = client.get(
            "/api/v1/cv-versions/CVV-SECURITY-TEST",
            headers={
                "X-User-ID": "USER-001",
            },
        )

    assert response.status_code == 401

    assert response.json() == {"detail": ("Missing or invalid service credentials.")}

    assert response.headers["X-Content-Type-Options"] == "nosniff"

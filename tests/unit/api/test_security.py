"""Tests for the CareerOps API authentication boundary."""

from typing import Annotated

import pytest
from fastapi import (
    Depends,
    FastAPI,
)
from fastapi.testclient import TestClient
from pydantic import (
    SecretStr,
    ValidationError,
)

from careerops_agent_engine.api.security import (
    get_authenticated_user_id,
)
from careerops_agent_engine.core.config import (
    Settings,
    get_settings,
)

TEST_SERVICE_KEY_VALUE = "careerops-test-service-key-1234567890"

TEST_SERVICE_KEY = SecretStr(TEST_SERVICE_KEY_VALUE)


def _client(
    settings: Settings,
) -> TestClient:
    """Create an isolated protected test endpoint."""

    application = FastAPI()

    application.dependency_overrides[get_settings] = lambda: settings

    @application.get("/protected")
    def protected(
        user_id: Annotated[
            str,
            Depends(get_authenticated_user_id),
        ],
    ) -> dict[str, str]:
        return {
            "user_id": user_id,
        }

    return TestClient(application)


def test_development_mode_accepts_user_header() -> None:
    """Development mode should preserve local workflow."""

    client = _client(
        Settings(
            environment="development",
            auth_mode="development",
        )
    )

    response = client.get(
        "/protected",
        headers={
            "X-User-ID": "USER-001",
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "user_id": "USER-001",
    }


def test_missing_user_header_returns_401() -> None:
    """User scope must always be present."""

    client = _client(
        Settings(
            environment="development",
            auth_mode="development",
        )
    )

    response = client.get("/protected")

    assert response.status_code == 401


def test_service_key_mode_rejects_missing_key() -> None:
    """Protected mode must authenticate the caller."""

    client = _client(
        Settings(
            environment="development",
            auth_mode="service_key",
            service_api_key=TEST_SERVICE_KEY,
        )
    )

    response = client.get(
        "/protected",
        headers={
            "X-User-ID": "USER-001",
        },
    )

    assert response.status_code == 401


def test_service_key_mode_rejects_wrong_key() -> None:
    """Incorrect service credentials must not pass."""

    client = _client(
        Settings(
            environment="development",
            auth_mode="service_key",
            service_api_key=TEST_SERVICE_KEY,
        )
    )

    response = client.get(
        "/protected",
        headers={
            "X-User-ID": "USER-001",
            "X-CareerOps-Service-Key": ("wrong-service-key"),
        },
    )

    assert response.status_code == 401

    assert TEST_SERVICE_KEY_VALUE not in response.text


def test_service_key_mode_accepts_correct_key() -> None:
    """Trusted service credentials should pass."""

    client = _client(
        Settings(
            environment="development",
            auth_mode="service_key",
            service_api_key=TEST_SERVICE_KEY,
        )
    )

    response = client.get(
        "/protected",
        headers={
            "X-User-ID": "USER-001",
            "X-CareerOps-Service-Key": (TEST_SERVICE_KEY_VALUE),
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "user_id": "USER-001",
    }


@pytest.mark.parametrize(
    "user_id",
    [
        " USER-001",
        "USER-001 ",
        "USER/001",
        "../USER-001",
        "A" * 65,
    ],
)
def test_malformed_user_id_is_rejected(
    user_id: str,
) -> None:
    """Invalid user scopes must not reach repositories."""

    client = _client(
        Settings(
            environment="development",
            auth_mode="development",
        )
    )

    response = client.get(
        "/protected",
        headers={
            "X-User-ID": user_id,
        },
    )

    assert response.status_code == 401


def test_service_key_mode_requires_secret() -> None:
    """Unsafe service-key configuration must fail."""

    with pytest.raises(
        ValidationError,
        match=("service_key authentication requires"),
    ):
        Settings(
            environment="development",
            auth_mode="service_key",
            service_api_key=None,
        )


@pytest.mark.parametrize(
    "environment",
    [
        "staging",
        "production",
    ],
)
def test_protected_environment_rejects_development_auth(
    environment: str,
) -> None:
    """Deployed environments cannot trust user headers alone."""

    with pytest.raises(
        ValidationError,
        match=("require service_key authentication"),
    ):
        Settings(
            environment=environment,  # type: ignore[arg-type]
            auth_mode="development",
        )


def test_production_rejects_debug_mode() -> None:
    """Production must never expose debug behaviour."""

    with pytest.raises(
        ValidationError,
        match="must not enable debug mode",
    ):
        Settings(
            environment="production",
            debug=True,
            auth_mode="service_key",
            service_api_key=TEST_SERVICE_KEY,
        )


def test_production_rejects_wildcard_trusted_host() -> None:
    """Production must not trust arbitrary Host headers."""

    with pytest.raises(
        ValidationError,
        match="must not trust every Host header",
    ):
        Settings(
            environment="production",
            auth_mode="service_key",
            service_api_key=TEST_SERVICE_KEY,
            trusted_hosts=["*"],
        )


def test_service_key_is_masked_in_settings_repr() -> None:
    """Configured secrets should not appear in diagnostics."""

    settings = Settings(
        environment="development",
        auth_mode="service_key",
        service_api_key=TEST_SERVICE_KEY,
    )

    assert TEST_SERVICE_KEY_VALUE not in repr(settings)

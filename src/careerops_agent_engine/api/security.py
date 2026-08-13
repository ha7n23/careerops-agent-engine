"""Authentication boundary for CareerOps API requests."""

from re import compile as compile_pattern
from secrets import compare_digest
from typing import Annotated

from fastapi import (
    Depends,
    Header,
    HTTPException,
    Security,
    status,
)
from fastapi.security import APIKeyHeader

from careerops_agent_engine.core.config import (
    Settings,
    get_settings,
)

SERVICE_KEY_HEADER_NAME = "X-CareerOps-Service-Key"

_USER_ID_PATTERN = compile_pattern(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")

_service_key_header = APIKeyHeader(
    name=SERVICE_KEY_HEADER_NAME,
    scheme_name="CareerOpsServiceKey",
    description=("Internal service credential used by trusted CareerOps callers."),
    auto_error=False,
)

SettingsDependency = Annotated[
    Settings,
    Depends(get_settings),
]

ServiceKeyDependency = Annotated[
    str | None,
    Security(_service_key_header),
]


def get_authenticated_user_id(
    settings: SettingsDependency,
    service_key: ServiceKeyDependency,
    x_user_id: Annotated[
        str | None,
        Header(alias="X-User-ID"),
    ] = None,
) -> str:
    """Authenticate the caller and return its user scope."""

    if settings.auth_mode == "service_key":
        _verify_service_key(
            supplied_key=service_key,
            settings=settings,
        )

    return _validate_user_id(x_user_id)


def _verify_service_key(
    *,
    supplied_key: str | None,
    settings: Settings,
) -> None:
    """Reject missing or incorrect service credentials."""

    configured_key = settings.service_api_key

    if supplied_key is None or configured_key is None:
        raise _authentication_error()

    expected_key = configured_key.get_secret_value()

    if not compare_digest(
        supplied_key.encode("utf-8"),
        expected_key.encode("utf-8"),
    ):
        raise _authentication_error()


def _validate_user_id(
    user_id: str | None,
) -> str:
    """Validate the authenticated user scope header."""

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=("The X-User-ID header is required."),
        )

    if user_id != user_id.strip() or _USER_ID_PATTERN.fullmatch(user_id) is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=("The X-User-ID header is invalid."),
        )

    return user_id


def _authentication_error() -> HTTPException:
    """Return one generic service-authentication failure."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=("Missing or invalid service credentials."),
        headers={
            "WWW-Authenticate": "APIKey",
        },
    )

"""Health, readiness, and service-information endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from careerops_agent_engine.api.dependencies import (
    get_database_engine,
)
from careerops_agent_engine.api.schemas.health import (
    HealthResponse,
    ReadinessResponse,
)
from careerops_agent_engine.core.config import get_settings
from careerops_agent_engine.infrastructure.database.readiness import (
    DatabaseSchemaNotReadyError,
    check_database_readiness,
)

router = APIRouter(tags=["System"])

DatabaseEngineDependency = Annotated[
    Engine,
    Depends(get_database_engine),
]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service liveness",
)
def health_check() -> HealthResponse:
    """Return process-level service health without external dependencies."""

    settings = get_settings()

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Check service readiness",
)
def readiness_check(
    engine: DatabaseEngineDependency,
) -> ReadinessResponse:
    """Return ready only when required infrastructure is reachable."""

    try:
        check_database_readiness(engine)

    except (
        SQLAlchemyError,
        DatabaseSchemaNotReadyError,
    ) as exc:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Service dependencies are not ready."),
        ) from exc

    return ReadinessResponse(
        status="ready",
        database="ok",
    )

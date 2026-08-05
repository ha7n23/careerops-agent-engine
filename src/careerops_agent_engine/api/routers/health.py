"""Health and service-information endpoints."""

from fastapi import APIRouter

from careerops_agent_engine.api.schemas.health import HealthResponse
from careerops_agent_engine.core.config import get_settings

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service health",
)
def health_check() -> HealthResponse:
    """Return the current service status."""

    settings = get_settings()

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

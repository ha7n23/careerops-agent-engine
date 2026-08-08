"""FastAPI application entry point."""

from fastapi import FastAPI

from careerops_agent_engine.api.routers.cv_documents import (
    router as cv_documents_router,
)
from careerops_agent_engine.api.routers.health import router as health_router
from careerops_agent_engine.api.routers.job_analysis import (
    router as job_analysis_router,
)
from careerops_agent_engine.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the CareerOps FastAPI application."""

    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        description=(
            "Stateful, evidence-grounded agent engine for the CareerOps platform."
        ),
    )

    application.include_router(health_router)
    application.include_router(job_analysis_router)
    application.include_router(cv_documents_router)
    return application


app = create_app()

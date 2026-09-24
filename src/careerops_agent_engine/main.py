"""FastAPI application entry point."""

from fastapi import FastAPI
from starlette.middleware.trustedhost import (
    TrustedHostMiddleware,
)

from careerops_agent_engine.api.errors import (
    register_exception_handlers,
)
from careerops_agent_engine.api.middleware import (
    SecurityHeadersMiddleware,
)
from careerops_agent_engine.api.routers.cv_documents import (
    router as cv_documents_router,
)
from careerops_agent_engine.api.routers.cv_evidence_reviews import (
    router as cv_evidence_reviews_router,
)
from careerops_agent_engine.api.routers.cv_versions import (
    router as cv_versions_router,
)
from careerops_agent_engine.api.routers.evidence import (
    router as evidence_router,
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

    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.trusted_hosts,
    )

    application.add_middleware(
        SecurityHeadersMiddleware,
    )

    register_exception_handlers(application)

    application.include_router(health_router)
    application.include_router(job_analysis_router)
    application.include_router(cv_documents_router)
    application.include_router(cv_evidence_reviews_router)
    application.include_router(evidence_router)
    application.include_router(cv_versions_router)

    return application


app = create_app()

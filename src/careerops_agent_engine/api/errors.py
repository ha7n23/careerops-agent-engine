"""Safe top-level API error handling."""

import logging

from fastapi import (
    FastAPI,
    Request,
    status,
)
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def register_exception_handlers(
    application: FastAPI,
) -> None:
    """Register sanitized API exception handling."""

    @application.exception_handler(Exception)
    async def handle_unexpected_exception(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        logger.error(
            ("Unhandled API exception method=%s path=%s"),
            request.method,
            request.url.path,
            exc_info=(
                type(error),
                error,
                error.__traceback__,
            ),
        )

        return JSONResponse(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            content={"detail": ("An internal server error occurred.")},
            headers={
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            },
        )

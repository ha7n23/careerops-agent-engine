"""Response models for service-health endpoints."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Current status and identity of the running service."""

    status: Literal["ok"]
    service: str
    version: str
    environment: str

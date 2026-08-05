"""Shared configuration for CareerOps domain models."""

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base model enforcing strict, predictable domain data."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

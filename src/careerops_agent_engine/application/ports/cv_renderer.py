"""Application port for deterministic CV rendering."""

from dataclasses import dataclass
from typing import Protocol

from careerops_agent_engine.domain.enums import (
    CVArtifactFormat,
)
from careerops_agent_engine.domain.models.cv_version import (
    CVVersion,
)


@dataclass(frozen=True)
class RenderedCVDocument:
    """Rendered CV bytes before secure artifact storage."""

    artifact_format: CVArtifactFormat
    media_type: str
    filename: str
    data: bytes

    def __post_init__(self) -> None:
        """Reject unusable renderer output."""

        if not self.filename:
            raise ValueError("Rendered CV filename cannot be blank.")

        if not self.media_type:
            raise ValueError("Rendered CV media type cannot be blank.")

        if not self.data:
            raise ValueError("Rendered CV bytes cannot be empty.")


class CVTemplateRenderer(Protocol):
    """Render one immutable CV version through a named template."""

    @property
    def template_id(self) -> str:
        """Return the renderer template identifier."""

        ...

    @property
    def template_version(self) -> str:
        """Return the renderer template version."""

        ...

    def render(
        self,
        *,
        version: CVVersion,
    ) -> RenderedCVDocument:
        """Render deterministic document bytes for one CV version."""

        ...

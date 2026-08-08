"""Application port for structured CV evidence extraction."""

from typing import Protocol

from careerops_agent_engine.domain.models.document import (
    ParsedCVDocument,
)
from careerops_agent_engine.domain.models.evidence import (
    CareerEvidenceCandidate,
)


class CVEvidenceExtractor(Protocol):
    """Extract candidate evidence from a parsed CV."""

    def extract(
        self,
        *,
        document: ParsedCVDocument,
    ) -> list[CareerEvidenceCandidate]:
        """Return model-extracted evidence candidates."""

        ...

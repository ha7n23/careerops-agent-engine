"""Load version-controlled CareerOps evaluation datasets."""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from careerops_agent_engine.evaluation.models import (
    RequirementExtractionEvalDataset,
)


def compute_dataset_sha256(
    path: Path,
) -> str:
    """Return the SHA-256 fingerprint of one committed dataset file."""

    return sha256(path.read_bytes()).hexdigest()


def load_requirement_extraction_dataset(
    path: Path,
) -> RequirementExtractionEvalDataset:
    """Load and validate one requirement-extraction dataset."""

    raw_data: Any = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    return RequirementExtractionEvalDataset.model_validate(raw_data)

"""Privacy-conscious LangSmith run configuration for CareerOps."""

from collections.abc import Mapping, Sequence

from langchain_core.runnables import RunnableConfig

type TraceMetadataValue = str | int | float | bool

SAFE_LANGSMITH_METADATA_KEYS = frozenset(
    {
        "component",
        "workflow",
        "workflow_version",
        "thread_id",
        "prompt_version",
        "ls_model_name",
        "requirement_id",
        "requirement_count",
        "proposal_id",
        "document_id",
    }
)

MAX_TRACE_RUN_NAME_LENGTH = 128
MAX_TRACE_TAG_LENGTH = 64
MAX_TRACE_METADATA_STRING_LENGTH = 128


def build_langsmith_run_config(
    *,
    run_name: str,
    tags: Sequence[str],
    metadata: Mapping[
        str,
        TraceMetadataValue,
    ],
) -> RunnableConfig:
    """Build one privacy-guarded LangSmith RunnableConfig."""

    clean_run_name = run_name.strip()

    if not clean_run_name:
        raise ValueError("LangSmith run name cannot be blank.")

    if len(clean_run_name) > MAX_TRACE_RUN_NAME_LENGTH:
        raise ValueError("LangSmith run name is too long.")

    unsafe_keys = set(metadata) - SAFE_LANGSMITH_METADATA_KEYS

    if unsafe_keys:
        raise ValueError(
            "Unsafe LangSmith metadata keys: " + ", ".join(sorted(unsafe_keys))
        )

    clean_metadata: dict[
        str,
        TraceMetadataValue,
    ] = {}

    for key, value in metadata.items():
        if isinstance(
            value,
            str,
        ):
            clean_value = value.strip()

            if not clean_value:
                raise ValueError("LangSmith metadata string values cannot be blank.")

            if len(clean_value) > MAX_TRACE_METADATA_STRING_LENGTH:
                raise ValueError("LangSmith metadata string value is too long.")

            clean_metadata[key] = clean_value

            continue

        if not isinstance(
            value,
            (
                int,
                float,
                bool,
            ),
        ):
            raise ValueError("LangSmith metadata values must be scalar.")

        clean_metadata[key] = value

    clean_tags: list[str] = []
    seen_tags: set[str] = set()

    for tag in (
        "careerops",
        *tags,
    ):
        clean_tag = tag.strip()

        if not clean_tag:
            raise ValueError("LangSmith tags cannot be blank.")

        if len(clean_tag) > MAX_TRACE_TAG_LENGTH:
            raise ValueError("LangSmith tag is too long.")

        if clean_tag in seen_tags:
            continue

        seen_tags.add(clean_tag)

        clean_tags.append(clean_tag)

    return {
        "run_name": clean_run_name,
        "tags": clean_tags,
        "metadata": clean_metadata,
    }

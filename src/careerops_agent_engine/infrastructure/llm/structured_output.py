"""Provider-neutral structured-output fallback construction."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable


def create_structured_output_runnable(
    *,
    model: BaseChatModel,
    schema: dict[str, Any],
    fallback_model: BaseChatModel | None = None,
    fallback_exceptions: tuple[type[BaseException], ...] = (),
) -> Runnable[Any, Any]:
    """Create structured output with an optional bounded fallback model."""

    primary = model.with_structured_output(
        schema=schema,
        method="json_schema",
    )

    if fallback_model is None:
        return primary

    if not fallback_exceptions:
        raise ValueError(
            "A structured-output fallback requires explicit exception types."
        )

    fallback = fallback_model.with_structured_output(
        schema=schema,
        method="json_schema",
    )

    return primary.with_fallbacks(
        [fallback],
        exceptions_to_handle=fallback_exceptions,
    )

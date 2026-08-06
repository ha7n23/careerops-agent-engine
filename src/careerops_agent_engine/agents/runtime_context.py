"""Trusted runtime context supplied to CareerOps agents."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvidenceAgentContext:
    """Authenticated context for one evidence-discovery run."""

    user_id: str

    def __post_init__(self) -> None:
        """Normalise and validate the trusted user identifier."""

        normalised_user_id = self.user_id.strip()

        if not normalised_user_id:
            raise ValueError("User identifier must not be empty.")

        object.__setattr__(
            self,
            "user_id",
            normalised_user_id,
        )

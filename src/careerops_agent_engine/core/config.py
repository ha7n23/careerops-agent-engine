"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CareerOps application settings."""

    app_name: str = "CareerOps Agent Engine"
    app_version: str = "0.1.0"
    environment: Literal[
        "development",
        "test",
        "staging",
        "production",
    ] = "development"

    debug: bool = False

    trusted_hosts: list[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
            "testserver",
        ]
    )

    auth_mode: Literal[
        "development",
        "service_key",
    ] = "development"

    service_api_key: SecretStr | None = None

    llm_provider: Literal["google"] = "google"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_requests_per_minute: float = Field(
        default=4.0,
        gt=0.0,
        le=600.0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CAREEROPS_",
        extra="ignore",
    )

    evidence_agent_max_model_calls: int = Field(
        default=6,
        ge=2,
        le=12,
    )
    evidence_agent_max_tool_calls: int = Field(
        default=4,
        ge=1,
        le=10,
    )
    evidence_agent_max_search_calls: int = Field(
        default=3,
        ge=1,
        le=6,
    )
    evidence_agent_recursion_limit: int = Field(
        default=50,
        ge=20,
        le=50,
    )

    database_url: str = (
        "postgresql+psycopg://careerops:careerops_local_dev@127.0.0.1:5432/careerops"
    )
    langgraph_database_uri: str = (
        "postgresql://careerops:careerops_local_dev@127.0.0.1:5432/careerops"
    )

    database_pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
    )
    database_max_overflow: int = Field(
        default=10,
        ge=0,
        le=50,
    )

    document_upload_max_bytes: int = Field(
        default=10 * 1024 * 1024,
        ge=1024,
        le=25 * 1024 * 1024,
    )

    document_storage_root: Path = Path(".careerops_data/documents")

    artifact_storage_root: Path = Path(".careerops_data/artifacts")

    libreoffice_executable: str = "soffice"

    pdf_conversion_timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        le=300.0,
    )

    @model_validator(mode="after")
    def validate_authentication_configuration(
        self,
    ) -> Self:
        """Reject unsafe authentication configuration."""

        protected_environment = self.environment in {
            "staging",
            "production",
        }

        if protected_environment and self.debug:
            raise ValueError("Staging and production must not enable debug mode.")

        if protected_environment and "*" in self.trusted_hosts:
            raise ValueError("Staging and production must not trust every Host header.")

        if protected_environment and self.auth_mode != "service_key":
            raise ValueError(
                "Staging and production require service_key authentication."
            )

        if self.auth_mode == "service_key":
            if self.service_api_key is None:
                raise ValueError(
                    "service_key authentication requires CAREEROPS_SERVICE_API_KEY."
                )

            secret = self.service_api_key.get_secret_value()

            if len(secret) < 32:
                raise ValueError(
                    "CAREEROPS_SERVICE_API_KEY must contain at least 32 characters."
                )

            if secret != secret.strip():
                raise ValueError(
                    "CAREEROPS_SERVICE_API_KEY must not contain "
                    "leading or trailing whitespace."
                )

        return self


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""

    return Settings()

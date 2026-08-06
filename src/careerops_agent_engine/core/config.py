"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CareerOps application settings."""

    app_name: str = "CareerOps Agent Engine"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    llm_model: str = "gemini-3.5-flash"
    llm_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)

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


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""

    return Settings()

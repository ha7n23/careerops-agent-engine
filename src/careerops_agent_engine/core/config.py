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


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""

    return Settings()

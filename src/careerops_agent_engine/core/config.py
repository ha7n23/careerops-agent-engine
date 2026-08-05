"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CareerOps application settings."""

    app_name: str = "CareerOps Agent Engine"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CAREEROPS_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance."""

    return Settings()

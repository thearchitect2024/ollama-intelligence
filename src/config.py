"""
Configuration management for Contributor Intelligence Platform.
Uses Pydantic Settings for type-safe configuration with environment variable support.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # DATABASE CONFIGURATION
    # ==========================================================================
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "contributor_intelligence"
    postgres_user: str = "postgres"
    postgres_password: str

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ==========================================================================
    # ACTIVITY ANALYSIS CONFIGURATION
    # ==========================================================================
    activity_window_days: int = 90
    weeks_in_90_days: int = 13
    min_hours_active: float = 1.0
    top_projects_count: int = 5
    db_batch_size: int = 10  # Number of intelligence updates to batch before DB commit

    # ==========================================================================
    # LLM CONFIGURATION (Ollama - Running Locally)
    # ==========================================================================
    ollama_model: str = "qwen2.5:7b-instruct-q4_0"  # Using locally available model
    ollama_base_url: str = "http://localhost:11434"  # Ollama running on localhost
    max_tokens: int = 320  # Increased from 256 to ensure room for skills section
    temperature: float = 0.05  # Lowered from 0.1 for better format compliance
    top_p: float = 0.9
    max_concurrent_llm: int = 10  # Process 10 profiles concurrently for async batch processing

    # Summary validation
    summary_min_words: int = 140
    summary_max_words: int = 170

    # ==========================================================================
    # LOGGING CONFIGURATION
    # ==========================================================================
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ==========================================================================
    # APPLICATION CONFIGURATION
    # ==========================================================================
    app_env: str = "development"
    debug: bool = False

    # Model configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env.lower() == "development"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get or create settings singleton instance.

    Returns:
        Settings: Application settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment.
    Useful for testing.

    Returns:
        Settings: Fresh settings instance
    """
    global _settings
    _settings = Settings()
    return _settings

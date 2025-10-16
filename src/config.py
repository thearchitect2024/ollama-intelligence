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
    # DATABASE CONFIGURATION (SQLite)
    # ==========================================================================
    database_url: str = "sqlite:////tmp/contributor_intelligence.db"
    
    def get_database_url(self) -> str:
        """Get SQLite database URL."""
        return self.database_url

    # ==========================================================================
    # ACTIVITY ANALYSIS CONFIGURATION
    # ==========================================================================
    activity_window_days: int = 90
    weeks_in_90_days: int = 13
    min_hours_active: float = 1.0
    top_projects_count: int = 5
    db_batch_size: int = 10  # Number of intelligence updates to batch before DB commit

    # ==========================================================================
    # LLM CONFIGURATION (Embedded GPU - Qwen2.5 7B)
    # ==========================================================================
    # Model runs in-process on GPU with 4-bit quantization, micro-batching, and Flash Attention
    ollama_model: str = "Qwen/Qwen2.5-7B-Instruct"  # Embedded model (no longer Ollama REST)
    ollama_base_url: str = "embedded"  # Kept for backward compatibility but unused
    max_tokens: int = 320  # Max new tokens to generate
    temperature: float = 0.05  # Low temperature for consistent format compliance
    top_p: float = 0.9  # Nucleus sampling parameter
    max_concurrent_llm: int = 1  # Worker threads for micro-batch collection (1-2 recommended for optimal bucketing)
    
    # GPU Optimization Parameters (new embedded engine)
    infer_concurrency: int = 3  # Max concurrent GPU batches (semaphore slots)
    micro_batch_size: int = 6  # Target prompts per batch
    batch_latency_ms: int = 100  # Max wait time to collect batch (ms)
    prefill_batch_tokens: int = 4096  # Max input tokens per prefill batch
    decode_concurrency: int = 8  # Max concurrent decode operations
    use_flash_attention: bool = True  # Enable FlashAttention-2 (if available)
    enable_compile: bool = True  # Enable torch.compile optimization

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

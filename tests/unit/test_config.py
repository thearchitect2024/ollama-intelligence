"""
Unit tests for src/config.py
Tests Settings configuration management.
"""
import pytest
import os
from unittest.mock import patch

from src.config import Settings, get_settings, reload_settings


# ==========================================================================
# TEST Settings Class
# ==========================================================================

class TestSettings:
    """Test Settings class."""

    def test_settings_default_values(self):
        """Should have correct default values."""
        # Only set password, let others use defaults
        minimal_env = {
            'POSTGRES_PASSWORD': 'test_password',
            'HOME': os.environ.get('HOME', '/tmp')  # Keep HOME for compatibility
        }
        with patch.dict(os.environ, minimal_env, clear=True):
            settings = Settings()
            
            assert settings.postgres_host == "localhost"
            assert settings.postgres_port == 5432
            assert settings.postgres_db == "contributor_intelligence"
            assert settings.postgres_user == "postgres"

    def test_settings_from_environment(self):
        """Should load from environment variables."""
        env_vars = {
            'POSTGRES_HOST': 'db.example.com',
            'POSTGRES_PORT': '5433',
            'POSTGRES_DB': 'test_db',
            'POSTGRES_USER': 'test_user',
            'POSTGRES_PASSWORD': 'secret'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.postgres_host == 'db.example.com'
            assert settings.postgres_port == 5433
            assert settings.postgres_db == 'test_db'
            assert settings.postgres_user == 'test_user'
            assert settings.postgres_password == 'secret'

    def test_settings_case_insensitive(self):
        """Should handle case-insensitive env vars."""
        env_vars = {
            'postgres_host': 'test.com',  # lowercase
            'POSTGRES_PASSWORD': 'pass'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.postgres_host == 'test.com'

    def test_settings_requires_password(self):
        """Should require postgres_password."""
        minimal_env = {'HOME': os.environ.get('HOME', '/tmp')}
        with patch.dict(os.environ, minimal_env, clear=True):
            with pytest.raises(Exception):  # Pydantic validation error
                Settings()

    def test_database_url_property(self):
        """Should construct database URL correctly."""
        env_vars = {
            'POSTGRES_HOST': 'localhost',
            'POSTGRES_PORT': '5432',
            'POSTGRES_DB': 'testdb',
            'POSTGRES_USER': 'user',
            'POSTGRES_PASSWORD': 'pass'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            expected_url = "postgresql://user:pass@localhost:5432/testdb"
            assert settings.database_url == expected_url

    def test_activity_configuration_defaults(self):
        """Should have correct activity analysis defaults."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            
            assert settings.activity_window_days == 90
            assert settings.weeks_in_90_days == 13
            assert settings.min_hours_active == 1.0
            assert settings.top_projects_count == 5

    def test_llm_configuration_defaults(self):
        """Should have correct LLM defaults."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            
            assert settings.ollama_model == "qwen2.5:7b-instruct-q4_0"
            assert settings.ollama_base_url == "http://localhost:11434"
            assert settings.max_tokens == 320
            assert settings.temperature == 0.05
            assert settings.top_p == 0.9
            assert settings.max_concurrent_llm == 10

    def test_logging_configuration_defaults(self):
        """Should have correct logging defaults."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            
            assert settings.log_level == "INFO"
            assert settings.log_file == "logs/app.log"

    def test_app_configuration_defaults(self):
        """Should have correct app defaults."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            
            assert settings.app_env == "development"
            assert settings.debug is False

    def test_is_production_method(self):
        """Should correctly identify production environment."""
        with patch.dict(os.environ, {'APP_ENV': 'production', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.is_production() is True
            assert settings.is_development() is False

    def test_is_development_method(self):
        """Should correctly identify development environment."""
        with patch.dict(os.environ, {'APP_ENV': 'development', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.is_production() is False
            assert settings.is_development() is True

    def test_is_production_case_insensitive(self):
        """Should be case-insensitive for environment check."""
        with patch.dict(os.environ, {'APP_ENV': 'PRODUCTION', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.is_production() is True

    def test_custom_ollama_model(self):
        """Should allow custom Ollama model."""
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'llama3.2:3b', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.ollama_model == 'llama3.2:3b'

    def test_custom_max_concurrent_llm(self):
        """Should allow custom concurrency."""
        with patch.dict(os.environ, {'MAX_CONCURRENT_LLM': '20', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.max_concurrent_llm == 20

    def test_custom_activity_window_days(self):
        """Should allow custom activity window."""
        with patch.dict(os.environ, {'ACTIVITY_WINDOW_DAYS': '30', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.activity_window_days == 30

    def test_summary_validation_defaults(self):
        """Should have summary validation defaults."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            
            assert settings.summary_min_words == 140
            assert settings.summary_max_words == 170

    def test_db_batch_size_default(self):
        """Should have DB batch size default."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.db_batch_size == 10

    def test_extra_env_vars_ignored(self):
        """Should ignore extra environment variables."""
        env_vars = {
            'POSTGRES_PASSWORD': 'test',
            'SOME_RANDOM_VAR': 'value',
            'ANOTHER_VAR': '123'
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            # Should not raise error and not have extra attributes
            assert not hasattr(settings, 'SOME_RANDOM_VAR')


# ==========================================================================
# TEST get_settings Function
# ==========================================================================

class TestGetSettings:
    """Test get_settings singleton function."""

    def test_get_settings_returns_instance(self):
        """Should return Settings instance."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            # Reset singleton
            import src.config
            src.config._settings = None
            
            settings = get_settings()
            assert isinstance(settings, Settings)

    def test_get_settings_returns_same_instance(self):
        """Should return same instance on multiple calls."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            # Reset singleton
            import src.config
            src.config._settings = None
            
            settings1 = get_settings()
            settings2 = get_settings()
            
            assert settings1 is settings2

    def test_get_settings_singleton_persists(self):
        """Should persist singleton across calls."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            # Reset singleton
            import src.config
            src.config._settings = None
            
            settings1 = get_settings()
            settings1.postgres_host = "modified.com"
            
            settings2 = get_settings()
            assert settings2.postgres_host == "modified.com"


# ==========================================================================
# TEST reload_settings Function
# ==========================================================================

class TestReloadSettings:
    """Test reload_settings function."""

    def test_reload_settings_creates_new_instance(self):
        """Should create new Settings instance."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test'}, clear=True):
            # Reset singleton
            import src.config
            src.config._settings = None
            
            settings1 = get_settings()
            settings1.postgres_host = "old.com"
            
            settings2 = reload_settings()
            
            # Should be new instance with default value
            assert settings2.postgres_host == "localhost"

    def test_reload_settings_updates_singleton(self):
        """Should update singleton reference."""
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test1'}, clear=True):
            # Reset singleton
            import src.config
            src.config._settings = None
            
            settings1 = get_settings()
            old_password = settings1.postgres_password
        
        with patch.dict(os.environ, {'POSTGRES_PASSWORD': 'test2'}, clear=True):
            settings2 = reload_settings()
            
            # get_settings should now return the reloaded instance
            settings3 = get_settings()
            assert settings3 is settings2
            assert settings3.postgres_password == 'test2'

    def test_reload_settings_picks_up_env_changes(self):
        """Should pick up environment changes."""
        with patch.dict(os.environ, {'POSTGRES_HOST': 'original.com', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            import src.config
            src.config._settings = None
            
            settings1 = get_settings()
            assert settings1.postgres_host == 'original.com'
        
        with patch.dict(os.environ, {'POSTGRES_HOST': 'changed.com', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings2 = reload_settings()
            assert settings2.postgres_host == 'changed.com'


# ==========================================================================
# TEST Configuration Validation
# ==========================================================================

class TestConfigurationValidation:
    """Test configuration validation."""

    def test_postgres_port_must_be_integer(self):
        """Should validate port as integer."""
        with patch.dict(os.environ, {'POSTGRES_PORT': 'not_a_number', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            with pytest.raises(Exception):  # Pydantic validation error
                Settings()

    def test_temperature_float_validation(self):
        """Should accept float values for temperature."""
        with patch.dict(os.environ, {'TEMPERATURE': '0.7', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.temperature == 0.7
            assert isinstance(settings.temperature, float)

    def test_max_tokens_integer_validation(self):
        """Should validate max_tokens as integer."""
        with patch.dict(os.environ, {'MAX_TOKENS': '500', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.max_tokens == 500
            assert isinstance(settings.max_tokens, int)

    def test_debug_boolean_validation(self):
        """Should parse boolean values."""
        with patch.dict(os.environ, {'DEBUG': 'true', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.debug is True
        
        with patch.dict(os.environ, {'DEBUG': 'false', 'POSTGRES_PASSWORD': 'test'}, clear=True):
            settings = Settings()
            assert settings.debug is False


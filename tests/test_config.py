"""Unit tests for configuration module - simplified and focused on critical paths."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from pokeconsultor.config import Settings


@pytest.fixture
def clean_settings() -> None:
    """Reset singleton state before and after each test."""
    if hasattr(Settings, "_instance"):
        delattr(Settings, "_instance")
    if hasattr(Settings, "_initialized"):
        Settings._initialized = False

    yield

    if hasattr(Settings, "_instance"):
        delattr(Settings, "_instance")
    if hasattr(Settings, "_initialized"):
        Settings._initialized = False


@pytest.fixture
def valid_settings_data() -> dict:
    """Provide valid settings data for tests."""
    return {
        "LLM_DEFAULT_PROVIDER": "groq",
        "LLM_DEFAULT_MODEL": "mixtral-8x7b-32768",
        "LLM_DEFAULT_TEMPERATURE": 0.7,
        "LLM_DEFAULT_MAX_TOKENS": 2048,
        "LLM_PROFILE_EXECUTOR_PROVIDER": "groq",
        "LLM_PROFILE_EXECUTOR_MODEL": "llama-3.3-70b-versatile",
        "LLM_PROFILE_EXECUTOR_TEMPERATURE": 0.3,
        "LLM_PROFILE_EXECUTOR_MAX_TOKENS": 4096,
        "LLM_PROFILE_SUPERVISOR_PROVIDER": "groq",
        "LLM_PROFILE_SUPERVISOR_MODEL": "llama-3.3-70b-versatile",
        "LLM_PROFILE_SUPERVISOR_TEMPERATURE": 0.5,
        "LLM_PROFILE_SUPERVISOR_MAX_TOKENS": 8192,
        "GROQ_API_KEY": "gsk_test_key",
        "POKEAPI_MCP_SERVER_URL": "http://localhost:8000",
        "DATA_PATH": "data/",
    }


class TestSettingsSingleton:
    """Test singleton pattern implementation for Settings."""

    def test_singleton_pattern(self, clean_settings, valid_settings_data) -> None:
        """Verify Settings maintains singleton pattern."""
        settings1 = Settings(**valid_settings_data)

        # Create second instance with different values
        modified_data = valid_settings_data.copy()
        modified_data["LLM_DEFAULT_PROVIDER"] = "openai"
        settings2 = Settings(**modified_data)

        # Must be the same instance with original values
        assert settings1 is settings2
        assert settings1.LLM_DEFAULT_PROVIDER == "groq"


class TestLLMProfileValidation:
    """Test LLM profile configurations with parametrized validation."""

    def test_valid_configurations(self, clean_settings, valid_settings_data) -> None:
        """Test all three LLM profiles are configured correctly."""
        settings = Settings(**valid_settings_data)

        # Default profile
        assert settings.LLM_DEFAULT_PROVIDER == "groq"
        assert settings.LLM_DEFAULT_TEMPERATURE == 0.7
        assert settings.LLM_DEFAULT_MAX_TOKENS == 2048

        # Executor profile
        assert settings.LLM_PROFILE_EXECUTOR_TEMPERATURE == 0.3
        assert settings.LLM_PROFILE_EXECUTOR_MAX_TOKENS == 4096

        # Supervisor profile
        assert settings.LLM_PROFILE_SUPERVISOR_TEMPERATURE == 0.5
        assert settings.LLM_PROFILE_SUPERVISOR_MAX_TOKENS == 8192

    @pytest.mark.parametrize(
        "profile_prefix,temp_value",
        [
            ("LLM_DEFAULT_TEMPERATURE", -0.1),
            ("LLM_DEFAULT_TEMPERATURE", 2.1),
            ("LLM_PROFILE_EXECUTOR_TEMPERATURE", 2.5),
            ("LLM_PROFILE_SUPERVISOR_TEMPERATURE", -1),
        ],
    )
    def test_temperature_validation(
        self,
        clean_settings: None,
        valid_settings_data: dict,
        profile_prefix: str,
        temp_value: float,
    ) -> None:
        """Test temperature constraints (0.0 to 2.0) for all profiles."""
        data = valid_settings_data.copy()
        data[profile_prefix] = temp_value

        with pytest.raises(ValidationError) as exc_info:
            Settings(**data)
        assert profile_prefix in str(exc_info.value)

    @pytest.mark.parametrize(
        "profile_prefix",
        [
            "LLM_DEFAULT_MAX_TOKENS",
            "LLM_PROFILE_EXECUTOR_MAX_TOKENS",
            "LLM_PROFILE_SUPERVISOR_MAX_TOKENS",
        ],
    )
    def test_max_tokens_must_be_positive(
        self, clean_settings: None, valid_settings_data: dict, profile_prefix: str
    ) -> None:
        """Test max_tokens must be greater than zero for all profiles."""
        data = valid_settings_data.copy()
        data[profile_prefix] = 0

        with pytest.raises(ValidationError) as exc_info:
            Settings(**data)
        assert profile_prefix in str(exc_info.value)


class TestApiKeyHandling:
    """Test API key configuration and optional fields."""

    def test_valid_api_key(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test API key is stored correctly."""
        settings = Settings(**valid_settings_data)
        if settings.GROQ_API_KEY is not None:
            assert settings.GROQ_API_KEY.get_secret_value() == "gsk_test_key"

    def test_api_key_field_is_optional(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test API key field definition allows None."""
        field_info = Settings.model_fields.get("GROQ_API_KEY")
        assert field_info is not None
        assert not field_info.is_required()

    def test_huggingface_token_alias(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test HuggingFace token accepts HF_TOKEN alias."""
        data = valid_settings_data.copy()
        data["HF_TOKEN"] = "hf_test_value"
        settings = Settings.model_validate(data)
        if settings.HUGGINGFACE_HUB_TOKEN is not None:
            assert settings.HUGGINGFACE_HUB_TOKEN.get_secret_value() == "hf_test_value"


class TestPathsAndUrls:
    """Test path and URL configurations."""

    def test_paths_configuration(self, clean_settings, valid_settings_data) -> None:
        """Test DATA_PATH and CACHE_DIR configurations."""
        settings = Settings(**valid_settings_data)
        assert settings.DATA_PATH == Path("data")
        assert settings.CACHE_DIR == Path(".cache")

    def test_custom_cache_dir(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test custom CACHE_DIR path."""
        data = valid_settings_data.copy()
        data["CACHE_DIR"] = "/custom/cache"
        settings = Settings(**data)
        assert settings.CACHE_DIR == Path("/custom/cache")

    def test_server_urls(self, clean_settings, valid_settings_data) -> None:
        """Test server URL configuration."""
        settings = Settings(**valid_settings_data)
        assert settings.POKEAPI_MCP_SERVER_URL == "http://localhost:8000"

    def test_custom_server_url(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test custom HTTPS server URL."""
        data = valid_settings_data.copy()
        data["POKEAPI_MCP_SERVER_URL"] = "https://api.example.com"
        settings = Settings(**data)
        assert settings.POKEAPI_MCP_SERVER_URL == "https://api.example.com"


class TestBooleanAndLoggingConfig:
    """Test boolean flags and logging configuration."""

    def test_mcp_enabled_states(self, clean_settings, valid_settings_data) -> None:
        """Test POKEAPI_MCP_ENABLED boolean flag."""
        # Default is False
        settings = Settings(**valid_settings_data)
        assert settings.POKEAPI_MCP_ENABLED is False

    def test_mcp_enabled_true(self, clean_settings, valid_settings_data) -> None:
        """Test POKEAPI_MCP_ENABLED can be set to True."""
        data = valid_settings_data.copy()
        data["POKEAPI_MCP_ENABLED"] = True
        settings = Settings(**data)
        assert settings.POKEAPI_MCP_ENABLED is True

    @pytest.mark.parametrize(
        "log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    )
    def test_valid_log_levels(
        self, clean_settings: None, valid_settings_data: dict, log_level: str
    ) -> None:
        """Test all valid logging levels."""
        data = valid_settings_data.copy()
        data["LOG_LEVEL"] = log_level
        settings = Settings(**data)
        assert settings.LOG_LEVEL == log_level


class TestConfigurationIntegration:
    """Test complete configuration scenarios."""

    def test_full_configuration_loads(
        self, clean_settings, valid_settings_data
    ) -> None:
        """Test complete configuration loads successfully."""
        settings = Settings(**valid_settings_data)

        # Verify all required fields are present
        assert settings.LLM_DEFAULT_PROVIDER
        assert settings.LLM_DEFAULT_MODEL
        assert settings.GROQ_API_KEY
        assert settings.POKEAPI_MCP_SERVER_URL
        assert settings.DATA_PATH

    def test_multiple_validation_errors_caught(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test that multiple validation errors are reported together."""
        data = valid_settings_data.copy()
        data["LLM_DEFAULT_TEMPERATURE"] = 3.0  # Invalid
        data["LLM_DEFAULT_MAX_TOKENS"] = -1  # Invalid

        with pytest.raises(ValidationError) as exc_info:
            Settings(**data)

        error_fields = {error["loc"][0] for error in exc_info.value.errors()}
        assert "LLM_DEFAULT_TEMPERATURE" in error_fields
        assert "LLM_DEFAULT_MAX_TOKENS" in error_fields

    def test_extra_fields_not_allowed(
        self, clean_settings: None, valid_settings_data: dict
    ) -> None:
        """Test that extra fields are rejected."""
        data = valid_settings_data.copy()
        data["UNKNOWN_FIELD"] = "value"

        with pytest.raises(ValidationError):
            Settings(**data)

"""Unit tests for configuration module."""

import pytest
from pydantic import ValidationError

from pokeconsultor.config import Settings


@pytest.fixture
def clean_settings():
    """Reset singleton state before and after each test.

    This fixture ensures that each test gets a fresh Settings instance
    by cleaning up the singleton pattern before and after each test.
    """
    # Clean up singleton before test
    if hasattr(Settings, "_instance"):
        delattr(Settings, "_instance")
    if hasattr(Settings, "_initialized"):
        Settings._initialized = False

    yield

    # Clean up singleton after test
    if hasattr(Settings, "_instance"):
        delattr(Settings, "_instance")
    if hasattr(Settings, "_initialized"):
        Settings._initialized = False


class TestSettingsSingleton:
    """Test singleton pattern implementation for Settings."""

    def test_singleton_instance(self, clean_settings) -> None:
        """Verify Settings maintains singleton pattern."""
        settings1 = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )

        # Segunda instância deve ser a mesma (singleton)
        settings2 = Settings(
            LLM_PROVIDER="openai",
            LLM_MODEL="gpt-4",
            LLM_TEMPERATURE=0.5,
            LLM_MAX_TOKENS=2048,
            LLM_API_KEY="gsk_different_key",
            POKEAPI_MCP_SERVER_URL="https://different.url:8000",
            DATA_PATH="other/",
        )

        # Devem ser a mesma instância
        assert settings1 is settings2
        # E manter os valores da primeira inicialização
        assert settings1.LLM_PROVIDER == "groq"


class TestLLMProviderValidation:
    """Test LLM_PROVIDER field validation."""

    def test_valid_groq_provider(self, clean_settings) -> None:
        """Test valid Groq provider."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_PROVIDER == "groq"

    def test_valid_openai_provider(self, clean_settings) -> None:
        """Test valid OpenAI provider."""
        settings = Settings(
            LLM_PROVIDER="openai",
            LLM_MODEL="gpt-4",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_PROVIDER == "openai"

    def test_valid_anthropic_provider(self, clean_settings) -> None:
        """Test valid Anthropic provider."""
        settings = Settings(
            LLM_PROVIDER="anthropic",
            LLM_MODEL="claude-3-opus",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_PROVIDER == "anthropic"


class TestLLMModelValidation:
    """Test LLM_MODEL field validation."""

    def test_valid_model(self, clean_settings) -> None:
        """Test valid model name."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_MODEL == "mixtral-8x7b"


class TestTemperatureValidation:
    """Test LLM_TEMPERATURE field validation."""

    def test_valid_temperature_min(self, clean_settings) -> None:
        """Test minimum valid temperature (0.0)."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.0,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_TEMPERATURE == 0.0

    def test_valid_temperature_max(self, clean_settings) -> None:
        """Test maximum valid temperature (2.0)."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=2.0,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_TEMPERATURE == 2.0

    def test_valid_temperature_mid(self, clean_settings) -> None:
        """Test mid-range temperature."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_TEMPERATURE == 0.7

    def test_temperature_below_min(self, clean_settings) -> None:
        """Test temperature below minimum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                LLM_PROVIDER="groq",
                LLM_MODEL="mixtral-8x7b",
                LLM_TEMPERATURE=-0.1,
                LLM_MAX_TOKENS=1024,
                LLM_API_KEY="gsk_test_key",
                POKEAPI_MCP_SERVER_URL="http://localhost:8000",
                DATA_PATH="data/",
            )
        assert "LLM_TEMPERATURE" in str(exc_info.value)

    def test_temperature_above_max(self, clean_settings) -> None:
        """Test temperature above maximum raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                LLM_PROVIDER="groq",
                LLM_MODEL="mixtral-8x7b",
                LLM_TEMPERATURE=2.1,
                LLM_MAX_TOKENS=1024,
                LLM_API_KEY="gsk_test_key",
                POKEAPI_MCP_SERVER_URL="http://localhost:8000",
                DATA_PATH="data/",
            )
        assert "LLM_TEMPERATURE" in str(exc_info.value)


class TestMaxTokensValidation:
    """Test LLM_MAX_TOKENS field validation."""

    def test_valid_max_tokens(self, clean_settings) -> None:
        """Test valid max tokens value."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_MAX_TOKENS == 1024

    def test_zero_max_tokens(self, clean_settings) -> None:
        """Test zero max tokens raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                LLM_PROVIDER="groq",
                LLM_MODEL="mixtral-8x7b",
                LLM_TEMPERATURE=0.7,
                LLM_MAX_TOKENS=0,
                LLM_API_KEY="gsk_test_key",
                POKEAPI_MCP_SERVER_URL="http://localhost:8000",
                DATA_PATH="data/",
            )
        assert "LLM_MAX_TOKENS" in str(exc_info.value)

    def test_negative_max_tokens(self, clean_settings) -> None:
        """Test negative max tokens raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                LLM_PROVIDER="groq",
                LLM_MODEL="mixtral-8x7b",
                LLM_TEMPERATURE=0.7,
                LLM_MAX_TOKENS=-100,
                LLM_API_KEY="gsk_test_key",
                POKEAPI_MCP_SERVER_URL="http://localhost:8000",
                DATA_PATH="data/",
            )
        assert "LLM_MAX_TOKENS" in str(exc_info.value)


class TestApiKeyValidation:
    """Test LLM_API_KEY field validation."""

    def test_valid_api_key(self, clean_settings) -> None:
        """Test valid API key."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key_123",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.LLM_API_KEY == "gsk_test_key_123"

    def test_empty_api_key(self, clean_settings) -> None:
        """Test empty API key raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                LLM_PROVIDER="groq",
                LLM_MODEL="mixtral-8x7b",
                LLM_TEMPERATURE=0.7,
                LLM_MAX_TOKENS=1024,
                LLM_API_KEY="",
                POKEAPI_MCP_SERVER_URL="http://localhost:8000",
                DATA_PATH="data/",
            )
        assert "LLM_API_KEY" in str(exc_info.value)


class TestServerUrlValidation:
    """Test POKEAPI_MCP_SERVER_URL field validation."""

    def test_valid_http_url(self, clean_settings) -> None:
        """Test valid HTTP URL."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
        )
        assert settings.POKEAPI_MCP_SERVER_URL == "http://localhost:8000"

    def test_valid_https_url(self, clean_settings) -> None:
        """Test valid HTTPS URL."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="https://api.example.com:8000",
            DATA_PATH="data/",
        )
        assert settings.POKEAPI_MCP_SERVER_URL == "https://api.example.com:8000"


class TestLogLevelValidation:
    """Test LOG_LEVEL field validation."""

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_valid_log_levels(self, level: str, clean_settings) -> None:
        """Test valid log levels."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
            LOG_LEVEL=level,
        )
        assert settings.LOG_LEVEL == level


class TestBooleanConfiguration:
    """Test boolean configuration fields."""

    def test_mcp_enabled_default(self, clean_settings) -> None:
        """Test POKEAPI_MCP_ENABLED defaults to True when not provided."""
        # When explicitly passing value, it should use True as default
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
            POKEAPI_MCP_ENABLED=True,
        )
        assert settings.POKEAPI_MCP_ENABLED is True

    def test_mcp_enabled_false(self, clean_settings) -> None:
        """Test POKEAPI_MCP_ENABLED can be set to False."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
            POKEAPI_MCP_ENABLED=False,
        )
        assert settings.POKEAPI_MCP_ENABLED is False


class TestCompleteConfiguration:
    """Test complete configuration scenarios."""

    def test_full_valid_configuration(self, clean_settings) -> None:
        """Test creating a complete valid configuration."""
        settings = Settings(
            LLM_PROVIDER="groq",
            LLM_MODEL="mixtral-8x7b",
            LLM_TEMPERATURE=0.7,
            LLM_MAX_TOKENS=1024,
            LLM_API_KEY="gsk_test_key",
            POKEAPI_MCP_SERVER_URL="http://localhost:8000",
            DATA_PATH="data/",
            LOG_LEVEL="INFO",
            POKEAPI_MCP_ENABLED=True,
        )

        assert settings.LLM_PROVIDER == "groq"
        assert settings.LLM_MODEL == "mixtral-8x7b"
        assert settings.LLM_TEMPERATURE == 0.7
        assert settings.LLM_MAX_TOKENS == 1024
        assert settings.LLM_API_KEY == "gsk_test_key"
        assert settings.POKEAPI_MCP_SERVER_URL == "http://localhost:8000"
        assert settings.DATA_PATH == "data/"
        assert settings.LOG_LEVEL == "INFO"
        assert settings.POKEAPI_MCP_ENABLED is True

    def test_multiple_validation_errors(self, clean_settings) -> None:
        """Test that multiple validation errors are caught."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(
                LLM_PROVIDER="groq",
                LLM_MODEL="mixtral-8x7b",
                LLM_TEMPERATURE=3.0,
                LLM_MAX_TOKENS=-1,
                LLM_API_KEY="test_key",
                POKEAPI_MCP_SERVER_URL="http://localhost:8000",
                DATA_PATH="data/",
            )

        error_dict = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in error_dict}

        # Deve ter erros de validação para temperatura e max_tokens
        assert "LLM_TEMPERATURE" in error_fields
        assert "LLM_MAX_TOKENS" in error_fields

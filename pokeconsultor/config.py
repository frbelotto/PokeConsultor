"""Configuration settings for PokeConsultor."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    The Settings class implements a singleton pattern to ensure a single
    instance is created and reused for the lifetime of the process.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM Configuration
    LLM_PROVIDER: str = Field(description="LLM provider name (e.g., 'groq', 'openai')")
    LLM_MODEL: str = Field(description="Model identifier for the LLM service")
    LLM_TEMPERATURE: float = Field(
        ge=0.0, le=2.0, description="Temperature parameter for LLM (0.0 to 2.0)"
    )
    LLM_MAX_TOKENS: int = Field(gt=0, description="Maximum tokens for LLM responses")

    # API Keys
    LLM_API_KEY: SecretStr = Field(min_length=1, description="API key for LLM service")

    # MCP Server Configuration
    POKEAPI_MCP_SERVER_URL: str = Field(description="URL for PokeAPI MCP server")
    POKEAPI_MCP_ENABLED: bool = Field(
        default=True, description="Enable/disable PokeAPI MCP server"
    )

    # CSV Data Configuration
    DATA_PATH: str = Field(
        description="Path to data directory containing data source files"
    )

    # Logging Configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    def __new__(cls, *args, **kwargs):
        """Ensure a single instance is created for the lifetime of the process."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(self, *args, **kwargs):
        """Initialize settings only once to avoid reloading environment data."""
        # Only initialize if this is the first time __init__ is called
        if not getattr(self.__class__, "_initialized", False):
            super().__init__(*args, **kwargs)
            self.__class__._initialized = True


# Global singleton instance
settings: Settings = Settings()

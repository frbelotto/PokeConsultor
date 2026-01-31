"""Configuration settings for PokeConsultor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
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
        extra="forbid",
    )

    # Optional API Keys for different providers
    GROQ_API_KEY: SecretStr | None = Field(
        default=None, description="Groq API key (if using Groq models)"
    )
    HUGGINGFACE_HUB_TOKEN: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"),
        description="Access token for Hugging Face gated repositories",
    )

    # LLM Configuration
    # ==============================================================#
    # Default LLM Profile
    LLM_DEFAULT_PROVIDER: str = Field(
        description="Default LLM provider name (e.g., 'groq', 'openai')"
    )
    LLM_DEFAULT_MODEL: str = Field(
        description="Default model identifier for the LLM service"
    )
    LLM_DEFAULT_TEMPERATURE: float = Field(
        ge=0.0, le=2.0, description="Default temperature parameter for LLM (0.0 to 2.0)"
    )
    LLM_DEFAULT_MAX_TOKENS: int = Field(
        gt=0, description="Default maximum tokens for LLM responses"
    )

    # Additional Profiles (optional)
    LLM_PROFILE_EXECUTOR_PROVIDER: str = Field(
        description="LLM provider name (e.g., 'groq', 'openai')"
    )
    LLM_PROFILE_EXECUTOR_MODEL: str = Field(
        description="LLM model identifier for the profile"
    )
    LLM_PROFILE_EXECUTOR_TEMPERATURE: float = Field(
        ge=0.0, le=2.0, description="Temperature parameter for LLM (0.0 to 2.0)"
    )
    LLM_PROFILE_EXECUTOR_MAX_TOKENS: int = Field(
        gt=0, description="Maximum tokens for LLM responses"
    )
    LLM_PROFILE_SUPERVISOR_PROVIDER: str = Field(
        description="LLM provider name (e.g., 'groq', 'openai')"
    )
    LLM_PROFILE_SUPERVISOR_MODEL: str = Field(
        description="LLM model identifier for the profile"
    )
    LLM_PROFILE_SUPERVISOR_TEMPERATURE: float = Field(
        ge=0.0, le=2.0, description="Temperature parameter for LLM (0.0 to 2.0)"
    )
    LLM_PROFILE_SUPERVISOR_MAX_TOKENS: int = Field(
        gt=0, description="Maximum tokens for LLM responses"
    )
    # ==============================================================#

    CACHE_DIR: Path = Field(
        default=Path(".cache"),
        description="Base path for application cache",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="pokeconsultor",
        description="Name of the ChromaDB collection",
    )
    DATA_PATH: Path = Field(
        description="Path to data directory containing data source files"
    )

    POKEAPI_MCP_SERVER_URL: str = Field(description="URL for PokeAPI MCP server")
    POKEAPI_MCP_ENABLED: bool = Field(
        default=False,
        description="Enable/disable PokeAPI MCP server",
    )

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Summarization Configuration
    # ==============================================================#
    SUMMARIZATION_ENABLED: bool = Field(
        default=True,
        description="Enable/disable automatic memory summarization",
    )
    SUMMARIZATION_TRIGGER_TOKENS: int = Field(
        default=4000,
        description="Token limit to trigger summarization",
    )
    SUMMARIZATION_KEEP_MESSAGES: int = Field(
        default=20,
        description="Number of recent messages to keep after summarization",
    )
    SUMMARIZATION_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Model identifier for generating summaries",
    )
    SUMMARIZATION_PROVIDER: str = Field(
        default="groq",
        description="Provider for the summarization model",
    )
    # ==============================================================#

    def __new__(cls, *args: Any, **kwargs: Any) -> Settings:
        """Ensure a single instance is created for the lifetime of the process."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._initialized = False
        return cls._instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize settings only once to avoid reloading environment data."""
        # Only initialize if this is the first time __init__ is called
        if not getattr(self.__class__, "_initialized", False):
            super().__init__(*args, **kwargs)
            self.__class__._initialized = True


# Global singleton instance
settings: Settings = Settings()

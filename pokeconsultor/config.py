"""Configuration settings for PokeConsultor."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

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

    # Singleton control (kept explicit to preserve current project pattern)
    _instance: ClassVar[Settings | None] = None
    _initialized: ClassVar[bool] = False
    _lock: ClassVar[Lock] = Lock()

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

    AGENT_RECURSION_LIMIT: int = Field(
        default=12,
        gt=0,
        description="Maximum number of graph steps allowed per agent interaction",
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
        instance = getattr(cls, "_instance", None)
        if instance is None:
            lock = getattr(cls, "_lock", None)
            if lock is None:
                lock = Lock()
                setattr(cls, "_lock", lock)

            with lock:
                instance = getattr(cls, "_instance", None)
                if instance is None:
                    instance = super().__new__(cls)
                    setattr(cls, "_instance", instance)
                    setattr(cls, "_initialized", False)

        return instance

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize settings only once to avoid reloading environment data."""
        if getattr(self.__class__, "_initialized", False):
            return

        with self.__class__._lock:
            if getattr(self.__class__, "_initialized", False):
                return
            super().__init__(*args, **kwargs)
            self.__class__._initialized = True

    def export_runtime_env(self) -> None:
        """Export selected settings values to process environment variables.

        This is useful for third-party libraries that only read API keys from
        ``os.environ`` during initialization.
        """

        env_map: dict[str, SecretStr | None] = {
            "GROQ_API_KEY": self.GROQ_API_KEY,
            "HUGGINGFACE_HUB_TOKEN": self.HUGGINGFACE_HUB_TOKEN,
        }

        for env_name, secret_value in env_map.items():
            if secret_value is None:
                continue

            value = secret_value.get_secret_value()
            # Export even empty-string secrets if explicitly configured, but skip unset ones.
            if value is not None and env_name not in os.environ:
                os.environ[env_name] = value


# Re-attach singleton controls explicitly after Pydantic class construction.
# This avoids edge cases where private-like class vars may be transformed.
Settings._instance = None
Settings._initialized = False
Settings._lock = Lock()

# Global singleton instance
settings: Settings = Settings()

"""Protocol definition for Language Models."""

from typing import Any
from pydantic import BaseModel, SecretStr, ConfigDict
from pokeconsultor.config import settings
from pokeconsultor.services.logger import logger


class LLMProfile(BaseModel):
    """Configuration profile for a specific LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int = 10
    api_key: SecretStr


class LLMProfiles(BaseModel):
    """Manager for multiple LLM profiles loaded from Settings configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _profiles: dict[str, LLMProfile] = {}

    def model_post_init(self, __context: Any) -> None:
        self._load_profiles()

    def _load_profiles(self) -> None:
        """Load LLM profiles from Settings configuration.

        Loads the default profile and any additional profiles defined
        in settings (e.g., executor, supervisor).
        """
        # Load default profile
        default_api_key = self._get_api_key_for_provider(settings.LLM_DEFAULT_PROVIDER)
        self._profiles["default"] = LLMProfile(
            provider=settings.LLM_DEFAULT_PROVIDER,
            model=settings.LLM_DEFAULT_MODEL,
            temperature=settings.LLM_DEFAULT_TEMPERATURE,
            max_tokens=settings.LLM_DEFAULT_MAX_TOKENS,
            api_key=default_api_key,
        )

        # Load executor profile
        executor_api_key = self._get_api_key_for_provider(
            settings.LLM_PROFILE_EXECUTOR_PROVIDER
        )
        self._profiles["executor"] = LLMProfile(
            provider=settings.LLM_PROFILE_EXECUTOR_PROVIDER,
            model=settings.LLM_PROFILE_EXECUTOR_MODEL,
            temperature=settings.LLM_PROFILE_EXECUTOR_TEMPERATURE,
            max_tokens=settings.LLM_PROFILE_EXECUTOR_MAX_TOKENS,
            api_key=executor_api_key,
        )

        # Load supervisor profile
        supervisor_api_key = self._get_api_key_for_provider(
            settings.LLM_PROFILE_SUPERVISOR_PROVIDER
        )
        self._profiles["supervisor"] = LLMProfile(
            provider=settings.LLM_PROFILE_SUPERVISOR_PROVIDER,
            model=settings.LLM_PROFILE_SUPERVISOR_MODEL,
            temperature=settings.LLM_PROFILE_SUPERVISOR_TEMPERATURE,
            max_tokens=settings.LLM_PROFILE_SUPERVISOR_MAX_TOKENS,
            api_key=supervisor_api_key,
        )

    def _get_api_key_for_provider(self, provider: str) -> SecretStr:
        """Get the API key for a specific provider from settings.

        Args:
            provider: Provider name (e.g., 'groq', 'openai').

        Returns:
            SecretStr with the API key or None if not available.
        """
        provider_upper = provider.upper()

        # Map common provider names to their settings attributes
        provider_map = {
            "GROQ": "GROQ_API_KEY",
            "OPENAI": "OPENAI_API_KEY",
            "ANTHROPIC": "ANTHROPIC_API_KEY",
            "HUGGINGFACE": "HUGGINGFACE_HUB_TOKEN",
        }

        attr_name = provider_map.get(provider_upper, f"{provider_upper}_API_KEY")
        api_key = getattr(settings, attr_name, None)

        return api_key if api_key else SecretStr("")

    def get_profile(self, name: str) -> LLMProfile:
        """Get a specific LLM profile by name."""
        profile_name = name.lower()
        if profile_name not in self._profiles:
            logger.warning(
                f"LLM profile '{profile_name}' not found. Using default profile."
            )
            return self._profiles["default"]

        return self._profiles[profile_name]

    def list_profiles(self) -> list[str]:
        """List all available profile names.

        Returns:
            List of profile names.
        """
        return list(self._profiles.keys())


llm_profiles = LLMProfiles()

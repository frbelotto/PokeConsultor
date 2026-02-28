"""Tokenizer utilities for accurate token counting and text trimming."""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.services.logger import logger

# Context window sizes for known models (tokens)
MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "llama-3.1-8b-instant": 8192,
    "llama-3.1-70b-versatile": 8192,
    "llama-3.2-90b-vision-preview": 8192,
    "mixtral-8x7b-32768": 32768,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
}


class TokenizerService(BaseModel):
    """Service for token counting and text manipulation.

    Provides accurate token counting using HuggingFace tokenizers when
    available, with fallback to character-based heuristics.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    llm_model: str | None = None
    context_ratio: float = 0.40  # Use 40% of context window for RAG

    _tokenizer: Any | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize tokenizer after model creation."""
        self._tokenizer = self._load_tokenizer()

    def get_max_context_tokens(self) -> int:
        """Calculate maximum context tokens based on configured LLM model.

        Uses the known context window for the model, applying the configured
        ratio to leave room for the LLM response. Falls back to 4000 tokens
        for unknown models.
        """
        model = self.llm_model or getattr(settings, "LLM_DEFAULT_MODEL", None)
        if not model:
            return 4000

        context_window = MODEL_CONTEXT_WINDOWS.get(model)
        if context_window:
            return max(1000, int(context_window * self.context_ratio))

        return 4000

    def _load_tokenizer(self) -> Any | None:
        """Load tokenizer for accurate token counting.

        Tries to load a matching tokenizer for the configured LLM model.
        Returns None if unavailable, and callers should use heuristics.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            logger.warning("Transformers not available for token counting: %s", exc)
            return None

        model_mapping = {
            "llama-3.1-8b-instant": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama-3.1-70b-versatile": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "llama-3.2-90b-vision-preview": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        }

        configured_model = self.llm_model or getattr(
            settings, "LLM_DEFAULT_MODEL", None
        )
        if not configured_model:
            logger.info("No LLM model configured; using heuristic token counting.")
            return None

        hf_model = model_mapping.get(configured_model, configured_model)

        gated_prefixes = (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        )

        hf_token = (
            settings.HUGGINGFACE_HUB_TOKEN.get_secret_value()
            if settings.HUGGINGFACE_HUB_TOKEN
            else None
        )

        if hf_model in gated_prefixes and not hf_token:
            logger.info(
                "Skipping tokenizer for gated model %s (no HF token); using heuristic.",
                hf_model,
            )
            return None

        try:
            return AutoTokenizer.from_pretrained(hf_model, token=hf_token)
        except Exception as exc:
            logger.warning(
                "Could not load tokenizer for %s: %s. Using heuristic.",
                hf_model,
                exc,
            )
            return None

    def count_tokens(self, text: str) -> int:
        """Count tokens for a text using tokenizer when available.

        For very large texts (>10k chars), uses heuristic to avoid
        tokenizer warnings about exceeding model max length.
        """
        if not text:
            return 0

        # For large texts, use heuristic to avoid tokenizer warnings
        # Most models have 512-131k token limits
        if len(text) > 10000:
            return max(1, len(text) // 4)  # ~4 chars per token heuristic

        if self._tokenizer is None:
            return max(1, len(text) // 4)  # ~4 chars per token heuristic

        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:
            logger.warning("Token counting failed, using heuristic: %s", exc)
            return max(1, len(text) // 4)

    def trim_to_tokens(self, text: str, token_budget: int) -> str:
        """Trim text to fit within a token budget."""
        if token_budget <= 0 or not text:
            return ""

        if self._tokenizer is None:
            approx_chars = max(0, token_budget * 4)
            return text[:approx_chars]

        try:
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            token_ids = token_ids[:token_budget]
            decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
            return cast(str, decoded)
        except Exception as exc:
            logger.warning("Token trim failed, using heuristic: %s", exc)
            return text[: max(0, token_budget * 4)]

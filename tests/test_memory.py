"""Unit tests for memory middleware configuration and PII protections."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware import PIIMiddleware
from langgraph.checkpoint.memory import InMemorySaver


class FakeSummarizationMiddleware:
    """Lightweight summarization middleware test double."""

    def __init__(self, model: str, trigger: tuple[str, int], keep: tuple[str, int]) -> None:
        self.model = model
        self.trigger = trigger
        self.keep = keep


@pytest.fixture
def load_memory_module(monkeypatch: pytest.MonkeyPatch):
    """Provide a helper to import memory module with controlled settings."""

    def _load(*, summarization_enabled: bool) -> Any:
        fake_settings = SimpleNamespace(
            SUMMARIZATION_ENABLED=summarization_enabled,
            SUMMARIZATION_PROVIDER="groq",
            SUMMARIZATION_MODEL="llama-3.1-8b-instant",
            SUMMARIZATION_TRIGGER_TOKENS=4000,
            SUMMARIZATION_KEEP_MESSAGES=20,
        )

        fake_config = ModuleType("pokeconsultor.config")
        fake_config.settings = fake_settings

        monkeypatch.setitem(sys.modules, "pokeconsultor.config", fake_config)

        import langchain.agents.middleware as langchain_middleware

        monkeypatch.setattr(
            langchain_middleware,
            "SummarizationMiddleware",
            FakeSummarizationMiddleware,
            raising=True,
        )

        sys.modules.pop("pokeconsultor.services.memory", None)
        return importlib.import_module("pokeconsultor.services.memory")

    return _load


class TestMemorySystem:
    """Validate memory/checkpointer setup and middleware composition."""

    def test_checkpointer_is_inmemorysaver(self, load_memory_module: Any) -> None:
        """Ensure the default checkpointer uses in-memory storage."""
        memory_module = load_memory_module(summarization_enabled=False)
        assert isinstance(memory_module.checkpointer, InMemorySaver)

    def test_pii_middleware_configuration(self, load_memory_module: Any) -> None:
        """Ensure all expected PII protections are configured with redaction."""
        memory_module = load_memory_module(summarization_enabled=False)

        pii_middlewares = [m for m in memory_module.middleware if isinstance(m, PIIMiddleware)]
        expected_types = {
            "email",
            "credit_card",
            "ip",
            "api_key",
            "bearer_token",
            "database_url",
        }

        assert len(pii_middlewares) == len(expected_types)
        assert {m.pii_type for m in pii_middlewares} == expected_types
        assert all(m.strategy == "redact" for m in pii_middlewares)
        assert all(m.apply_to_input is True for m in pii_middlewares)
        assert all(m.apply_to_tool_results is True for m in pii_middlewares)

    def test_summarization_middleware_added_when_enabled(
        self, load_memory_module: Any
    ) -> None:
        """Ensure summarization middleware is appended when feature is enabled."""
        memory_module = load_memory_module(summarization_enabled=True)

        assert len(memory_module.middleware) == 7
        assert isinstance(memory_module.middleware[-1], FakeSummarizationMiddleware)
        assert memory_module.middleware[-1].model == "groq:llama-3.1-8b-instant"
        assert memory_module.middleware[-1].trigger == ("tokens", 4000)
        assert memory_module.middleware[-1].keep == ("messages", 20)


class TestPIIMiddlewareFunctions:
    """Validate custom PII detectors used by memory middleware."""

    @staticmethod
    def _get_middleware_by_type(middlewares: list[Any], pii_type: str) -> PIIMiddleware:
        """Find a PIIMiddleware instance by its configured pii_type."""
        for middleware in middlewares:
            if isinstance(middleware, PIIMiddleware) and middleware.pii_type == pii_type:
                return middleware
        raise AssertionError(f"PIIMiddleware not found for pii_type={pii_type}")

    def test_api_key_detector_matches_expected_pattern(self, load_memory_module: Any) -> None:
        """API key detector should identify OpenAI-like 'sk-' secret patterns."""
        memory_module = load_memory_module(summarization_enabled=False)
        api_key_middleware = self._get_middleware_by_type(memory_module.middleware, "api_key")

        matches = api_key_middleware.detector(
            "token=sk-abcdefghijklmnopqrstuvwxyz1234567890"
        )

        assert len(matches) >= 1

    def test_bearer_detector_matches_expected_pattern(self, load_memory_module: Any) -> None:
        """Bearer token detector should identify authorization header patterns."""
        memory_module = load_memory_module(summarization_enabled=False)
        bearer_middleware = self._get_middleware_by_type(memory_module.middleware, "bearer_token")

        matches = bearer_middleware.detector(
            "Authorization: Bearer abc123.XYZ-987_token+/="
        )

        assert len(matches) >= 1

    @pytest.mark.parametrize(
        "value",
        [
            "postgres://user:pass@localhost:5432/db",
            "mysql://root:pass@127.0.0.1:3306/db",
            "mongodb://user:pass@mongo:27017/db",
            "redis://:pass@cache:6379/0",
        ],
    )
    def test_database_url_detector_matches_supported_schemes(
        self, load_memory_module: Any, value: str
    ) -> None:
        """Database URL detector should match all configured DB schemes."""
        memory_module = load_memory_module(summarization_enabled=False)
        db_url_middleware = self._get_middleware_by_type(memory_module.middleware, "database_url")

        matches = db_url_middleware.detector(value)

        assert len(matches) >= 1

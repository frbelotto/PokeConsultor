"""Integration test for PokeConsultor main entrypoint.

This minimal test verifies that:
- The application initializes (RAG and Agent)
- It answers a question
- It maintains conversation memory across interactions
- The RAG service is instantiated (basic load)

The test uses monkeypatch to stub heavy external dependencies.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterator

import pytest


@dataclass
class _Role:
    """Simple role representation mimicking Message.role in the app."""

    value: str


@dataclass
class _Message:
    """Simple message representation used by FakeMemory."""

    role: _Role
    timestamp: datetime
    content: str


class FakeMemory:
    """Minimal in-memory conversation store for the FakeAIAgent."""

    def __init__(self) -> None:
        self._messages: list[_Message] = []

    def add_pair(self, user_text: str, assistant_text: str) -> None:
        now = datetime.now()
        self._messages.append(
            _Message(role=_Role("user"), timestamp=now, content=user_text)
        )
        self._messages.append(
            _Message(role=_Role("assistant"), timestamp=now, content=assistant_text)
        )

    def get_summary(self) -> str:
        count = len(self._messages)
        pairs = count // 2
        return f"Conversation with {pairs} exchanges ({count} messages)."  # comentário: resumo simples

    def get_messages(self) -> list[_Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()


class FakeAIAgent:
    """Lightweight fake for AIAgent to avoid real LLM calls in integration test."""

    def __init__(self, llm: object) -> None:  # noqa: D401 - succinct
        self.llm = llm
        self.memory = FakeMemory()

    def respond(self, request: object) -> str:
        # The real code uses LLMRequest; we only need its 'prompt' attribute here
        prompt = getattr(request, "prompt", "")
        answer = f"[fake] Answer to: {prompt}"
        self.memory.add_pair(user_text=prompt, assistant_text=answer)
        return answer


class FakeRAGService:
    """Lightweight fake RAG service; counts initializations and stubs methods used."""

    init_count: int = 0

    def __init__(self, use_cache: bool, llm_model: str) -> None:  # noqa: D401 - succinct
        type(self).init_count += 1
        self.use_cache = use_cache
        self.llm_model = llm_model

    def retrieve(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        # Not used in non-debug path; provided for completeness
        return [(f"Doc about: {query}", 0.9)]

    def format_results(self, results: list[tuple[str, float]]) -> str:
        # Minimal formatting similar to the app's expectation
        formatted = []
        for i, (doc, _score) in enumerate(results, 1):
            formatted.append(f"[Resultado {i}] {doc}")
        return "\n\n".join(formatted)

    def _count_tokens(
        self, text: str
    ) -> int:  # right-side comment: compatível com uso interno
        return len(text.split())


def _input_sequence(*items: str) -> Callable[[str], str]:
    """Build a fake input() function yielding a fixed sequence of user entries."""

    iterator: Iterator[str] = iter(items)

    def _fake_input(prompt: str) -> str:  # noqa: D401 - succinct
        del prompt  # unused
        try:
            return next(iterator)
        except StopIteration:
            # If called more than the provided items, exit gracefully
            return "exit"

    return _fake_input


def test_main_integration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run main() with stubbed dependencies to verify basic interactive flow."""

    # Import the main module fresh to ensure our monkeypatches apply to its globals
    main = importlib.import_module("main")

    # Patch heavy components with fakes
    monkeypatch.setattr(main, "AIAgent", FakeAIAgent, raising=True)
    monkeypatch.setattr(main, "RAGService", FakeRAGService, raising=True)

    # Stub llm profile resolution to return a simple object
    def _fake_get_profile(cls: object, name: str) -> object:
        assert name == "default"
        return {"profile": "default"}

    monkeypatch.setattr(
        main.LLMProfiles, "get_profile", classmethod(_fake_get_profile), raising=True
    )

    # Provide an input sequence: ask, inspect memory, then exit
    fake_input = _input_sequence(
        "What is Pikachu?",  # question
        "memory",  # show memory
        "exit",  # quit
    )
    monkeypatch.setattr("builtins.input", fake_input, raising=True)

    # Execute main flow
    main.main()

    # Capture output and perform minimal assertions
    out = capsys.readouterr().out

    # Initialization messages
    assert "INICIALIZANDO POKECONSULTOR" in out
    assert "Carregando RAG service" in out
    assert "Sistema pronto para consultas" in out

    # Interaction and response
    assert "QUERY: What is Pikachu?" in out
    assert "RESPOSTA DA IA" in out
    assert "[fake] Answer to: What is Pikachu?" in out

    # Memory inspection
    assert "HISTÓRICO COMPLETO DE CONVERSAS" in out

    # RAG instantiated at least once
    assert FakeRAGService.init_count >= 1

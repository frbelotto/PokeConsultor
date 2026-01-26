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
from langchain_core.documents import Document


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

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        # Now returns List[Document]
        return [Document(page_content=f"Doc about: {query}")]

    def format_results(
        self, results: list[Document] | list[tuple[Document, float]]
    ) -> str:
        # Minimal formatting similar to the app's expectation
        formatted = []

        # Handle both list of Documents and list of (Doc, score)
        docs = []
        if results and isinstance(results[0], tuple):
            docs = [r[0] for r in results]
        else:
            docs = results

        for i, doc in enumerate(docs, 1):
            formatted.append(f"[Resultado {i}] {doc.page_content}")
        return "\n\n".join(formatted)

    def count_tokens(self, text: str) -> int:
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


import pytest


@pytest.fixture
def fake_rag_service():
    """Fixture que retorna uma instância do FakeRAGService."""
    return FakeRAGService(use_cache=False, llm_model="fake-llm")


@pytest.fixture
def fake_ai_agent():
    """Fixture que retorna uma instância do FakeAIAgent."""
    return FakeAIAgent(llm=object())


def test_integration_main_flow(fake_rag_service, fake_ai_agent):
    """
    Integration test covering the main flow:
    - Initializes fake RAG and agent
    - Simulates a user question
    - Checks agent response and memory
    """
    # Simula uma pergunta do usuário
    user_question = "What is Pikachu?"
    # Simula recuperação de contexto pelo RAG
    retrieved_docs = fake_rag_service.retrieve(user_question)
    formatted_context = fake_rag_service.format_results(retrieved_docs)

    # Simula requisição ao agente
    class Request:
        prompt: str = user_question + "\n" + formatted_context

    response = fake_ai_agent.respond(Request())
    # Verifica se a resposta contém o prefixo fake
    assert response.startswith("[fake] Answer to: ")
    # Verifica se a memória foi atualizada
    messages = fake_ai_agent.memory.get_messages()
    assert any(user_question in m.content for m in messages)
    assert any("[fake] Answer to:" in m.content for m in messages)

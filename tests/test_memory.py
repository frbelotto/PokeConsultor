"""Unit tests for conversation memory service."""

from unittest.mock import MagicMock

import pytest
from langchain.messages import AIMessage, HumanMessage, SystemMessage

from pokeconsultor.services.memory import ConversationMemory


@pytest.fixture
def conversation_memory() -> ConversationMemory:
    """Provide a ConversationMemory instance with a small window for trimming."""
    return ConversationMemory(max_messages=3)


def test_add_messages_and_trim_history(conversation_memory: ConversationMemory) -> None:
    """Ensure messages are stored with correct roles and trimmed when exceeding limits."""

    conversation_memory.add_user_message([HumanMessage(content="Hi")])
    conversation_memory.add_assistant_message([AIMessage(content="Hello")])
    conversation_memory.add_user_message([HumanMessage(content="How are you?")])
    conversation_memory.add_user_message([HumanMessage(content="Tell me more")])

    messages = conversation_memory.get_messages()

    assert len(messages) == 3
    assert [msg.content for msg in messages] == [
        "Hello",
        "How are you?",
        "Tell me more",
    ]
    assert [type(msg) for msg in messages] == [
        AIMessage,
        HumanMessage,
        HumanMessage,
    ]


def test_get_messages_returns_copy(conversation_memory: ConversationMemory) -> None:
    """Verify get_messages returns a defensive copy that does not affect internal state."""

    conversation_memory.add_user_message([HumanMessage(content="Original")])

    external_copy = conversation_memory.get_messages()
    external_copy.append(HumanMessage(content="Injected"))

    assert len(conversation_memory.get_messages()) == 1
    assert external_copy is not conversation_memory.get_messages()


def test_get_messages_for_llm_format(conversation_memory: ConversationMemory) -> None:
    """Ensure messages are formatted correctly for LLM consumption."""

    conversation_memory.add_system_message([SystemMessage(content="System guidance")])
    conversation_memory.add_user_message([HumanMessage(content="Question")])

    formatted = conversation_memory.get_history()
    assert formatted == [
        {"role": "system", "content": "System guidance"},
        {"role": "user", "content": "Question"},
    ]


def test_clear_resets_history_and_logs(
    conversation_memory: ConversationMemory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify clear removes all messages and logs removal count."""

    conversation_memory.add_user_message([HumanMessage(content="One")])
    conversation_memory.add_assistant_message([AIMessage(content="Two")])

    mock_logger = MagicMock()
    monkeypatch.setattr("pokeconsultor.services.memory.logger", mock_logger)

    conversation_memory.clear()

    assert conversation_memory.get_messages() == []
    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args
    assert call_args.args[0] == "Conversation history cleared (%d messages removed)"
    assert call_args.args[1] == 2


def test_get_summary_counts_by_role(conversation_memory: ConversationMemory) -> None:
    """Check that summary reflects totals and per-role counts."""

    conversation_memory.add_system_message([SystemMessage(content="Sys")])
    conversation_memory.add_user_message([HumanMessage(content="User question")])
    conversation_memory.add_assistant_message([AIMessage(content="Assistant reply")])

    summary = conversation_memory.get_summary()

    assert "3 messages" in summary
    assert "1 user" in summary
    assert "1 assistant" in summary
    assert "1 system" in summary

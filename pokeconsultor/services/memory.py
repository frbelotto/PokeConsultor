from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver

from pokeconsultor.config import settings

checkpointer = InMemorySaver()

# PII protection is always active — covers user input and RAG tool results.
# apply_to_tool_results=True ensures context returned by retrieve_context is also sanitized.
middleware: list[AgentMiddleware[Any, Any]] = [
    PIIMiddleware("email", strategy="redact", apply_to_tool_results=True),
    PIIMiddleware("credit_card", strategy="redact", apply_to_tool_results=True),
    PIIMiddleware("ip", strategy="redact", apply_to_tool_results=True),
    PIIMiddleware(
        "api_key",
        detector=r"sk-[a-zA-Z0-9]{32,64}",
        strategy="redact",
        apply_to_tool_results=True,
    ),
    PIIMiddleware(
        "bearer_token",
        detector=r"\bBearer\s+[a-zA-Z0-9\-._~+/]+=*\b",
        strategy="redact",
        apply_to_tool_results=True,
    ),
    PIIMiddleware(
        "database_url",
        detector=(
            r"(?:postgres|mysql|mongodb|redis)://"
            r"(?:[^\s@/]*:[^\s@/]+@|[^\s:@/]+(?::[^\s@/]+)?@)?"
            r"[A-Za-z0-9.\-]+"
            r"(?::\d+)?"
            r"(?:/[A-Za-z0-9_\-./]+)?"
            r"(?:\?[^\s#]+)?"
        ),
        strategy="redact",
        apply_to_tool_results=True,
    ),
]

if settings.SUMMARIZATION_ENABLED:
    middleware.append(
        SummarizationMiddleware(
            model=f"{settings.SUMMARIZATION_PROVIDER}:{settings.SUMMARIZATION_MODEL}",
            trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
            keep=("messages", settings.SUMMARIZATION_KEEP_MESSAGES),
        )
    )

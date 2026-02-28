from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

from pokeconsultor.config import settings

checkpointer = InMemorySaver()

middleware = []

if settings.SUMMARIZATION_ENABLED:
    middleware.append(
        SummarizationMiddleware(
            model=f"{settings.SUMMARIZATION_PROVIDER}:{settings.SUMMARIZATION_MODEL}",
            trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
            keep=("messages", settings.SUMMARIZATION_KEEP_MESSAGES),
        )
    )

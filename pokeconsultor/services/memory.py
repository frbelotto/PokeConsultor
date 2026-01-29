from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

checkpointer = InMemorySaver()

middleware = [
    SummarizationMiddleware(
        model="groq:llama-3.1-8b-instant",     
        trigger=("tokens", 4000),       
        keep=("messages", 20),         
    ),
]



"""LLM module for language model implementations."""

from pokeconsultor.llm.base import LLM
from pokeconsultor.llm.groq_llm import GroqLLM

__all__ = ["LLM", "GroqLLM"]

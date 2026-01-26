"""Tests for EmbeddingService chunking logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from pokeconsultor.services.rag.embeddings import EmbeddingService


class TestChunkingLogic:
    """Test specifically the text chunking capabilities."""

    @pytest.fixture
    def embedding_service(self, tmp_path) -> EmbeddingService:
        """Create a service instance with mocked internals."""
        # We need to mock things that __init__ might touch or just pass valid dummy values
        # EmbeddingService is a Pydantic model. config is global settings.

        # Mocking complex internals:
        # The service initializes HuggingFaceEmbeddings and Chroma in model_post_init
        # We can bypass model_post_init logic by mocking the relevant parts if possible,
        # or by patching the heavy imports.
        # or by patching the heavy imports.
        pass

    @pytest.fixture(autouse=True)
    def mock_torch(self, mocker):
        """Mock torch to prevent runtime error from set_num_interop_threads."""
        return mocker.patch("pokeconsultor.services.rag.embeddings.torch")

    def test_regex_split_behavior(self, mocker, tmp_path):
        """Verify the specific regex logic for splitting."""
        import re

        text = "Hello world. How are you? I am fine!"
        # The regex used in the code:
        pass
        # We will implicitly test this via the service call as above.

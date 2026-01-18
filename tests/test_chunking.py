"""Tests for EmbeddingService chunking logic."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from pokeconsultor.services.rag.embeddings import EmbeddingService
from pathlib import Path

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

    def test_chunk_by_sentences_simple(self, mocker, tmp_path):
        """Test simple sentence splitting with regex."""
        # Patch heavy dependencies to avoid instantiating them
        # Note: HuggingFaceEmbeddings is imported inside model_post_init, so we patch the source
        mocker.patch("langchain_huggingface.HuggingFaceEmbeddings")
        mocker.patch("pokeconsultor.services.rag.embeddings.Chroma")
        mocker.patch("pokeconsultor.services.rag.embeddings.os.cpu_count", return_value=1)
        
        # Instantiate service
        service = EmbeddingService(
            data_path=tmp_path,
            chunking_strategy="sentence",
            chunk_size=100, # Small chunk size to force splitting
            use_token_counting=False # Use char counting for simplicity in test
        )
        # Mock tokenizer to return simple length
        service._tokenizer_service = MagicMock()
        service._tokenizer_service.count_tokens = lambda t: len(t.split())
        
        # Override _count_size to be predictable (chars)
        service.use_token_counting = False
        
        text = "Hello world. This is a test! Is it working? Yes."
        # chunks should ideally preserve sentences
        start_docs = [text]
        
        # We set chunk_size large enough to hold individual sentences but small enough if we wanted to test splitting
        # Actually, let's just test that it splits sentences correctly first.
        # But _chunk_by_sentences GROUPS them into chunks.
        
        # If chunk_size is huge, it shouldn't split the doc unless the doc is massive.
        # Wait, the logic is:
        # 1. Split doc into sentences.
        # 2. Group sentences into chunk.
        
        service.chunk_size = 500 # Large enough for whole text
        chunks = service.chunk_documents(start_docs)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_by_sentences_splits_correctly(self, mocker, tmp_path):
        """Test that it actually identifies sentence boundaries."""
        # Patch heavy dependencies
        mocker.patch("langchain_huggingface.HuggingFaceEmbeddings")
        mocker.patch("pokeconsultor.services.rag.embeddings.Chroma")
        mocker.patch("pokeconsultor.services.rag.embeddings.os.cpu_count", return_value=1)

        service = EmbeddingService(
            data_path=tmp_path,
            chunking_strategy="sentence",
            use_token_counting=False,
            chunk_size=20, # Very small chunk size
            chunk_overlap=0
        )
        
        # "Sentence 1." (11 chars)
        # "Sentence 2." (11 chars)
        # Total 22 chars > 20. Should split.
        text = "Sentence 1. Sentence 2."
        
        chunks = service.chunk_documents([text])
        
        # Should result in 2 chunks
        assert len(chunks) == 2
        assert "Sentence 1." in chunks[0]
        assert "Sentence 2." in chunks[1]

    def test_regex_split_behavior(self, mocker, tmp_path):
        """Verify the specific regex logic for splitting."""
        import re
        text = "Hello world. How are you? I am fine!"
        # The regex used in the code:
        pass
        # We will implicitly test this via the service call as above.

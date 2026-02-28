"""Tests for EmbeddingService chunking behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from pokeconsultor.services.rag.embeddings import EmbeddingService
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService


@pytest.fixture
def tokenizer_service() -> TokenizerService:
    """Provide a tokenizer service instance for chunking tests."""
    return TokenizerService(llm_model="llama-3.1-8b-instant")


@pytest.fixture
def embedding_service(
    tmp_path: Path,
    tokenizer_service: TokenizerService,
    monkeypatch: pytest.MonkeyPatch,
) -> EmbeddingService:
    """Create an EmbeddingService instance without loading heavy models."""

    def _noop_model_post_init(self: EmbeddingService, _context: object) -> None:
        return None

    monkeypatch.setattr(EmbeddingService, "model_post_init", _noop_model_post_init)

    return EmbeddingService(
        data_path=tmp_path,
        tokenizer_service=tokenizer_service,
    )


class TestChunkingLogic:
    """Test chunking dispatch and size counting logic."""

    @pytest.mark.parametrize(
        ("strategy", "expected_method"),
        [("sentence", "_chunk_by_sentences"), ("character", "_chunk_by_characters")],
    )
    def test_chunk_documents_dispatches_to_strategy(
        self,
        embedding_service: EmbeddingService,
        monkeypatch: pytest.MonkeyPatch,
        strategy: str,
        expected_method: str,
    ) -> None:
        """Ensure strategy selection calls the expected internal method."""
        embedding_service.chunking_strategy = strategy
        docs = [Document(page_content="alpha beta gamma")]

        called: dict[str, bool] = {"sentence": False, "character": False}

        def _fake_sentence(_docs: list[Document]) -> list[Document]:
            called["sentence"] = True
            return _docs

        def _fake_character(_docs: list[Document]) -> list[Document]:
            called["character"] = True
            return _docs

        monkeypatch.setattr(embedding_service, "_chunk_by_sentences", _fake_sentence)
        monkeypatch.setattr(embedding_service, "_chunk_by_characters", _fake_character)

        result = embedding_service.chunk_documents(docs)

        assert result == docs
        assert called["sentence"] is (expected_method == "_chunk_by_sentences")
        assert called["character"] is (expected_method == "_chunk_by_characters")

    def test_chunk_documents_empty_returns_empty_list(
        self, embedding_service: EmbeddingService
    ) -> None:
        """Empty input should not produce chunks."""
        assert embedding_service.chunk_documents([]) == []

    @pytest.mark.parametrize(
        ("use_token_counting", "text", "expected"),
        [
            (True, "one two three", 3),
            (False, "one two three", len("one two three")),
        ],
    )
    def test_count_size_modes(
        self,
        embedding_service: EmbeddingService,
        use_token_counting: bool,
        text: str,
        expected: int,
    ) -> None:
        """_count_size should support token and character counting modes."""
        embedding_service.use_token_counting = use_token_counting
        assert embedding_service._count_size(text) == expected

    def test_sentence_chunking_splits_long_text(
        self, embedding_service: EmbeddingService
    ) -> None:
        """Sentence chunking should split sufficiently long content."""
        embedding_service.chunking_strategy = "sentence"
        embedding_service.chunk_size = 6
        embedding_service.chunk_overlap = 1
        embedding_service.use_token_counting = True

        docs = [
            Document(
                page_content=(
                    "Pikachu is electric. Charizard is fire flying. "
                    "Bulbasaur is grass poison. Squirtle is water."
                )
            )
        ]

        chunks = embedding_service.chunk_documents(docs)

        assert len(chunks) >= 2
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all(chunk.page_content.strip() for chunk in chunks)

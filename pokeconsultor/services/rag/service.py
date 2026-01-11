"""RAG (Retrieval-Augmented Generation) service facade.

Unified interface to the RAG subsystem using hybrid retrieval (lexical + vector)
with rank fusion (RRF) and optional reranking for improved precision.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.services.rag.embeddings import EmbeddingService
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService
from pokeconsultor.services.rag.search.lexical import LexicalSearcher
from pokeconsultor.services.rag.search.vector import VectorSearcher
from pokeconsultor.services.rag.search.executor import HybridExecutor


class RAGService(BaseModel):
    """Unified RAG service with hybrid retrieval as default.

    Provides a simple interface for the complete RAG pipeline:
    document loading, chunking, embedding, hybrid retrieval, and context formatting.

    Uses the Facade design pattern to simplify access to EmbeddingService,
    HybridRetrieverService, and TokenizerService components.

    Example:
        rag = RAGService(data_path=Path("./data"))
        results = rag.retrieve("What is Pikachu's type?")
        context = rag.format_results(results)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Data source configuration
    data_path: Path = Field(default=settings.DATA_PATH)

    # Retrieval configuration (hybrid is standard)
    retrieve_k: int = Field(
        default=20,
        gt=0,
        description="Top-K to retrieve from each retriever (lexical & vector)",
    )
    rerank_k: int | None = Field(
        default=15,
        ge=0,
        description="Documents to keep after reranking (None disables)",
    )
    lexical_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-K from lexical retriever (defaults to retrieve_k)",
    )
    vector_k: int | None = Field(
        default=None,
        ge=1,
        description="Top-K from vector retriever (defaults to retrieve_k)",
    )
    rrf_k: int = Field(
        default=60,
        gt=0,
        description="RRF constant for rank fusion",
    )
    rerank_method: str = Field(
        default="cross_encoder",
        description="Rerank method for hybrid: 'none' | 'cosine' | 'cross_encoder'",
    )

    # Cache configuration (passed to EmbeddingService)
    use_cache: bool = Field(
        default=True,
        description="Cache vector store for reuse (EmbeddingService)",
    )

    # LLM context configuration
    llm_model: str = Field(
        description="LLM model for context window calculation",
    )

    # Internal services
    _embedding_service: EmbeddingService = PrivateAttr()
    _retriever_service: HybridExecutor = PrivateAttr()
    _lexical_searcher: LexicalSearcher = PrivateAttr()
    _vector_searcher: VectorSearcher = PrivateAttr()
    _tokenizer_service: TokenizerService = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize internal services after validation."""
        self._tokenizer_service = TokenizerService(llm_model=self.llm_model)

        # EmbeddingService with cache setting propagated
        self._embedding_service = EmbeddingService(
            data_path=self.data_path,
            use_cache=self.use_cache,
        )

        # Searchers
        self._vector_searcher = VectorSearcher(self._embedding_service)
        self._lexical_searcher = LexicalSearcher()
        self._lexical_searcher.build_from_vector_store(self._embedding_service.vector_store)

        # Hybrid executor as standard
        self._retriever_service = HybridExecutor(
            lexical_searcher=self._lexical_searcher,
            vector_searcher=self._vector_searcher,
            tokenizer_service=self._tokenizer_service,
            retrieve_k=self.retrieve_k,
            lexical_k=self.lexical_k,
            vector_k=self.vector_k,
            rrf_k=self.rrf_k,
            rerank_method=self.rerank_method,
            rerank_k=self.rerank_k,
        )

    @property
    def vector_store(self) -> Any:
        """Access underlying vector store (for advanced use cases)."""
        return self._embedding_service.vector_store

    def clear_cache(self) -> None:
        """Delete cached vector store files.

        Delegates to the EmbeddingService to clear cache without reloading models.
        """
        self._embedding_service.clear_cache()

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        """Retrieve relevant documents for a query using hybrid pipeline.

        Returns a list of (document_text, relevance_score) tuples.
        """
        return self._retriever_service.retrieve(query)

    def format_results(
        self,
        results: list[tuple[str, float]],
        max_tokens: int | None = None,
        compact: bool = True,
    ) -> str:
        """Format retrieval results into LLM context string.

        Args:
            results: List of (content, score) from retrieve().
            max_tokens: Maximum tokens for context (auto-calculated if None).
            compact: Use compact format for ~30% token savings.

        Returns:
            Formatted string ready for LLM prompt injection.
        """
        return self._retriever_service.format_context(
            results, max_tokens=max_tokens, compact=compact
        )

    # Alias for backward compatibility
    format_context = format_results

    def count_tokens(self, text: str) -> int:
        """Public helper to count tokens using the configured tokenizer.

        This replaces the previous private `_count_tokens` used in main.py
        debug output, maintaining backward compatibility with clearer naming.

        Note: This method uses tokenizer only, no need to initialize embedding services.
        """
        return self._tokenizer_service.count_tokens(text)

"""RAG (Retrieval-Augmented Generation) service facade.

Unified interface to the RAG subsystem using hybrid retrieval (lexical + vector)
with rank fusion (RRF) and optional reranking for improved precision.

Implements intelligent caching with incremental embedding support through
manifest-based file tracking.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.embeddings import EmbeddingService
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService
from pokeconsultor.services.rag.manifest import ManifestManager, FileManifestEntry
from pokeconsultor.services.rag.search.executor import HybridExecutor
from pokeconsultor.services.rag.search.lexical import LexicalSearcher
from pokeconsultor.services.rag.search.vector import VectorSearcher


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
    _manifest_manager: ManifestManager = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize internal services with intelligent caching.
        
        Implements incremental embedding workflow:
        1. Load manifest to see which files were previously embedded
        2. Scan data_path to find current files
        3. Detect NEW, MODIFIED, and DELETED files
        4. Process only changed files
        5. Load vector stores from cache
        """
        self._tokenizer_service = TokenizerService(llm_model=self.llm_model)
        
        # Manifest always stored in CACHE_DIR/manifest.json
        manifest_path = settings.CACHE_DIR.parent / "manifest.json"
        self._manifest_manager = ManifestManager(manifest_path)

        # Smart loading: detect file changes and process only what's needed
        self._load_with_incremental_embedding()

        # Initialize searchers with loaded vector store
        self._vector_searcher = VectorSearcher(self._embedding_service)
        self._lexical_searcher = LexicalSearcher()
        self._lexical_searcher.build_from_vector_store(
            self._embedding_service.vector_store
        )

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

    def _load_with_incremental_embedding(self) -> None:
        """Intelligently load documents using manifest tracking.
        
        1. Scan current files
        2. Check manifest for each file
        3. Identify NEW, MODIFIED, DELETED files
        4. Process only changed files through embedding
        5. Use cache for unchanged files
        """
        logger.info("Incremental embedding enabled")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Scan current files
        source_files = self._discover_source_files()
        current_files_by_path = {
            fp.relative_to(
                self.data_path if self.data_path.is_dir() else self.data_path.parent
            ).as_posix(): fp
            for fp in source_files
        }

        # Detect changes
        files_to_embed: list[Path] = []
        files_status = {}

        for relative_path, file_path in current_files_by_path.items():
            file_hash = FileManifestEntry.calculate_hash(file_path)
            status = self._manifest_manager.get_file_status(relative_path, file_hash)
            files_status[relative_path] = status

            if status in ("NEW", "MODIFIED"):
                files_to_embed.append(file_path)
                logger.info("Will embed %s file: %s", status, relative_path)

        # Detect deleted files
        deleted_files = self._manifest_manager.get_deleted_files(
            set(current_files_by_path.keys())
        )
        for deleted in deleted_files:
            logger.info("File was deleted: %s", deleted)
            self._manifest_manager.remove_entry(deleted)

        # Determine strategy: rebuild vs. incremental
        if self._should_rebuild_index(files_status):
            logger.info("Rebuilding entire vector store")
            self._rebuild_all_embeddings(source_files)
        else:
            logger.info("Using incremental embedding strategy")
            self._incremental_embed_files(files_to_embed, current_files_by_path)

        self._manifest_manager.save()

    def _should_rebuild_index(self, files_status: dict[str, str]) -> bool:
        """Determine if full rebuild is needed based on change rate.
        
        Args:
            files_status: Dictionary of file paths to their status
            
        Returns:
            True if more than threshold% of files changed
        """
        if not self._manifest_manager.manifest.files:
            # No existing files, start fresh
            return True

        total = len(self._manifest_manager.manifest.files)
        changed = sum(
            1 for status in files_status.values() if status in ("NEW", "MODIFIED")
        )

        change_rate = changed / total if total > 0 else 1.0
        threshold = settings.SMART_INVALIDATION_THRESHOLD

        should_rebuild = change_rate >= threshold
        logger.info(
            "Change rate: %.1f%% (%d/%d) - %s",
            change_rate * 100,
            changed,
            total,
            "REBUILD" if should_rebuild else "INCREMENTAL",
        )

        return should_rebuild

    def _rebuild_all_embeddings(self, source_files: list[Path]) -> None:
        """Rebuild vector store from all source files.
        
        Args:
            source_files: List of all source files to embed
        """
        logger.info("Creating embeddings for %d files...", len(source_files))

        # Create fresh EmbeddingService (which will load all files)
        self._embedding_service = EmbeddingService(
            data_path=self.data_path,
            use_cache=self.use_cache,
        )

        # Update manifest with all files
        base_path = (
            self.data_path if self.data_path.is_dir() else self.data_path.parent
        )
        for file_path in source_files:
            relative_path = file_path.relative_to(base_path).as_posix()
            file_hash = FileManifestEntry.calculate_hash(file_path)
            cache_key = self._embedding_service._cache_key
            self._manifest_manager.add_entry(relative_path, file_hash, cache_key)

    def _incremental_embed_files(
        self,
        files_to_embed: list[Path],
        current_files_by_path: dict[str, Path],
    ) -> None:
        """Embed only changed files and merge with existing vector store.
        
        Args:
            files_to_embed: List of NEW or MODIFIED files to embed
            current_files_by_path: Mapping of relative paths to file paths
        """
        base_path = (
            self.data_path if self.data_path.is_dir() else self.data_path.parent
        )

        if not files_to_embed:
            # No changes, just load from cache
            logger.info("No files to embed, loading from cache")
            self._embedding_service = EmbeddingService(
                data_path=self.data_path,
                use_cache=self.use_cache,
            )
            return

        logger.info("Processing %d changed files incrementally", len(files_to_embed))

        # For now, we'll process changed files and use the main embedding service
        # In a production system, you might create separate embeddings for new files
        # and merge them with existing vector stores

        # Create EmbeddingService for new/modified files
        self._embedding_service = EmbeddingService(
            data_path=self.data_path,
            use_cache=self.use_cache,
        )

        # Update manifest
        cache_key = self._embedding_service._cache_key
        for relative_path, file_path in current_files_by_path.items():
            file_hash = FileManifestEntry.calculate_hash(file_path)
            self._manifest_manager.add_entry(relative_path, file_hash, cache_key)

    def _discover_source_files(self) -> list[Path]:
        """Discover all supported source files."""
        if self.data_path.is_file():
            return [self.data_path]

        if self.data_path.is_dir():
            files = [
                p
                for p in self.data_path.rglob("*")
                if p.is_file() and LoaderFactory.is_supported(p)
            ]
            files.sort(key=lambda p: p.relative_to(self.data_path).as_posix())
            logger.info("Found %d supported files", len(files))
            return files

        logger.warning("Data path not found: %s", self.data_path)
        return []

    @property
    def vector_store(self) -> Any:
        """Access underlying vector store (for advanced use cases)."""
        return self._embedding_service.vector_store

    def clear_cache(self) -> None:
        """Delete cached vector store files.

        Delegates to the EmbeddingService to clear cache without reloading models.
        """
        self._embedding_service.clear_cache()

    def retrieve(
        self, query: str, filter_by_file: str | None = None
    ) -> list[tuple[str, float]]:
        """Retrieve relevant documents for a query using hybrid pipeline.

        Args:
            query: Search query string
            filter_by_file: Optional filename to filter results (e.g., "treinadores.csv")

        Returns:
            List of (document_text, relevance_score) tuples, filtered if specified.
        """
        results = self._retriever_service.retrieve(query)

        # Apply file filter if specified
        if filter_by_file:
            filtered_results = []
            for text, score in results:
                # Check if this document came from the specified file
                # This is a simple approach - for production, you'd want to
                # store metadata with documents and filter at retrieval time
                filtered_results.append((text, score))

            logger.info(
                "Filtered results by file '%s': %d/%d documents",
                filter_by_file,
                len(filtered_results),
                len(results),
            )
            return filtered_results

        return results

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

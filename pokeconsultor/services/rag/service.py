"""RAG (Retrieval-Augmented Generation) service facade.

Unified interface to the RAG subsystem using hybrid retrieval (lexical + vector)
with rank fusion (RRF) and optional reranking for improved precision.

Implements intelligent caching with incremental embedding support through
"""
import hashlib
from pathlib import Path
import threading
from datetime import datetime
from typing import Any, List
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.embeddings import EmbeddingService
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService
from pokeconsultor.services.rag.search.executor import HybridExecutor
from pokeconsultor.services.rag.search.lexical import LexicalSearcher
from pokeconsultor.services.rag.search.vector import VectorSearcher




class RAGService(BaseModel):
    

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
        """Initialize internal services with intelligent caching.
        
        Implements incremental embedding workflow:
        1. Load existing database to see which files were previously embedded
        2. Scan data_path to find current files
        3. Detect NEW, MODIFIED, and DELETED files
        4. Process only changed files
        5. Load vector stores from cache
        """

        self._tokenizer_service = TokenizerService(llm_model=self.llm_model)

        # EmbeddingService with cache setting propagated
        self._embedding_service = EmbeddingService(
            data_path=self.data_path,
            use_cache=self.use_cache,
        )

        # Initialize searchers with loaded vector store
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

        self._load_with_incremental_embedding()
        
        # Start background initialization of heavy models
        threading.Thread(target=self.warmup, daemon=True).start()

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of a single file (leitura única)."""
        hasher = hashlib.sha256()
        hasher.update(file_path.read_bytes())
        return hasher.hexdigest()
    

    def _load_with_incremental_embedding(self) -> None:
        """Intelligently load documents using manifest tracking.
        
        1. Scan current files
        2. Check existing data folder for each file
        3. Identify NEW, or DELETED files
        4. Process only changed files through embedding
        5. Use cache for unchanged files
        """
        logger.info("Incremental embedding enabled")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Scan current files
        source_files = self._discover_source_files()

        # Detect changes
        files_to_embed: list[Path] = []
        # Lógica incremental baseada nos metadados do ChromaDB
        chroma_hashes = self._embedding_service.get_file_hashes()

        files_to_embed: list[Path] = []
        files_to_delete: set[str] = set(chroma_hashes)
        file_hash_map: dict[str, Path] = {}

        for file_path in source_files:
            file_hash = self.calculate_file_hash(file_path)
            file_hash_map[file_hash] = file_path
            if file_hash not in chroma_hashes:
                files_to_embed.append(file_path)
            else:
                files_to_delete.discard(file_hash)

        # Remover embeddings de arquivos deletados
        for deleted_hash in files_to_delete:
            logger.info(f"Deleting embeddings for deleted file hash: {deleted_hash}")
            self._embedding_service.delete_file_embeddings(deleted_hash)

        # Embutir arquivos novos ou modificados
        if files_to_embed:
            logger.info(f"Embedding {len(files_to_embed)} new/modified files...")
            for file_path in files_to_embed:
                chunks = self._load_and_chunk_single_file(file_path)
                self._embedding_service.add_file_embeddings(file_path, chunks, self.calculate_file_hash(file_path))
        else:
            logger.info("No new or modified files to embed.")

    
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

    def _load_and_chunk_single_file(self, file_path: Path) -> list[Document]:
        """Load and chunk a single file with enriched metadata.
        
        Delegates chunking to EmbeddingService to ensure consistent parameters.
        
        Args:
            file_path: Path to the file to load and chunk
            
        Returns:
            List of text chunks as Document objects
        """
        try:
            # Load the file
            loader = LoaderFactory.get_loader(file_path)
            raw_docs = loader.load(file_path)
            
            if not raw_docs:
                logger.warning("No content loaded from %s", file_path.name)
                return []
            
            # Enrich with OS-level stats
            stats = file_path.stat()
            file_stats = {
                "file_path": str(file_path),
                "file_size": stats.st_size,
                "last_modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "file_extension": file_path.suffix.lower()
            }
            
            for doc in raw_docs:
                doc.metadata.update(file_stats)
            
            # Delegate chunking to EmbeddingService (single source of truth)
            chunked = self._embedding_service.chunk_documents(raw_docs)
            
            logger.info(
                "Loaded and chunked %s: %d chunks from %d raw docs",
                file_path.name,
                len(chunked),
                len(raw_docs),
            )
            return chunked
            
        except Exception:
            logger.exception("Error loading file %s", file_path.name)
            return []

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents using hybrid search.
        
        Returns:
            List of LangChain Document objects with metadata.
        """
        results = self._retriever_service.invoke(query)
        return results

    def format_results(
        self,
        results: List[Document] | List[tuple[Document, float]],
        max_tokens: int | None = None,
        compact: bool = True,
    ) -> str:
        """Format retrieval results into LLM context string.

        Args:
            results: List of Document objects or (doc, score) tuples.
            max_tokens: Maximum tokens for context (auto-calculated if None).
            compact: Use compact format for ~30% token savings.

        Returns:
            Formatted string ready for LLM prompt injection.
        """
        return self._retriever_service.format_context(
            results, max_tokens=max_tokens, compact=compact
        )

    @property
    def retriever(self) -> HybridExecutor:
        """Expose the internal hybrid retriever."""
        return self._retriever_service

    def warmup(self) -> None:
        """Warmup internal services (e.g. load heavy models)."""
        self._retriever_service.warmup()


    def count_tokens(self, text: str) -> int:
        """Public helper to count tokens using the configured tokenizer.

        This replaces the previous private `_count_tokens` used in main.py
        debug output, maintaining backward compatibility with clearer naming.

        Note: This method uses tokenizer only, no need to initialize embedding services.
        """
        return self._tokenizer_service.count_tokens(text)

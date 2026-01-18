"""RAG (Retrieval-Augmented Generation) service facade.

Unified interface to the RAG subsystem using hybrid retrieval (lexical + vector)
with rank fusion (RRF) and optional reranking for improved precision.

Implements intelligent caching with incremental embedding support through
"""
import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag import embeddings
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

    def _load_and_chunk_single_file(self, file_path: Path) -> list[str]:
        """Load and chunk a single file.
        
        Args:
            file_path: Path to the file to load and chunk
            
        Returns:
            List of text chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        try:
            # Load the file
            loader = LoaderFactory.get_loader(file_path)
            raw_docs = loader.load(file_path)
            
            if not raw_docs:
                logger.warning("No content loaded from %s", file_path.name)
                return []
            
            # Use the same chunking parameters as EmbeddingService
            # Default chunk_size=512, chunk_overlap=50
            chunk_size = 512
            chunk_overlap = 50
            
            # Apply chunking with RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size * 4,  # Character-based chunking
                chunk_overlap=chunk_overlap * 4,
            )
            
            chunked: list[str] = []
            for doc in raw_docs:
                if not doc:
                    continue
                if len(doc) <= chunk_size * 4:
                    chunked.append(doc)
                else:
                    chunked.extend(splitter.split_text(doc))
            
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

    def retrieve(self, query: str ) -> list[tuple[str, float]]:
        results = self._retriever_service.retrieve(query)
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


    def count_tokens(self, text: str) -> int:
        """Public helper to count tokens using the configured tokenizer.

        This replaces the previous private `_count_tokens` used in main.py
        debug output, maintaining backward compatibility with clearer naming.

        Note: This method uses tokenizer only, no need to initialize embedding services.
        """
        return self._tokenizer_service.count_tokens(text)

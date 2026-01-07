"""RAG (Retrieval-Augmented Generation) service for Pokemon data."""

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pokeconsultor.config import settings
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.logger import logger

# Context window sizes for known models (tokens)
MODEL_CONTEXT_WINDOWS = {
    "llama-3.1-8b-instant": 8192,
    "llama-3.1-70b-versatile": 8192,
    "mixtral-8x7b-32768": 32768,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 4096,
}


class RAGService(BaseModel):
    """Service for retrieval-augmented generation from multiple data sources.

    Supports multiple file formats: CSV, XLSX/XLS, JSON, TXT/MD, and PDF.
    Features:
    - Automatic file format detection via LoaderFactory
    - Persistent storage of embeddings on disk (FAISS indices)
    - Automatic cache invalidation when source data changes
    - Lazy loading of embeddings from disk cache
    - Unified index across multiple data sources
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data_path: Path = Field(default=settings.DATA_PATH)
    chunk_size: int = Field(default=1200)
    chunk_overlap: int = Field(default=250)
    use_cache: bool = Field(default=True)
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    llm_model: str | None = Field(
        default=None,
        description="LLM model name to calculate context window. If None, uses LLM_DEFAULT_MODEL from settings.",
    )
    vector_store: FAISS | None = None

    _embeddings: Any = PrivateAttr(default=None)
    _tokenizer: Any | None = PrivateAttr(default=None)
    _tokenizer_unavailable: bool = PrivateAttr(default=False)

    def model_post_init(self, __context: Any) -> None:
        """Initialize embeddings and load data after model validation.

        This hook runs after Pydantic validation is complete, making it the
        recommended way to run custom initialization logic in Pydantic v2.
        """
        # Lazy import HuggingFaceEmbeddings to avoid slow loading on module import
        from langchain_huggingface import HuggingFaceEmbeddings

        self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        self._cache_key = self._generate_cache_key()
        self._load_and_index_data()

    def _get_max_context_tokens(self) -> int:
        """Calculate maximum context tokens based on configured LLM model.

        Uses the known context window for the model, applying a 30% safety
        margin to leave room for the LLM response. Falls back to 4000 tokens
        for unknown models.
        """
        model = self.llm_model or getattr(settings, "LLM_DEFAULT_MODEL", None)
        if not model:
            return 4000

        # Look up in known models; use 30% of context window as safe margin
        context_window = MODEL_CONTEXT_WINDOWS.get(model)
        if context_window:
            return max(1000, int(context_window * 0.30))

        return 4000

    def _get_cache_path(self) -> Path:
        """Return the directory path used to cache the vector store on disk."""
        return settings.CACHE_DIR / self._cache_key

    def clear_cache(self) -> None:
        """Delete cached vector store files for the current cache key."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info("Cleared cache directory: %s", cache_path)

    def _load_and_index_data(self) -> None:
        """Load data from files and create vector store for similarity search.

        Discovers supported file formats in the data source and loads them.
        Uses cached embeddings from disk if available, otherwise creates and caches them.
        """
        if not self.data_path.exists():
            logger.error("Data file not found: %s", self.data_path)
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        if self.use_cache and self._load_from_cache():
            return

        documents = self._load_documents_from_source()
        documents = self._chunk_documents(documents)

        if not documents:
            logger.warning("No documents loaded from data source")
            return

        logger.info(f"Creating embeddings for {len(documents)} documents...")
        self.vector_store = FAISS.from_texts(documents, self._embeddings)

        if self.use_cache:
            self._save_to_cache()

    def _load_documents_from_source(self) -> list[str]:
        """Load documents from data source.

        Discovers all supported files and loads their content.
        """
        source_files = self._discover_source_files()

        if not source_files:
            logger.warning("No supported data files found to load documents")
            return []

        documents: list[str] = []

        for file_path in source_files:
            try:
                loader = LoaderFactory.get_loader(file_path)
                file_documents = loader.load(file_path)
                if file_documents:
                    documents.extend(file_documents)
                logger.info(
                    "Loaded %d documents from %s",
                    len(file_documents),
                    file_path.name,
                )
            except Exception as exc:
                logger.exception("Error loading file %s", file_path.name)
                raise

        logger.info(
            "Loaded %d documents from %d files", len(documents), len(source_files)
        )
        return documents

    def _chunk_documents(self, documents: list[str]) -> list[str]:
        """Split large documents into smaller overlapping chunks.

        This prevents very large files (e.g., PDFs) from producing single huge
        retrieval results that exceed the LLM context window.
        """
        if not documents:
            return []

        # Lazy import to avoid slow loading on module import
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunked: list[str] = []
        for doc in documents:
            if not doc:
                continue
            if len(doc) <= self.chunk_size:
                chunked.append(doc)
                continue
            chunked.extend(splitter.split_text(doc))

        if len(chunked) != len(documents):
            logger.info(
                "Chunked %d raw documents into %d chunks (chunk_size=%d, overlap=%d)",
                len(documents),
                len(chunked),
                self.chunk_size,
                self.chunk_overlap,
            )

        return chunked

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on data source content.

        Supports both single files and directories containing multiple files.
        The cache key changes whenever any supported source file content changes
        or when configuration parameters change.

        Returns:
            Deterministic SHA256 hash prefix for cache directory naming.
        """
        source_files = self._discover_source_files()

        if not source_files:
            raise FileNotFoundError(
                f"No supported data files found in {self.data_path}"
            )

        hasher = hashlib.sha256()
        base_path = self.data_path if self.data_path.is_dir() else self.data_path.parent

        for file_path in source_files:
            relative = file_path.relative_to(base_path).as_posix()
            hasher.update(relative.encode())
            hasher.update(file_path.read_bytes())

        hasher.update(f"{self.chunk_size}:{self.chunk_overlap}".encode())
        hasher.update(self.embedding_model.encode())
        return hasher.hexdigest()[:16]

    def _discover_source_files(self) -> list[Path]:
        """Discover supported data source files."""
        if self.data_path.is_file():
            return [self.data_path]

        if self.data_path.is_dir():
            files = [
                path
                for path in self.data_path.rglob("*")
                if path.is_file() and LoaderFactory.is_supported(path)
            ]
            files.sort(key=lambda p: p.relative_to(self.data_path).as_posix())
            logger.info(
                f"Discovered {len(files)} supported data files in directory: {self.data_path}"
            )
            return files

        raise FileNotFoundError(f"Data path not found: {self.data_path}")

    def _load_from_cache(self) -> bool:
        """Load vector store from disk cache if available."""
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path}")
            return False

        self.vector_store = FAISS.load_local(
            str(cache_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded embeddings from cache: %s", cache_path)
        return True

    def _save_to_cache(self) -> None:
        """Save vector store to disk cache."""
        if not self.vector_store:
            logger.warning("Vector store not initialized, cannot save cache")
            return

        cache_path = self._get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(cache_path))
        logger.info("Saved embeddings to cache: %s", cache_path)

    def format_results(
        self,
        results: list[tuple[str, float]],
        max_tokens: int | None = None,
    ) -> str:
        """Format retrieval results into a single context string with token limit.

        Token counting is performed with a tokenizer when available for the
        configured LLM; otherwise falls back to a character-based heuristic.

        Args:
            results: List of tuples (document_content, relevance_score).
            max_tokens: Maximum tokens to include in the final context. Defaults
                to settings.RAG_MAX_CONTEXT_TOKENS when not provided.

        Returns:
            A formatted string suitable to be injected into the LLM prompt.
        """
        if not results:
            return ""

        max_context_tokens = self._get_max_context_tokens()
        token_limit = max_tokens if max_tokens is not None else max_context_tokens
        remaining_tokens = max(0, int(token_limit))

        parts: list[str] = []
        for i, (doc, _score) in enumerate(results, 1):
            header = f"[Resultado {i}]\n"
            header_tokens = self._count_tokens(header)
            if remaining_tokens <= header_tokens:
                break

            body_budget = remaining_tokens - header_tokens
            body = self._trim_to_tokens(doc or "", body_budget)
            body_tokens = self._count_tokens(body)

            parts.append(header + body)
            remaining_tokens -= header_tokens + body_tokens

            if remaining_tokens <= 0:
                break

        return "\n\n".join(parts)

    def _get_tokenizer(self) -> Any | None:
        """Lazy-load tokenizer for accurate token counting.

        Tries to load a matching tokenizer for the configured LLM model. If
        unavailable or loading fails, returns None and callers should fall back
        to heuristics.
        """
        if self._tokenizer_unavailable:
            return None

        if self._tokenizer is not None:
            return self._tokenizer

        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Transformers not available for token counting: %s", exc)
            self._tokenizer_unavailable = True
            self._tokenizer = None
            return None

        model_mapping = {
            "llama-3.1-8b-instant": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama-3.1-70b-versatile": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        }

        configured_model = getattr(settings, "LLM_DEFAULT_MODEL", None) or getattr(
            settings, "LLM_MODEL", None
        )
        if not configured_model:
            logger.info(
                "No LLM model configured in settings; skipping tokenizer load and using heuristic token counting."
            )
            self._tokenizer_unavailable = True
            self._tokenizer = None
            return None

        hf_model = model_mapping.get(configured_model, configured_model)

        gated_prefixes = (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        )

        hf_token = (
            settings.HUGGINGFACE_HUB_TOKEN.get_secret_value()
            if settings.HUGGINGFACE_HUB_TOKEN
            else None
        )

        has_hf_token = bool(hf_token)

        if hf_model in gated_prefixes and not has_hf_token:
            logger.info(
                "Skipping tokenizer download for gated model %s (no HF token found); using heuristic token counting instead.",
                hf_model,
            )
            self._tokenizer_unavailable = True
            self._tokenizer = None
            return None

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Could not load tokenizer for %s: %s. Falling back to heuristic token counting.",
                hf_model,
                exc,
            )
            self._tokenizer_unavailable = True
            self._tokenizer = None

        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens for a text using tokenizer when available."""
        if not text:
            return 0

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            # Heuristic fallback: ~4 chars per token
            return max(1, len(text) // 4)

        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Token counting failed, using heuristic: %s", exc)
            return max(1, len(text) // 4)

    def _trim_to_tokens(self, text: str, token_budget: int) -> str:
        """Trim text to fit within a token budget using tokenizer when possible."""
        if token_budget <= 0 or not text:
            return ""

        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            approx_chars = max(0, token_budget * 4)
            return text[:approx_chars]

        try:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            token_ids = token_ids[:token_budget]
            result: str = tokenizer.decode(token_ids, skip_special_tokens=True)
            return result
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Token trim failed, using heuristic: %s", exc)
            approx_chars = max(0, token_budget * 4)
            return text[:approx_chars]

    def retrieve(
        self, query: str, k: int = 7, rerank_k: int | None = 3
    ) -> list[tuple[str, float]]:
        """Retrieve relevant documents with optional lightweight reranking.

        Args:
            query: User query to search for relevant documents.
            k: Number of documents to retrieve from the vector store.
            rerank_k: When provided, reranks the top-k results using cosine similarity
                on fresh embeddings and returns only the top ``rerank_k``. Set to None
                to disable reranking.

        Returns:
            List of tuples (document_content, relevance_score).
        """
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error("Error retrieving documents: %s", exc)
            return []

        if not results:
            return []

        if not self._should_rerank(results, rerank_k):
            return [(doc.page_content, score) for doc, score in results]

        reranked = self._rerank_results(query, results, rerank_k)  # type: ignore[arg-type]
        return reranked

    def _should_rerank(self, results: Any, rerank_k: int | None) -> bool:
        """Decide whether reranking should run."""
        return bool(rerank_k is not None and rerank_k > 0 and len(results) > rerank_k)

    def _rerank_results(
        self,
        query: str,
        results: Any,
        rerank_k: int,
    ) -> list[tuple[str, float]]:
        """Rerank FAISS results using cosine similarity on fresh embeddings."""
        try:
            query_vec = np.array(self._embeddings.embed_query(query), dtype=float)
            doc_texts = [doc.page_content for doc, _ in results]
            doc_vecs = np.array(
                self._embeddings.embed_documents(doc_texts), dtype=float
            )

            if doc_vecs.ndim != 2 or query_vec.ndim != 1:
                logger.debug("Unexpected embedding shapes; skipping rerank")
                return [(doc.page_content, score) for doc, score in results]

            cosine_scores = self._cosine_scores(query_vec, doc_vecs)
            top_indexes = np.argsort(cosine_scores)[::-1][:rerank_k]

            return [
                (results[i][0].page_content, float(cosine_scores[i]))
                for i in top_indexes
            ]
        except Exception as exc:
            logger.warning("Rerank failed, returning base results: %s", exc)
            return [(doc.page_content, score) for doc, score in results]

    @staticmethod
    def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity scores between query and document vectors."""
        query_norm = np.linalg.norm(query_vec)
        doc_norms = np.linalg.norm(doc_vecs, axis=1)
        denom = np.clip(query_norm * doc_norms, a_min=1e-12, a_max=None)
        scores: np.ndarray = (doc_vecs @ query_vec) / denom
        return scores

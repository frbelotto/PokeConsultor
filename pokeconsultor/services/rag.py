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
    chunk_size: int = Field(
        default=500,
        gt=0,
        description="Maximum chunk size in tokens (when use_token_counting=True) or characters",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap size in tokens (when use_token_counting=True) or characters",
    )
    chunking_strategy: str = Field(
        default="sentence",
        description="Chunking strategy: 'character' (fixed size) or 'sentence' (semantic boundaries)",
    )
    use_token_counting: bool = Field(
        default=True,
        description="Use token counting for chunk sizes instead of character counting",
    )
    use_cache: bool = Field(default=True)
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    llm_model: str | None = Field(
        default=None,
        description="LLM model name to calculate context window. If None, uses LLM_DEFAULT_MODEL from settings.",
    )
    retrieve_k: int = Field(
        default=20,
        gt=0,
        description="Default number of documents to retrieve from the vector store",
    )
    rerank_k: int | None = Field(
        default=5,
        ge=0,
        description="Default number of documents to keep after reranking. None disables reranking.",
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

        # Safety: filter out any chunks that are too large for the embedding model
        # Most sentence-transformers models have 512 token limit
        max_embedding_size = 100000  # chars (safety: ~25k tokens)
        oversized = [i for i, d in enumerate(documents) if len(d) > max_embedding_size]
        if oversized:
            logger.error(
                "Found %d chunks exceeding max size (%d chars). First oversized: %d chars. "
                "This indicates chunking failed. Please check chunking configuration.",
                len(oversized),
                max_embedding_size,
                len(documents[oversized[0]]),
            )
            # Truncate oversized chunks to prevent crash
            for idx in oversized:
                doc_text = str(documents[idx])
                documents[idx] = doc_text[:max_embedding_size]
                logger.warning(
                    "Truncated chunk %d to %d chars", idx, max_embedding_size
                )

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

        Supports two strategies:
        - 'character': Fixed-size chunking (traditional approach)
        - 'sentence': Semantic chunking respecting sentence boundaries

        This prevents very large files (e.g., PDFs) from producing single huge
        retrieval results that exceed the LLM context window.
        """
        if not documents:
            return []

        if self.chunking_strategy == "sentence":
            return self._chunk_by_sentences(documents)
        else:
            return self._chunk_by_characters(documents)

    def _chunk_by_characters(self, documents: list[str]) -> list[str]:
        """Traditional fixed-size chunking using RecursiveCharacterTextSplitter."""
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
                "Chunked %d raw documents into %d chunks (strategy=character, chunk_size=%d, overlap=%d)",
                len(documents),
                len(chunked),
                self.chunk_size,
                self.chunk_overlap,
            )

        return chunked

    def _chunk_by_sentences(self, documents: list[str]) -> list[str]:
        """Semantic chunking that respects sentence boundaries.

        Splits text into sentences using nltk, then groups sentences into
        chunks respecting the configured size limit. This produces more
        coherent chunks compared to fixed-size splitting.
        """
        try:
            import nltk
            from nltk.tokenize import sent_tokenize

            # Ensure nltk punkt tokenizer is available
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                logger.info("Downloading nltk punkt tokenizer...")
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)  # For newer nltk versions
        except ImportError:
            logger.warning(
                "nltk not available for sentence chunking, falling back to character chunking"
            )
            return self._chunk_by_characters(documents)

        chunked: list[str] = []

        for doc in documents:
            if not doc or not doc.strip():
                continue

            # Safety check: use both token counting and character length
            doc_size = self._count_tokens(doc) if self.use_token_counting else len(doc)
            doc_chars = len(doc)

            # For very small documents, keep as-is
            # But add safety: if doc has >10k chars, always chunk it (safety against token counting errors)
            max_char_threshold = 10000
            if doc_size <= self.chunk_size and doc_chars <= max_char_threshold:
                chunked.append(doc)
                continue

            # Log warning for very large documents
            if doc_chars > 50000:
                logger.warning(
                    "Processing very large document (%d chars, ~%d tokens). This may take time.",
                    doc_chars,
                    doc_size,
                )

            # Split into sentences
            sentences = sent_tokenize(doc, language="portuguese")
            if not sentences:
                logger.warning(
                    "sent_tokenize returned empty for document (%d chars). Using character-based fallback.",
                    len(doc),
                )
                # Fallback to character chunking for this document
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size
                    if not self.use_token_counting
                    else self.chunk_size * 4,
                    chunk_overlap=self.chunk_overlap
                    if not self.use_token_counting
                    else self.chunk_overlap * 4,
                )
                chunked.extend(splitter.split_text(doc))
                continue

            # Group sentences into chunks
            current_chunk: list[str] = []
            current_size = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sent_size = (
                    self._count_tokens(sentence)
                    if self.use_token_counting
                    else len(sentence)
                )

                # If single sentence exceeds chunk_size, split it using character-based chunking
                if sent_size > self.chunk_size:
                    if current_chunk:
                        chunked.append(" ".join(current_chunk))
                        current_chunk = []
                        current_size = 0

                    logger.debug(
                        "Oversized sentence (%d tokens/chars), splitting with character chunker",
                        sent_size,
                    )
                    # Split oversized sentence
                    from langchain_text_splitters import RecursiveCharacterTextSplitter

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size
                        if not self.use_token_counting
                        else self.chunk_size * 4,
                        chunk_overlap=self.chunk_overlap
                        if not self.use_token_counting
                        else self.chunk_overlap * 4,
                    )
                    chunked.extend(splitter.split_text(sentence))
                    continue

                # Check if adding this sentence would exceed the limit
                if current_chunk and (current_size + sent_size > self.chunk_size):
                    chunked.append(" ".join(current_chunk))
                    # Start new chunk with overlap: keep last few sentences
                    overlap_sentences = self._get_overlap_sentences(
                        current_chunk, self.chunk_overlap
                    )
                    current_chunk = overlap_sentences
                    current_size = sum(
                        self._count_tokens(s) if self.use_token_counting else len(s)
                        for s in current_chunk
                    )

                current_chunk.append(sentence)
                current_size += sent_size

            # Add remaining sentences
            if current_chunk:
                chunked.append(" ".join(current_chunk))

        if len(chunked) != len(documents):
            unit = "tokens" if self.use_token_counting else "chars"
            logger.info(
                "Chunked %d raw documents into %d chunks (strategy=sentence, chunk_size=%d %s, overlap=%d %s)",
                len(documents),
                len(chunked),
                self.chunk_size,
                unit,
                self.chunk_overlap,
                unit,
            )

        return chunked

    def _get_overlap_sentences(
        self, sentences: list[str], overlap_budget: int
    ) -> list[str]:
        """Get the last few sentences that fit within overlap budget."""
        if not sentences or overlap_budget <= 0:
            return []

        overlap: list[str] = []
        overlap_size = 0

        # Work backwards from the end
        for sentence in reversed(sentences):
            sent_size = (
                self._count_tokens(sentence)
                if self.use_token_counting
                else len(sentence)
            )
            if overlap_size + sent_size <= overlap_budget:
                overlap.insert(0, sentence)
                overlap_size += sent_size
            else:
                break

        return overlap

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

        hasher.update(
            f"{self.chunk_size}:{self.chunk_overlap}:{self.chunking_strategy}:{self.use_token_counting}".encode()
        )
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
        self,
        query: str,
        k: int | None = None,
        rerank_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve relevant documents with optional lightweight reranking.

        Args:
            query: User query to search for relevant documents.
            k: Number of documents to retrieve from the vector store. If None, uses self.retrieve_k.
            rerank_k: When provided, reranks the top-k results using cosine similarity
                on fresh embeddings and returns only the top ``rerank_k``. If not provided,
                uses self.rerank_k (which defaults to None to disable reranking).

        Returns:
            List of tuples (document_content, relevance_score).
        """
        k = k if k is not None else self.retrieve_k
        if rerank_k is None:
            rerank_k = self.rerank_k
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

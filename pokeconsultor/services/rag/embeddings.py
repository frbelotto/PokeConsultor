"""Embedding service for document indexing and vector store management."""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from blingfire import text_to_sentences
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import torch
from tqdm import tqdm

from pokeconsultor.config import settings
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService

# Embedding model limits (multilingual-e5-large has 512 token max)
EMBEDDING_MAX_TOKENS = 500  # Safety margin below 512
EMBEDDING_MAX_CHARS = 1800  # ~3.6 chars/token average for multilingual


class EmbeddingService(BaseModel):
    """Service for creating and managing document embeddings.

    Handles document loading, chunking, embedding generation, and
    persistent caching of vector stores.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data_path: Path
    chunk_size: int = Field(
        default=350,
        gt=0,
        description="Target chunk size in tokens (optimal: 300-400 for semantic search)",
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap size (~12-15% of chunk_size recommended)",
    )
    chunking_strategy: str = Field(
        default="sentence",
        description="Chunking strategy: 'character' or 'sentence' (semantic)",
    )
    use_token_counting: bool = Field(
        default=True,
        description="Use token counting instead of character counting",
    )
    use_cache: bool = Field(
        default=True,
        description="Cache vector store for reuse",
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        description="HuggingFace embedding model for semantic search",
    )

    vector_store: FAISS | None = None

    _embeddings: Any = PrivateAttr()
    _tokenizer_service: TokenizerService = PrivateAttr()
    _cache_key: str = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize embeddings and load data after validation."""
        from langchain_huggingface import HuggingFaceEmbeddings

        self._configure_torch_threads()
        logger.info("Loading embedding model...")

        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self._tokenizer_service = TokenizerService()
        self._cache_key = self._generate_cache_key()
        self._load_and_index_data()

    def _configure_torch_threads(self) -> None:
        """Configure PyTorch to use all available CPU threads."""
        max_threads = os.cpu_count() or 1
        try:
            torch.set_num_threads(max_threads)
            torch.set_num_interop_threads(max_threads)
            logger.info("Torch threads configured: %d", max_threads)
        except Exception:
            logger.exception("Failed to configure torch threads")  # Log error when configuring threads

    @property
    def embeddings(self) -> Any:
        """Expose embeddings for retriever use."""
        return self._embeddings

    def _get_cache_path(self) -> Path:
        """Return the cache directory path for the vector store."""
        return settings.CACHE_DIR / self._cache_key

    def clear_cache(self) -> None:
        """Delete cached vector store files."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            shutil.rmtree(cache_path)
            logger.info("Cleared cache: %s", cache_path)

    def _load_and_index_data(self) -> None:
        """Load documents and create vector store for similarity search."""
        # Ensure data directory exists
        self.data_path.mkdir(parents=True, exist_ok=True)

        if self.use_cache and self._load_from_cache():
            return

        documents = self._load_documents_from_source()
        documents = self._chunk_documents(documents)

        if not documents:
            logger.warning("No documents loaded from data source")
            return

        logger.info("Creating embeddings for %d chunks...", len(documents))
        self._create_embeddings_with_progress(documents)

        if self.use_cache:
            self._save_to_cache()

    def _create_embeddings_with_progress(self, documents: list[str]) -> None:
        """Create FAISS vector store with minimal memory overhead.

        Batches embeddings to avoid holding the entire embedding matrix
        in RAM and builds the FAISS index incrementally.
        """
        batch_size = 32
        faiss = dependable_faiss_import()
        docstore = InMemoryDocstore()
        index = None

        with tqdm(
            total=len(documents), desc="Embedding documents", unit="doc"
        ) as progress:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                if not batch:
                    continue

                batch_embeddings = self._embeddings.embed_documents(batch)
                if not batch_embeddings:
                    continue

                if index is None:
                    dimension = len(batch_embeddings[0])
                    index = faiss.IndexFlatL2(dimension)
                    self.vector_store = FAISS(
                        embedding_function=self._embeddings,
                        index=index,
                        docstore=docstore,
                        index_to_docstore_id={},
                    )

                assert self.vector_store is not None

                self.vector_store.add_embeddings(zip(batch, batch_embeddings))
                progress.update(len(batch))

    def _load_documents_from_source(self) -> list[str]:
        """Load documents from all supported files in data path."""
        source_files = self._discover_source_files()

        if not source_files:
            logger.warning("No supported files found")
            return []

        documents: list[str] = []

        for file_path in source_files:
            try:
                loader = LoaderFactory.get_loader(file_path)
                file_docs = loader.load(file_path)
                if file_docs:
                    documents.extend(file_docs)
                logger.info("Loaded %d docs from %s", len(file_docs), file_path.name)
            except Exception:
                logger.exception("Error loading %s", file_path.name)
                raise

        logger.info(
            "Total: %d documents from %d files", len(documents), len(source_files)
        )
        return documents

    def _chunk_documents(self, documents: list[str]) -> list[str]:
        """Split documents into overlapping chunks for better retrieval."""
        if not documents:
            return []

        if self.chunking_strategy == "sentence":
            return self._chunk_by_sentences(documents)
        return self._chunk_by_characters(documents)

    def _chunk_by_characters(self, documents: list[str]) -> list[str]:
        """Fixed-size chunking using RecursiveCharacterTextSplitter."""
        # Use safe char limit for embedding model, capped at EMBEDDING_MAX_CHARS
        char_size = min(
            self.chunk_size * 4 if self.use_token_counting else self.chunk_size,
            EMBEDDING_MAX_CHARS,
        )
        char_overlap = min(
            self.chunk_overlap * 4 if self.use_token_counting else self.chunk_overlap,
            char_size // 4,
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=char_size,
            chunk_overlap=char_overlap,
        )

        chunked: list[str] = []
        for doc in documents:
            if not doc:
                continue
            if len(doc) <= char_size:
                chunked.append(doc)
            else:
                chunked.extend(splitter.split_text(doc))

        if len(chunked) != len(documents):
            logger.info(
                "Chunked %d docs → %d chunks (character, size=%d)",
                len(documents),
                len(chunked),
                char_size,
            )

        return chunked

    def _chunk_by_sentences(self, documents: list[str]) -> list[str]:
        """Semantic chunking respecting sentence boundaries."""
        chunked: list[str] = []

        for doc in documents:
            if not doc or not doc.strip():
                continue

            doc_size = self._count_size(doc)

            # Small docs: keep as-is
            if doc_size <= self.chunk_size:
                chunked.append(doc)
                continue

            sentences = text_to_sentences(doc).split("\n")
            if not sentences:
                raise RuntimeError(
                    f"Sentence tokenization returned no sentences for a document of "
                    f"length {len(doc)} characters. Consider using character-based "
                    "chunking instead (e.g., set chunking_strategy='character')."
                )

            current_chunk: list[str] = []
            current_size = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                sent_size = self._count_size(sentence)

                # Oversized sentence: split with character chunker
                if sent_size > self.chunk_size:
                    if current_chunk:
                        chunked.append(" ".join(current_chunk))
                        current_chunk = []
                        current_size = 0
                    chunked.extend(self._chunk_by_characters([sentence]))
                    continue

                # Would exceed limit: save current and start new with overlap
                if current_chunk and (current_size + sent_size > self.chunk_size):
                    chunked.append(" ".join(current_chunk))
                    overlap = self._get_overlap_sentences(current_chunk)
                    current_chunk = overlap
                    current_size = sum(self._count_size(s) for s in current_chunk)

                current_chunk.append(sentence)
                current_size += sent_size

            if current_chunk:
                chunked.append(" ".join(current_chunk))

        if len(chunked) != len(documents):
            logger.info(
                "Chunked %d docs → %d chunks (sentence, size=%d tokens)",
                len(documents),
                len(chunked),
                self.chunk_size,
            )

        return chunked

    def _count_size(self, text: str) -> int:
        """Count size in tokens or characters based on configuration."""
        if self.use_token_counting:
            assert self._tokenizer_service is not None
            return self._tokenizer_service.count_tokens(text)
        return len(text)

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        """Get last sentences fitting within overlap budget."""
        if not sentences or self.chunk_overlap <= 0:
            return []

        overlap: list[str] = []
        overlap_size = 0

        for sentence in reversed(sentences):
            sent_size = self._count_size(sentence)
            if overlap_size + sent_size <= self.chunk_overlap:
                overlap.insert(0, sentence)
                overlap_size += sent_size
            else:
                break

        return overlap

    @staticmethod
    def _calculate_files_hash(data_path: Path) -> str:
        """Calculate hash of source files for cache key.

        Static method so it can be used by both EmbeddingService and RAGService
        without creating a full instance.
        """
        # Discover source files
        if data_path.is_file():
            source_files = [data_path]
        elif data_path.is_dir():
            source_files = [
                p
                for p in data_path.rglob("*")
                if p.is_file() and LoaderFactory.is_supported(p)
            ]
            source_files.sort(key=lambda p: p.relative_to(data_path).as_posix())
        else:
            return ""

        if not source_files:
            return ""

        hasher = hashlib.sha256()
        base_path = data_path if data_path.is_dir() else data_path.parent

        for file_path in source_files:
            relative = file_path.relative_to(base_path).as_posix()
            hasher.update(relative.encode())
            hasher.update(file_path.read_bytes())

        return hasher.hexdigest()[:16]

    def _generate_cache_key(self) -> str:
        """Generate unique cache key based on data and config."""
        files_hash = self._calculate_files_hash(self.data_path)
        if not files_hash:
            logger.info("No supported files in %s", self.data_path)

        # Add config to hash for uniqueness
        config_str = f"{self.chunk_size}:{self.chunk_overlap}:{self.chunking_strategy}:{self.use_token_counting}:{self.embedding_model}"
        hasher = hashlib.sha256()
        hasher.update(files_hash.encode())
        hasher.update(config_str.encode())

        return hasher.hexdigest()[:16]

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
            logger.info("Found %d supported files in %s", len(files), self.data_path)
            return files

        logger.info("Data path not found: %s", self.data_path)
        return []

    def _load_from_cache(self) -> bool:
        """Load vector store from disk cache."""
        cache_path = self._get_cache_path()

        if not cache_path.exists():
            logger.debug("Cache miss: %s", cache_path)
            return False

        self.vector_store = FAISS.load_local(
            str(cache_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("Loaded from cache: %s", cache_path)
        return True

    def _save_to_cache(self) -> None:
        """Save vector store to disk cache."""
        if not self.vector_store:
            logger.warning("No vector store to cache")
            return

        cache_path = self._get_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(cache_path))
        logger.info("Saved to cache: %s", cache_path)

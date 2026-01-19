"""Embedding service for document indexing and vector store management."""

import os
from pathlib import Path
from typing import Any


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
import torch
import chromadb

from pokeconsultor.config import settings
from pokeconsultor.services.data_loaders.factory import LoaderFactory
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService
from langchain_chroma import Chroma

# Embedding model limits (multilingual-e5-large has 512 token max)
EMBEDDING_MAX_TOKENS = 500  # Safety margin below 512
EMBEDDING_MAX_CHARS = 1800  # ~3.6 chars/token average for multilingual


class EmbeddingService(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    data_path: Path
    chunk_size: int = Field(
        default=450,
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

    _vector_store: Chroma = PrivateAttr()
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
        self._vector_store = self.configure_vector_store()


    def configure_vector_store(self) -> Chroma:
        # User requested: .cache/chromadb
        persist_path = settings.CACHE_DIR / "chroma"
        persist_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Connecting to ChromaDB at {persist_path}")
        
        client = chromadb.PersistentClient(path=str(persist_path))
        
        vector_store = Chroma(
            client=client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
        )
        return vector_store
    
    @property
    def vector_store(self) -> Chroma:
        """Public accessor for the internal vector store."""
        return self._vector_store


    def _configure_torch_threads(self) -> None:
        """Configure PyTorch to use all available CPU threads."""
        max_threads = os.cpu_count() or 1
        try:
            torch.set_num_threads(max_threads)
            torch.set_num_interop_threads(max_threads)
            logger.info("Torch threads configured: %d", max_threads)
        except Exception:
            logger.exception("Failed to configure torch threads")  # Log error when configuring threads

    def add_file_embeddings(self, file_path: Path, chunks: list[Document], file_hash: str) -> None:
        """Add embeddings for a file, associating each chunk with the file hash and path."""
        for doc in chunks:
            doc.metadata["file_hash"] = file_hash
            doc.metadata["file_path"] = str(file_path)
            
        self.vector_store.add_documents(chunks)
        # Persistência automática pelo ChromaDB

    def delete_file_embeddings(self, file_hash: str) -> None:
        """Delete all embeddings associated with a file hash."""
        # ChromaDB permite deletar por filtro de metadados
        self.vector_store.delete(where={"file_hash": file_hash})
        # Persistência automática pelo ChromaDB

    def get_file_hashes(self) -> set[str]:
        """Return all file hashes present in the vector store."""
        # ChromaDB permite buscar metadados
        results = self.vector_store.get(include=["metadatas"])
        hashes = set()
        for meta in results["metadatas"]:
            if meta and "file_hash" in meta:
                hashes.add(meta["file_hash"])
        return hashes
  
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

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split documents into overlapping chunks for better retrieval.
        
        Public method to allow external services to use the same chunking logic.
        """
        if not documents:
            return []

        if self.chunking_strategy == "sentence":
            return self._chunk_by_sentences(documents)
        return self._chunk_by_characters(documents)

    def _chunk_by_characters(self, documents: list[Document]) -> list[Document]:
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

        chunked = splitter.split_documents(documents)

        if len(chunked) != len(documents):
            logger.info(
                "Chunked %d docs → %d chunks (character, size=%d)",
                len(documents),
                len(chunked),
                char_size,
            )

        return chunked

    def _chunk_by_sentences(self, documents: list[Document]) -> list[Document]:
        """Semantic chunking using RecursiveCharacterTextSplitter with sentence priorities."""
        # Define separators to prioritize: paragraphs -> sentences -> words
        separators = ["\n\n", "\n", r"(?<=\. )", r"(?<=\! )", r"(?<=\? )", " ", ""]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_size,
            separators=separators,
            is_separator_regex=True,
            keep_separator=True,
        )

        chunked = splitter.split_documents(documents)

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









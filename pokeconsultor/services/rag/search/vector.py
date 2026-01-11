"""Vector search wrapper for FAISS-based retrieval."""

from __future__ import annotations

from typing import Any, List, Tuple

from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.embeddings import EmbeddingService


class VectorSearcher:
    """Thin wrapper over FAISS vector store similarity search."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def search(self, query: str, k: int) -> List[Tuple[Any, float]]:
        vector_store = self.embedding_service.vector_store
        if not vector_store:
            logger.warning("Vector store not initialized")
            return []
        try:
            return vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.error("Vector retrieval error: %s", exc)
            return []

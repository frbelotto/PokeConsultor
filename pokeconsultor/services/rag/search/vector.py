
from __future__ import annotations

from typing import Any, List, Tuple

from pokeconsultor.services.rag.embeddings import EmbeddingService


class VectorSearcher:

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def search(self, query: str, k: int) -> List[Tuple[Any, float]]:
        vector_store = self.embedding_service.vector_store
        
        return vector_store.similarity_search_with_score(query, k=k)

"""Hybrid executor orchestrating lexical + vector search with RRF and rerank."""

from __future__ import annotations

from typing import Any, List, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import numpy as np
from pydantic import ConfigDict, Field

from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.formatting.context import format_context
from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService
from pokeconsultor.services.rag.search.lexical import LexicalSearcher
from pokeconsultor.services.rag.search.vector import VectorSearcher


def rrf_fuse(
    vector_pairs: List[Tuple[Document, float]],
    lexical_pairs: List[Tuple[Document, float]],
    k: int,
) -> List[Tuple[Document, float]]:
    """Reciprocal Rank Fusion over two rankings."""
    # Use page_content as key, but keep track of the Document object
    vec_rank: dict[str, int] = {
        doc.page_content: i for i, (doc, _s) in enumerate(vector_pairs)
    }
    lex_rank: dict[str, int] = {
        doc.page_content: i for i, (doc, _s) in enumerate(lexical_pairs)
    }

    # Map content to Document to avoid losing metadata
    content_to_doc: dict[str, Document] = {}
    for doc, _ in vector_pairs:
        content_to_doc[doc.page_content] = doc
    for doc, _ in lexical_pairs:
        content_to_doc[doc.page_content] = doc

    all_contents = set(vec_rank.keys()) | set(lex_rank.keys())
    fused_scores: list[tuple[str, float]] = []
    for content in all_contents:
        score = 0.0
        if content in vec_rank:
            score += 1.0 / (k + vec_rank[content] + 1)
        if content in lex_rank:
            score += 1.0 / (k + lex_rank[content] + 1)
        fused_scores.append((content, score))

    fused_scores.sort(key=lambda x: x[1], reverse=True)
    return [(content_to_doc[content], score) for content, score in fused_scores]


class HybridExecutor(BaseRetriever):
    """Orchestrates lexical + vector retrieval, RRF fusion, and rerank."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    lexical_searcher: LexicalSearcher
    vector_searcher: VectorSearcher
    tokenizer_service: TokenizerService

    retrieve_k: int = Field(
        default=10, gt=0, description="Top-K to retrieve from each retriever"
    )
    lexical_k: int | None = Field(default=None, description="Override K for lexical")
    vector_k: int | None = Field(default=None, description="Override K for vector")
    rrf_k: int = Field(default=60, gt=0, description="RRF constant")
    rerank_method: str = Field(
        default="cross_encoder", description="'none' | 'cosine' | 'cross_encoder'"
    )
    rerank_k: int | None = Field(
        default=10, ge=0, description="Keep top-K after rerank"
    )

    _cross_encoder: Any | None = None

    def _ensure_cross_encoder(self) -> None:
        if self._cross_encoder is not None:
            return
        if self.rerank_method != "cross_encoder":
            return
        try:
            from langchain_community.cross_encoders.huggingface import (
                HuggingFaceCrossEncoder,
            )

            self._cross_encoder = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-base",
                model_kwargs={"device": "cpu"},
            )
            logger.debug("Cross-encoder reranker initialized (BAAI/bge-reranker-base)")
        except Exception as exc:
            raise RuntimeError(
                "Cross-encoder reranker requested but unavailable. "
                "Install 'sentence-transformers' or configure rerank_method='cosine' or 'none'."
            ) from exc

    def warmup(self) -> None:
        """Pre-initialize heavy models."""
        self._ensure_cross_encoder()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Run hybrid retrieval for a single query."""
        return self._get_relevant_documents_multi([query])

    def _get_relevant_documents_multi(self, queries: list[str]) -> List[Document]:
        """Run hybrid retrieval for multiple queries, fusing results."""
        self._ensure_cross_encoder()

        k_lex = self.lexical_k or self.retrieve_k
        k_vec = self.vector_k or self.retrieve_k

        all_vec_pairs: List[Tuple[Document, float]] = []
        all_lex_pairs: List[Tuple[Document, float]] = []

        for q in queries:
            vector_results = self.vector_searcher.search(q, k_vec)
            lexical_results = self.lexical_searcher.search(q, k_lex)

            if vector_results:
                for doc, score in vector_results:
                    if not isinstance(doc, Document):
                        all_vec_pairs.append(
                            (Document(page_content=str(doc)), float(score))
                        )
                    else:
                        all_vec_pairs.append((doc, float(score)))

            if lexical_results:
                all_lex_pairs.extend(lexical_results)

        if not all_vec_pairs and not all_lex_pairs:
            return []

        fused = rrf_fuse(all_vec_pairs, lexical_pairs=all_lex_pairs, k=self.rrf_k)
        if not fused:
            return []

        if self.rerank_k and self.rerank_k > 0:
            # Rerank against the ORIGINAL query (first in the list)
            fused = self._rerank_results(queries[0], fused, self.rerank_k)

        return [doc for doc, score in fused]

    def retrieve(self, query: str) -> list[tuple[str, float]]:
        """Deprecated: use invoke() or _get_relevant_documents.

        Maintaining for backward compatibility during transition.
        """
        docs = self.invoke(query)
        return [(doc.page_content, 1.0) for doc in docs]

    def _rerank_results(
        self, query: str, results: list[tuple[Document, float]], rerank_k: int
    ) -> list[tuple[Document, float]]:
        if self.rerank_method == "none":
            return results[:rerank_k]

        if self.rerank_method == "cross_encoder":
            if self._cross_encoder is None:
                raise RuntimeError(
                    "Cross-encoder reranker not initialized; check dependencies."
                )
            pairs: List[Tuple[str, str]] = [
                (query, doc.page_content) for doc, _ in results
            ]
            scores = list(self._cross_encoder.score(pairs))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top = order[:rerank_k]
            return [(results[i][0], float(scores[i])) for i in top]

        if self.rerank_method == "cosine":
            embeddings = self.vector_searcher.embedding_service.embeddings
            q = np.array(embeddings.embed_query(query), dtype=float)
            docs = np.array(
                embeddings.embed_documents([doc.page_content for doc, _ in results]),
                dtype=float,
            )
            q_norm = np.linalg.norm(q)
            d_norm = np.linalg.norm(docs, axis=1)
            denom = np.clip(q_norm * d_norm, a_min=1e-12, a_max=None)
            scores = (docs @ q) / denom
            order = np.argsort(scores)[::-1][:rerank_k]
            return [
                (results[int(i)][0], float(scores[int(i)]))
                for i in order  # type: ignore[arg-type]
            ]

        return results[:rerank_k]

    def format_context(
        self,
        results: list[Document] | list[tuple[Document, float]],
        max_tokens: int | None = None,
        compact: bool = True,
    ) -> str:
        # Handle both list of Documents and list of (Doc, score)
        if results and isinstance(results[0], tuple):
            formatted_results = results  # format_context handles tuples
        else:
            # format_context expects List[Tuple[Document | str, float]]
            formatted_results = [(doc, 1.0) for doc in results]

        return format_context(
            formatted_results, self.tokenizer_service, max_tokens, compact
        )

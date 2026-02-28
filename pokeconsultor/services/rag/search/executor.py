"""Hybrid executor orchestrating lexical + vector search with RRF and rerank."""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
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
    """Orchestrates lexical + vector retrieval with RRF fusion."""

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

    def warmup(self) -> None:
        """No-op warmup kept for interface stability."""
        logger.debug("HybridExecutor warmup completed (no reranker to pre-load)")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Run hybrid retrieval for a single query.

        The ``run_manager`` callback manager is accepted to comply with the
        ``BaseRetriever`` interface but is not used in this implementation.
        """
        return self._get_relevant_documents_multi([query])

    def _get_relevant_documents_multi(self, queries: list[str]) -> List[Document]:
        """Run hybrid retrieval for multiple queries, fusing results."""
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
        return [doc for doc, score in fused]

    def format_context(
        self,
        results: list[Document] | list[tuple[Document, float]],
        max_tokens: int | None = None,
        compact: bool = True,
    ) -> str:
        formatted_results: list[tuple[Document | str, float]] = []
        for item in results:
            if isinstance(item, tuple):
                doc, score = item
                formatted_results.append((doc, float(score)))
            else:
                formatted_results.append((item, 1.0))

        return format_context(
            formatted_results, self.tokenizer_service, max_tokens, compact
        )

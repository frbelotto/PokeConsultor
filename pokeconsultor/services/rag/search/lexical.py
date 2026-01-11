"""Lightweight lexical search for hybrid retrieval (TF-IDF-like)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Any

from pokeconsultor.services.logger import logger


@dataclass
class LexicalDoc:
    """Container for lexical statistics."""

    content: str
    tf: dict[str, float]
    length: int


class SimpleLexicalIndex:
    """Minimal TF-IDF-like index using whitespace tokenization."""

    def __init__(self) -> None:
        self.docs: list[LexicalDoc] = []
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.vocab_size: int = 0
        self._ready: bool = False

    _token_re = re.compile(r"\w+", re.UNICODE)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t.lower() for t in SimpleLexicalIndex._token_re.findall(text)]

    def build(self, documents: Iterable[str]) -> None:
        self.docs = []
        self.df.clear()
        for content in documents:
            tokens = self._tokenize(content)
            if not tokens:
                self.docs.append(LexicalDoc(content=content, tf={}, length=0))
                continue
            tf: dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0.0) + 1.0
            for tok in tf:
                tf[tok] = 1.0 + math.log(tf[tok])
            for tok in tf.keys():
                self.df[tok] = self.df.get(tok, 0) + 1
            self.docs.append(LexicalDoc(content=content, tf=tf, length=len(tokens)))
        self.vocab_size = len(self.df)
        n_docs = max(1, len(self.docs))
        self.idf = {
            tok: math.log((n_docs + 1) / (df + 1)) + 1.0 for tok, df in self.df.items()
        }
        self._ready = True
        logger.info("Lexical index built: %d docs, %d terms", len(self.docs), self.vocab_size)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        if not self._ready:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        q_weights = {tok: self.idf.get(tok, 0.0) for tok in q_tokens}
        scores: list[tuple[int, float]] = []
        for i, doc in enumerate(self.docs):
            if not doc.tf:
                continue
            s = 0.0
            for tok, w in q_weights.items():
                if w == 0.0:
                    continue
                s += doc.tf.get(tok, 0.0) * w
            if s > 0.0:
                scores.append((i, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:k]
        return [(self.docs[i].content, score) for i, score in top]


class LexicalSearcher:
    """Thin wrapper around SimpleLexicalIndex for retrieval."""

    def __init__(self) -> None:
        self.index = SimpleLexicalIndex()

    def build_from_documents(self, documents: Iterable[str]) -> None:
        self.index.build(documents)

    def build_from_vector_store(self, vector_store: Any) -> None:
        """Extract documents from a LangChain docstore-compatible vector store."""
        contents: list[str] = []
        try:
            docstore = vector_store.docstore
            idx_map = vector_store.index_to_docstore_id
            for _idx, doc_id in idx_map.items():
                doc = docstore.search(doc_id) if hasattr(docstore, "search") else None
                if not doc or isinstance(doc, str):
                    continue
                contents.append(getattr(doc, "page_content", "") or "")
        except Exception as exc:
            logger.exception("Failed building lexical index from vector store: %s", exc)
        self.index.build(contents)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        return self.index.search(query, k)

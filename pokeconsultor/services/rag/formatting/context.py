"""Context formatting utilities for retrieval results."""

from __future__ import annotations

from typing import List, Tuple

from pokeconsultor.services.rag.formatting.tokenizer import TokenizerService


def format_context(
    results: List[Tuple[str, float]],
    tokenizer_service: TokenizerService,
    max_tokens: int | None = None,
    compact: bool = True,
) -> str:
    """Format retrieval results into optimized context string."""
    if not results:
        return ""

    token_limit = max_tokens or tokenizer_service.get_max_context_tokens()
    remaining = max(0, token_limit)

    if compact:
        return _format_compact(results, tokenizer_service, remaining)
    return _format_verbose(results, tokenizer_service, remaining)


def _format_compact(
    results: List[Tuple[str, float]],
    tokenizer_service: TokenizerService,
    token_budget: int,
) -> str:
    """Compact format: minimal headers, deduplicated content."""
    seen_hashes: set[int] = set()
    unique_results: List[Tuple[str, float]] = []

    for content, score in results:
        content_hash = hash(content[:100].strip().lower())
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_results.append((content, score))

    parts: list[str] = []
    remaining = token_budget
    separator_tokens = 2  # "\n---\n" ~2 tokens

    for i, (content, _score) in enumerate(unique_results):
        if remaining <= separator_tokens + 10:
            break

        header = f"[{i + 1}] "
        header_tokens = tokenizer_service.count_tokens(header)

        if remaining <= header_tokens:
            break

        body_budget = remaining - header_tokens - separator_tokens
        body = _clean_content(content)
        body = tokenizer_service.trim_to_tokens(body, body_budget)
        body_tokens = tokenizer_service.count_tokens(body)

        if body:
            parts.append(header + body)
            remaining -= header_tokens + body_tokens + separator_tokens

    return "\n---\n".join(parts)


def _format_verbose(
    results: List[Tuple[str, float]],
    tokenizer_service: TokenizerService,
    token_budget: int,
) -> str:
    """Verbose format: full headers with result numbers."""
    parts: list[str] = []
    remaining = token_budget

    for i, (content, _score) in enumerate(results, 1):
        header = f"[Resultado {i}]\n"
        header_tokens = tokenizer_service.count_tokens(header)

        if remaining <= header_tokens:
            break

        body_budget = remaining - header_tokens
        body = tokenizer_service.trim_to_tokens(content or "", body_budget)
        body_tokens = tokenizer_service.count_tokens(body)

        parts.append(header + body)
        remaining -= header_tokens + body_tokens

        if remaining <= 0:
            break

    return "\n\n".join(parts)


def _clean_content(text: str) -> str:
    """Normalize whitespace to reduce token usage without losing meaning."""
    if not text:
        return ""
    lines = text.split("\n")
    cleaned_lines = [" ".join(line.split()) for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

"""Token-aware text chunking with page tracking."""

from __future__ import annotations

import re

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.embeddings import get_model, get_tokenizer
from src.extraction import page_at_position


def _page_end_offset(page_boundaries: list[int], pos: int) -> int:
    """Return the character offset where the page containing pos ends."""

    for boundary in page_boundaries:
        if pos < boundary:
            return boundary
    return page_boundaries[-1] if page_boundaries else 0


def chunk_text(
    text: str, page_boundaries: list[int], page_nums: list[int]
) -> list[dict]:
    """Split text into token-aware chunks with page tracking.

    Uses the actual tokenizer to count tokens (not character heuristics).
    Applies binary-search-style adjustment to hit target_tokens per chunk.
    Overlaps chunks by CHUNK_OVERLAP tokens to preserve context.

    Args:
        text: Full concatenated text from all pages.
        page_boundaries: Cumulative character offsets per page.
        page_nums: Page numbers corresponding to boundaries.

    Returns:
        List of dicts with 'text', 'start_page', and 'end_page' per chunk.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    max_tokens = model.max_seq_length or 512
    target_tokens = min(CHUNK_SIZE, max_tokens - 10)
    chunks = []

    init_chars = int(target_tokens * 2.5)
    overlap_chars = int(CHUNK_OVERLAP * 4)
    char_pos = 0

    while char_pos < len(text):
        end_pos = min(char_pos + init_chars, len(text))

        # Clamp end_pos to current page boundary to avoid cross-page chunks
        page_end = _page_end_offset(page_boundaries, char_pos)
        end_pos = min(end_pos, page_end - 1)

        raw = text[char_pos:end_pos]

        token_count = len(tokenizer.encode(raw))
        while token_count > target_tokens and end_pos > char_pos + 50:
            end_pos -= max(1, (token_count - target_tokens) * 2)
            end_pos = max(end_pos, char_pos + 50)
            raw = text[char_pos:end_pos]
            token_count = len(tokenizer.encode(raw))

        raw = raw.strip()
        if not raw:
            char_pos = end_pos
            continue

        start_page = page_at_position(page_boundaries, page_nums, char_pos)
        end_page = page_at_position(page_boundaries, page_nums, end_pos)

        chunks.append(
            {
                "text": raw,
                "start_page": start_page,
                "end_page": end_page,
            }
        )

        # If we've reached the end of a page, jump to the next page
        if end_pos >= page_end - 1:
            char_pos = page_end
            continue

        next_pos = end_pos - overlap_chars
        if next_pos <= char_pos:
            next_pos = end_pos
        char_pos = next_pos

    return chunks


def chunk_markdown(text: str, source_file: str = "") -> list[dict]:
    """Split markdown text into token-aware chunks without page boundaries.

    Treats the entire document as a single page. Used for benchmark
    evaluation with plain markdown files (no PDF extraction).

    Args:
        text: Raw markdown text.
        source_file: Filename for attribution in returned chunks.

    Returns:
        List of dicts with 'text', 'source_file', 'start_page', 'end_page'.
    """
    tokenizer = get_tokenizer()
    model = get_model()
    max_tokens = model.max_seq_length or 512
    target_tokens = min(CHUNK_SIZE, max_tokens - 10)

    init_chars = int(target_tokens * 2.5)
    overlap_chars = int(CHUNK_OVERLAP * 4)
    chunks = []
    char_pos = 0

    while char_pos < len(text):
        end_pos = min(char_pos + init_chars, len(text))
        raw = text[char_pos:end_pos]

        token_count = len(tokenizer.encode(raw))
        while token_count > target_tokens and end_pos > char_pos + 50:
            end_pos -= max(1, (token_count - target_tokens) * 2)
            end_pos = max(end_pos, char_pos + 50)
            raw = text[char_pos:end_pos]
            token_count = len(tokenizer.encode(raw))

        raw = raw.strip()
        if not raw:
            char_pos = end_pos
            continue

        chunks.append(
            {
                "text": raw,
                "source_file": source_file,
                "start_page": 1,
                "end_page": 1,
            }
        )

        next_pos = end_pos - overlap_chars
        if next_pos <= char_pos:
            next_pos = end_pos
        char_pos = next_pos

    return chunks

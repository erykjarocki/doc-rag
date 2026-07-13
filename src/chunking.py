"""Token-aware text chunking with page tracking."""

from __future__ import annotations

from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.embeddings import get_model, get_tokenizer
from src.extraction import page_at_position


def chunk_text(text: str, page_boundaries: list[int], page_nums: list[int]) -> list[dict]:
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

        chunks.append({
            "text": raw,
            "start_page": start_page,
            "end_page": end_page,
        })

        next_pos = end_pos - overlap_chars
        if next_pos <= char_pos:
            next_pos = end_pos
        char_pos = next_pos

    return chunks

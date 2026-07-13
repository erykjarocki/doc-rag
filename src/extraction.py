"""PDF text extraction and page-boundary utilities."""

from __future__ import annotations

import fitz


def extract_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with 'page_num' (1-indexed) and 'text' per page.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,
            "text": text.strip(),
        })
    doc.close()
    return pages


def get_page_boundaries(pages_data: list[dict]) -> list[int]:
    """Calculate cumulative character positions marking page boundaries.

    Args:
        pages_data: List of dicts with 'text' per page.

    Returns:
        List of cumulative character offsets (one per page).
    """
    boundaries = []
    total = 0
    for p in pages_data:
        total += len(p["text"]) + 1
        boundaries.append(total)
    return boundaries


def get_full_text_with_page_info(pages_data: list[dict]) -> tuple[str, list[int]]:
    """Join all page texts into one string and compute page boundary offsets.

    Args:
        pages_data: List of dicts with 'page_num' and 'text' per page.

    Returns:
        Tuple of (full_text, page_boundaries) where page_boundaries is a
        list of cumulative character positions marking where each page ends.
    """
    segments = []
    page_nums = []
    for p in pages_data:
        segments.append(p["text"])
        page_nums.append(p["page_num"])
    full_text = "\n".join(segments)
    boundaries = get_page_boundaries(pages_data)
    return full_text, boundaries


def page_at_position(page_boundaries: list[int], page_nums: list[int], pos: int) -> int:
    """Find which page a character position falls on.

    Args:
        page_boundaries: Cumulative character offsets per page.
        page_nums: Corresponding page numbers (1-indexed).
        pos: Character position in the full text.

    Returns:
        Page number containing the given position.
    """
    for i, boundary in enumerate(page_boundaries):
        if pos < boundary:
            return page_nums[i]
    return page_nums[-1] if page_nums else 1

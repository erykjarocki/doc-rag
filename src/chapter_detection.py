"""Universal chapter detection for PDF documents.

Uses a three-layer fallback strategy to detect chapters/sections in any PDF:

1. **PDF Outline (TOC)** — reads the document's built-in bookmark structure.
   Most professionally produced PDFs include this metadata. Fastest and most
   reliable when available.

2. **Font-size analysis** — extracts per-span font metadata from each page and
   classifies headings by comparing font sizes to the document's body-text
   baseline. Works for PDFs without bookmarks that use visual hierarchy
   (larger/bolder fonts for headings).

3. **Regex fallback** — pattern-matches common heading formats in extracted
   text. Language-agnostic patterns cover English, Polish, and generic
   numbered/structured headings. Last resort when no structural data exists.

The detector is lazy: each strategy is only evaluated if all previous ones
produce no results for the document.
"""

from __future__ import annotations

import re
from collections import defaultdict

import fitz

# ---------------------------------------------------------------------------
# Regex patterns for fallback detection (language-agnostic)
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    # English: "Chapter 1", "Section 2.1", "Part I", "Article 3", "Appendix A"
    re.compile(
        r"^(?:Chapter|Section|Part|Article|Appendix)\s+[\dIVXLCDM]+(?:\.\d+)*",
        re.IGNORECASE,
    ),
    # Polish: "Rozdział V", "CZĘŚĆ II", "Tom 1", "Dział 3", "Artykuł 7"
    re.compile(
        r"^(?:Rozdział|CZĘŚĆ|Tom|Dział|Artykuł|Załącznik)\s+[\dIVXLCDM]+",
        re.IGNORECASE,
    ),
    # Generic numbered headings: "1. Introduction", "1.1 Overview", "2.3.1 Details"
    re.compile(r"^\d{1,3}(?:\.\d{1,3}){0,3}\.?\s+.*"),
    # Legal/special: "§ 42", "Paragraph 3", "Paragraf 7"
    re.compile(r"^(?:§|Paragraph|Paragraf)\s+[\dIVXLCDM]+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# TOC helpers
# ---------------------------------------------------------------------------

def _build_toc_map(doc: fitz.Document) -> dict[int, str]:
    """Build a page_number → chapter breadcrumb mapping from the PDF outline.

    Args:
        doc: Opened PyMuPDF document.

    Returns:
        Dict mapping 1-based page numbers to chapter breadcrumb strings
        (e.g. "Book I > Chapter 3 > Section 1"). Pages between two TOC
        entries inherit the preceding entry's chapter path.
    """
    toc = doc.get_toc(simple=True)
    if not toc:
        return {}

    # Flatten hierarchical TOC into a stack and build page→path mapping
    page_chapters: dict[int, str] = {}
    stack: list[str] = []  # tracks current hierarchy at each level

    for level, title, page_num in toc:
        if page_num < 1:
            continue
        # Trim stack to current level (1-indexed)
        stack = stack[: level - 1]
        stack.append(title.strip())
        page_chapters[page_num] = " > ".join(stack)

    if not page_chapters:
        return {}

    # Fill gaps: every page between two TOC entries inherits the previous entry
    max_page = doc.page_count
    filled: dict[int, str] = {}
    current_chapter = ""

    for page in range(1, max_page + 1):
        if page in page_chapters:
            current_chapter = page_chapters[page]
        if current_chapter:
            filled[page] = current_chapter

    return filled


# ---------------------------------------------------------------------------
# Font-size analysis
# ---------------------------------------------------------------------------

def _extract_font_sizes(doc: fitz.Document) -> tuple[float, float, float]:
    """Compute body-text font size statistics across the entire document.

    Returns:
        Tuple of (mean_size, heading_threshold, subheading_threshold).
        heading_threshold = mean + 4pt, subheading_threshold = mean + 2pt.
        These offsets are calibrated for formally structured documents where
        body text dominates the font-size distribution.
    """
    sizes: list[float] = []
    for page in doc:
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        sizes.append(span["size"])

    if not sizes:
        return (11.0, 15.0, 13.0)  # safe defaults

    mean_size = sum(sizes) / len(sizes)
    return (mean_size, mean_size + 4.0, mean_size + 2.0)


def _is_bold(span: dict) -> bool:
    """Check if a span's font is bold (bit 4 of flags field)."""
    return bool(span.get("flags", 0) & 16)


def _classify_span(
    span: dict,
    mean_size: float,
    heading_threshold: float,
    subheading_threshold: float,
) -> str:
    """Classify a text span as heading level or body content.

    Classification rules (in priority order):
    1. font_size >= heading_threshold → "heading"
    2. font_size >= subheading_threshold AND bold AND ALL-CAPS → "heading"
    3. font_size >= subheading_threshold → "subheading"
    4. Everything else → "content"
    """
    size = span.get("size", 0)
    text = span.get("text", "").strip()

    if not text:
        return "content"

    if size >= heading_threshold:
        return "heading"

    if size >= subheading_threshold:
        if _is_bold(span) and text.isupper() and len(text) <= 100:
            return "heading"
        return "subheading"

    return "content"


def _build_font_map(doc: fitz.Document) -> dict[int, str]:
    """Build a page_number → chapter mapping using font-size analysis.

    For each page, collects heading-classified spans and builds a chapter
    string from them. Returns the most specific (deepest) heading found
    on each page.
    """
    mean_size, heading_threshold, subheading_threshold = _extract_font_sizes(doc)

    page_headings: dict[int, list[str]] = defaultdict(list)

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        blocks = page.get_text("dict", sort=True)["blocks"]

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text_parts = []
                line_role = "content"
                max_role_priority = 0

                for span in line.get("spans", []):
                    role = _classify_span(
                        span, mean_size, heading_threshold, subheading_threshold
                    )
                    role_priority = {"heading": 3, "subheading": 2, "content": 1}
                    if role_priority.get(role, 0) > max_role_priority:
                        max_role_priority = role_priority.get(role, 0)
                        line_role = role
                    line_text_parts.append(span.get("text", ""))

                if line_role in ("heading", "subheading"):
                    line_text = " ".join(line_text_parts).strip()
                    if line_text:
                        page_headings[page_idx + 1].append(line_text)

    # Build page→chapter from collected headings
    result: dict[int, str] = {}
    for page_num, headings in page_headings.items():
        # Use the most specific heading on the page
        # If multiple headings, join the last two (parent > child)
        if len(headings) >= 2:
            result[page_num] = f"{headings[-2]} > {headings[-1]}"
        else:
            result[page_num] = headings[0]

    return result


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------

def _regex_fallback(doc: fitz.Document) -> dict[int, str]:
    """Build a page→chapter mapping using regex pattern matching.

    Scans the first ~500 characters of each page for heading patterns.
    Returns a mapping only for pages where a heading is actually found.
    Unlike TOC-based detection, does NOT carry forward chapter names across
    pages — each page must independently match a heading pattern.
    """
    result: dict[int, str] = {}

    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text = page.get_text("text")[:500]

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for pattern in _HEADING_PATTERNS:
                match = pattern.match(line)
                if match:
                    result[page_idx + 1] = match.group(0).strip()
                    break
            else:
                continue
            break  # found a heading on this page, stop scanning

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ChapterDetector:
    """Detects chapter/section boundaries in a PDF document.

    Uses a three-layer fallback:
    1. PDF outline/bookmarks (TOC) — fastest, most reliable
    2. Font-size analysis — works for visually structured PDFs
    3. Regex pattern matching — last resort

    Example::

        detector = ChapterDetector("book.pdf")
        chapter = detector.get_chapter_for_page(42)
        print(chapter)  # "Rozdział III > Podrozdział 1"
        detector.close()

    Or as a context manager::

        with ChapterDetector("book.pdf") as detector:
            for page_num in range(1, 100):
                ch = detector.get_chapter_for_page(page_num)
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self._toc_map: dict[int, str] | None = None
        self._font_map: dict[int, str] | None = None
        self._regex_map: dict[int, str] | None = None
        self._active_map: dict[int, str] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the underlying PDF document."""
        if self.doc:
            self.doc.close()

    def detect_strategy(self) -> str:
        """Return which detection strategy is active for this document.

        Returns one of: "toc", "font", "regex", "none".
        """
        self._ensure_loaded()
        if self._active_map is self._toc_map and self._toc_map:
            return "toc"
        if self._active_map is self._font_map and self._font_map:
            return "font"
        if self._active_map is self._regex_map and self._regex_map:
            return "regex"
        return "none"

    def get_chapter_for_page(self, page_num: int) -> str | None:
        """Return the chapter name for a 1-based page number.

        Args:
            page_num: 1-based page number.

        Returns:
            Chapter name string (may include " > " hierarchy), or None if
            no chapter could be detected for this page.
        """
        self._ensure_loaded()
        if self._active_map is None:
            return None
        return self._active_map.get(page_num)

    def get_all_chapters(self) -> dict[int, str]:
        """Return the full page→chapter mapping for this document.

        Returns:
            Dict mapping 1-based page numbers to chapter names.
        """
        self._ensure_loaded()
        return dict(self._active_map) if self._active_map else {}

    def _ensure_loaded(self):
        """Lazy-load the chapter maps using fallback strategy."""
        if self._active_map is not None:
            return

        # Strategy 1: TOC
        self._toc_map = _build_toc_map(self.doc)
        if self._toc_map:
            self._active_map = self._toc_map
            return

        # Strategy 2: Font analysis
        self._font_map = _build_font_map(self.doc)
        if self._font_map:
            self._active_map = self._font_map
            return

        # Strategy 3: Regex fallback (always returns something or empty dict)
        self._regex_map = _regex_fallback(self.doc)
        self._active_map = self._regex_map

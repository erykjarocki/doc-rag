# Plan: Universal Chapter Detection Upgrade

## Problem

`src/ingest.py` detects chapters using regex patterns (`Rozdział X`, `Chapter X`, `CZĘŚĆ X`) on extracted text. This is fragile:

- Only works if heading text matches exact regex patterns
- Language-specific (Polish/English only)
- Misses chapters that rely on font styling (bold, size) rather than text patterns
- Ignores PDF structural metadata (bookmarks/outlines) that many PDFs contain
- Chapter detection runs after chunking, so headings landing mid-chunk may be missed

Production RAG systems (pymupdf4llm, ChunkNorris, RAGFlow) use a **layered approach**: TOC → font analysis → regex fallback.

## Approach

Create a new `src/chapter_detection.py` module with a `ChapterDetector` class implementing three detection strategies tried in order:

### Strategy 1: PDF Outline/TOC (Best)
- Use `doc.get_toc()` to read the PDF's built-in bookmark structure
- Build a `page_number → chapter_path` mapping (e.g., `{7: "Ksiega Rodzaju", 90: "Ksiega Wyjscia"}`)
- For each chunk, look up its page number to get the chapter
- Handles hierarchical TOC (Level 1 > Level 2 > Level 3) by joining with " > "
- **Works for**: PDFs with bookmarks (most professionally produced PDFs)

### Strategy 2: Font-Size Analysis (Good fallback)
- Extract span-level metadata via `page.get_text("dict")` for each page
- Compute the mean font size across the document (body text dominates, anchoring the mean)
- Classify spans: `font_size > mean + 4pt` = heading, `font_size > mean + 2pt` = subheading
- Also check for bold flag (`flags & 16`) as additional heading signal
- Merge consecutive heading lines within 25pt vertical proximity
- Build a page→heading mapping from classified spans
- **Works for**: PDFs without TOC but with visually distinct headings (most books, papers, reports)

### Strategy 3: Regex Fallback (Last resort)
- Keep existing regex patterns plus add generic patterns for common formats:
  - `Part X`, `Section X`, `Article X`, `§ X`, `Tom X`
  - Numbered patterns: `1.`, `1.1`, `1.1.1`
- Applied to the first line(s) of each page's text
- **Works for**: Any PDF where text extraction succeeds but no structural metadata exists

### Orchestration

```python
class ChapterDetector:
    def __init__(self, pdf_path: str):
        self.doc = fitz.open(pdf_path)
        self._toc_map = None   # lazy
        self._font_map = None  # lazy

    def get_chapter_for_page(self, page_num: int) -> str | None:
        """Return chapter name for a 1-based page number."""
        # Try TOC first
        if self._toc_map is None:
            self._toc_map = self._build_toc_map()
        if self._toc_map:
            return self._toc_map.get(page_num)

        # Fall back to font analysis
        if self._font_map is None:
            self._font_map = self._build_font_map()
        if self._font_map:
            return self._font_map.get(page_num)

        # Fall back to regex on page text
        return self._regex_fallback(page_num)

    def close(self):
        self.doc.close()
```

## Files to Change

### 1. NEW: `src/chapter_detection.py`
- `ChapterDetector` class with three strategies
- `_build_toc_map() → dict[int, str]` — TOC-based page→chapter mapping
- `_build_font_map() → dict[int, str]` — font-size-based page→chapter mapping
- `_regex_fallback(page_num) → str | None` — regex on page text
- `_classify_heading_spans(spans, mean_size, heading_threshold, subheading_threshold)` — span classifier
- Helper: `_merge_toc_hierarchy(toc_entries) → dict[int, str]` — flatten TOC to breadcrumb paths

### 2. MODIFY: `src/ingest.py`
- Remove `CHAPTER_PATTERN` regex constant
- Remove `detect_chapter()` function
- In `process_book()`: create `ChapterDetector`, use `get_chapter_for_page(page_num)` for each chunk instead of scanning chunk text
- Pass `chapter` based on `chunk["start_page"]` lookup
- Keep backward-compatible output format (same `chapter` field in payload)

### 3. MODIFY: `pyproject.toml`
- No new dependencies (PyMuPDF already installed)

### 4. NEW: `tests/unit/test_chapter_detection.py`
- `TestTocMapBuilder` — mock `doc.get_toc()` return values, verify page→chapter mapping
- `TestFontMapBuilder` — mock `page.get_text("dict")` returns, verify heading classification
- `TestRegexFallback` — test generic patterns (Part, Section, Article, §, numbered)
- `TestChapterDetectorOrchestration` — verify fallback order (TOC → font → regex)
- `TestTocHierarchy` — verify nested TOC produces breadcrumb paths

### 5. MODIFY: `tests/unit/test_ingest.py`
- Remove `TestDetectChapter` class (function no longer exists)
- Add tests for chapter assignment in `process_book()` via mocked `ChapterDetector`
- Keep `TestPageBoundaries`, `TestGetFullTextWithPageInfo`, `TestPageAtPosition`

### 6. MODIFY: `tests/conftest.py`
- Add `sample_toc` fixture: `[(1, "Chapter 1", 1), (2, "Section 1.1", 3), ...]`
- Add `sample_font_spans` fixture: mock span dicts with `size`, `flags`, `text`, `bbox`

### 7. MODIFY: `docs/architecture.md`
- Update ingestion pipeline diagram to show `ChapterDetector` instead of `detect_chapter()`
- Add "Chapter Detection" subsection under Design Decisions explaining the three strategies

### 8. MODIFY: `docs/improvements.md`
- Mark improvement #9 (better chunking) as partially addressed
- Add new entry for "Universal chapter detection" and mark it done

## Implementation Order

1. Create `src/chapter_detection.py` with all three strategies + tests
2. Update `tests/conftest.py` with new fixtures
3. Update `src/ingest.py` to use `ChapterDetector`
4. Update `tests/unit/test_ingest.py`
5. Run tests to verify
6. Update documentation
7. Run linter (`ruff`)

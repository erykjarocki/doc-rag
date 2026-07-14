"""Document format adapters for multi-format ingestion.

Provides a unified Document abstraction across PDF, plain text, Markdown,
and source code files. Each adapter extracts text into sections with
metadata, producing a Document that the ingestion pipeline can process
uniformly.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from src.extraction import extract_pdf, get_full_text_with_page_info


@dataclass
class DocumentSection:
    """A logical section within a document (chapter, heading, function, etc.)."""

    text: str
    section_name: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive


@dataclass
class Document:
    """Unified document representation produced by adapters."""

    name: str
    full_text: str
    page_boundaries: list[int]
    page_nums: list[int]
    sections: list[DocumentSection]


def section_for_position(
    sections: list[DocumentSection], char_pos: int, full_text: str
) -> str:
    """Map a character position in full_text to its section name.

    Builds cumulative line offsets from section boundaries and checks which
    section contains the given position. Falls back to the last section.
    """
    if not sections:
        return "unknown"

    lines = full_text.split("\n")
    cum_chars = 0
    section_boundaries: list[tuple[int, str]] = []

    for sec in sections:
        section_boundaries.append((cum_chars, sec.section_name))
        for line in lines[sec.start_line - 1 : sec.end_line]:
            cum_chars += len(line) + 1  # +1 for newline

    for i, (offset, name) in enumerate(section_boundaries):
        if i + 1 < len(section_boundaries):
            next_offset = section_boundaries[i + 1][0]
            if char_pos < next_offset:
                return name
        else:
            return name

    return sections[-1].section_name if sections else "unknown"


# ---------------------------------------------------------------------------
# PDF Adapter
# ---------------------------------------------------------------------------

class PDFAdapter:
    """Adapter for PDF files using PyMuPDF extraction and chapter detection."""

    format_name = "pdf"

    def extract(self, file_path: str) -> Document:
        from src.chapter_detection import ChapterDetector

        name = os.path.splitext(os.path.basename(file_path))[0]
        pages_data = extract_pdf(file_path)
        full_text, page_boundaries = get_full_text_with_page_info(pages_data)
        page_nums = [p["page_num"] for p in pages_data]

        # Build sections from chapter detection
        sections: list[DocumentSection] = []
        with ChapterDetector(file_path) as detector:
            detector.detect_strategy()
            current_chapter = None
            chapter_start_page = None

            for page_num in page_nums:
                chapter = detector.get_chapter_for_page(page_num) or "unknown"
                if chapter != current_chapter:
                    if current_chapter is not None:
                        # Find line range for previous chapter
                        start_char = sum(
                            len(p["text"]) + 1 for p in pages_data[: chapter_start_page - 1]
                        )
                        end_char = sum(
                            len(p["text"]) + 1 for p in pages_data[: page_num - 1]
                        )
                        sections.append(
                            DocumentSection(
                                text=full_text[start_char:end_char],
                                section_name=current_chapter,
                                start_line=chapter_start_page,
                                end_line=page_num - 1,
                            )
                        )
                    current_chapter = chapter
                    chapter_start_page = page_num

            # Last chapter
            if current_chapter is not None and chapter_start_page is not None:
                start_char = sum(
                    len(p["text"]) + 1 for p in pages_data[: chapter_start_page - 1]
                )
                sections.append(
                    DocumentSection(
                        text=full_text[start_char:],
                        section_name=current_chapter,
                        start_line=chapter_start_page,
                        end_line=len(page_nums),
                    )
                )

        return Document(
            name=name,
            full_text=full_text,
            page_boundaries=page_boundaries,
            page_nums=page_nums,
            sections=sections,
        )


# ---------------------------------------------------------------------------
# Plain Text Adapter
# ---------------------------------------------------------------------------

class PlainTextAdapter:
    """Adapter for plain .txt files. Treats the entire file as one section."""

    format_name = "text"

    def extract(self, file_path: str) -> Document:
        name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, encoding="utf-8") as f:
            full_text = f.read()

        lines = full_text.split("\n")
        total_lines = len(lines)

        # Single section for the entire file
        sections = [
            DocumentSection(
                text=full_text,
                section_name="full_text",
                start_line=1,
                end_line=total_lines,
            )
        ]

        # Non-PDF: treat entire document as one "page"
        page_boundaries = [len(full_text) + 1]
        page_nums = [1]

        return Document(
            name=name,
            full_text=full_text,
            page_boundaries=page_boundaries,
            page_nums=page_nums,
            sections=sections,
        )


# ---------------------------------------------------------------------------
# Markdown Adapter
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownAdapter:
    """Adapter for Markdown files. Parses headings as sections."""

    format_name = "markdown"

    def extract(self, file_path: str) -> Document:
        name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, encoding="utf-8") as f:
            full_text = f.read()

        lines = full_text.split("\n")
        total_lines = len(lines)

        # Find heading positions
        heading_positions: list[tuple[int, str]] = []  # (line_idx, title)
        for i, line in enumerate(lines):
            m = _HEADING_RE.match(line)
            if m:
                heading_positions.append((i, m.group(2).strip()))

        sections: list[DocumentSection] = []
        if not heading_positions:
            sections.append(
                DocumentSection(
                    text=full_text,
                    section_name="full_text",
                    start_line=1,
                    end_line=total_lines,
                )
            )
        else:
            # Content before first heading
            if heading_positions[0][0] > 0:
                pre_text = "\n".join(lines[: heading_positions[0][0]])
                sections.append(
                    DocumentSection(
                        text=pre_text,
                        section_name="preamble",
                        start_line=1,
                        end_line=heading_positions[0][0],
                    )
                )

            for idx, (line_idx, title) in enumerate(heading_positions):
                start = line_idx
                if idx + 1 < len(heading_positions):
                    end = heading_positions[idx + 1][0] - 1
                else:
                    end = total_lines
                sec_text = "\n".join(lines[start : end + 1])
                sections.append(
                    DocumentSection(
                        text=sec_text,
                        section_name=title,
                        start_line=start + 1,
                        end_line=end + 1,
                    )
                )

        # Non-PDF: single "page"
        page_boundaries = [len(full_text) + 1]
        page_nums = [1]

        return Document(
            name=name,
            full_text=full_text,
            page_boundaries=page_boundaries,
            page_nums=page_nums,
            sections=sections,
        )


# ---------------------------------------------------------------------------
# Code Adapter
# ---------------------------------------------------------------------------

_CODE_SECTION_RE = re.compile(
    r"^(?:"
    r"(?:async\s+)?def\s+\w+"  # Python / generic
    r"|class\s+\w+"  # class
    r"|fn\s+\w+"  # Rust
    r"|func\s+(?:\([^)]*\)\s+)?\w+"  # Go
    r"|function\s+\w+"  # JS
    r"|export\s+(?:default\s+)?(?:async\s+)?function\s+\w+"  # JS export
    r"|interface\s+\w+"  # TypeScript / Java
    r"|type\s+\w+"  # Go / TS
    r"|struct\s+\w+"  # Rust / C
    r"|impl(?:es)?\s+[\w<>]+"  # Rust / Java
    r"|trait\s+\w+"  # Rust
    r"|enum\s+\w+"  # Rust / TS
    r"|module\s+\w+"  # Ruby
    r"|@\w+"  # Decorators (Python/TS)
    r")",
    re.MULTILINE,
)


class CodeAdapter:
    """Adapter for source code files. Detects functions/classes as sections."""

    format_name = "code"

    def extract(self, file_path: str) -> Document:
        name = os.path.splitext(os.path.basename(file_path))[0]

        with open(file_path, encoding="utf-8") as f:
            full_text = f.read()

        lines = full_text.split("\n")
        total_lines = len(lines)

        # Find section boundaries
        section_starts: list[tuple[int, str]] = []
        for i, line in enumerate(lines):
            m = _CODE_SECTION_RE.match(line.strip())
            if m:
                # Extract the name: first word after keyword
                raw = m.group(0).strip()
                parts = raw.split()
                # Skip keywords like 'def', 'class', 'fn', 'func', 'function', etc.
                sec_name = parts[-1] if len(parts) > 1 else parts[0]
                # Clean trailing colons, parens, etc.
                sec_name = re.sub(r"[:({\[].*$", "", sec_name)
                section_starts.append((i, sec_name))

        sections: list[DocumentSection] = []

        if not section_starts:
            sections.append(
                DocumentSection(
                    text=full_text,
                    section_name="full_file",
                    start_line=1,
                    end_line=total_lines,
                )
            )
        else:
            # Content before first section
            if section_starts[0][0] > 0:
                pre_text = "\n".join(lines[: section_starts[0][0]])
                if pre_text.strip():
                    sections.append(
                        DocumentSection(
                            text=pre_text,
                            section_name="imports_and_preamble",
                            start_line=1,
                            end_line=section_starts[0][0],
                        )
                    )

            for idx, (line_idx, sec_name) in enumerate(section_starts):
                start = line_idx
                if idx + 1 < len(section_starts):
                    end = section_starts[idx + 1][0] - 1
                else:
                    end = total_lines
                sec_text = "\n".join(lines[start : end + 1])
                sections.append(
                    DocumentSection(
                        text=sec_text,
                        section_name=sec_name,
                        start_line=start + 1,
                        end_line=end + 1,
                    )
                )

        # Non-PDF: single "page"
        page_boundaries = [len(full_text) + 1]
        page_nums = [1]

        return Document(
            name=name,
            full_text=full_text,
            page_boundaries=page_boundaries,
            page_nums=page_nums,
            sections=sections,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTERS = {
    ".pdf": PDFAdapter,
    ".txt": PlainTextAdapter,
    ".md": MarkdownAdapter,
    ".markdown": MarkdownAdapter,
    ".py": CodeAdapter,
    ".js": CodeAdapter,
    ".ts": CodeAdapter,
    ".jsx": CodeAdapter,
    ".tsx": CodeAdapter,
    ".rs": CodeAdapter,
    ".go": CodeAdapter,
    ".java": CodeAdapter,
    ".rb": CodeAdapter,
    ".c": CodeAdapter,
    ".cpp": CodeAdapter,
    ".h": CodeAdapter,
    ".hpp": CodeAdapter,
    ".cs": CodeAdapter,
    ".php": CodeAdapter,
    ".swift": CodeAdapter,
    ".kt": CodeAdapter,
    ".scala": CodeAdapter,
    ".r": CodeAdapter,
    ".R": CodeAdapter,
    ".lua": CodeAdapter,
    ".sh": CodeAdapter,
    ".bash": CodeAdapter,
    ".zsh": CodeAdapter,
    ".sql": CodeAdapter,
    ".html": CodeAdapter,
    ".css": CodeAdapter,
    ".scss": CodeAdapter,
    ".yaml": CodeAdapter,
    ".yml": CodeAdapter,
    ".toml": CodeAdapter,
    ".json": CodeAdapter,
    ".xml": CodeAdapter,
    ".csv": PlainTextAdapter,
    ".log": PlainTextAdapter,
}


def get_adapter(file_path: str):
    """Return the appropriate adapter for a file based on its extension.

    Raises ValueError if the file extension is not supported.
    """
    ext = os.path.splitext(file_path)[1].lower()
    adapter_cls = _ADAPTERS.get(ext)
    if adapter_cls is None:
        supported = sorted(set(_ADAPTERS.keys()))
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported: {', '.join(supported)}"
        )
    return adapter_cls()


def supported_extensions() -> list[str]:
    """Return sorted list of all supported file extensions."""
    return sorted(set(_ADAPTERS.keys()))

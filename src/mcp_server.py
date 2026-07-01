#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from src.retriever import search_book, format_fragments_for_prompt
from src.qdrant_store import list_collections

mcp = FastMCP("pdf-rag")


@mcp.tool()
def search_book_tool(question: str, book: str | None = None) -> str:
    """Search the knowledge base for relevant fragments from PDF documents.

    Use this whenever you need information from the indexed PDFs.
    Optionally filter to a specific book (use list_books_tool to see available books).
    Provide a detailed question to get the most relevant excerpts.
    Returns fragments with text and source (book, chapter, page).
    """
    fragments = search_book(question, book=book)
    if not fragments:
        return "No relevant fragments found in the knowledge base."
    return format_fragments_for_prompt(fragments)


@mcp.tool()
def search_book_raw(question: str, book: str | None = None) -> str:
    """Search the knowledge base and return structured JSON.

    Use when you need the raw data including relevance scores.
    Optionally filter to a specific book.
    Returns JSON with text, book, chapter, page, and score.
    """
    import json
    fragments = search_book(question, book=book)
    return json.dumps(fragments, ensure_ascii=False, indent=2)


@mcp.tool()
def list_books_tool() -> str:
    """List all books currently indexed in the knowledge base.
    Use this to discover what documents are available before searching.
    """
    collections = list_collections()
    if not collections:
        return "No books in the knowledge base."
    lines = ["Available books:"]
    for c in sorted(collections):
        lines.append(f"  - {c}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()

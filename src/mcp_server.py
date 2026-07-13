#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

from src.qdrant_store import list_collections
from src.retriever import format_fragments_for_prompt, search_book

mcp = FastMCP("pdf-rag")


@mcp.tool()
def search_book_tool(question: str, book: str | None = None) -> str:
    """Search indexed PDF books for relevant text fragments using semantic similarity.

    Use this tool whenever the user asks a question that might be answered by the
    indexed book collection (e.g. "What does X say about Y?", "Summarize the chapter
    on Z", "Find information about X in the books"). Always prefer this over
    guessing or fabricating book content.

    Args:
        question: A detailed natural-language query. More specific queries yield
            better results. Example: "What are the safety protocols for chemical
            storage?" rather than just "safety".
        book: Optional book/volume name to restrict search. Use list_books_tool()
            first to discover exact names. If omitted, searches all indexed books.

    Returns: Formatted text fragments with source references (book, chapter, page).
        Each fragment includes enough context to answer the query. If nothing
        relevant is found, returns "No relevant fragments found."
    """
    fragments = search_book(question, book=book)
    if not fragments:
        return "No relevant fragments found in the knowledge base."
    return format_fragments_for_prompt(fragments)


@mcp.tool()
def search_book_raw(question: str, book: str | None = None) -> str:
    """Search indexed PDF books and return raw structured JSON with relevance scores.

    Use this instead of search_book_tool when you need machine-readable output
    with relevance scores for programmatic comparison, filtering, or ranking.
    For normal Q&A about book content, prefer search_book_tool which returns
    human-readable formatted output.

    Args:
        question: A detailed natural-language query (same as search_book_tool).
        book: Optional book/volume name to restrict search. Use list_books_tool()
            to discover available names.

    Returns: JSON array of fragments, each with keys: text, book, chapter, page,
        score (0-1, higher = more relevant). Useful for thresholding on score
        or building ranked answer lists.
    """
    import json
    fragments = search_book(question, book=book)
    return json.dumps(fragments, ensure_ascii=False, indent=2)


@mcp.tool()
def list_books_tool() -> str:
    """List all books/collections currently indexed in the knowledge base.

    Call this first to discover what documents are available before searching.
    Returns a list of book/volume names that can be used as the `book` filter
    argument in search_book_tool and search_book_raw. Always invoke this when
    the user asks about "all books", wants to know what's available, or when
    you need the exact book name string for a filtered search.
    """
    collections = list_collections()
    if not collections:
        return "No books in the knowledge base."
    lines = ["Available books:"]
    for c in sorted(collections):
        lines.append(f"  - {c}")
    return "\n".join(lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()

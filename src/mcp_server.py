#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

from src.qdrant_store import list_collections
from src.retriever import format_fragments_for_prompt, search_book
from src.trace import format_trace

mcp = FastMCP("doc-rag")


@mcp.tool()
def search_book_tool(question: str, book: str | None = None, rerank: bool | None = None) -> str:
    """Search indexed documents for relevant text fragments using semantic similarity.

    Use this tool whenever the user asks a question that might be answered by the
    indexed document collection (e.g. "What does X say about Y?", "Summarize the
    chapter on Z", "Find information about X in the documents"). Always prefer
    this over guessing or fabricating content.

    Args:
        question: A detailed natural-language query. More specific queries yield
            better results. Example: "What are the safety protocols for chemical
            storage?" rather than just "safety".
        book: Optional document/collection name to restrict search. Use
            list_books_tool() first to discover exact names. If omitted, searches
            all indexed documents.
        rerank: Whether to apply cross-encoder re-ranking for higher precision.
            If None, uses the default from config. Enable for complex queries
            where precision matters more than speed.

    Returns: Formatted text fragments with source references (book, chapter, page).
        Each fragment includes enough context to answer the query. If nothing
        relevant is found, returns "No relevant fragments found."
    """
    fragments = search_book(question, book=book, rerank=rerank)
    if not fragments:
        return "No relevant fragments found in the knowledge base."
    return format_fragments_for_prompt(fragments)


@mcp.tool()
def search_book_raw(question: str, book: str | None = None, rerank: bool | None = None) -> str:
    """Search indexed documents and return raw structured JSON with relevance scores.

    Use this instead of search_book_tool when you need machine-readable output
    with relevance scores for programmatic comparison, filtering, or ranking.
    For normal Q&A about document content, prefer search_book_tool which returns
    human-readable formatted output.

    Args:
        question: A detailed natural-language query (same as search_book_tool).
        book: Optional document/collection name to restrict search. Use
            list_books_tool() to discover available names.
        rerank: Whether to apply cross-encoder re-ranking. If None, uses config default.

    Returns: JSON array of fragments, each with keys: text, book, chapter, page,
        score (0-1, higher = more relevant), and optionally rerank_score when
        re-ranking is enabled. Useful for thresholding on score or building
        ranked answer lists.
    """
    import json

    fragments = search_book(question, book=book, rerank=rerank)
    return json.dumps(fragments, ensure_ascii=False, indent=2)


@mcp.tool()
def search_book_trace(question: str, book: str | None = None, rerank: bool | None = None) -> str:
    """Search documents with full pipeline trace showing retrieval and reranking details.

    Returns a detailed trace showing: which text chunks were retrieved by the
    bi-encoder (with cosine scores), and how the cross-encoder reranker
    reordered them (before/after rank positions with score deltas).
    Use this for debugging retrieval quality or understanding why certain
    results appear at the top.

    Args:
        question: A detailed natural-language query.
        book: Optional document/collection name to restrict search.
        rerank: Whether to apply cross-encoder re-ranking. If None, uses config default.

    Returns: Human-readable trace report with per-stage timing, retrieved
        candidates, and rerank rank changes.
    """
    result = search_book(question, book=book, rerank=rerank, trace=True)
    if not result.trace:
        return "No trace available."
    return format_trace(result.trace)


@mcp.tool()
def list_books_tool() -> str:
    """List all documents/collections currently indexed in the knowledge base.

    Call this first to discover what documents are available before searching.
    Returns a list of collection names that can be used as the `book` filter
    argument in search_book_tool and search_book_raw. Always invoke this when
    the user asks about "all books", wants to know what's available, or when
    you need the exact collection name string for a filtered search.
    """
    collections = list_collections()
    if not collections:
        return "No documents in the knowledge base."
    lines = ["Available documents:"]
    for c in sorted(collections):
        lines.append(f"  - {c}")
    return "\n".join(lines)


@mcp.tool()
def get_collection_info(collection_name: str) -> str:
    """Get details about a specific indexed collection.

    Shows the number of chunks, which documents it contains, and what
    chapters/sections are indexed.

    Args:
        collection_name: Name of the collection (use list_books_tool to discover names).

    Returns: Summary with chunk count, document names, and chapter list.
    """
    from src.qdrant_store import get_qdrant_client

    collections = list_collections()
    if collection_name not in collections:
        return f"Collection '{collection_name}' not found. Available: {sorted(collections)}"

    client = get_qdrant_client()
    count_result = client.count(collection_name=collection_name, exact=True)
    total = count_result.count if hasattr(count_result, "count") else 0

    from src.config import EMBED_DIM

    sample = client.query_points(
        collection_name=collection_name,
        query=[0.0] * EMBED_DIM,
        limit=min(total, 100),
    )

    books = set()
    chapters = set()
    for point in sample.points:
        if point.payload:
            books.add(point.payload.get("book", ""))
            chapters.add(point.payload.get("chapter", ""))

    lines = [
        f"Collection: {collection_name}",
        f"Chunks: {total}",
        f"Documents: {', '.join(sorted(books))}",
    ]
    if chapters - {""}:
        lines.append(f"Chapters: {', '.join(sorted(chapters - {''}))}")
    return "\n".join(lines)


@mcp.tool()
def delete_document(collection_name: str) -> str:
    """Delete a document/collection from the knowledge base.

    Removes the collection and all its chunks permanently. Use
    list_books_tool() first to see what's available.

    Args:
        collection_name: Name of the collection to delete.

    Returns: Confirmation message or error if not found.
    """
    from src.qdrant_store import delete_collection as _delete

    collections = list_collections()
    if collection_name not in collections:
        return f"Collection '{collection_name}' not found. Available: {sorted(collections)}"

    _delete(collection_name)
    return f"Deleted collection '{collection_name}' from the knowledge base."


@mcp.tool()
def ingest_document(file_path: str, reindex: bool = False) -> str:
    """Ingest a document into the knowledge base for semantic search.

    Supports PDF, plain text, Markdown, and source code files (Python,
    JavaScript, TypeScript, Rust, Go, Java, and many more). The document
    is chunked, embedded, and stored in a Qdrant collection.

    Args:
        file_path: Absolute or relative path to the document file on disk.
        reindex: If True, replaces any existing collection with the same name.
            If False and the document is already indexed, returns a message
            indicating it's already in the knowledge base.

    Returns: Status message with the document name, number of chunks created,
        and format type. Includes error details if the file is not found or
        the format is unsupported.
    """
    from src.adapters import get_adapter
    from src.ingest import index_document

    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        adapter = get_adapter(file_path)
    except ValueError as e:
        return f"Error: {e}"

    try:
        result = index_document(file_path, reindex=reindex)
    except Exception as e:
        return f"Error during ingestion: {e}"

    return (
        f"Successfully indexed '{result['book']}' ({adapter.format_name}).\n"
        f"Chunks: {len(result['chunks'])}, Format: {adapter.format_name}"
    )


@mcp.tool()
def ingest_folder(directory: str, reindex: bool = False) -> str:
    """Ingest all supported documents from a directory into the knowledge base.

    Scans the directory for files with supported extensions (PDF, Markdown,
    source code, plain text) and indexes each one. Skips already-indexed
    documents unless reindex is True.

    Args:
        directory: Absolute or relative path to the directory on disk.
        reindex: If True, re-indexes all documents (deletes existing collections
            first). If False, skips documents that are already indexed.

    Returns: Summary with counts of indexed, skipped, and errored documents,
        plus per-file details.
    """
    import os

    from src.ingest import ingest_folder as _ingest_folder

    if not os.path.isdir(directory):
        return f"Error: Not a directory: {directory}"

    try:
        results = _ingest_folder(directory, reindex=reindex)
    except Exception as e:
        return f"Error during folder ingestion: {e}"

    if not results:
        return f"No supported files found in {directory}"

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")

    lines = [f"Folder ingestion complete: {indexed} indexed, {skipped} skipped, {errors} errors"]
    for r in results:
        if r["status"] == "indexed":
            lines.append(f"  + {r['name']} ({r['chunks']} chunks)")
        elif r["status"] == "skipped":
            lines.append(f"  - {r['name']} (already indexed)")
        else:
            lines.append(f"  ! {r['name']}: {r.get('error', 'unknown error')}")
    return "\n".join(lines)


def main():
    mcp.run()


if __name__ == "__main__":
    main()

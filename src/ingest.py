"""Ingestion pipeline: orchestrate extraction, chunking, and indexing."""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chapter_detection import ChapterDetector
from src.chunking import chunk_text
from src.config import BOOKS_DIR, EXTRACTED_DIR, collection_name
from src.embeddings import embed
from src.extraction import extract_pdf, get_full_text_with_page_info
from src.qdrant_store import (
    delete_collection,
    ensure_collection,
    get_qdrant_client,
    list_collections,
)


def process_book(pdf_path: str) -> dict:
    """Extract, chunk, and annotate a single PDF book.

    Reads the PDF, splits into chunks with page tracking, detects chapter
    headings via structural metadata (TOC, font analysis) or regex fallback,
    and saves raw extracted text to disk.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with 'book' name, 'chunks' list, and 'total_pages' count.
    """
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"  Processing: {book_name}")

    pages_data = extract_pdf(pdf_path)
    full_text, page_boundaries = get_full_text_with_page_info(pages_data)
    page_nums = [p["page_num"] for p in pages_data]

    print(f"    Pages: {len(pages_data)}, text length: {len(full_text)} chars")

    chunks = chunk_text(full_text, page_boundaries, page_nums)

    with ChapterDetector(pdf_path) as detector:
        strategy = detector.detect_strategy()
        print(f"    Chapter detection: {strategy} strategy")

        result_chunks = []
        for chunk in chunks:
            chapter = detector.get_chapter_for_page(chunk["start_page"])
            result_chunks.append({
                "text": chunk["text"],
                "book": book_name,
                "chapter": chapter or "unknown",
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
            })

    extracted_path = os.path.join(EXTRACTED_DIR, f"{book_name}.txt")
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"    Chunks created: {len(result_chunks)}")
    return {"book": book_name, "chunks": result_chunks, "total_pages": len(pages_data)}


def index_book(pdf_path: str, reindex: bool = False):
    """Process a PDF and upsert its chunks into a Qdrant collection.

    Creates the collection if needed, generates embeddings for all chunks,
    and stores them in batches of 500.

    Args:
        pdf_path: Path to the PDF file.
        reindex: If True, delete existing collection before re-indexing.

    Returns:
        Dict with 'book', 'chunks', and 'total_pages' from process_book().
    """
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    coll = collection_name(book_name)

    qdrant = get_qdrant_client()

    if reindex:
        if coll in list_collections(qdrant):
            delete_collection(coll, qdrant)

    ensure_collection(coll, qdrant)

    result = process_book(pdf_path)
    chunks = result["chunks"]

    if not chunks:
        print(f"  No chunks to index for '{book_name}'.")
        return result

    print(f"  Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    print(f"  Storing in Qdrant collection '{coll}'...")
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append({
            "id": i + 1,
            "vector": vector,
            "payload": {
                "text": chunk["text"],
                "book": chunk["book"],
                "chapter": chunk["chapter"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
            },
        })

    batch_size = 500
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        qdrant.upsert(
            collection_name=coll,
            points=batch,
        )
        print(f"    Upserted {start + len(batch)}/{len(points)} points")

    print(f"  Done! Indexed {len(chunks)} chunks into '{coll}'.")
    return result


def ingest_all(reindex: str | None = None, book: str | None = None):
    """Index PDFs in the books/ directory.

    Skips already-indexed books unless reindex is specified.
    Can target a specific book with the book parameter.

    Args:
        reindex: If provided, only re-index this specific book name.
        book: If provided, only index this specific book name.
    """
    pdf_files = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.pdf")))
    if not pdf_files:
        print("No PDF files found in books/ directory.")
        return

    qdrant = get_qdrant_client()

    if reindex:
        pdf_path = os.path.join(BOOKS_DIR, f"{reindex}.pdf")
        if not os.path.exists(pdf_path):
            possible = [os.path.splitext(os.path.basename(p))[0] for p in pdf_files]
            print(f"Book '{reindex}' not found. Available: {possible}")
            return
        print(f"Re-indexing: {reindex}")
        index_book(pdf_path, reindex=True)
        return

    if book:
        pdf_path = os.path.join(BOOKS_DIR, f"{book}.pdf")
        if not os.path.exists(pdf_path):
            possible = [os.path.splitext(os.path.basename(p))[0] for p in pdf_files]
            print(f"Book '{book}' not found. Available: {possible}")
            return
        coll = collection_name(book)
        existing = list_collections(qdrant)
        if coll in existing:
            print(f"Book '{book}' is already indexed. Use --reindex to re-index.")
            return
        print(f"Indexing: {book}")
        index_book(pdf_path, reindex=False)
        return

    existing_collections = set(list_collections(qdrant))

    all_results = []
    for pdf_path in pdf_files:
        book_name = os.path.splitext(os.path.basename(pdf_path))[0]
        coll = collection_name(book_name)
        if coll in existing_collections:
            print(f"\nSkipping: {os.path.basename(pdf_path)} (already indexed)")
            continue
        print(f"\nIndexing: {os.path.basename(pdf_path)}")
        result = index_book(pdf_path, reindex=False)
        all_results.append(result)

    total = sum(r["total_pages"] for r in all_results)
    print(f"\nDone! Processed {len(all_results)} books, {total} total pages.")


def delete_book(book_name: str):
    """Delete a book's collection from the Qdrant knowledge base.

    Args:
        book_name: Name of the book to remove (filename without extension).
    """
    coll = collection_name(book_name)
    qdrant = get_qdrant_client()
    collections = list_collections(qdrant)

    if coll not in collections:
        possible = [c for c in collections if c != "_point_vector"]  # skip internal
        print(f"Collection '{coll}' not found. Available: {possible}")
        return

    delete_collection(coll, qdrant)
    print(f"Book '{book_name}' removed from knowledge base.")


def list_books():
    """Print all indexed collections with their chunk counts."""
    qdrant = get_qdrant_client()
    collections = list_collections(qdrant)
    if not collections:
        print("No books in the knowledge base.")
        return
    print("Books in knowledge base:")
    for c in sorted(collections):
        count_result = qdrant.count(collection_name=c, exact=True)
        total = count_result.count if hasattr(count_result, 'count') else 0
        print(f"  - {c} ({total} chunks)")


def main():
    parser = argparse.ArgumentParser(description="PDF-RAG ingestion pipeline")
    parser.add_argument("--reindex", type=str, help="Re-index a specific book by name")
    parser.add_argument("--book", type=str, help="Index a specific book by name")
    parser.add_argument("--delete", type=str, help="Delete a book from the knowledge base")
    parser.add_argument("--list", action="store_true", help="List all books in the knowledge base")
    args = parser.parse_args()

    if args.list:
        list_books()
    elif args.delete:
        delete_book(args.delete)
    else:
        ingest_all(reindex=args.reindex, book=args.book)


if __name__ == "__main__":
    main()

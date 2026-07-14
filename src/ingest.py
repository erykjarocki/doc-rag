"""Ingestion pipeline: orchestrate extraction, chunking, and indexing."""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters import get_adapter, section_for_position, supported_extensions
from src.chunking import chunk_text
from src.config import EXTRACTED_DIR
from src.embeddings import embed
from src.log import get_logger
from src.qdrant_store import (
    delete_collection,
    ensure_collection,
    get_qdrant_client,
    list_collections,
)
from src.utils import collection_name

logger = get_logger(__name__)


def process_document(file_path: str) -> dict:
    """Extract, chunk, and annotate any supported document format.

    Uses the adapter pattern to handle PDF, text, Markdown, and code files
    through a unified pipeline.

    Args:
        file_path: Path to any supported document file.

    Returns:
        Dict with 'book' name, 'chunks' list, and 'total_pages' count.
    """
    adapter = get_adapter(file_path)
    doc = adapter.extract(file_path)
    logger.info(
        "Processing document",
        extra={"file": doc.name, "format": adapter.format_name},
    )
    logger.debug(
        "Document details",
        extra={
            "sections": len(doc.sections),
            "text_length": len(doc.full_text),
            "tables": len(doc.tables),
        },
    )

    chunks = chunk_text(doc.full_text, doc.page_boundaries, doc.page_nums)

    result_chunks = []
    for chunk in chunks:
        chapter = section_for_position(doc.sections, 0, doc.full_text)
        char_offset = doc.full_text.find(chunk["text"][:50])
        if char_offset >= 0:
            chapter = section_for_position(doc.sections, char_offset, doc.full_text)

        result_chunks.append(
            {
                "text": chunk["text"],
                "book": doc.name,
                "chapter": chapter or "unknown",
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
            }
        )

    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    extracted_path = os.path.join(EXTRACTED_DIR, f"{doc.name}.txt")
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(doc.full_text)

    total_pages = len(doc.page_nums)
    logger.info(
        "Chunks created",
        extra={"file": doc.name, "chunks": len(result_chunks), "pages": total_pages},
    )
    return {"book": doc.name, "chunks": result_chunks, "total_pages": total_pages}


process_book = process_document


def index_document(file_path: str, reindex: bool = False) -> dict:
    """Process a document and upsert its chunks into a Qdrant collection.

    Works with any format supported by the adapter system.

    Args:
        file_path: Path to any supported document file.
        reindex: If True, delete existing collection before re-indexing.

    Returns:
        Dict with 'book', 'chunks', and 'total_pages' from process_document().
    """
    adapter = get_adapter(file_path)
    doc = adapter.extract(file_path)
    book_name = doc.name
    coll = collection_name(book_name)

    qdrant = get_qdrant_client()

    if reindex:
        if coll in list_collections(qdrant):
            delete_collection(coll, qdrant)

    ensure_collection(coll, qdrant)

    result = process_document(file_path)
    chunks = result["chunks"]

    if not chunks:
        logger.warning("No chunks to index for '%s'", book_name, extra={"book": book_name})
        return result

    logger.info(
        "Generating embeddings",
        extra={"book": book_name, "chunk_count": len(chunks)},
    )
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    logger.info("Storing in Qdrant", extra={"book": book_name, "collection": coll})
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            {
                "id": i + 1,
                "vector": vector,
                "payload": {
                    "text": chunk["text"],
                    "book": chunk["book"],
                    "chapter": chunk["chapter"],
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                },
            }
        )

    batch_size = 500
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        qdrant.upsert(
            collection_name=coll,
            points=batch,
        )
        logger.debug(
            "Upserted batch",
            extra={"collection": coll, "upserted": start + len(batch), "total": len(points)},
        )

    logger.info(
        "Indexing complete",
        extra={"book": book_name, "collection": coll, "chunks": len(chunks)},
    )
    return result


index_book = index_document


def ingest_folder(directory: str, reindex: bool = False) -> list[dict]:
    """Index all supported documents from a directory.

    Scans the directory for files with supported extensions and indexes each.
    Skips already-indexed documents unless reindex is True.

    Args:
        directory: Path to directory containing documents.
        reindex: If True, re-index all documents (delete + re-create collections).

    Returns:
        List of result dicts with 'name', 'status', and optional 'chunks'/'error'.
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")

    all_files: list[str] = []
    for ext in supported_extensions():
        all_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    all_files.sort()

    if not all_files:
        logger.info("No supported files found in %s", directory)
        return []

    qdrant = get_qdrant_client()
    existing = set(list_collections(qdrant))
    results = []

    for file_path in all_files:
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        coll = collection_name(doc_name)

        if not reindex and coll in existing:
            logger.info(
                "Skipping already-indexed file",
                extra={"file": os.path.basename(file_path)},
            )
            results.append({"name": doc_name, "status": "skipped"})
            continue

        try:
            logger.info("Indexing file", extra={"file": os.path.basename(file_path)})
            result = index_document(file_path, reindex=reindex)
            results.append({
                "name": doc_name,
                "status": "indexed",
                "chunks": len(result["chunks"]),
            })
        except Exception as e:
            logger.error(
                "Error indexing %s: %s", os.path.basename(file_path), e,
                extra={"file": os.path.basename(file_path)},
            )
            results.append({"name": doc_name, "status": "error", "error": str(e)})

    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    logger.info(
        "Folder ingestion complete",
        extra={"directory": directory, "indexed": indexed, "skipped": skipped, "errors": errors},
    )
    return results


def delete_book(book_name: str):
    """Delete a document's collection from the Qdrant knowledge base.

    Args:
        book_name: Name of the document to remove (filename without extension).
    """
    coll = collection_name(book_name)
    qdrant = get_qdrant_client()
    collections = list_collections(qdrant)

    if coll not in collections:
        possible = [c for c in collections if c != "_point_vector"]
        logger.warning(
            "Collection '%s' not found. Available: %s",
            coll,
            possible,
            extra={"collection": coll},
        )
        return

    delete_collection(coll, qdrant)
    logger.info("Document '%s' removed from knowledge base", book_name, extra={"book": book_name})


def list_books():
    """Print all indexed collections with their chunk counts."""
    qdrant = get_qdrant_client()
    collections = list_collections(qdrant)
    if not collections:
        logger.info("No documents in the knowledge base")
        return
    logger.info("Documents in knowledge base:")
    for c in sorted(collections):
        count_result = qdrant.count(collection_name=c, exact=True)
        total = count_result.count if hasattr(count_result, "count") else 0
        logger.info("  - %s (%d chunks)", c, total, extra={"collection": c, "chunks": total})


def main():
    parser = argparse.ArgumentParser(description="DOC-RAG ingestion pipeline")
    parser.add_argument("file", nargs="?", help="Path to a document file to index")
    parser.add_argument("--reindex", action="store_true", help="Re-index the given file")
    parser.add_argument("--folder", type=str, help="Index all supported files in a directory")
    parser.add_argument(
        "--delete", type=str, help="Delete a document from the knowledge base"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all documents in the knowledge base"
    )
    args = parser.parse_args()

    if args.list:
        list_books()
    elif args.delete:
        delete_book(args.delete)
    elif args.folder:
        ingest_folder(args.folder, reindex=args.reindex)
    elif args.file:
        if not os.path.exists(args.file):
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        index_document(args.file, reindex=args.reindex)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
